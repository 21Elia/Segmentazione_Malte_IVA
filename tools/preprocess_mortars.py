import os
import glob
import argparse
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from scipy.ndimage import shift

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


def load_image(filepath, mode="RGB"):
    """Carica un'immagine usando PIL e restituisce un array NumPy."""
    img = Image.open(filepath)
    if mode == "RGB":
        return np.array(img.convert("RGB"))
    elif mode == "L":
        return np.array(img.convert("L"))
    return np.array(img)


def pad_to_shape(img, target_shape, fill_val=0):
    """Applica padding a destra e in basso per raggiungere la dimensione target_shape."""
    h, w = img.shape[:2]
    th, tw = target_shape[:2]
    dh = max(0, th - h)
    dw = max(0, tw - w)

    if dh == 0 and dw == 0:
        return img

    if img.ndim == 3:
        padding = ((0, dh), (0, dw), (0, 0))
    else:
        padding = ((0, dh), (0, dw))

    return np.pad(img, padding, mode="constant", constant_values=fill_val)


def phase_correlation_fft(img1, img2):
    """
    Calcola lo spostamento (dy, dx) tra img1 (moving) e img2 (reference) usando pure NumPy FFT.
    """
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)
    cross_power = f1 * np.conj(f2)
    cross_power /= (np.abs(cross_power) + 1e-8)
    r = np.fft.ifft2(cross_power).real

    dy, dx = np.unravel_index(np.argmax(r), r.shape)

    if dy > r.shape[0] // 2:
        dy -= r.shape[0]
    if dx > r.shape[1] // 2:
        dx -= r.shape[1]

    return dy, dx


def align_paralleli_patch(paralleli_patch, incrociati_patch):
    """
    Allinea la patch paralleli alla patch incrociati mediante Phase Correlation (FFT).
    """
    if paralleli_patch.ndim == 3:
        p_gray = np.mean(paralleli_patch, axis=2).astype(np.float32)
    else:
        p_gray = paralleli_patch.astype(np.float32)

    if incrociati_patch.ndim == 3:
        i_gray = np.mean(incrociati_patch, axis=2).astype(np.float32)
    else:
        i_gray = incrociati_patch.astype(np.float32)

    dy, dx = phase_correlation_fft(p_gray, i_gray)

    # Se lo shift è irragionevolmente grande (es. > 50px), evita traslazioni distruttive
    if abs(dy) > 50 or abs(dx) > 50 or np.isnan(dx) or np.isnan(dy):
        return paralleli_patch

    # Applica traslazione 2D a ciascun canale RGB usando scipy.ndimage.shift
    aligned = np.zeros_like(paralleli_patch)
    for c in range(3):
        aligned[:, :, c] = shift(paralleli_patch[:, :, c], shift=(dy, dx), mode='constant', cval=0)

    return aligned


def find_file(files_dict, pattern):
    """Trova un file nella cartella dando priorità assoluta alle estensioni .TIF rispetto a .JPG."""
    tif_matches = [k for k in files_dict if pattern in k and k.endswith('.TIF')]
    if tif_matches:
        return files_dict[tif_matches[0]]
    jpg_matches = [k for k in files_dict if pattern in k and k.endswith('.JPG')]
    if jpg_matches:
        return files_dict[jpg_matches[0]]
    return None


def process_section(
    folder_path,
    output_dir,
    patch_size=512,
    apply_phase_corr=True,
    exclude_images=None
):
    folder_name = os.path.basename(folder_path)

    # Esclusione immagine (es. Sezione 5 come indicato nel report per massimizzare le performance)
    if exclude_images:
        exclude_str = [str(x) for x in exclude_images]
        if folder_name in exclude_str:
            print(f"Skipping cartella {folder_name} (esclusa dalle opzioni).")
            return 0

    # Ricerca dei 5 file dando priorità ai TIF rispetto ai JPG di anteprima
    files = {f.upper(): f for f in os.listdir(folder_path)}

    par_file = find_file(files, "NICOLS PARALLELI")
    inc_file = find_file(files, "NICOLS INCROCIATI")
    agg_file = find_file(files, "AGGREGATE")
    por_file = find_file(files, "POROSITY")
    tot_file = find_file(files, "TOTAL")

    if not (par_file and inc_file and agg_file and por_file and tot_file):
        print(f" Cartella {folder_name} saltata: Mancano una o più modali/maschere.")
        return 0

    print(f"\nProcessing sezione {folder_name} (AGG: {agg_file}, TOT: {tot_file})...")

    par_img = load_image(os.path.join(folder_path, par_file), "RGB")
    inc_img = load_image(os.path.join(folder_path, inc_file), "RGB")
    agg_img = load_image(os.path.join(folder_path, agg_file), "L")
    por_img = load_image(os.path.join(folder_path, por_file), "L")
    tot_img = load_image(os.path.join(folder_path, tot_file), "L")

    # Passaggio 1: Uniformare le dimensioni
    max_h = max(par_img.shape[0], inc_img.shape[0], agg_img.shape[0], por_img.shape[0], tot_img.shape[0])
    max_w = max(par_img.shape[1], inc_img.shape[1], agg_img.shape[1], por_img.shape[1], tot_img.shape[1])

    par_img = pad_to_shape(par_img, (max_h, max_w, 3))
    inc_img = pad_to_shape(inc_img, (max_h, max_w, 3))
    agg_img = pad_to_shape(agg_img, (max_h, max_w))
    por_img = pad_to_shape(por_img, (max_h, max_w))
    tot_img = pad_to_shape(tot_img, (max_h, max_w))

    # Sottocartelle di output
    par_out = os.path.join(output_dir, "paralleli")
    inc_out = os.path.join(output_dir, "incrociati")
    lbl_out = os.path.join(output_dir, "label")

    os.makedirs(par_out, exist_ok=True)
    os.makedirs(inc_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    saved_count = 0

    # Passaggio 2: Suddivisione a griglia 512x512
    n_rows = max_h // patch_size
    n_cols = max_w // patch_size

    for r in range(n_rows):
        for c in range(n_cols):
            r_start, r_end = r * patch_size, (r + 1) * patch_size
            c_start, c_end = c * patch_size, (c + 1) * patch_size

            tot_patch = tot_img[r_start:r_end, c_start:c_end]

            # Scarta la patch se TOTAL è interamente 0 (regione non valida/sfondo)
            if np.all(tot_patch == 0):
                continue

            par_patch = par_img[r_start:r_end, c_start:c_end]
            inc_patch = inc_img[r_start:r_end, c_start:c_end]
            agg_patch = agg_img[r_start:r_end, c_start:c_end]
            por_patch = por_img[r_start:r_end, c_start:c_end]

            # Passaggio 3: Phase Correlation per allineare la patch paralleli
            if apply_phase_corr:
                par_patch = align_paralleli_patch(par_patch, inc_patch)

            # Passaggio 4: Generazione Mappa Etichette Multiclasse
            # 0: Legante (Matrix/Binder)
            # 1: Porosità
            # 2: Aggregati
            # 3: Ignore / Padding
            label_patch = np.zeros((patch_size, patch_size), dtype=np.uint8)

            # Porosità (> 0) -> Classe 1
            label_patch[por_patch > 0] = 1

            # Aggregati (> 0) -> Classe 2 (sovrascrive la porosità in caso di conflitti)
            label_patch[agg_patch > 0] = 2

            # Maschera Ignore (dove TOTAL == 0) -> Classe 3
            label_patch[tot_patch == 0] = 3

            # Salvataggio patch con nome univoco condiviso
            patch_name = f"sec{int(folder_name):02d}_r{r:02d}_c{c:02d}.tif"

            Image.fromarray(par_patch).save(os.path.join(par_out, patch_name))
            Image.fromarray(inc_patch).save(os.path.join(inc_out, patch_name))
            Image.fromarray(label_patch).save(os.path.join(lbl_out, patch_name))

            saved_count += 1

    print(f" Generati {saved_count} patch per la sezione {folder_name}.")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Preprocessing del dataset delle malte (TIF RAW -> Patch 512x512)")
    parser.add_argument("--input_dir", type=str, required=True, help="Percorso alla cartella contenente le cartelle 1..12 (es. DIA_FIRENZE)")
    parser.add_argument("--output_dir", type=str, default="data/mortars", help="Percorso di output dove salvare le patch (es. data/mortars)")
    parser.add_argument("--patch_size", type=int, default=512, help="Dimensione delle patch (default: 512)")
    parser.add_argument("--no_phase_corr", action="store_true", help="Disabilita l'allineamento tramite Phase Correlation")
    parser.add_argument("--exclude_img5", action="store_true", help="Esclude l'Immagine 5 (migliore configurazione report)")
    args = parser.parse_args()

    exclude_list = [5] if args.exclude_img5 else []

    # Cerca le sottocartelle numeriche 1..12
    section_folders = []
    for item in os.listdir(args.input_dir):
        full_path = os.path.join(args.input_dir, item)
        if os.path.isdir(full_path) and item.isdigit():
            section_folders.append((int(item), full_path))

    section_folders.sort(key=lambda x: x[0])

    if not section_folders:
        print(f" Nessuna cartella numerata trovata in {args.input_dir}")
        return

    print(f" Trovate {len(section_folders)} sezioni da processare.")
    total_patches = 0

    for sec_num, sec_path in section_folders:
        count = process_section(
            sec_path,
            args.output_dir,
            patch_size=args.patch_size,
            apply_phase_corr=not args.no_phase_corr,
            exclude_images=exclude_list
        )
        total_patches += count

    print(f"\n==========================================")
    print(f" Preprocessing completato con successo!")
    print(f" Patch totali create: {total_patches}")
    print(f" Output salvato in: {os.path.abspath(args.output_dir)}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
