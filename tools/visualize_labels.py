import os
import argparse
import numpy as np
from PIL import Image

def colorize_label(label_np):
    """
    Mappa i valori interi 0, 1, 2, 3 in colori RGB ben visibili:
    0 (Legante)   -> Nero       [0, 0, 0]
    1 (Porosità)  -> Rosso      [255, 0, 0]
    2 (Aggregati) -> Verde      [0, 255, 0]
    3 (Ignore)    -> Blu scuro  [0, 0, 180]
    """
    h, w = label_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    rgb[label_np == 0] = [0, 0, 0]        # Legante (Nero)
    rgb[label_np == 1] = [255, 0, 0]      # Porosità (Rosso)
    rgb[label_np == 2] = [0, 255, 0]      # Aggregati (Verde)
    rgb[label_np == 3] = [0, 0, 180]      # Ignore / Padding (Blu)

    return rgb

def overlay_mask(img_rgb, mask_rgb, alpha=0.5):
    """Sovrappone la maschera colorata all'immagine originale con trasparenza alpha."""
    blended = (img_rgb.astype(np.float32) * (1 - alpha) + mask_rgb.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Visualizzatore a colori delle etichette del dataset malte")
    parser.add_argument("--data_dir", type=str, default="data/mortars", help="Cartella radice del dataset (default: data/mortars)")
    parser.add_argument("--output_dir", type=str, default="data/mortars_visualized", help="Cartella dove salvare le anteprime a colori")
    parser.add_argument("--num_samples", type=int, default=20, help="Numero di patch da visualizzare (default: 20)")
    args = parser.parse_args()

    label_dir = os.path.join(args.data_dir, "label")
    par_dir = os.path.join(args.data_dir, "paralleli")
    inc_dir = os.path.join(args.data_dir, "incrociati")

    if not os.path.exists(label_dir):
        print(f" Cartella label non trovata in {label_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".tif") or f.endswith(".png")])

    print(f" Trovate {len(label_files)} label. Generazione anteprime a colori in corso...")

    # Seleziona campioni distribuite uniformemente nel dataset
    step = max(1, len(label_files) // args.num_samples)
    sampled_files = label_files[::step][:args.num_samples]

    for fname in sampled_files:
        lbl_path = os.path.join(label_dir, fname)
        par_path = os.path.join(par_dir, fname)
        inc_path = os.path.join(inc_dir, fname)

        label_np = np.array(Image.open(lbl_path))
        colored_mask = colorize_label(label_np)

        out_name = os.path.splitext(fname)[0] + "_colored.png"
        Image.fromarray(colored_mask).save(os.path.join(args.output_dir, out_name))

        # Se esistono le immagini paralleli/incrociati, crea anche l'overlay 50% trasparente
        if os.path.exists(par_path) and os.path.exists(inc_path):
            par_np = np.array(Image.open(par_path).convert("RGB"))
            inc_np = np.array(Image.open(inc_path).convert("RGB"))

            overlay_par = overlay_mask(par_np, colored_mask, alpha=0.4)
            overlay_inc = overlay_mask(inc_np, colored_mask, alpha=0.4)

            # Salva affiancati: [Paralleli Overlay | Incrociati Overlay | Maschera Colori]
            combined = np.hstack([par_np, overlay_par, colored_mask])
            combo_name = os.path.splitext(fname)[0] + "_preview_combo.png"
            Image.fromarray(combined).save(os.path.join(args.output_dir, combo_name))

    print(f"\n Visualizzazione completata! {len(sampled_files)} anteprime salvate in:")
    print(f"   📂 {os.path.abspath(args.output_dir)}")
    print(f"\n Legenda colori applicata:")
    print(f"   ⬛ Legante (Classe 0)")
    print(f"   🟥 Porosità (Classe 1)")
    print(f"   🟩 Aggregati (Classe 2)")
    print(f"   🟦 Ignore / Padding (Classe 3)")

if __name__ == "__main__":
    main()
