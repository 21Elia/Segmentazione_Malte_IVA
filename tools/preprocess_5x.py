#!/usr/bin/env python3
"""
Mortar Dataset Pre-Processing and Alignment Script for 5X Images.

Workflow:
1. Loads NP and NX 5X gigapixel images and optionally rescales to 1X equivalent (scale=0.20) with cv2.INTER_AREA.
2. Loads corresponding 1:1 validity mask (0=Mortar, 255=Background) and auto-adapts mask resolution.
3. Pads NP and NX to identical dimensions using centered symmetric padding (Notari method).
4. Extracts grid patches (512x512) strictly within the valid mortar region.
5. Aligns NP relative to NX patch-by-patch via Phase Cross-Correlation.
6. Generates PRE and POST alignment false-color RGB preview samples (R=NP, G=NX, B=0).
7. Saves aligned patches to data/patches_5x/<section_name>/(paralleli|incrociati)
   and serializes grid_metadata.json for section reconstruction.
"""

import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "-8"

import sys
import glob
import json
import time
import argparse
import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_OFF)
except Exception:
    pass

import numpy as np
from PIL import Image
from skimage.registration import phase_cross_correlation

Image.MAX_IMAGE_PIXELS = None


def get_5x_image_pairs(src_dir):
    """
    Locates matching 5X image pairs (Parallel and Crossed Nicols) in the source directory.
    """
    pairs = []
    all_files = glob.glob(os.path.join(src_dir, "*.*"))
    
    np_files = [f for f in all_files if 'paralleli' in os.path.basename(f).lower() and f.lower().endswith(('.jpg', '.jpeg', '.tif', '.png'))]
    nx_files = [f for f in all_files if 'incrociati' in os.path.basename(f).lower() and f.lower().endswith(('.jpg', '.jpeg', '.tif', '.png'))]
    
    for np_path in np_files:
        np_base = os.path.basename(np_path)
        prefix = np_base.lower().split('nicols')[0].strip().replace(' ', '_').rstrip('_')
        if not prefix:
            prefix = "ARCHEO_02"
        
        matching_nx = [nx for nx in nx_files if os.path.basename(nx).lower().split('nicols')[0].strip().replace(' ', '_').rstrip('_') == prefix]
        if matching_nx:
            pairs.append({
                'name': prefix.upper(),
                'np': np_path,
                'nx': matching_nx[0]
            })
            
    return pairs


def load_and_rescale_5x(image_path, scale=0.20):
    """
    Loads native 5X image and rescales via cv2.INTER_AREA if scale != 1.0.
    """
    print(f"Loading: {image_path} ...", flush=True)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image file: {image_path}")
        
    h_orig, w_orig = img.shape[:2]
    if abs(scale - 1.0) < 1e-4:
        print(f"Native 5X resolution (S=1.0): {w_orig} x {h_orig} px ({w_orig*h_orig/1e6:.1f} MP)", flush=True)
        return img

    target_w = int(round(w_orig * scale))
    target_h = int(round(h_orig * scale))
    print(f"Native 5X dimensions: {w_orig} x {h_orig} px -> Rescaling (S={scale}): {target_w} x {target_h} px", flush=True)
    img_rescaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return img_rescaled


def pad_image_to_size(img, target_h, target_w, fill_value=0):
    """
    Applies centered symmetric padding (Notari method) to match target dimensions.
    """
    h, w = img.shape[:2]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    if pad_h == 0 and pad_w == 0:
        return img

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if len(img.shape) == 3:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))

    return np.pad(img, padding, mode='constant', constant_values=fill_value)


def is_valid_patch_position(validity_mask, y, x, patch_size=512):
    """
    Verifies if patch at (y, x) is valid (no background pixels > 127 within window).
    """
    h, w = validity_mask.shape[:2]
    if y + patch_size > h or x + patch_size > w:
        return False

    patch_region = validity_mask[y:y + patch_size, x:x + patch_size]
    return bool(np.sum(patch_region > 127) == 0)


def create_false_color_rgb(img_np, img_nx):
    """
    Creates false-color RGB preview:
      - Red channel (BGR index 2) = Parallel Nicols grayscale
      - Green channel (BGR index 1) = Crossed Nicols grayscale
      - Blue channel (BGR index 0) = 0
    """
    np_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) if len(img_np.shape) == 3 else img_np
    nx_gray = cv2.cvtColor(img_nx, cv2.COLOR_BGR2GRAY) if len(img_nx.shape) == 3 else img_nx

    false_color = np.zeros((np_gray.shape[0], np_gray.shape[1], 3), dtype=np.uint8)
    false_color[:, :, 2] = np_gray  # Red = NP
    false_color[:, :, 1] = nx_gray  # Green = NX
    false_color[:, :, 0] = 0        # Blue = 0
    return false_color


def compute_global_shift(img_np, img_nx, max_dim=2048):
    """
    Computes global rigid translation (shift_y, shift_x) to register NP onto NX
    using multi-scale Phase Cross-Correlation on downsampled previews.
    """
    h, w = img_np.shape[:2]
    downscale = min(1.0, max_dim / max(h, w))

    np_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) if len(img_np.shape) == 3 else img_np
    nx_gray = cv2.cvtColor(img_nx, cv2.COLOR_BGR2GRAY) if len(img_nx.shape) == 3 else img_nx

    small_np = cv2.resize(np_gray, (0, 0), fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    small_nx = cv2.resize(nx_gray, (0, 0), fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)

    min_h = min(small_np.shape[0], small_nx.shape[0])
    min_w = min(small_np.shape[1], small_nx.shape[1])

    try:
        shift_values, error, _ = phase_cross_correlation(small_nx[:min_h, :min_w], small_np[:min_h, :min_w])
        global_shift_y = int(round(shift_values[0] / downscale))
        global_shift_x = int(round(shift_values[1] / downscale))
    except Exception as e:
        print(f"[Warning] Global phase cross-correlation failed: {e}. Defaulting to (0, 0).")
        global_shift_y, global_shift_x = 0, 0

    return global_shift_y, global_shift_x


def apply_global_shift(img, shift_y, shift_x, fill_value=0):
    """
    Applies global rigid 2D translation to an image using warpAffine.
    """
    if shift_y == 0 and shift_x == 0:
        return img
    h, w = img.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill_value)
    return shifted


def align_patch_pair(p_np, p_nx, patch_size=512, max_shift=50):
    """
    Aligns NP relative to NX patch using Phase Cross Correlation,
    crops the overlapping region, and restores 512x512 with symmetric zero padding.
    Identical methodology to preprocess_new_images.py (1_SCALA and UNITO_B).
    """
    np_gray = cv2.cvtColor(p_np, cv2.COLOR_BGR2GRAY) if len(p_np.shape) == 3 else p_np
    nx_gray = cv2.cvtColor(p_nx, cv2.COLOR_BGR2GRAY) if len(p_nx.shape) == 3 else p_nx

    try:
        shift_values, error, _ = phase_cross_correlation(nx_gray, np_gray)
        shift_y, shift_x = shift_values
    except Exception:
        shift_y, shift_x = 0.0, 0.0

    shift_x_int = int(round(shift_x))
    shift_y_int = int(round(shift_y))

    # Safety check for excessive shifts
    if abs(shift_x_int) > max_shift or abs(shift_y_int) > max_shift:
        shift_x_int, shift_y_int = 0, 0

    h, w = patch_size, patch_size

    np_left = max(0, -shift_x_int)
    np_top = max(0, -shift_y_int)
    np_right = w - max(0, shift_x_int)
    np_bottom = h - max(0, shift_y_int)

    nx_left = max(0, shift_x_int)
    nx_top = max(0, shift_y_int)
    nx_right = w - max(0, -shift_x_int)
    nx_bottom = h - max(0, -shift_y_int)

    if (np_right <= np_left or np_bottom <= np_top or nx_right <= nx_left or nx_bottom <= nx_top):
        np_left, np_top, np_right, np_bottom = 0, 0, w, h
        nx_left, nx_top, nx_right, nx_bottom = 0, 0, w, h

    crop_np = p_np[np_top:np_bottom, np_left:np_right]
    crop_nx = p_nx[nx_top:nx_bottom, nx_left:nx_right]

    crop_h, crop_w = crop_np.shape[:2]
    pad_h = max(0, patch_size - crop_h)
    pad_w = max(0, patch_size - crop_w)

    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        out_np = np.pad(crop_np, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
        out_nx = np.pad(crop_nx, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
    else:
        out_np = crop_np
        out_nx = crop_nx

    return out_np, out_nx, (shift_x_int, shift_y_int)


def process_5x_section(pair_info, mask_path, dst_dir, samples_dir, scale=0.20, patch_size=512, grid_size=512):
    """
    Processes full 5X section generating aligned patch dataset and metadata.
    """
    sec_name = pair_info['name']
    print(f"\n{'=' * 60}", flush=True)
    print(f"PREPROCESSING 5X SECTION: {sec_name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # 1. Load and optionally rescale NP and NX
    img_np = load_and_rescale_5x(pair_info['np'], scale=scale)
    img_nx = load_and_rescale_5x(pair_info['nx'], scale=scale)
    
    # 2. Load validity mask
    print(f"Loading validity mask: {mask_path} ...", flush=True)
    img_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img_mask is None:
        raise FileNotFoundError(f"Validity mask not found: {mask_path}")

    max_h = max(img_np.shape[0], img_nx.shape[0])
    max_w = max(img_np.shape[1], img_nx.shape[1])

    # If mask resolution differs from target dimensions, adapt using nearest neighbor
    if img_mask.shape[:2] != (max_h, max_w):
        print(f"Adapting mask resolution from {img_mask.shape[1]}x{img_mask.shape[0]} to {max_w}x{max_h} px (INTER_NEAREST)...", flush=True)
        img_mask = cv2.resize(img_mask, (max_w, max_h), interpolation=cv2.INTER_NEAREST)

    # Equalize image dimensions via centered padding
    img_np = pad_image_to_size(img_np, max_h, max_w, 0)
    img_nx = pad_image_to_size(img_nx, max_h, max_w, 0)
    img_mask = pad_image_to_size(img_mask, max_h, max_w, 255)

    # 3. Global coarse co-registration between NP and NX
    print("Computing global rigid alignment between NP and NX (Coarse stage)...", flush=True)
    global_sy, global_sx = compute_global_shift(img_np, img_nx)
    print(f"Global shift detected: delta_y = {global_sy:+d} px, delta_x = {global_sx:+d} px", flush=True)
    if global_sy != 0 or global_sx != 0:
        print(f"Applying global translation to NP image...", flush=True)
        img_np = apply_global_shift(img_np, global_sy, global_sx, fill_value=0)

    dir_paralleli = os.path.join(dst_dir, 'paralleli')
    dir_incrociati = os.path.join(dst_dir, 'incrociati')
    os.makedirs(dir_paralleli, exist_ok=True)
    os.makedirs(dir_incrociati, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    grid_rows = max_h // grid_size
    grid_cols = max_w // grid_size

    print(f"Target grid dimensions : {max_h} x {max_w} px", flush=True)
    print(f"Grid scan layout       : {grid_rows} rows x {grid_cols} columns ({grid_rows*grid_cols} total cells)", flush=True)

    generated_patches = 0
    grid_metadata = {
        'section_name': sec_name,
        'scale_factor': scale,
        'image_height': max_h,
        'image_width': max_w,
        'global_shift_y': global_sy,
        'global_shift_x': global_sx,
        'patch_size': patch_size,
        'grid_size': grid_size,
        'grid_rows': grid_rows,
        'grid_cols': grid_cols,
        'patches': []
    }

    for row in range(grid_rows):
        for col in range(grid_cols):
            y = row * grid_size
            x = col * grid_size

            if is_valid_patch_position(img_mask, y, x, patch_size):
                p_np = img_np[y:y + patch_size, x:x + patch_size]
                p_nx = img_nx[y:y + patch_size, x:x + patch_size]

                # PRE-alignment preview
                pre_align_rgb = create_false_color_rgb(p_np, p_nx)

                # Local Phase Correlation alignment
                out_np, out_nx, (shift_x, shift_y) = align_patch_pair(p_np, p_nx, patch_size=patch_size)

                # POST-alignment preview
                post_align_rgb = create_false_color_rgb(out_np, out_nx)

                patch_stem = f"sec_{sec_name}_p{generated_patches:04d}_r{row:02d}_c{col:02d}_y{y}_x{x}"
                filename_tif = f"{patch_stem}.tif"

                # Save alignment comparison previews (sampled for large datasets)
                if generated_patches < 30 or generated_patches % 25 == 0:
                    cv2.imwrite(os.path.join(samples_dir, f"{patch_stem}_PRE.png"), pre_align_rgb)
                    cv2.imwrite(os.path.join(samples_dir, f"{patch_stem}_POST.png"), post_align_rgb)

                # Save aligned TIFF patches
                cv2.imwrite(os.path.join(dir_paralleli, filename_tif), out_np)
                cv2.imwrite(os.path.join(dir_incrociati, filename_tif), out_nx)

                grid_metadata['patches'].append({
                    'patch_id': generated_patches,
                    'filename': filename_tif,
                    'row': row,
                    'col': col,
                    'y': y,
                    'x': x,
                    'shift_y': shift_y,
                    'shift_x': shift_x
                })

                generated_patches += 1
                if generated_patches % 20 == 0:
                    print(f"[{sec_name}] Generated {generated_patches} valid patches...", flush=True)

    # Save grid metadata JSON
    meta_path = os.path.join(dst_dir, 'grid_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(grid_metadata, f, indent=2)

    print(f"\nCompleted! Generated {generated_patches} valid patches.", flush=True)
    print(f"Metadata saved to: {meta_path}", flush=True)
    print(f"Alignment samples saved to: {samples_dir}", flush=True)
    return generated_patches


def main():
    parser = argparse.ArgumentParser(description="Pre-processing and Patch Alignment for 5X Images")
    parser.add_argument('--src', type=str, default='../DIA_FIRENZE/Prova immagini 5X', help="Source 5X images directory")
    parser.add_argument('--mask', type=str, default='output/inspection/ARCHEO_02_validity_mask.tif', help="Path to validity mask TIFF")
    parser.add_argument('--dst', type=str, default='data/patches_5x/ARCHEO_02', help="Output patch directory")
    parser.add_argument('--samples-dir', type=str, default='output/alignment_samples/ARCHEO_02', help="Output directory for alignment samples")
    parser.add_argument('--scale', type=float, default=0.20, help="Scale factor (default: 0.20)")
    parser.add_argument('--patch-size', type=int, default=512, help="Patch size in pixels (default: 512)")
    args = parser.parse_args()

    pairs = get_5x_image_pairs(args.src)
    if not pairs:
        print(f"No 5X pairs found in: {args.src}")
        sys.exit(1)

    for p in pairs:
        process_5x_section(
            pair_info=p,
            mask_path=args.mask,
            dst_dir=args.dst,
            samples_dir=args.samples_dir,
            scale=args.scale,
            patch_size=args.patch_size,
            grid_size=args.patch_size
        )


if __name__ == '__main__':
    main()
