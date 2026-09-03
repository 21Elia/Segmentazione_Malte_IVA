#!/usr/bin/env python3
"""
Mortar Dataset Pre-Processing Script for New Images (Nuove Immagini).

This script:
1. Locates matching NP (paralleli) and NX (incrociati) image pairs in DIA_FIRENZE/Nuove Immagini.
2. Loads corresponding global validity masks (0=Valid Mortar, 255=Background) from output/inspection/.
3. Resizes the validity mask to match full resolution if downsampled previews were used.
4. Extracts grid patches (default 512x512) for valid section regions.
5. Aligns NP relative to NX patch-by-patch using Phase Cross Correlation.
6. Generates false-color RGB preview images (R=NP, G=NX, B=0) BEFORE and AFTER alignment
   to visualize and evaluate the shift correction.
7. Saves aligned patches to data/nuove_patches/<section_name>/(paralleli|incrociati)
   along with grid_metadata.json for high-resolution reconstruction during inference.
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
from skimage.registration import phase_cross_correlation


def get_new_image_pairs(nuove_img_dir, masks_dir):
    """
    Locates matching NP and NX image pairs and their corresponding validity mask.
    """
    pairs = []
    
    # Pair 1: 1_SCALA
    np1 = os.path.join(nuove_img_dir, '1NP_SCALA.tif')
    nx1 = os.path.join(nuove_img_dir, '1NX_SCALA.tif')
    mask1_tif = os.path.join(masks_dir, '1_SCALA_validity_mask.tif')
    mask1_png = os.path.join(masks_dir, '1_SCALA_validity_mask.png')
    mask1 = mask1_tif if os.path.exists(mask1_tif) else mask1_png
    if os.path.exists(np1) and os.path.exists(nx1) and os.path.exists(mask1):
        pairs.append({
            'name': '1_SCALA',
            'np': np1,
            'nx': nx1,
            'mask': mask1
        })
        
    # Pair 2: UNITO_B
    np2 = os.path.join(nuove_img_dir, 'NP_UNITO_BW_B.tif')
    nx2 = os.path.join(nuove_img_dir, 'NX_UNITO_B.tif')
    mask2_tif = os.path.join(masks_dir, 'UNITO_B_validity_mask.tif')
    mask2_png = os.path.join(masks_dir, 'UNITO_B_validity_mask.png')
    mask2 = mask2_tif if os.path.exists(mask2_tif) else mask2_png
    if os.path.exists(np2) and os.path.exists(nx2) and os.path.exists(mask2):
        pairs.append({
            'name': 'UNITO_B',
            'np': np2,
            'nx': nx2,
            'mask': mask2
        })

    return pairs


def pad_image_to_size(img, target_h, target_w, fill_value=0):
    """
    Pads image to target_h, target_w using centered padding (matching Notari's pad_image method).
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
    Checks if a patch extracted at (y, x) is valid (no background/white pixels > 127 in validity_mask).
    """
    h, w = validity_mask.shape[:2]
    if y + patch_size > h or x + patch_size > w:
        return False

    patch_region = validity_mask[y:y + patch_size, x:x + patch_size]
    return bool(np.sum(patch_region > 127) == 0)


def create_false_color_rgb(img_np, img_nx):
    """
    Creates a false-color RGB preview to visualize alignment:
    - Red Channel (index 2 in BGR) = NP grayscale
    - Green Channel (index 1 in BGR) = NX grayscale
    - Blue Channel (index 0 in BGR) = 0
    """
    np_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) if len(img_np.shape) == 3 else img_np
    nx_gray = cv2.cvtColor(img_nx, cv2.COLOR_BGR2GRAY) if len(img_nx.shape) == 3 else img_nx

    false_color = np.zeros((np_gray.shape[0], np_gray.shape[1], 3), dtype=np.uint8)
    false_color[:, :, 2] = np_gray  # Red = NP
    false_color[:, :, 1] = nx_gray  # Green = NX
    false_color[:, :, 0] = 0        # Blue = 0
    return false_color


def align_patch_pair(p_np, p_nx, patch_size=512, max_shift=50):
    """
    Aligns p_np (paralleli) relative to p_nx (incrociati) using Phase Cross Correlation,
    crops both patches to the overlapping region, and pads back to (patch_size, patch_size).
    Returns (out_np, out_nx, (shift_x, shift_y)).
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


def process_new_section(pair_info, dst_dir, samples_dir, patch_size=512, grid_size=512):
    """
    Processes a new section pair (NP, NX, validity_mask).
    """
    sec_name = pair_info['name']
    print(f"\n{'=' * 60}", flush=True)
    print(f"PROCESSING NEW SECTION: {sec_name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    print("Loading NP, NX, and validity mask images...", flush=True)
    img_np = cv2.imread(pair_info['np'])
    img_nx = cv2.imread(pair_info['nx'])
    img_mask = cv2.imread(pair_info['mask'], cv2.IMREAD_GRAYSCALE)

    if any(x is None for x in [img_np, img_nx, img_mask]):
        print(f"Error: Could not load images for section {sec_name}", flush=True)
        return 0

    max_h = max(img_np.shape[0], img_nx.shape[0])
    max_w = max(img_np.shape[1], img_nx.shape[1])

    # Pad NP and NX if dimensions differ slightly
    img_np = pad_image_to_size(img_np, max_h, max_w, 0)
    img_nx = pad_image_to_size(img_nx, max_h, max_w, 0)

    # Resize validity mask to match full image dimensions if it was generated at downsampled resolution
    if img_mask.shape[:2] != (max_h, max_w):
        print(f"Resizing validity mask from {img_mask.shape[1]}x{img_mask.shape[0]} to full resolution {max_w}x{max_h}...", flush=True)
        img_mask = cv2.resize(img_mask, (max_w, max_h), interpolation=cv2.INTER_NEAREST)

    dir_paralleli = os.path.join(dst_dir, sec_name, 'paralleli')
    dir_incrociati = os.path.join(dst_dir, sec_name, 'incrociati')
    sec_samples_dir = os.path.join(samples_dir, sec_name)

    os.makedirs(dir_paralleli, exist_ok=True)
    os.makedirs(dir_incrociati, exist_ok=True)
    os.makedirs(sec_samples_dir, exist_ok=True)

    grid_rows = max_h // grid_size
    grid_cols = max_w // grid_size

    print(f"Full dimensions : {max_h}x{max_w}", flush=True)
    print(f"Grid dimensions : {grid_rows} rows x {grid_cols} columns", flush=True)

    generated_patches = 0
    grid_metadata = {
        'section_name': sec_name,
        'image_height': max_h,
        'image_width': max_w,
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

                # 1. False-color PRE-alignment preview
                pre_align_rgb = create_false_color_rgb(p_np, p_nx)

                # 2. Local alignment via Phase Correlation
                out_np, out_nx, (shift_x, shift_y) = align_patch_pair(p_np, p_nx, patch_size=patch_size)

                # 3. False-color POST-alignment preview
                post_align_rgb = create_false_color_rgb(out_np, out_nx)

                patch_stem = f"sec_{sec_name}_p{generated_patches:04d}_r{row}_c{col}_y{y}_x{x}"
                filename_tif = f"{patch_stem}.tif"

                # Save alignment comparison previews
                cv2.imwrite(os.path.join(sec_samples_dir, f"{patch_stem}_PRE.png"), pre_align_rgb)
                cv2.imwrite(os.path.join(sec_samples_dir, f"{patch_stem}_POST.png"), post_align_rgb)

                # Save aligned patches for dataset
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
                if generated_patches % 10 == 0:
                    print(f"[{sec_name}] Generated {generated_patches} valid patches...", flush=True)

    # Save grid metadata JSON
    meta_path = os.path.join(dst_dir, sec_name, 'grid_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(grid_metadata, f, indent=2)

    print(f"Done section {sec_name}: generated {generated_patches} valid patches.", flush=True)
    print(f"Metadata saved to: {meta_path}", flush=True)
    print(f"Alignment samples saved to: {sec_samples_dir}", flush=True)
    return generated_patches


def main():
    parser = argparse.ArgumentParser(description="Pre-process New Mortar Images for MMSFormer")
    parser.add_argument('--src', type=str, default='../DIA_FIRENZE/Nuove Immagini', help="Path to Nuove Immagini folder")
    parser.add_argument('--masks-dir', type=str, default='output/inspection', help="Path to validity masks folder")
    parser.add_argument('--dst', type=str, default='data/nuove_patches', help="Output directory for processed patches")
    parser.add_argument('--samples-dir', type=str, default='output/alignment_samples', help="Output directory for alignment samples")
    parser.add_argument('--patch-size', type=int, default=512, help="Patch size in pixels")
    parser.add_argument('--grid-size', type=int, default=512, help="Grid step in pixels")
    parser.add_argument('--section', type=str, default=None, help="Filter to process only a specific section (e.g. 'UNITO_B' or '1_SCALA')")
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src)
    masks_dir = os.path.abspath(args.masks_dir)
    dst_dir = os.path.abspath(args.dst)
    samples_dir = os.path.abspath(args.samples_dir)

    print("=" * 70, flush=True)
    print("NEW MORTAR IMAGES PRE-PROCESSING & ALIGNMENT", flush=True)
    print(f"Source Directory    : {src_dir}", flush=True)
    print(f"Masks Directory     : {masks_dir}", flush=True)
    print(f"Output Dataset Dir  : {dst_dir}", flush=True)
    print(f"Alignment Samples   : {samples_dir}", flush=True)
    print(f"Patch Size          : {args.patch_size}x{args.patch_size}", flush=True)
    print(f"Grid Size           : {args.grid_size}x{args.grid_size}", flush=True)
    if args.section:
        print(f"Target Section Filter: {args.section}", flush=True)
    print("=" * 70, flush=True)

    pairs = get_new_image_pairs(src_dir, masks_dir)
    if not pairs:
        print("Error: No matching NP, NX and validity_mask pairs found!", flush=True)
        sys.exit(1)

    if args.section:
        pairs = [p for p in pairs if p['name'].lower() == args.section.lower()]
        if not pairs:
            print(f"Error: Section '{args.section}' not found in available pairs! Available: {[p['name'] for p in get_new_image_pairs(src_dir, masks_dir)]}", flush=True)
            sys.exit(1)

    print(f"Found {len(pairs)} section pair(s) to process:", flush=True)
    for p in pairs:
        print(f" - {p['name']}: NP={os.path.basename(p['np'])}, NX={os.path.basename(p['nx'])}, Mask={os.path.basename(p['mask'])}", flush=True)

    total_patches = 0
    start_time = time.time()

    for pair_info in pairs:
        patches_count = process_new_section(
            pair_info, dst_dir, samples_dir,
            patch_size=args.patch_size, grid_size=args.grid_size
        )
        total_patches += patches_count

    elapsed = time.time() - start_time

    print("\n" + "=" * 70, flush=True)
    print("PRE-PROCESSING COMPLETE", flush=True)
    print("=" * 70, flush=True)
    print(f"Total sections processed : {len(pairs)}", flush=True)
    print(f"Total patches generated  : {total_patches}", flush=True)
    print(f"Elapsed time             : {elapsed:.2f} seconds", flush=True)
    print(f"Patches output location  : {dst_dir}", flush=True)
    print(f"Alignment samples        : {samples_dir}", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()
