#!/usr/bin/env python3
"""
Mortar Dataset Pre-Processing Script for MMSFormer (Historical 12 Sections).

This script implements the preprocessing pipeline:
1. Scan section directories in DIA_FIRENZE (or specified source folder).
2. Check for the required 5 image files per section:
   - nicols paralleli (RGB)
   - nicols incrociati (RGB)
   - aggregate (binary mask)
   - porosity (binary mask)
   - total (valid region mask)
   If any required file is missing (e.g. section 4), the section is skipped.
3. Standardize dimensions across all 5 images (padding to max_h, max_w or upscaling validity mask).
4. Grid-based patch extraction (default 512x512) excluding invalid regions indicated by total.tif.
5. Fine alignment of 'paralleli' relative to 'incrociati' via Phase Cross Correlation.
6. Generates false-color RGB preview images (R=NP, G=NX, B=0) BEFORE and AFTER alignment.
7. Multiclass ground truth generation:
   - 0: Binder (Matrix)
   - 1: Porosity
   - 2: Aggregates (resolving overlaps in favor of Aggregates)
   - 3: Ignore Label (for padding region resulting from alignment shift)
8. Save patches to:
   - dst_dir/paralleli/
   - dst_dir/incrociati/
   - dst_dir/label/
   And alignment samples to output/alignment_samples/<section_name>/
   And grid metadata to dst_dir/<section_name>/grid_metadata.json
"""

import os
# Suppress OpenCV C++ LibTIFF warnings regarding unknown metadata tags
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "-8"

import sys
import glob
import json
import time
import argparse
import numpy as np
import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_OFF)
except Exception:
    pass

from skimage.registration import phase_cross_correlation


def find_section_files(section_dir):
    """
    Locates the 5 required files inside a section directory.
    
    Returns:
        dict: File paths for all 5 modalities, or None if any file is missing.
    """
    if not os.path.isdir(section_dir):
        return None

    files_in_dir = os.listdir(section_dir)

    def find_matching_file(keyword):
        for f in files_in_dir:
            if keyword.lower() in f.lower() and f.lower().endswith(('.tif', '.tiff')):
                return os.path.join(section_dir, f)
        return None

    file_map = {
        'paralleli': find_matching_file('paralleli'),
        'incrociati': find_matching_file('incrociati'),
        'aggregate': find_matching_file('aggregate'),
        'porosity': find_matching_file('porosity'),
        'total': find_matching_file('total')
    }

    # Verify that all 5 files were successfully found
    missing = [k for k, v in file_map.items() if v is None]
    if missing:
        return None

    return file_map


def pad_image_to_size(img, target_h, target_w, fill_value=0):
    """
    Pads bottom and right borders of an image up to target_h, target_w.
    """
    h, w = img.shape[:2]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    if pad_h == 0 and pad_w == 0:
        return img

    if len(img.shape) == 3:
        padding = ((0, pad_h), (0, pad_w), (0, 0))
    else:
        padding = ((0, pad_h), (0, pad_w))

    return np.pad(img, padding, mode='constant', constant_values=fill_value)


def is_valid_patch_position(total_mask, y, x, patch_size=512):
    """
    Checks if a patch extracted at (y, x) with size patch_size is valid
    (i.e. does not contain white/excluded background pixels > 127 in total_mask).
    """
    h, w = total_mask.shape[:2]
    if y + patch_size > h or x + patch_size > w:
        return False

    patch_region = total_mask[y:y + patch_size, x:x + patch_size]
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


def align_and_build_gt(img1_patch, img2_patch, porosity_patch, aggregate_patch, patch_size=512, max_shift=50):
    """
    Aligns img1_patch (paralleli) to img2_patch (incrociati) using Phase Correlation,
    crops all patches to the overlapping area, constructs the multiclass GT mask,
    and pads all arrays back to (patch_size, patch_size).
    Also generates pre and post false-color RGB previews for visualization.
    
    GT Classes:
      0: Binder (porosity == 0 and aggregate == 0)
      1: Porosity (porosity > 127 and aggregate <= 127)
      2: Aggregates (aggregate > 127, overwriting porosity on overlaps)
      3: Ignore label (padding region)
    """
    # 1. False-color PRE-alignment preview
    pre_align_rgb = create_false_color_rgb(img1_patch, img2_patch)

    # Convert to grayscale for phase correlation calculation
    img1_gray = cv2.cvtColor(img1_patch, cv2.COLOR_BGR2GRAY) if len(img1_patch.shape) == 3 else img1_patch
    img2_gray = cv2.cvtColor(img2_patch, cv2.COLOR_BGR2GRAY) if len(img2_patch.shape) == 3 else img2_patch

    # Calculate translation shift
    try:
        shift_values, error, diffphase = phase_cross_correlation(img2_gray, img1_gray)
        shift_y, shift_x = shift_values
    except Exception:
        shift_y, shift_x = 0.0, 0.0

    shift_x_int = int(round(shift_x))
    shift_y_int = int(round(shift_y))

    # Safety threshold for excessive shifts
    if abs(shift_x_int) > max_shift or abs(shift_y_int) > max_shift:
        shift_x_int, shift_y_int = 0, 0

    h, w = patch_size, patch_size

    # Calculate crop coordinates
    img1_left = max(0, -shift_x_int)
    img1_top = max(0, -shift_y_int)
    img1_right = w - max(0, shift_x_int)
    img1_bottom = h - max(0, shift_y_int)

    img2_left = max(0, shift_x_int)
    img2_top = max(0, shift_y_int)
    img2_right = w - max(0, -shift_x_int)
    img2_bottom = h - max(0, -shift_y_int)

    # Validate crop boundaries
    if (img1_right <= img1_left or img1_bottom <= img1_top or
        img2_right <= img2_left or img2_bottom <= img2_top):
        img1_left, img1_top, img1_right, img1_bottom = 0, 0, w, h
        img2_left, img2_top, img2_right, img2_bottom = 0, 0, w, h

    # Crop images
    crop1 = img1_patch[img1_top:img1_bottom, img1_left:img1_right]
    crop2 = img2_patch[img2_top:img2_bottom, img2_left:img2_right]
    crop_porosity = porosity_patch[img2_top:img2_bottom, img2_left:img2_right]
    crop_aggregate = aggregate_patch[img2_top:img2_bottom, img2_left:img2_right]

    crop_h, crop_w = crop1.shape[:2]

    # Construct multiclass ground truth map on cropped region
    gt_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)  # Default 0 = Binder
    gt_crop[crop_porosity > 127] = 1                      # 1 = Porosity
    gt_crop[crop_aggregate > 127] = 2                     # 2 = Aggregates (overwrites porosity in case of overlap)

    # Pad back to exact patch_size x patch_size
    pad_bottom = patch_size - crop_h
    pad_right = patch_size - crop_w

    if pad_bottom > 0 or pad_right > 0:
        padded_img1 = np.pad(crop1, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='constant', constant_values=0)
        padded_img2 = np.pad(crop2, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='constant', constant_values=0)
        padded_gt = np.pad(gt_crop, ((0, pad_bottom), (0, pad_right)), mode='constant', constant_values=3)  # 3 = Ignore label
    else:
        padded_img1 = crop1
        padded_img2 = crop2
        padded_gt = gt_crop

    # 2. False-color POST-alignment preview
    post_align_rgb = create_false_color_rgb(padded_img1, padded_img2)

    return padded_img1, padded_img2, padded_gt, pre_align_rgb, post_align_rgb, (shift_x_int, shift_y_int)


def process_section(section_name, file_map, output_dir, samples_dir, patch_size=512, grid_size=512):
    """
    Processes a complete section: loads the 5 images, standardizes dimensions,
    extracts grid patches, aligns them, builds ground truth masks, saves alignment previews,
    and writes dataset files + grid_metadata.json.
    """
    print(f"\n{'=' * 60}", flush=True)
    print(f"PROCESSING HISTORICAL SECTION: {section_name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    print(f"[{section_name}] Loading paralleli: {os.path.basename(file_map['paralleli'])}...", flush=True)
    img_paralleli = cv2.imread(file_map['paralleli'])

    print(f"[{section_name}] Loading incrociati: {os.path.basename(file_map['incrociati'])}...", flush=True)
    img_incrociati = cv2.imread(file_map['incrociati'])

    print(f"[{section_name}] Loading aggregate: {os.path.basename(file_map['aggregate'])}...", flush=True)
    img_aggregate = cv2.imread(file_map['aggregate'], cv2.IMREAD_GRAYSCALE)

    print(f"[{section_name}] Loading porosity: {os.path.basename(file_map['porosity'])}...", flush=True)
    img_porosity = cv2.imread(file_map['porosity'], cv2.IMREAD_GRAYSCALE)

    print(f"[{section_name}] Loading total mask: {os.path.basename(file_map['total'])}...", flush=True)
    img_total = cv2.imread(file_map['total'], cv2.IMREAD_GRAYSCALE)

    if any(x is None for x in [img_paralleli, img_incrociati, img_aggregate, img_porosity, img_total]):
        print(f"Error loading images for section {section_name}", flush=True)
        return 0

    # Determine maximum dimensions for full-resolution alignment
    max_h = max(img_paralleli.shape[0], img_incrociati.shape[0], img_aggregate.shape[0], img_porosity.shape[0])
    max_w = max(img_paralleli.shape[1], img_incrociati.shape[1], img_aggregate.shape[1], img_porosity.shape[1])

    # Apply initial padding to full-resolution images if needed
    img_paralleli = pad_image_to_size(img_paralleli, max_h, max_w, 0)
    img_incrociati = pad_image_to_size(img_incrociati, max_h, max_w, 0)
    img_aggregate = pad_image_to_size(img_aggregate, max_h, max_w, 0)
    img_porosity = pad_image_to_size(img_porosity, max_h, max_w, 0)

    # If img_total was downsampled, upscale with INTER_NEAREST to match max_h, max_w 1:1
    if img_total.shape[:2] != (max_h, max_w):
        print(f"Resizing total mask from {img_total.shape[1]}x{img_total.shape[0]} to full resolution {max_w}x{max_h}...", flush=True)
        img_total = cv2.resize(img_total, (max_w, max_h), interpolation=cv2.INTER_NEAREST)

    # Create output directories for dataset
    dir_paralleli = os.path.join(output_dir, 'paralleli')
    dir_incrociati = os.path.join(output_dir, 'incrociati')
    dir_label = os.path.join(output_dir, 'label')
    sec_samples_dir = os.path.join(samples_dir, section_name)
    sec_meta_dir = os.path.join(output_dir, section_name)

    os.makedirs(dir_paralleli, exist_ok=True)
    os.makedirs(dir_incrociati, exist_ok=True)
    os.makedirs(dir_label, exist_ok=True)
    os.makedirs(sec_samples_dir, exist_ok=True)
    os.makedirs(sec_meta_dir, exist_ok=True)

    grid_rows = max_h // grid_size
    grid_cols = max_w // grid_size

    print(f"Full dimensions : {max_h}x{max_w}", flush=True)
    print(f"Grid dimensions : {grid_rows} rows x {grid_cols} columns", flush=True)

    generated_patches = 0
    grid_metadata = {
        'section_name': section_name,
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

            if is_valid_patch_position(img_total, y, x, patch_size):
                p_paralleli = img_paralleli[y:y + patch_size, x:x + patch_size]
                p_incrociati = img_incrociati[y:y + patch_size, x:x + patch_size]
                p_porosity = img_porosity[y:y + patch_size, x:x + patch_size]
                p_aggregate = img_aggregate[y:y + patch_size, x:x + patch_size]

                # Fine alignment, multiclass GT generation, and false-color RGB previews
                out_paralleli, out_incrociati, out_label, pre_rgb, post_rgb, (shift_x, shift_y) = align_and_build_gt(
                    p_paralleli, p_incrociati, p_porosity, p_aggregate, patch_size=patch_size
                )

                patch_stem = f"sec{section_name}_patch_{generated_patches:04d}_r{row}_c{col}_y{y}_x{x}"
                filename_tif = f"{patch_stem}.tif"

                # Save alignment comparison previews (PRE and POST)
                cv2.imwrite(os.path.join(sec_samples_dir, f"{patch_stem}_PRE.png"), pre_rgb)
                cv2.imwrite(os.path.join(sec_samples_dir, f"{patch_stem}_POST.png"), post_rgb)

                # Save dataset files
                cv2.imwrite(os.path.join(dir_paralleli, filename_tif), out_paralleli)
                cv2.imwrite(os.path.join(dir_incrociati, filename_tif), out_incrociati)
                cv2.imwrite(os.path.join(dir_label, filename_tif), out_label)

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
                    print(f"[{section_name}] Generated {generated_patches} valid patches...", flush=True)

    # Save grid metadata JSON
    meta_path = os.path.join(sec_meta_dir, 'grid_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(grid_metadata, f, indent=2)

    print(f"Done section {section_name}: generated {generated_patches} valid patches.", flush=True)
    print(f"Metadata saved to: {meta_path}", flush=True)
    print(f"Alignment samples saved to: {sec_samples_dir}", flush=True)
    return generated_patches


def main():
    parser = argparse.ArgumentParser(description="Mortar dataset pre-processing script for MMSFormer")
    parser.add_argument('--src', type=str, default='../DIA_FIRENZE', help="Path to raw mortar sections directory")
    parser.add_argument('--dst', type=str, default='data/mortars', help="Output directory for processed dataset")
    parser.add_argument('--samples-dir', type=str, default='output/alignment_samples', help="Output directory for alignment samples")
    parser.add_argument('--patch-size', type=int, default=512, help="Square patch dimension in pixels (default: 512)")
    parser.add_argument('--grid-size', type=int, default=512, help="Grid step size for patch extraction in pixels (default: 512)")
    args = parser.parse_args()

    # Resolve absolute paths
    src_dir = os.path.abspath(args.src)
    dst_dir = os.path.abspath(args.dst)
    samples_dir = os.path.abspath(args.samples_dir)

    print("=" * 70, flush=True)
    print("MORTAR DATASET PRE-PROCESSING & ALIGNMENT (MMSFormer)")
    print(f"Source Directory    : {src_dir}", flush=True)
    print(f"Output Directory    : {dst_dir}", flush=True)
    print(f"Alignment Samples   : {samples_dir}", flush=True)
    print(f"Patch Size          : {args.patch_size}x{args.patch_size}", flush=True)
    print(f"Grid Size           : {args.grid_size}x{args.grid_size}", flush=True)
    print("=" * 70, flush=True)

    if not os.path.exists(src_dir):
        print(f"Error: Source directory {src_dir} does not exist!", flush=True)
        sys.exit(1)

    subdirs = sorted([d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))])
    
    total_processed_sections = 0
    total_skipped_sections = 0
    total_patches = 0

    start_time = time.time()

    for subdir in subdirs:
        # Ignore Nuove Immagini or 5X if present in DIA_FIRENZE
        if 'nuove' in subdir.lower() or '5x' in subdir.lower():
            continue

        section_path = os.path.join(src_dir, subdir)
        file_map = find_section_files(section_path)

        if file_map is None:
            print(f"\nSkipping section '{subdir}': incomplete file set (missing required .tif modalities).", flush=True)
            total_skipped_sections += 1
            continue

        num_patches = process_section(
            subdir, file_map, dst_dir, samples_dir,
            patch_size=args.patch_size, grid_size=args.grid_size
        )
        total_processed_sections += 1
        total_patches += num_patches

    elapsed = time.time() - start_time

    print("\n" + "=" * 70, flush=True)
    print("SUMMARY PRE-PROCESSING REPORT")
    print("=" * 70, flush=True)
    print(f"Total sections checked   : {len(subdirs)}", flush=True)
    print(f"Sections processed       : {total_processed_sections}", flush=True)
    print(f"Sections skipped         : {total_skipped_sections}", flush=True)
    print(f"Total patches generated  : {total_patches}", flush=True)
    print(f"Elapsed time             : {elapsed:.2f} seconds", flush=True)
    print(f"Dataset location         : {dst_dir}", flush=True)
    print(f"Alignment samples        : {samples_dir}", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()
