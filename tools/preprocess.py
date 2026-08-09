#!/usr/bin/env python3
"""
Mortar Dataset Pre-Processing Script for MMSFormer.

This script implements the preprocessing pipeline used by Leonardo Notari
(based on Progetto_malte_bacci.ipynb and described in Report_segmentazione_malte_Notari.pdf).

Main steps:
1. Scan section directories in DIA_FIRENZE (or specified source folder).
2. Check for the required 5 image files per section:
   - nicols paralleli (RGB)
   - nicols incrociati (RGB)
   - aggregate (binary mask)
   - porosity (binary mask)
   - total (valid region mask)
   If any required file is missing (e.g. section 4), the section is skipped.
3. Check and apply initial padding to full-resolution images if dimensions mismatch.
4. Grid-based patch extraction (default 512x512) excluding invalid regions indicated by total.tif.
5. Fine alignment of the 'paralleli' patch relative to 'incrociati' via Phase Correlation.
6. Multiclass ground truth generation:
   - 0: Binder (Matrix)
   - 1: Porosity
   - 2: Aggregates (resolving overlaps in favor of Aggregates)
   - 3: Ignore Label (for padding region resulting from alignment shift)
7. Save patches to:
   - output_dir/paralleli/
   - output_dir/incrociati/
   - output_dir/label/
"""

import os
# Suppress OpenCV C++ LibTIFF warnings regarding unknown metadata tags
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import sys
import glob
import time
import argparse
import numpy as np
import cv2
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
    # Pixels > 127 in total_mask indicate background/resin regions to be ignored
    return bool(np.sum(patch_region > 127) == 0)


def align_and_build_gt(img1_patch, img2_patch, porosity_patch, aggregate_patch, patch_size=512, max_shift=50):
    """
    Aligns img1_patch (paralleli) to img2_patch (incrociati) using Phase Correlation,
    crops all patches to the overlapping area, constructs the multiclass GT mask,
    and pads all arrays back to (patch_size, patch_size).
    
    GT Classes:
      0: Binder (porosity == 0 and aggregate == 0)
      1: Porosity (porosity > 127 and aggregate <= 127)
      2: Aggregates (aggregate > 127, overwriting porosity on overlaps)
      3: Ignore label (padding region)
    """
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
        # Fallback to no shift if bounds are invalid
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

    return padded_img1, padded_img2, padded_gt, (shift_x_int, shift_y_int)


def process_section(section_name, file_map, output_dir, patch_size=512, grid_size=512):
    """
    Processes a complete section: loads the 5 images, standardizes dimensions,
    extracts grid patches, aligns them, builds ground truth masks, and saves outputs.
    """
    print(f"\nProcessing section: {section_name}")

    # Load images
    img_paralleli = cv2.imread(file_map['paralleli'])
    img_incrociati = cv2.imread(file_map['incrociati'])
    img_aggregate = cv2.imread(file_map['aggregate'], cv2.IMREAD_GRAYSCALE)
    img_porosity = cv2.imread(file_map['porosity'], cv2.IMREAD_GRAYSCALE)
    img_total = cv2.imread(file_map['total'], cv2.IMREAD_GRAYSCALE)

    if any(x is None for x in [img_paralleli, img_incrociati, img_aggregate, img_porosity, img_total]):
        print(f"Error loading images for section {section_name}")
        return 0

    # Determine maximum dimensions for initial full-resolution padding
    max_h = max(img_paralleli.shape[0], img_incrociati.shape[0], img_aggregate.shape[0], img_porosity.shape[0], img_total.shape[0])
    max_w = max(img_paralleli.shape[1], img_incrociati.shape[1], img_aggregate.shape[1], img_porosity.shape[1], img_total.shape[1])

    # Apply initial padding to full-resolution images
    img_paralleli = pad_image_to_size(img_paralleli, max_h, max_w, 0)
    img_incrociati = pad_image_to_size(img_incrociati, max_h, max_w, 0)
    img_aggregate = pad_image_to_size(img_aggregate, max_h, max_w, 0)
    img_porosity = pad_image_to_size(img_porosity, max_h, max_w, 0)
    img_total = pad_image_to_size(img_total, max_h, max_w, 255)  # 255 = invalid area for initial padding

    # Create output directories for modalities and labels
    dir_paralleli = os.path.join(output_dir, 'paralleli')
    dir_incrociati = os.path.join(output_dir, 'incrociati')
    dir_label = os.path.join(output_dir, 'label')

    os.makedirs(dir_paralleli, exist_ok=True)
    os.makedirs(dir_incrociati, exist_ok=True)
    os.makedirs(dir_label, exist_ok=True)

    grid_rows = max_h // grid_size
    grid_cols = max_w // grid_size

    generated_patches = 0

    for row in range(grid_rows):
        for col in range(grid_cols):
            y = row * grid_size
            x = col * grid_size

            if is_valid_patch_position(img_total, y, x, patch_size):
                p_paralleli = img_paralleli[y:y + patch_size, x:x + patch_size]
                p_incrociati = img_incrociati[y:y + patch_size, x:x + patch_size]
                p_porosity = img_porosity[y:y + patch_size, x:x + patch_size]
                p_aggregate = img_aggregate[y:y + patch_size, x:x + patch_size]

                # Fine alignment and multiclass ground truth generation
                out_paralleli, out_incrociati, out_label, shift = align_and_build_gt(
                    p_paralleli, p_incrociati, p_porosity, p_aggregate, patch_size=patch_size
                )

                filename = f"sec{section_name}_patch_{generated_patches:04d}_r{row}_c{col}.tif"

                cv2.imwrite(os.path.join(dir_paralleli, filename), out_paralleli)
                cv2.imwrite(os.path.join(dir_incrociati, filename), out_incrociati)
                cv2.imwrite(os.path.join(dir_label, filename), out_label)

                generated_patches += 1

    print(f"Done section {section_name}: generated {generated_patches} valid patches.")
    return generated_patches


def main():
    parser = argparse.ArgumentParser(description="Mortar dataset pre-processing script for MMSFormer")
    parser.add_argument('--src', type=str, default='../DIA_FIRENZE', help="Path to raw mortar sections directory")
    parser.add_argument('--dst', type=str, default='data/mortars', help="Output directory for processed dataset")
    parser.add_argument('--patch-size', type=int, default=512, help="Square patch dimension in pixels (default: 512)")
    parser.add_argument('--grid-size', type=int, default=512, help="Grid step size for patch extraction in pixels (default: 512)")
    args = parser.parse_args()

    # Resolve absolute paths
    src_dir = os.path.abspath(args.src)
    dst_dir = os.path.abspath(args.dst)

    print("=" * 70)
    print("MORTAR DATASET PRE-PROCESSING (MMSFormer)")
    print(f"Source Directory : {src_dir}")
    print(f"Output Directory : {dst_dir}")
    print(f"Patch Size       : {args.patch_size}x{args.patch_size}")
    print(f"Grid Size        : {args.grid_size}x{args.grid_size}")
    print("=" * 70)

    if not os.path.exists(src_dir):
        print(f"Error: Source directory {src_dir} does not exist!")
        sys.exit(1)

    subdirs = sorted([d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))])
    
    total_processed_sections = 0
    total_skipped_sections = 0
    total_patches = 0

    start_time = time.time()

    for subdir in subdirs:
        section_path = os.path.join(src_dir, subdir)
        file_map = find_section_files(section_path)

        if file_map is None:
            print(f"\nSkipping section '{subdir}': incomplete file set (missing required .tif modalities).")
            total_skipped_sections += 1
            continue

        num_patches = process_section(subdir, file_map, dst_dir, patch_size=args.patch_size, grid_size=args.grid_size)
        total_processed_sections += 1
        total_patches += num_patches

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY PRE-PROCESSING REPORT")
    print("=" * 70)
    print(f"Total sections checked   : {len(subdirs)}")
    print(f"Sections processed       : {total_processed_sections}")
    print(f"Sections skipped         : {total_skipped_sections}")
    print(f"Total patches generated  : {total_patches}")
    print(f"Elapsed time             : {elapsed:.2f} seconds")
    print(f"Dataset location         : {dst_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
