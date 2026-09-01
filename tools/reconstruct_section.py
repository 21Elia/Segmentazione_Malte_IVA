#!/usr/bin/env python3
"""
Mortar High-Resolution Section Reconstruction & Overlay Tool.

Reconstructs the full high-resolution segmentation map of a mortar section
from predicted patch images, discarding alignment padding bands to prevent
stitching artifacts. Optionally blends the segmentation map over the original
high-resolution photo for visual inspection.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import glob


def reconstruct_section(metadata_path, pred_dir, output_path, bg_color=(0, 0, 0),
                        bg_path=None, overlay_output=None, overlay_alpha=0.4):
    """
    Reconstructs full-resolution segmentation map from grid metadata and patch predictions.

    Args:
        metadata_path (str): Path to grid_metadata.json
        pred_dir (str): Path to directory containing predicted PNG patches
        output_path (str): Output PNG/TIFF path for reconstructed image
        bg_color (tuple): RGB background color for unclassified/padding regions (default: (0,0,0))
        bg_path (str): Optional path to original full-res photo OR background patch directory for overlay
        overlay_output (str): Optional path to save blended overlay image
        overlay_alpha (float): Opacity of green segmentation mask (default: 0.4)
    """
    print("=" * 70, flush=True)
    print("HIGH-RESOLUTION MORTAR SECTION RECONSTRUCTION", flush=True)
    print(f"Metadata path       : {metadata_path}", flush=True)
    print(f"Predictions dir     : {pred_dir}", flush=True)
    print(f"Output path         : {output_path}", flush=True)
    if overlay_output:
        print(f"Overlay Output      : {overlay_output}", flush=True)
    print("=" * 70, flush=True)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    sec_name = meta.get('section_name', 'Unknown')
    img_h = meta['image_height']
    img_w = meta['image_width']
    patch_size = meta['patch_size']
    patches = meta['patches']

    print(f"Section Name        : {sec_name}", flush=True)
    print(f"Full Dimensions     : {img_w} x {img_h} pixels", flush=True)
    print(f"Total Grid Patches  : {len(patches)}", flush=True)

    # Initialize canvas with background color (OpenCV uses BGR order)
    bgr_bg = [bg_color[2], bg_color[1], bg_color[0]]
    canvas = np.full((img_h, img_w, 3), bgr_bg, dtype=np.uint8)

    reconstructed_count = 0
    missing_count = 0

    for idx, p in enumerate(patches):
        filename_png = p['filename'].replace('.tif', '.png')
        pred_path = os.path.join(pred_dir, filename_png)

        if not os.path.exists(pred_path):
            # Check masks/ subfolder first
            masks_sub = os.path.join(pred_dir, 'masks', filename_png)
            if os.path.exists(masks_sub):
                pred_path = masks_sub
            else:
                # Fallback to search subdirectories (e.g. MMSFormer-B3) or original .tif filename
                sub_matches = glob.glob(os.path.join(pred_dir, '**', 'masks', filename_png), recursive=True)
                if not sub_matches:
                    sub_matches = glob.glob(os.path.join(pred_dir, '**', filename_png), recursive=True)
                if not sub_matches:
                    sub_matches = glob.glob(os.path.join(pred_dir, '**', p['filename']), recursive=True)
                if sub_matches:
                    pred_path = sub_matches[0]
                elif os.path.exists(os.path.join(pred_dir, p['filename'])):
                    pred_path = os.path.join(pred_dir, p['filename'])
                else:
                    missing_count += 1
                    continue


        pred_patch = cv2.imread(pred_path)
        if pred_patch is None:
            print(f"Warning: Failed to load prediction image: {pred_path}", flush=True)
            missing_count += 1
            continue

        # Extract shift values
        shift_y = p.get('shift_y', 0)
        shift_x = p.get('shift_x', 0)

        pad_h = abs(shift_y)
        pad_w = abs(shift_x)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        crop_y_end = patch_size - pad_bottom
        crop_x_end = patch_size - pad_right

        # Crop out the artificial zero-padding bands
        valid_pred = pred_patch[pad_top:crop_y_end, pad_left:crop_x_end]

        # Calculate exact target coordinates in the global canvas
        y_start = p['y'] + pad_top
        y_end = p['y'] + patch_size - pad_bottom
        x_start = p['x'] + pad_left
        x_end = p['x'] + patch_size - pad_right

        # Safety boundary checks
        y_start = max(0, min(img_h, y_start))
        y_end = max(0, min(img_h, y_end))
        x_start = max(0, min(img_w, x_start))
        x_end = max(0, min(img_w, x_end))

        # Insert crop into canvas
        h_crop = y_end - y_start
        w_crop = x_end - x_start

        if h_crop > 0 and w_crop > 0:
            canvas[y_start:y_end, x_start:x_end] = valid_pred[:h_crop, :w_crop]
            reconstructed_count += 1

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print(f"Saved Reconstructed Map: {output_path}", flush=True)

    # Optional Overlay Generation
    if overlay_output and bg_path:
        print("\nGenerating blended overlay image...", flush=True)
        bg_img = None
        if os.path.isfile(bg_path):
            bg_img = cv2.imread(bg_path)
            if bg_img is not None and (bg_img.shape[0] != img_h or bg_img.shape[1] != img_w):
                bg_img = cv2.resize(bg_img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        elif os.path.isdir(bg_path):
            print(f"Reconstructing background photo from patches in {bg_path}...", flush=True)
            # Reconstruct background photo from patch directory
            temp_bg_path = output_path + "_temp_bg.png"
            reconstruct_section(metadata_path, bg_path, temp_bg_path, bg_color=bg_color)
            bg_img = cv2.imread(temp_bg_path)
            if os.path.exists(temp_bg_path):
                os.remove(temp_bg_path)

        if bg_img is not None:
            overlay_result = bg_img.copy()
            # Identify non-background classified pixels (green aggregates)
            mask = np.any(canvas > 0, axis=2)
            overlay_result[mask] = cv2.addWeighted(
                bg_img[mask], 1.0 - overlay_alpha,
                canvas[mask], overlay_alpha, 0
            )

            os.makedirs(os.path.dirname(os.path.abspath(overlay_output)), exist_ok=True)
            cv2.imwrite(overlay_output, overlay_result)
            print(f"Saved Blended Overlay Map: {overlay_output}", flush=True)
        else:
            print(f"Warning: Could not load background image from {bg_path}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("RECONSTRUCTION COMPLETE", flush=True)
    print(f"Patches Reconstructed : {reconstructed_count} / {len(patches)}", flush=True)
    if missing_count > 0:
        print(f"Missing Predictions   : {missing_count}", flush=True)
    print("=" * 70, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct High-Resolution Mortar Section Map & Overlay")
    parser.add_argument('--meta', type=str, required=True, help="Path to grid_metadata.json")
    parser.add_argument('--pred-dir', type=str, required=True, help="Path to predicted patches folder")
    parser.add_argument('--output', type=str, required=True, help="Path to save reconstructed image")
    parser.add_argument('--bg-path', type=str, default=None, help="Path to original full-res photo OR background patch dir")
    parser.add_argument('--overlay-output', type=str, default=None, help="Path to save blended overlay image")
    parser.add_argument('--overlay-alpha', type=float, default=0.4, help="Opacity of green mask in overlay (default: 0.4)")
    parser.add_argument('--bg-color', type=str, default='0,0,0', help="RGB background color (e.g. '0,0,0')")
    args = parser.parse_args()

    bg_tuple = tuple(int(x.strip()) for x in args.bg_color.split(','))
    reconstruct_section(
        metadata_path=args.meta,
        pred_dir=args.pred_dir,
        output_path=args.output,
        bg_color=bg_tuple,
        bg_path=args.bg_path,
        overlay_output=args.overlay_output,
        overlay_alpha=args.overlay_alpha
    )


if __name__ == '__main__':
    main()
