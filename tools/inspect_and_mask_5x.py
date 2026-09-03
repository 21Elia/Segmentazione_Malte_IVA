#!/usr/bin/env python3
"""
Inspect and Mask Tool for 5X Gigapixel Mortar Images.

Workflow:
1. Disables image size limits for gigapixel handling.
2. Locates matching NP and NX 5X image pairs in DIA_FIRENZE/Prova immagini 5X.
3. Loads and rescales 5X NP & NX images to 1X equivalent (scale=0.20) using cv2.INTER_AREA.
4. Generates a solid binary validity mask (0=Valid Mortar, 255=Background/Resin)
   leveraging multi-modal polarization birefringence (NX) and solid contour filling.
5. Saves 1:1 TIFF mask and preview overlay visualizations.
"""

import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "-8"

import sys
import glob
import argparse
import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_OFF)
except Exception:
    pass

import numpy as np
from PIL import Image

# Disable PIL limit for decompression bomb protection on gigapixel images
Image.MAX_IMAGE_PIXELS = None


def get_5x_image_pairs(src_dir):
    """
    Finds matching 5X image pairs (Parallel and Crossed Nicols) in the source directory.
    Returns a list of dictionaries: [{'name': 'ARCHEO_02', 'np': path_np, 'nx': path_nx}, ...]
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


def load_and_rescale_image(image_path, scale=0.20):
    """
    Loads native 5X image from disk and rescales to 1X equivalent using cv2.INTER_AREA (Area-based Box Filtering).
    """
    print(f"Loading 5X image from disk: {image_path} ...", flush=True)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image file: {image_path}")
        
    h_orig, w_orig = img.shape[:2]
    target_w = int(round(w_orig * scale))
    target_h = int(round(h_orig * scale))
    
    print(f"Native 5X dimensions : {w_orig} x {h_orig} px ({w_orig*h_orig/1e6:.1f} MP)", flush=True)
    print(f"Rescaling to 1X (S={scale}): {target_w} x {target_h} px with cv2.INTER_AREA...", flush=True)
    
    img_rescaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return img_rescaled


def pad_image_to_size(img, target_h, target_w, fill_value=0):
    """
    Pads image to target dimensions using centered symmetric padding (Notari method).
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


def generate_validity_mask_5x(np_img_1x, nx_img_1x=None, min_area_ratio=0.01, kernel_size=135):
    """
    Generates a solid full-body binary validity mask (0=Valid Mortar, 255=Background/Resin).
    Leverages optical birefringence and polarization activity of minerals under Crossed Nicols (NX)
    to precisely discriminate the mortar body from isotropic extinct resin and scanner margins.
    """
    h, w = np_img_1x.shape[:2]
    
    if nx_img_1x is not None:
        # 1. Multi-modal pipeline guided by Crossed Nicols (NX)
        nx_gray = cv2.cvtColor(nx_img_1x, cv2.COLOR_BGR2GRAY)
        nx_blur = cv2.GaussianBlur(nx_gray, (31, 31), 0)
        
        # Otsu threshold on NX (separates optical activity of mortar from extinct resin)
        otsu_th, nx_bin = cv2.threshold(nx_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological closing to seal internal pores and bridge peripheral indentations
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(nx_bin, cv2.MORPH_CLOSE, kernel_close)
        
        # Morphological opening to eliminate isolated external noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    else:
        # Unimodal fallback (Parallel Nicols only)
        gray = cv2.cvtColor(np_img_1x, cv2.COLOR_BGR2GRAY)
        is_bg = (gray < 20)
        mortar_candidates = (~is_bg).astype(np.uint8) * 255
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(mortar_candidates, cv2.MORPH_CLOSE, kernel_close)
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    
    # Topological extraction of external contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter dominant contours above minimum area threshold
    total_area = h * w
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area_ratio * total_area]
    
    # Initialize background canvas with 255 (Ignored / Resin)
    mask_1x = np.full((h, w), 255, dtype=np.uint8)
    
    # Solid fill inside valid contours with 0 (Valid Mortar)
    if valid_contours:
        cv2.drawContours(mask_1x, valid_contours, -1, 0, thickness=-1)
        
    return mask_1x


def create_mask_overlay(np_img, mask, max_preview_dim=2000):
    """
    Creates diagnostic preview overlay:
      - Valid mortar: original color with green outline.
      - Ignored background: semi-transparent red tint (alpha=0.45).
    """
    h, w = np_img.shape[:2]
    scale = min(1.0, max_preview_dim / max(h, w))
    np_small = cv2.resize(np_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    
    overlay = np_small.copy()
    bg_indices = (mask_small == 255)
    
    red_tint = np_small.copy()
    red_tint[:, :, 2] = 255  # Red channel (BGR index 2)
    red_tint[:, :, 0] = 0    # Blue channel
    red_tint[:, :, 1] = 0    # Green channel
    
    alpha = 0.45
    overlay[bg_indices] = cv2.addWeighted(np_small[bg_indices], 1 - alpha, red_tint[bg_indices], alpha, 0)
    
    # Draw green outline (0, 255, 0) around valid mortar boundary (where mask == 0)
    binary_mortar = (mask_small == 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mortar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Inspection and Validity Mask Generation for 5X Images")
    parser.add_argument('--src', type=str, default='../DIA_FIRENZE/Prova immagini 5X', help="Source folder for 5X images")
    parser.add_argument('--output', type=str, default='output/inspection', help="Output directory for masks and previews")
    parser.add_argument('--scale', type=float, default=0.20, help="Linear scale reduction factor (default: 0.20)")
    parser.add_argument('--kernel-size', type=int, default=135, help="Morphological closing kernel size (default: 135)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    pairs = get_5x_image_pairs(args.src)
    
    if not pairs:
        print(f"No valid 5X image pairs found in: {args.src}")
        sys.exit(1)
        
    print(f"Found {len(pairs)} 5X section(s) to process.")
    for p in pairs:
        sec_name = p['name']
        print(f"\n{'='*60}")
        print(f"PROCESSING SECTION: {sec_name}")
        print(f"{'='*60}")
        
        # 1. Load and rescale NP and NX
        np_rescaled = load_and_rescale_image(p['np'], scale=args.scale)
        nx_rescaled = load_and_rescale_image(p['nx'], scale=args.scale)
        
        # Equalize dimensions if they differ slightly
        max_h = max(np_rescaled.shape[0], nx_rescaled.shape[0])
        max_w = max(np_rescaled.shape[1], nx_rescaled.shape[1])
        np_rescaled = pad_image_to_size(np_rescaled, max_h, max_w, 0)
        nx_rescaled = pad_image_to_size(nx_rescaled, max_h, max_w, 0)
        
        # 2. Generate multi-modal validity mask (NX-guided)
        print("Generating solid validity mask (Crossed Nicols guided)...", flush=True)
        mask_1x = generate_validity_mask_5x(np_rescaled, nx_rescaled, kernel_size=args.kernel_size)
        
        h_1x, w_1x = mask_1x.shape[:2]
        valid_pct = (np.sum(mask_1x == 0) / (h_1x * w_1x)) * 100.0
        print(f"Valid Mortar Region (0=Black): {valid_pct:.2f}% of full 1X equivalent area")
        
        # 3. Save full 1:1 TIFF mask (native 1X equivalent resolution)
        mask_tif_path = os.path.join(args.output, f"{sec_name}_validity_mask.tif")
        cv2.imwrite(mask_tif_path, mask_1x)
        print(f"Saved 1:1 TIFF mask to: {mask_tif_path}")
        
        # 4. Save PNG preview of the mask
        mask_png_path = os.path.join(args.output, f"{sec_name}_validity_mask.png")
        cv2.imwrite(mask_png_path, mask_1x)
        
        # 5. Save visual overlay preview
        overlay_bgr = create_mask_overlay(np_rescaled, mask_1x)
        overlay_path = os.path.join(args.output, f"{sec_name}_mask_overlay.png")
        cv2.imwrite(overlay_path, overlay_bgr)
        print(f"Saved overlay preview to: {overlay_path}")


if __name__ == '__main__':
    main()
