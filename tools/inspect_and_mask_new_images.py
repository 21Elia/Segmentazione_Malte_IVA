#!/usr/bin/env python3
"""
Inspect and Mask New Images Tool for MMSFormer.

This script:
1. Locates NP (paralleli) and NX (incrociati) image pairs in DIA_FIRENZE/Nuove Immagini (or its subfolders).
2. Evaluates alignment between NP and NX using Phase Correlation and saves overlay diff visualization.
3. Generates a global validity mask (Otsu thresholding + morphological cleanup) to separate mortar section from background/resin.
4. Saves preview visualizations showing the mask overlay in semi-transparent color over NP image.

Outputs (default output/inspection/): <SECTION>_validity_mask.tif (full resolution, 0 = mortar,
255 = background), read by tools/preprocess_new_images.py, plus a downscaled preview and an overlay.

Usage (from the repository root):
    python tools/inspect_and_mask_new_images.py
"""

import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "-8"

import sys
import argparse
import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_OFF)
except Exception:
    pass

import numpy as np
from PIL import Image
from skimage.registration import phase_cross_correlation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from semseg.preprocessing import find_image

def get_image_pairs(nuove_img_dir):
    """
    Finds matching NP and NX image pairs in nuove_img_dir.
    Returns a list of dicts: [{'name': ..., 'np': path, 'nx': path}, ...]
    """
    pairs = []
    # Same lookup as tools/preprocess_new_images.py: the folder itself, then its subfolders
    for name, np_file, nx_file in (('1_SCALA', '1NP_SCALA.tif', '1NX_SCALA.tif'),
                                   ('UNITO_B', 'NP_UNITO_BW_B.tif', 'NX_UNITO_B.tif')):
        np_path, nx_path = find_image(nuove_img_dir, np_file), find_image(nuove_img_dir, nx_file)
        if np_path and nx_path:
            pairs.append({'name': name, 'np': np_path, 'nx': nx_path})
    return pairs

def inspect_alignment(np_img, nx_img):
    """
    Computes Phase Correlation shift between NP and NX grayscale images.
    Returns (shift_y, shift_x), error.
    """
    np_gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    nx_gray = cv2.cvtColor(nx_img, cv2.COLOR_BGR2GRAY)
    
    # Downsample if image is huge for fast calculation
    h, w = np_gray.shape
    scale = 1.0
    if max(h, w) > 4000:
        scale = 4000.0 / max(h, w)
        np_sub = cv2.resize(np_gray, (0, 0), fx=scale, fy=scale)
        nx_sub = cv2.resize(nx_gray, (0, 0), fx=scale, fy=scale)
    else:
        np_sub, nx_sub = np_gray, nx_gray
        
    shift, error, _ = phase_cross_correlation(nx_sub, np_sub)
    shift_y, shift_x = shift / scale
    return (float(shift_y), float(shift_x)), float(error)

def create_alignment_overlay(np_img, nx_img, max_preview_dim=2000):
    """
    Creates a false-color overlay to inspect alignment:
    Channel R = NP, Channel G = NX, Channel B = NX
    """
    h, w = np_img.shape[:2]
    scale = min(1.0, max_preview_dim / max(h, w))
    np_small = cv2.resize(np_img, (0, 0), fx=scale, fy=scale)
    nx_small = cv2.resize(nx_img, (0, 0), fx=scale, fy=scale)
    
    np_gray = cv2.cvtColor(np_small, cv2.COLOR_BGR2GRAY)
    nx_gray = cv2.cvtColor(nx_small, cv2.COLOR_BGR2GRAY)
    
    # False color: R = NP, G = NX, B = NX
    false_color = np.zeros((np_gray.shape[0], np_gray.shape[1], 3), dtype=np.uint8)
    false_color[:, :, 2] = np_gray  # Red channel in BGR is index 2
    false_color[:, :, 1] = nx_gray  # Green channel
    false_color[:, :, 0] = nx_gray  # Blue channel
    
    return false_color

def generate_validity_mask(np_img, min_area_ratio=0.01, kernel_size=21):
    """
    Generates a solid binary validity mask for the mortar section.
    Handles both dark background slides (UNITO_B) and bright resin slides (1_SCALA),
    properly excluding pure black scanner border/padding alongside bright resin.
    
    Convention (matching Notari's total.tif):
      0: Valid Mortar Section (Black)
    255: Ignored Background/Resin/Scan Padding (White)
    """
    h, w = np_img.shape[:2]
    
    scale = 2000.0 / max(h, w) if max(h, w) > 2000 else 1.0
    img_sub = cv2.resize(np_img, (0, 0), fx=scale, fy=scale)
    
    gray = cv2.cvtColor(img_sub, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_sub, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    sh, sw = gray.shape
    
    # Inset border sampling (3% inset) ignoring dark scan padding (<20) to detect true background lightness
    inset_y = max(1, int(sh * 0.03))
    inset_x = max(1, int(sw * 0.03))
    sample_ring = np.concatenate([
        gray[inset_y, inset_x:-inset_x],
        gray[-inset_y, inset_x:-inset_x],
        gray[inset_y:-inset_y, inset_x],
        gray[inset_y:-inset_y, -inset_x]
    ])
    valid_samples = sample_ring[sample_ring > 20]
    bg_mean = np.mean(valid_samples) if len(valid_samples) > 0 else np.mean(sample_ring)
    bg_is_dark = bg_mean < 127
    
    if bg_is_dark:
        # Dark background (UNITO_B): Mortar is brighter than dark resin background
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_c)
    else:
        # Bright background (1_SCALA):
        # Background consists of TWO parts:
        # 1. Pure black scanning margins (gray < 20)
        # 2. Bright/off-white resin (val > 165 AND sat < 45)
        is_bg = (gray < 20) | ((val > 165) & (sat < 45))
        thresh = (~is_bg).astype(np.uint8) * 255
        kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_c)
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sub_area = sh * sw
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area_ratio * sub_area]
    
    mask_full = np.full((h, w), 255, dtype=np.uint8)
    if valid_contours:
        if scale != 1.0:
            full_contours = [(c / scale).astype(np.int32) for c in valid_contours]
        else:
            full_contours = valid_contours
        cv2.drawContours(mask_full, full_contours, -1, 0, thickness=-1)
        
    return mask_full







def create_mask_overlay(np_img, mask, max_preview_dim=2000):
    """
    Creates an overlay showing the valid mortar region (mask == 0) enclosed in green
    and ignored background (mask == 255) in semi-transparent red tint.
    """
    h, w = np_img.shape[:2]
    scale = min(1.0, max_preview_dim / max(h, w))
    np_small = cv2.resize(np_img, (0, 0), fx=scale, fy=scale)
    mask_small = cv2.resize(mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    
    overlay = np_small.copy()
    bg_indices = (mask_small == 255)
    
    red_tint = np_small.copy()
    red_tint[:, :, 2] = 255  # Red channel in BGR
    red_tint[:, :, 0] = 0    # Blue channel
    red_tint[:, :, 1] = 0    # Green channel
    
    alpha = 0.45
    overlay[bg_indices] = cv2.addWeighted(np_small[bg_indices], 1 - alpha, red_tint[bg_indices], alpha, 0)
    
    # Draw green outline around valid region boundary (where mask == 0)
    binary_mortar = (mask_small == 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mortar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Validity masks of 1_SCALA and UNITO_B")
    parser.add_argument('--src', type=str, default='../DIA_FIRENZE/Nuove Immagini', help="Path to the Nuove Immagini folder")
    parser.add_argument('--output', type=str, default='output/inspection', help="Output directory for masks and previews")
    args = parser.parse_args()
    nuove_dir, output_dir = args.src, args.output
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Scanning for image pairs in: {nuove_dir}")
    pairs = get_image_pairs(nuove_dir)
    
    if not pairs:
        print("No valid NP/NX image pairs found in Nuove Immagini directory.")
        return

        
    for item in pairs:
        name = item['name']
        print(f"\n--- Processing Section: {name} ---")
        print(f"Loading NP: {item['np']}")
        np_img = cv2.imread(item['np'])
        
        if np_img is None:
            print(f"Error loading image for section {name}")
            continue
            
        h, w = np_img.shape[:2]
        print(f"Image Dimensions: {w} x {h} pixels")
        
        # Validity mask generation (Criterio 2 - Solid Contour Fill)
        print("Generating global validity mask (Criterio 2 - Solid Contour Fill)...")
        mask = generate_validity_mask(np_img)
        valid_percentage = (np.sum(mask == 0) / (h * w)) * 100.0
        print(f"Valid Mortar Region (0=Black): {valid_percentage:.2f}% of full image area")

        
        # Save full 1:1 binary mask as TIFF (100% native resolution)
        mask_tif_path = os.path.join(output_dir, f"{name}_validity_mask.tif")
        cv2.imwrite(mask_tif_path, mask)
        print(f"Saved full-resolution 1:1 TIFF validity mask to: {mask_tif_path}")

        # Save downsampled preview mask for quick visual inspection
        mask_small = cv2.resize(mask, (0, 0), fx=0.2, fy=0.2)
        mask_save_path = os.path.join(output_dir, f"{name}_validity_mask.png")
        img_mask = Image.fromarray(mask_small)
        img_mask.save(mask_save_path)
        print(f"Saved mask preview to: {mask_save_path}")
        
        # Save mask overlay visualization
        mask_overlay_bgr = create_mask_overlay(np_img, mask)
        mask_overlay_rgb = cv2.cvtColor(mask_overlay_bgr, cv2.COLOR_BGR2RGB)
        mask_overlay_path = os.path.join(output_dir, f"{name}_mask_overlay.png")
        img_overlay = Image.fromarray(mask_overlay_rgb)
        img_overlay.save(mask_overlay_path)
        print(f"Saved mask overlay preview to: {mask_overlay_path}")

if __name__ == '__main__':
    main()



