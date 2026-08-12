#!/usr/bin/env python3
"""
Inspect and Mask New Images Tool for MMSFormer.

This script:
1. Locates NP (paralleli) and NX (incrociati) image pairs in DIA_FIRENZE/Nuove Immagini.
2. Evaluates alignment between NP and NX using Phase Correlation and saves overlay diff visualization.
3. Generates a global validity mask (Otsu thresholding + morphological cleanup) to separate mortar section from background/resin.
4. Saves preview visualizations showing the mask overlay in semi-transparent color over NP image.
"""

import os
import sys
import glob
import cv2
import numpy as np
from PIL import Image
from skimage.registration import phase_cross_correlation

def get_image_pairs(nuove_img_dir):
    """
    Finds matching NP and NX image pairs in nuove_img_dir.
    Returns a list of dicts: [{'name': ..., 'np': path, 'nx': path}, ...]
    """
    pairs = []
    # Pair 1: 1NP_SCALA.tif and 1NX_SCALA.tif
    np1 = os.path.join(nuove_img_dir, '1NP_SCALA.tif')
    nx1 = os.path.join(nuove_img_dir, '1NX_SCALA.tif')
    if os.path.exists(np1) and os.path.exists(nx1):
        pairs.append({'name': '1_SCALA', 'np': np1, 'nx': nx1})
        
    # Pair 2: NP_UNITO_BW_B.tif and NX_UNITO_B.tif
    np2 = os.path.join(nuove_img_dir, 'NP_UNITO_BW_B.tif')
    nx2 = os.path.join(nuove_img_dir, 'NX_UNITO_B.tif')
    if os.path.exists(np2) and os.path.exists(nx2):
        pairs.append({'name': 'UNITO_B', 'np': np2, 'nx': nx2})
        
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    nuove_dir = os.path.join(project_root, "..", "DIA_FIRENZE", "Nuove Immagini")
    output_dir = os.path.join(project_root, "output", "inspection")
    artifact_dir = "/Users/eliaawad/.gemini/antigravity/brain/adfaffe3-b868-4bfb-9af1-1f650f9816e3/inspection"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    
    print(f"Scanning for image pairs in: {nuove_dir}")
    pairs = get_image_pairs(nuove_dir)
    
    if not pairs:
        print("No valid NP/NX image pairs found in Nuove Immagini directory.")
        return

        
    for item in pairs:
        name = item['name']
        print(f"\n--- Processing Pair: {name} ---")
        print(f"Loading NP: {item['np']}")
        np_img = cv2.imread(item['np'])
        print(f"Loading NX: {item['nx']}")
        nx_img = cv2.imread(item['nx'])
        
        if np_img is None or nx_img is None:
            print(f"Error loading images for pair {name}")
            continue
            
        h, w = np_img.shape[:2]
        print(f"Image Dimensions: {w} x {h} pixels")
        
        # 1. Alignment check
        print("Calculating Phase Cross Correlation shift...")
        shift, error = inspect_alignment(np_img, nx_img)
        print(f"Detected Shift (Y, X): ({shift[0]:.2f}, {shift[1]:.2f}) pixels | Error: {error:.4f}")
        
        align_overlay_bgr = create_alignment_overlay(np_img, nx_img)
        align_overlay_rgb = cv2.cvtColor(align_overlay_bgr, cv2.COLOR_BGR2RGB)
        align_save_path = os.path.join(output_dir, f"{name}_alignment_overlay.png")
        align_art_path = os.path.join(artifact_dir, f"{name}_alignment_overlay.png")
        img_align = Image.fromarray(align_overlay_rgb)
        img_align.save(align_save_path)
        img_align.save(align_art_path)
        print(f"Saved alignment preview to: {align_save_path}")
        
        # 2. Validity mask generation (Criterio 2)
        print("Generating global validity mask (Criterio 2 - Solid Contour Fill)...")
        mask = generate_validity_mask(np_img)
        valid_percentage = (np.sum(mask == 0) / (h * w)) * 100.0
        print(f"Valid Mortar Region (0=Black): {valid_percentage:.2f}% of full image area")

        
        # Save full binary mask (downsampled for inspection)
        mask_small = cv2.resize(mask, (0, 0), fx=0.2, fy=0.2)
        mask_save_path = os.path.join(output_dir, f"{name}_validity_mask.png")
        mask_art_path = os.path.join(artifact_dir, f"{name}_validity_mask.png")
        img_mask = Image.fromarray(mask_small)
        img_mask.save(mask_save_path)
        img_mask.save(mask_art_path)
        print(f"Saved mask preview to: {mask_save_path}")
        
        # Save mask overlay visualization
        mask_overlay_bgr = create_mask_overlay(np_img, mask)
        mask_overlay_rgb = cv2.cvtColor(mask_overlay_bgr, cv2.COLOR_BGR2RGB)
        mask_overlay_path = os.path.join(output_dir, f"{name}_mask_overlay.png")
        mask_overlay_art_path = os.path.join(artifact_dir, f"{name}_mask_overlay.png")
        img_overlay = Image.fromarray(mask_overlay_rgb)
        img_overlay.save(mask_overlay_path)
        img_overlay.save(mask_overlay_art_path)
        print(f"Saved mask overlay preview to: {mask_overlay_path}")

if __name__ == '__main__':
    main()



