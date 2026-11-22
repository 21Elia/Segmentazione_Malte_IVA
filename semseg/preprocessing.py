"""
Helpers shared by the preprocessing tools (thesis, Chapter 2): centered padding, validity of a grid cell,
NP/NX alignment of a patch pair by phase correlation, false-colour alignment previews.

  tools/preprocessing/preprocess.py             historical sections (labelled), data/mortars
  tools/preprocessing/preprocess_new_images.py  1_SCALA and UNITO_B, data/nuove_patches
  tools/preprocessing/preprocess_5x.py          ARCHEO_02 (5X objective, rescaled), data/patches_5x_scale=057
  tools/preprocessing/inspect_and_mask_5x.py    validity mask of ARCHEO_02

The labelled historical sections are aligned by tools/preprocessing/preprocess.py (align_and_build_gt), which applies
the same shift to the annotation masks.
"""

import os
import glob

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation


def find_image(folder, filename):
    """Looks for filename in folder and then in its immediate subfolders (e.g. 1/, 2/ with the GT).

    Returns the first match, or None.
    """
    candidates = [os.path.join(folder, filename)]
    candidates += sorted(glob.glob(os.path.join(folder, '*', filename)))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


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
