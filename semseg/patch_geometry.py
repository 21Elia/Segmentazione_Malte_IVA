"""
Geometry of the aligned 512x512 patches described by grid_metadata.json.

During preprocessing every patch extracted at (y, x) is aligned NP->NX with an integer shift
(shift_y, shift_x): NP and NX are cropped to their overlap of size
(patch - |shift_y|) x (patch - |shift_x|) and padded back to patch x patch with zeros,
centred (pad_top = |shift_y| // 2, pad_left = |shift_x| // 2).

So the real content of a patch occupies [pad_top : pad_top + crop_h, pad_left : pad_left + crop_w]
and corresponds, in the full-section NP image, to the rectangle starting at
(y + np_top, x + np_left) with np_top = max(0, -shift_y), np_left = max(0, -shift_x).
This module is the single place where these offsets are computed.

For ARCHEO_02 the "NP frame" is the globally shifted NP image used by preprocess_5x.py.
"""

import numpy as np


def patch_regions(patch: dict, patch_size: int):
    """Returns (pad_top, pad_left, crop_h, crop_w, np_top, np_left) for one metadata entry."""
    sy, sx = int(patch.get('shift_y', 0)), int(patch.get('shift_x', 0))
    crop_h, crop_w = patch_size - abs(sy), patch_size - abs(sx)
    return abs(sy) // 2, abs(sx) // 2, crop_h, crop_w, max(0, -sy), max(0, -sx)


def content_slices(patch: dict, patch_size: int):
    """(row_slice, col_slice) of the real (non-padding) content inside the patch array."""
    pad_top, pad_left, crop_h, crop_w, _, _ = patch_regions(patch, patch_size)
    return slice(pad_top, pad_top + crop_h), slice(pad_left, pad_left + crop_w)


def np_rect(patch: dict, patch_size: int):
    """(y0, x0, h, w) of the patch content in the full-section NP frame."""
    _, _, crop_h, crop_w, np_top, np_left = patch_regions(patch, patch_size)
    return patch['y'] + np_top, patch['x'] + np_left, crop_h, crop_w


def valid_mask(patch: dict, patch_size: int) -> np.ndarray:
    """Boolean patch_size x patch_size mask: True on real content, False on alignment padding."""
    mask = np.zeros((patch_size, patch_size), dtype=bool)
    rows, cols = content_slices(patch, patch_size)
    mask[rows, cols] = True
    return mask


def paste_content(canvas: np.ndarray, patch_array: np.ndarray, patch: dict, patch_size: int) -> bool:
    """
    Copies the real content of a patch-shaped array (prediction, probability map, ...) into a
    full-section canvas at its NP-frame position, clipping at the canvas border.
    Works for 2D (H, W) and channels-last (H, W, C) arrays. Returns False if nothing was pasted.
    """
    rows, cols = content_slices(patch, patch_size)
    content = patch_array[rows, cols]
    y0, x0, h, w = np_rect(patch, patch_size)
    H, W = canvas.shape[:2]
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + h, H), min(x0 + w, W)
    if ye <= ys or xe <= xs:
        return False
    canvas[ys:ye, xs:xe] = content[ys - y0:ye - y0, xs - x0:xe - x0]
    return True


def coverage_mask(meta: dict) -> np.ndarray:
    """Full-section boolean mask of the NP-frame pixels covered by the content of any patch."""
    H, W, ps = meta['image_height'], meta['image_width'], meta['patch_size']
    cov = np.zeros((H, W), dtype=bool)
    for p in meta['patches']:
        y0, x0, h, w = np_rect(p, ps)
        cov[max(y0, 0):min(y0 + h, H), max(x0, 0):min(x0 + w, W)] = True
    return cov
