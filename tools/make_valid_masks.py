#!/usr/bin/env python3
"""
Writes a validity mask for every aligned patch: valid/<patch>.png, 255 = real image content,
0 = zero padding introduced by the NP/NX alignment (see semseg/patch_geometry.py).

The masks are derived from grid_metadata.json (shift_y, shift_x of each patch), so the
alignment does not need to be recomputed. Analyses on the images (domain shift statistics,
sharpness, GLCM) use them to ignore the padding. For datasets with labels, the masks are
cross-checked against the ignore label 3 written by preprocess.py.

Usage (from the repository root):
    python tools/make_valid_masks.py
    python tools/make_valid_masks.py --datasets data/mortars_v2
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from semseg.patch_geometry import valid_mask

# dataset root -> where its grid_metadata.json files live (one per section)
DEFAULT_DATASETS = [
    'data/mortars_v2',
    'data/nuove_patches/1_SCALA',
    'data/nuove_patches/UNITO_B',
    'data/patches_5x_scale=057/ARCHEO_02',
]


def find_metadata(root):
    """grid_metadata.json either directly in root (one section) or in root/<section>/."""
    direct = os.path.join(root, 'grid_metadata.json')
    if os.path.isfile(direct):
        return [direct]
    return sorted(glob.glob(os.path.join(root, '*', 'grid_metadata.json')))


def process_dataset(root):
    metas = find_metadata(root)
    if not metas:
        print(f"  [skip] no grid_metadata.json under {root}")
        return
    out_dir = os.path.join(root, 'valid')
    os.makedirs(out_dir, exist_ok=True)
    label_dir = os.path.join(root, 'label')
    written = missing = 0
    agree = total = 0
    for meta_path in metas:
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        ps = meta['patch_size']
        for p in meta['patches']:
            if not os.path.isfile(os.path.join(root, 'paralleli', p['filename'])):
                missing += 1
                continue
            mask = valid_mask(p, ps)
            Image.fromarray((mask * 255).astype(np.uint8)).save(
                os.path.join(out_dir, p['filename'].replace('.tif', '.png')))
            written += 1
            label_path = os.path.join(label_dir, p['filename'])
            if os.path.isfile(label_path):
                label_valid = np.array(Image.open(label_path)) != 3
                agree += int((label_valid == mask).sum())
                total += mask.size
    print(f"  {root}: {written} masks written to {out_dir}"
          + (f", {missing} metadata entries without patch file" if missing else ""))
    if total:
        print(f"    agreement with label != 3: {agree / total * 100:.3f}% of pixels")


def main():
    parser = argparse.ArgumentParser(description="Write per-patch validity masks (alignment padding = 0)")
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    args = parser.parse_args()
    for root in args.datasets:
        process_dataset(root)


if __name__ == '__main__':
    main()
