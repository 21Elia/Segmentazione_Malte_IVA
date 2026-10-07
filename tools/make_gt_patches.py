#!/usr/bin/env python3
"""
Builds a labelled patch dataset from the partial ground truth (RITAGLIO) of 1_SCALA and UNITO_B,
so that second-campaign models can be scored on the new domains with tools/val_mm.py
(split 'all'), exactly like the in-domain test set.

For every existing aligned patch of data/nuove_patches/<SECTION> whose content overlaps the
annotated crop, the script copies the NP/NX patch images and writes a label in the dataset
convention of preprocess.py:
    0 = binder, 1 = porosity, 2 = aggregates, 3 = ignore
Ignore covers the alignment padding, the part of the patch outside the crop and the
resin/background (TOTAL white). Masks follow the DIA_FIRENZE convention: black = class.

Output: data/nuove_patches_gt/<SECTION>/{paralleli, incrociati, label, valid} + grid_metadata.json

Usage (from the repository root):
    python tools/make_gt_patches.py
    python tools/make_gt_patches.py --min-labelled 0.5
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from semseg.patch_geometry import content_slices, np_rect, valid_mask
from tools.eval_ood_gt import DOMAINS, load_gray, locate_crop

Image.MAX_IMAGE_PIXELS = None
IGNORE_LABEL = 3


def build_label_crop(gt_dir):
    """4-class label in the crop frame (dataset convention), from AGGREGATE / POROSITY / TOTAL."""
    aggregate = load_gray(os.path.join(gt_dir, 'AGGREGATE.tif')) < 128
    porosity = load_gray(os.path.join(gt_dir, 'POROSITY.tif')) < 128
    valid = load_gray(os.path.join(gt_dir, 'TOTAL.tif')) < 128
    label = np.full(aggregate.shape, IGNORE_LABEL, dtype=np.uint8)
    label[valid] = 0
    label[valid & porosity] = 1
    label[valid & aggregate] = 2          # aggregates win over porosity, as in preprocess.py
    return label


def crop_offset(domain, cfg, offsets_path):
    """Crop offset from output/gt_ood/crop_offsets.json if present, otherwise located again."""
    if os.path.isfile(offsets_path):
        with open(offsets_path, encoding='utf-8') as f:
            offsets = json.load(f)
        if domain in offsets:
            return offsets[domain]['x0'], offsets[domain]['y0']
    full = load_gray(os.path.join(cfg['gt_dir'], cfg['full_np']))
    crop = load_gray(os.path.join(cfg['gt_dir'], cfg['crop_np']))
    x0, y0, diff = locate_crop(full, crop)
    print(f"  crop located at ({x0}, {y0}), mean |diff| = {diff:.3f}")
    return x0, y0


def main():
    parser = argparse.ArgumentParser(description="Labelled patches from the partial GT of the new sections")
    parser.add_argument('--domains', nargs='+', default=list(DOMAINS))
    parser.add_argument('--src-root', default=os.path.join('data', 'nuove_patches'))
    parser.add_argument('--dst-root', default=os.path.join('data', 'nuove_patches_gt'))
    parser.add_argument('--offsets', default=os.path.join('output', 'gt_ood', 'crop_offsets.json'))
    parser.add_argument('--min-labelled', type=float, default=0.25,
                        help="Keep a patch only if at least this fraction of its pixels is labelled (not ignore)")
    args = parser.parse_args()

    for dom in args.domains:
        cfg = DOMAINS[dom]
        print(f"\n=== {dom} ===")
        src = os.path.join(args.src_root, dom)
        dst = os.path.join(args.dst_root, dom)
        with open(os.path.join(src, 'grid_metadata.json'), encoding='utf-8') as f:
            meta = json.load(f)
        ps = meta['patch_size']

        x0, y0 = crop_offset(dom, cfg, args.offsets)
        label_crop = build_label_crop(cfg['gt_dir'])
        ch_crop, cw_crop = label_crop.shape

        for sub in ('paralleli', 'incrociati', 'label', 'valid'):
            os.makedirs(os.path.join(dst, sub), exist_ok=True)

        kept, counts = [], np.zeros(4, dtype=np.int64)
        for p in meta['patches']:
            ry0, rx0, h, w = np_rect(p, ps)
            # intersection of the patch content with the crop, in crop coordinates
            cy0, cx0 = max(ry0 - y0, 0), max(rx0 - x0, 0)
            cy1, cx1 = min(ry0 - y0 + h, ch_crop), min(rx0 - x0 + w, cw_crop)
            if cy1 <= cy0 or cx1 <= cx0:
                continue

            content = np.full((h, w), IGNORE_LABEL, dtype=np.uint8)
            content[cy0 - (ry0 - y0):cy1 - (ry0 - y0), cx0 - (rx0 - x0):cx1 - (rx0 - x0)] = label_crop[cy0:cy1, cx0:cx1]
            label = np.full((ps, ps), IGNORE_LABEL, dtype=np.uint8)
            rows, cols = content_slices(p, ps)
            label[rows, cols] = content

            labelled = (label != IGNORE_LABEL).mean()
            if labelled < args.min_labelled:
                continue

            name = p['filename']
            shutil.copy(os.path.join(src, 'paralleli', name), os.path.join(dst, 'paralleli', name))
            shutil.copy(os.path.join(src, 'incrociati', name), os.path.join(dst, 'incrociati', name))
            Image.fromarray(label).save(os.path.join(dst, 'label', name))
            Image.fromarray((valid_mask(p, ps) * 255).astype(np.uint8)).save(
                os.path.join(dst, 'valid', name.replace('.tif', '.png')))
            counts += np.bincount(label.ravel(), minlength=4)[:4]
            kept.append(p)

        out_meta = dict(meta, patches=kept, gt_crop={'x0': x0, 'y0': y0, 'width': cw_crop, 'height': ch_crop},
                        min_labelled=args.min_labelled)
        with open(os.path.join(dst, 'grid_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(out_meta, f, indent=2)

        lab = counts[:3].sum()
        print(f"  {len(kept)} patches written to {dst} (min labelled fraction {args.min_labelled})")
        if lab:
            print(f"  class fractions of labelled pixels: binder {counts[0] / lab * 100:.1f}% | "
                  f"porosity {counts[1] / lab * 100:.1f}% | aggregates {counts[2] / lab * 100:.1f}% "
                  f"| labelled pixels {lab / counts.sum() * 100:.1f}% of all patch pixels")


if __name__ == '__main__':
    main()
