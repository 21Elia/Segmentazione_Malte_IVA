#!/usr/bin/env python3
"""
Out-of-domain evaluation against the partial ground truth of 1_SCALA and UNITO_B (thesis, Sections 2.6 and 5.5).

The annotators delivered a crop (RITAGLIO) of each section with AGGREGATE, POROSITY and
TOTAL masks (black = class / valid mortar). This tool:

1. Locates the crop inside the full NP image (template matching + exact refinement)
   and saves the offsets to <output-dir>/crop_offsets.json.
2. Builds the 2-class ground truth in the crop frame:
   0 = binder (porosity merged into binder), 1 = aggregates, 255 = ignore (resin/background).
3. Maps the predictions of each model into the same NP frame, patch by patch,
   using grid_metadata.json (alignment shift and centered padding of every patch).
4. Computes IoU / F1 / accuracy per class, mIoU and the aggregate fraction (GT vs predicted)
   on the pixels that are both annotated and predicted, and saves error maps.

Predictions are the per-patch masks written by tools/run_all_inferences_softmax.py
(green = aggregates), one source per sub-folder:
    single : masks/        single forward pass
    tta    : masks_tta/    8-transform test-time augmentation
    corr   : masks_corr/   inverse photometric correction (--inverse-correction)

Usage (from the repository root):
    python tools/eval_ood_gt.py                                     # every model, every source found
    python tools/eval_ood_gt.py --experiments v3_baseline_med v3_gen_060_p080 --sources single tta
    python tools/eval_ood_gt.py --experiments v3_baseline_med v3_baseline_med_seed42 v3_baseline_med_seed3408 \
        --sources single corr --output-dir output/campaign3/gt_ood_corr
"""

import os
import sys
import csv
import json
import glob
import argparse
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from semseg.patch_geometry import patch_regions, content_slices, np_rect

Image.MAX_IMAGE_PIXELS = None

IGNORE = 255  # label value for pixels that are not annotated or not predicted

DEFAULT_GT_ROOT = os.path.join('..', 'DIA_FIRENZE', 'Nuove Immagini')
DEFAULT_RUNS_DIR = os.path.join('output', 'campaign3', 'MMSFormer')
DEFAULT_PREDICTIONS_ROOT = os.path.join('output', 'campaign3', 'inference_nuove')
DEFAULT_OUTPUT_DIR = os.path.join('output', 'campaign3', 'gt_ood')

# Ground truth sub-folder (inside --gt-root), full NP image, NP image of the annotated crop, patch metadata
DOMAINS = {
    '1_SCALA': {
        'gt_dir': '1',
        'full_np': '1NP_SCALA.tif',
        'crop_np': '1NP_SCALA_RITAGLIO.tif',
        'meta': os.path.join('data', 'nuove_patches', '1_SCALA', 'grid_metadata.json'),
    },
    'UNITO_B': {
        'gt_dir': '2',
        'full_np': 'NP_UNITO_BW_B.tif',
        'crop_np': 'RITAGLIO.tif',
        'meta': os.path.join('data', 'nuove_patches', 'UNITO_B', 'grid_metadata.json'),
    },
}

# Prediction sources: sub-folder of <predictions-root>/inference_nuove_<model>/<SECTION>/MMSFormer-B3/
SOURCES = {
    'single': 'masks',
    'tta': 'masks_tta',
    'corr': 'masks_corr',
}


def model_keys(runs_dir):
    """Model keys of the run folders, as in tools/run_all_inferences_softmax.py."""
    return [os.path.basename(d).lower().replace('mmsf-', '', 1).replace('-', '_')
            for d in sorted(glob.glob(os.path.join(runs_dir, 'MMSF-*'))) if os.path.isdir(d)]


def load_gray(path):
    """Loads an image as uint8 grayscale, applying the palette of palette TIFFs."""
    img = Image.open(path)
    if img.mode not in ('L', 'RGB'):
        img = img.convert('RGB')
    arr = np.array(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def load_gray_shape(path):
    """(height, width) of an image without decoding the pixels."""
    with Image.open(path) as img:
        return img.size[1], img.size[0]


def locate_crop(full_gray, crop_gray, scale=0.125, refine=16):
    """
    Finds the top-left corner (x0, y0) of crop_gray inside full_gray.
    Coarse normalized cross-correlation on downscaled images, then exhaustive refinement
    of +-refine px at full resolution minimizing the mean absolute difference (on every 7th
    pixel in both directions). Returns (x0, y0, mean_abs_diff); ~0 means an exact sub-rectangle.
    """
    small_full = cv2.resize(full_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_crop = cv2.resize(crop_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(small_full, small_crop, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(res)
    cx, cy = int(round(loc[0] / scale)), int(round(loc[1] / scale))

    ch, cw = crop_gray.shape
    fh, fw = full_gray.shape
    sub = crop_gray[::7, ::7].astype(np.int16)
    best = (np.inf, cx, cy)
    for dy in range(-refine, refine + 1):
        for dx in range(-refine, refine + 1):
            x, y = cx + dx, cy + dy
            if x < 0 or y < 0 or x + cw > fw or y + ch > fh:
                continue
            diff = np.abs(full_gray[y:y + ch:7, x:x + cw:7].astype(np.int16) - sub).mean()
            if diff < best[0]:
                best = (diff, x, y)
    return best[1], best[2], float(best[0])


def build_gt(gt_dir):
    """2-class GT in the crop frame: 0 binder (+ porosity), 1 aggregates, 255 ignore."""
    aggregate = load_gray(os.path.join(gt_dir, 'AGGREGATE.tif')) < 128   # black = aggregate
    total = load_gray(os.path.join(gt_dir, 'TOTAL.tif')) < 128           # black = valid mortar
    gt = np.full(aggregate.shape, IGNORE, dtype=np.uint8)
    gt[total] = 0
    gt[total & aggregate] = 1
    return gt


def predictions_in_crop(meta, roi, mask_dir):
    """
    Builds the predicted aggregate map (1 = aggregate, 0 = binder, 255 = not predicted) in the crop frame.
    roi = (x0, y0, w, h) of the crop in full-section NP coordinates. Returns (map, patches_used),
    or (None, mask_dir) if the folder does not exist.
    """
    if not os.path.isdir(mask_dir):
        return None, mask_dir
    x0, y0, w, h = roi
    ps = meta['patch_size']
    out = np.full((h, w), IGNORE, dtype=np.uint8)

    used = 0
    for p in meta['patches']:
        _, _, ch, cw, _, _ = patch_regions(p, ps)
        ny0, nx0, _, _ = np_rect(p, ps)   # NP-frame position of the patch content
        ix0, iy0 = max(nx0, x0), max(ny0, y0)
        ix1, iy1 = min(nx0 + cw, x0 + w), min(ny0 + ch, y0 + h)
        if ix1 <= ix0 or iy1 <= iy0:
            continue

        mpath = os.path.join(mask_dir, p['filename'].replace('.tif', '.png'))
        if not os.path.exists(mpath):
            continue
        rows, cols = content_slices(p, ps)
        content = (np.array(Image.open(mpath).convert('RGB'))[:, :, 1] > 127)[rows, cols]   # green = aggregate
        if content.shape != (ch, cw):
            continue

        out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = \
            content[iy0 - ny0:iy1 - ny0, ix0 - nx0:ix1 - nx0].astype(np.uint8)
        used += 1
    return out, used


def scores(gt, pred_agg):
    """Confusion-based metrics on pixels annotated in GT and predicted (rows = GT)."""
    keep = (gt != IGNORE) & (pred_agg != IGNORE)
    g, p = gt[keep].astype(np.int64), pred_agg[keep].astype(np.int64)
    cm = np.bincount(g * 2 + p, minlength=4).reshape(2, 2)
    tp = np.diag(cm).astype(float)
    iou = tp / (cm.sum(0) + cm.sum(1) - tp)
    f1 = 2 * tp / (cm.sum(0) + cm.sum(1))
    acc = tp / cm.sum(1)
    return {
        'evaluated_px': int(keep.sum()),
        'coverage_of_gt': float(keep.sum() / max((gt != IGNORE).sum(), 1)),
        'iou_binder': float(iou[0] * 100), 'iou_aggregates': float(iou[1] * 100), 'miou': float(iou.mean() * 100),
        'f1_binder': float(f1[0] * 100), 'f1_aggregates': float(f1[1] * 100), 'mf1': float(f1.mean() * 100),
        'acc_binder': float(acc[0] * 100), 'acc_aggregates': float(acc[1] * 100),
        'gt_aggregate_fraction': float(cm[1].sum() / cm.sum() * 100),
        'pred_aggregate_fraction': float(cm[:, 1].sum() / cm.sum() * 100),
        'confusion_matrix': cm.tolist(),
    }


def save_error_map(gt, pred_agg, path, scale=0.25):
    """Correct binder = dark gray, correct aggregate = light gray, missed aggregate = red,
    false aggregate = blue, ignored / not predicted = black."""
    vis = np.zeros(gt.shape + (3,), dtype=np.uint8)
    valid = (gt != IGNORE) & (pred_agg != IGNORE)
    vis[valid & (gt == 0) & (pred_agg == 0)] = (70, 70, 70)
    vis[valid & (gt == 1) & (pred_agg == 1)] = (200, 200, 200)
    vis[valid & (gt == 1) & (pred_agg == 0)] = (0, 0, 255)      # BGR red: missed aggregate
    vis[valid & (gt == 0) & (pred_agg == 1)] = (255, 0, 0)      # BGR blue: false aggregate
    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, vis)


def main():
    parser = argparse.ArgumentParser(description="OOD evaluation against the partial GT of 1_SCALA and UNITO_B")
    parser.add_argument('--domains', nargs='+', default=list(DOMAINS), choices=list(DOMAINS))
    parser.add_argument('--experiments', nargs='+', default=None,
                        help="Model keys (as in run_all_inferences_softmax.py); default: every run in --runs-dir")
    parser.add_argument('--sources', nargs='+', default=None, choices=list(SOURCES),
                        help="Prediction sources to evaluate (default: every source found)")
    parser.add_argument('--gt-root', default=DEFAULT_GT_ROOT, help="Folder with the ground truth of the new sections")
    parser.add_argument('--runs-dir', default=DEFAULT_RUNS_DIR, help="Folder with one sub-folder per training run")
    parser.add_argument('--predictions-root', default=DEFAULT_PREDICTIONS_ROOT,
                        help="Folder with the inference_nuove_<model> prediction folders")
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    experiments = args.experiments or model_keys(args.runs_dir)
    sources = args.sources or list(SOURCES)

    offsets, rows = {}, []
    for dom in args.domains:
        cfg = DOMAINS[dom]
        gt_dir = os.path.join(args.gt_root, cfg['gt_dir'])
        print(f"\n{'=' * 70}\nDOMAIN {dom}\n{'=' * 70}", flush=True)

        full = load_gray(os.path.join(gt_dir, cfg['full_np']))
        crop = load_gray(os.path.join(gt_dir, cfg['crop_np']))
        x0, y0, diff = locate_crop(full, crop)
        h, w = crop.shape
        offsets[dom] = {'x0': x0, 'y0': y0, 'width': w, 'height': h, 'mean_abs_diff': round(diff, 3)}
        print(f"  Crop located at x0={x0}, y0={y0} ({w}x{h}), mean |diff| = {diff:.3f}"
              f"{'  (exact sub-rectangle)' if diff < 0.5 else '  WARNING: not an exact match'}", flush=True)
        del full, crop

        with open(cfg['meta'], encoding='utf-8') as f:
            meta = json.load(f)
        if (meta['image_height'], meta['image_width']) != load_gray_shape(os.path.join(gt_dir, cfg['full_np'])):
            print("  WARNING: grid_metadata size differs from the full NP image used for the GT", flush=True)

        gt = build_gt(gt_dir)
        if gt.shape != (h, w):
            print(f"  ERROR: GT masks {gt.shape} do not match the crop {(h, w)}", flush=True)
            continue
        annotated = gt != IGNORE
        print(f"  GT: {annotated.mean() * 100:.1f}% of the crop is mortar, "
              f"aggregates = {gt[annotated].mean() * 100:.1f}% of the mortar", flush=True)
        Image.fromarray(gt).save(os.path.join(args.output_dir, f'{dom}_gt_crop.png'))

        for exp in experiments:
            for src_name in sources:
                mask_dir = os.path.join(args.predictions_root, f'inference_nuove_{exp}', dom, 'MMSFormer-B3', SOURCES[src_name])
                pred, info = predictions_in_crop(meta, (x0, y0, w, h), mask_dir)
                if pred is None:
                    if args.sources:   # only report sources that were asked for explicitly
                        print(f"  [skip] {exp}/{src_name}: missing {info}", flush=True)
                    continue
                s = scores(gt, pred)
                save_error_map(gt, pred, os.path.join(args.output_dir, f'{dom}_{exp}_{src_name}_errors.png'))
                rows.append({'domain': dom, 'experiment': exp, 'source': src_name, 'patches_used': info, **s})
                print(f"  {exp:32s} {src_name:6s} mIoU={s['miou']:6.2f}  IoU binder={s['iou_binder']:6.2f}  "
                      f"IoU aggr={s['iou_aggregates']:6.2f}  aggr% GT={s['gt_aggregate_fraction']:5.1f} "
                      f"pred={s['pred_aggregate_fraction']:5.1f}  coverage={s['coverage_of_gt'] * 100:5.1f}%", flush=True)

    with open(os.path.join(args.output_dir, 'crop_offsets.json'), 'w', encoding='utf-8') as f:
        json.dump(offsets, f, indent=2)
    with open(os.path.join(args.output_dir, 'gt_ood_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    if rows:
        cols = [k for k in rows[0] if k != 'confusion_matrix']
        with open(os.path.join(args.output_dir, 'gt_ood_metrics.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nResults saved to {args.output_dir}", flush=True)


if __name__ == '__main__':
    sys.exit(main())
