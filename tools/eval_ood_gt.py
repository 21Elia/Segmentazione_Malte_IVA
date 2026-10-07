#!/usr/bin/env python3
"""
Out-of-Domain evaluation against the partial ground truth of 1_SCALA and UNITO_B.

The annotators delivered a crop (RITAGLIO) of each section with AGGREGATE, POROSITY and
TOTAL masks (black = class / valid mortar). This tool:

1. Locates the crop inside the full NP image (template matching + exact refinement)
   and saves the offsets to output/gt_ood/crop_offsets.json.
2. Builds the 2-class ground truth in the crop frame:
   0 = binder (porosity merged into binder), 1 = aggregates, 255 = ignore (resin/background).
3. Maps the predictions of each experiment into the same NP frame, patch by patch,
   using grid_metadata.json (alignment shift and centered padding of every patch).
4. Computes IoU / F1 / accuracy per class, mIoU and the aggregate fraction (GT vs predicted)
   on the pixels that are both annotated and predicted, and saves error maps.

Prediction sources (declared per experiment in EXPERIMENTS):
  First campaign (baseline, exp1-exp3, exp4a):
    - 'recon': reconstructed single-pass maps written by the old reconstruct_section.py
               (patches pasted half an alignment shift off the NP frame, compensated here);
               infer_mm.py ran with TEST.INVERT_PALETTE = true, so green = model class 0
    - 'tta'  : per-patch masks/ folder, overwritten with the argmax of the 8-transform TTA
               average by run_all_inferences_softmax.py --tta; green = model class 1
    These checkpoints were trained on labels with binder and aggregates swapped (see the
    tools/preprocess.py fix): their class 1 ("Aggregati") is binder, so recon green = real
    aggregates and tta green = real binder.
  Second campaign (v2_*): corrected labels and standard palette (green = aggregates)
    - 'single': per-patch masks/ (single forward pass)
    - 'tta'   : per-patch masks_tta/ (8-transform TTA average)

Usage (run from the repository root):
    python tools/eval_ood_gt.py                                   # first-campaign models
    python tools/eval_ood_gt.py --experiments v2_baseline v2_exp1   # second-campaign models
    python tools/eval_ood_gt.py --domains UNITO_B --experiments exp1 --sources recon
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from semseg.patch_geometry import patch_regions, content_slices, np_rect

Image.MAX_IMAGE_PIXELS = None

GT_ROOT = os.path.join('..', 'DIA_FIRENZE', 'Nuove Immagini')
IGNORE = 255  # label value for pixels that are not annotated or not predicted

DOMAINS = {
    '1_SCALA': {
        'gt_dir': os.path.join(GT_ROOT, '1'),
        'full_np': '1NP_SCALA.tif',
        'crop_np': '1NP_SCALA_RITAGLIO.tif',
        'meta': os.path.join('data', 'nuove_patches', '1_SCALA', 'grid_metadata.json'),
    },
    'UNITO_B': {
        'gt_dir': os.path.join(GT_ROOT, '2'),
        'full_np': 'NP_UNITO_BW_B.tif',
        'crop_np': 'RITAGLIO.tif',
        'meta': os.path.join('data', 'nuove_patches', 'UNITO_B', 'grid_metadata.json'),
    },
}

# Prediction sources of every experiment:
#   kind 'recon_legacy': full-section map written by the OLD reconstruct_section.py, which pasted
#                        each patch at (y + pad_top, x + pad_left) (half the alignment shift)
#   kind 'recon'       : full-section map in the NP frame (fixed reconstruct_section.py)
#   kind 'masks'       : per-patch prediction PNGs in <root>/<SECTION>/MMSFormer-B3/<subdir>/
# green_class: model class index painted green. labels_swapped: model class 1 is binder
# (first-campaign checkpoints, trained on the swapped labels of the old data/mortars).
_OLD_SOURCES = {
    'recon': {'kind': 'recon_legacy', 'green_class': 0},          # infer_mm.py with INVERT_PALETTE: true
    'tta':   {'kind': 'masks', 'subdir': 'masks', 'green_class': 1},  # overwritten by the --tta run
}
_V2_SOURCES = {
    'single': {'kind': 'masks', 'subdir': 'masks', 'green_class': 1},
    'tta':    {'kind': 'masks', 'subdir': 'masks_tta', 'green_class': 1},
}
EXPERIMENTS = {
    'baseline': {'root': 'output/inference_nuove/inference_nuove_baseline', 'labels_swapped': True, 'sources': _OLD_SOURCES},
    'exp1':     {'root': 'output/inference_nuove/inference_nuove_exp1', 'labels_swapped': True, 'sources': _OLD_SOURCES},
    'exp2':     {'root': 'output/inference_nuove/inference_nuove_exp2', 'labels_swapped': True, 'sources': _OLD_SOURCES},
    'exp3':     {'root': 'output/inference_nuove/inference_nuove_exp3', 'labels_swapped': True, 'sources': _OLD_SOURCES},
    'exp4a':    {'root': 'output/inference_nuove/inference_nuove_exp4a', 'labels_swapped': True, 'sources': _OLD_SOURCES},
    # Second campaign: corrected labels, standard palette (no INVERT_PALETTE)
    'v2_baseline': {'root': 'output/inference_nuove/inference_nuove_v2_baseline', 'labels_swapped': False, 'sources': _V2_SOURCES},
    'v2_exp1':     {'root': 'output/inference_nuove/inference_nuove_v2_exp1', 'labels_swapped': False, 'sources': _V2_SOURCES},
    'v2_exp2':     {'root': 'output/inference_nuove/inference_nuove_v2_exp2', 'labels_swapped': False, 'sources': _V2_SOURCES},
    'v2_exp4b':    {'root': 'output/inference_nuove/inference_nuove_v2_exp4b', 'labels_swapped': False, 'sources': _V2_SOURCES},
}
FIRST_CAMPAIGN = ['baseline', 'exp1', 'exp2', 'exp3', 'exp4a']


def green_to_aggregate(green, green_class, labels_swapped):
    """Converts a green/not-green map (0/1, 255 = not predicted) into real aggregates (1) vs binder (0)."""
    model_class = green if green_class == 1 else 1 - green   # model class index of each pixel
    aggregate_class = 0 if labels_swapped else 1
    aggr = (model_class == aggregate_class).astype(np.uint8)
    aggr[green == IGNORE] = IGNORE
    return aggr


def load_gray(path):
    """Loads an image as uint8 grayscale, applying the palette of palette TIFFs."""
    img = Image.open(path)
    if img.mode not in ('L', 'RGB'):
        img = img.convert('RGB')
    arr = np.array(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def locate_crop(full_gray, crop_gray, scale=0.125, refine=16):
    """
    Finds the top-left corner (x0, y0) of crop_gray inside full_gray.
    Coarse normalized cross-correlation on downscaled images, then exhaustive refinement
    of +-refine px at full resolution minimizing the mean absolute difference.
    Returns (x0, y0, mean_abs_diff); a difference of ~0 means an exact sub-rectangle.
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


def predictions_in_crop(meta, roi, source, exp_root, domain):
    """
    Builds the predicted green map (1 = green, 0 = black, 255 = not predicted) in the crop frame.
    roi = (x0, y0, w, h) of the crop in full-section NP coordinates. Returns (map, patches_used)
    or (None, missing_path).
    """
    x0, y0, w, h = roi
    ps = meta['patch_size']
    out = np.full((h, w), IGNORE, dtype=np.uint8)

    canvas = None
    if source['kind'] in ('recon', 'recon_legacy'):
        path = os.path.join(exp_root, f'{domain}_reconstructed.png')
        if not os.path.exists(path):
            return None, path
        canvas = np.array(Image.open(path).convert('RGB'))[:, :, 1] > 127
    else:
        mask_dir = os.path.join(exp_root, domain, 'MMSFormer-B3', source['subdir'])
        if not os.path.isdir(mask_dir):
            return None, mask_dir

    used = 0
    for p in meta['patches']:
        pad_top, pad_left, ch, cw, _, _ = patch_regions(p, ps)
        ny0, nx0, _, _ = np_rect(p, ps)   # NP-frame position of the patch content
        ix0, iy0 = max(nx0, x0), max(ny0, y0)
        ix1, iy1 = min(nx0 + cw, x0 + w), min(ny0 + ch, y0 + h)
        if ix1 <= ix0 or iy1 <= iy0:
            continue

        if source['kind'] == 'recon_legacy':
            ry0, rx0 = p['y'] + pad_top, p['x'] + pad_left   # where the old reconstruct pasted it
            content = canvas[ry0:ry0 + ch, rx0:rx0 + cw]
        elif source['kind'] == 'recon':
            content = canvas[ny0:ny0 + ch, nx0:nx0 + cw]
        else:
            mpath = os.path.join(mask_dir, p['filename'].replace('.tif', '.png'))
            if not os.path.exists(mpath):
                continue
            rows, cols = content_slices(p, ps)
            content = (np.array(Image.open(mpath).convert('RGB'))[:, :, 1] > 127)[rows, cols]
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
    parser.add_argument('--domains', nargs='+', default=list(DOMAINS))
    parser.add_argument('--experiments', nargs='+', default=FIRST_CAMPAIGN, choices=list(EXPERIMENTS))
    parser.add_argument('--sources', nargs='+', default=None,
                        help="Source names to evaluate (default: all sources of each experiment)")
    parser.add_argument('--output-dir', default=os.path.join('output', 'gt_ood'))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    offsets, rows = {}, []
    for dom in args.domains:
        cfg = DOMAINS[dom]
        print(f"\n{'=' * 70}\nDOMAIN {dom}\n{'=' * 70}", flush=True)

        full = load_gray(os.path.join(cfg['gt_dir'], cfg['full_np']))
        crop = load_gray(os.path.join(cfg['gt_dir'], cfg['crop_np']))
        x0, y0, diff = locate_crop(full, crop)
        h, w = crop.shape
        offsets[dom] = {'x0': x0, 'y0': y0, 'width': w, 'height': h, 'mean_abs_diff': round(diff, 3)}
        print(f"  Crop located at x0={x0}, y0={y0} ({w}x{h}), mean |diff| = {diff:.3f}"
              f"{'  (exact sub-rectangle)' if diff < 0.5 else '  WARNING: not an exact match'}", flush=True)
        del full, crop

        with open(cfg['meta'], encoding='utf-8') as f:
            meta = json.load(f)
        if (meta['image_height'], meta['image_width']) != load_gray_shape(os.path.join(cfg['gt_dir'], cfg['full_np'])):
            print("  WARNING: grid_metadata size differs from the full NP image used for the GT", flush=True)

        gt = build_gt(cfg['gt_dir'])
        if gt.shape != (h, w):
            print(f"  ERROR: GT masks {gt.shape} do not match the crop {(h, w)}", flush=True)
            continue
        annotated = gt != IGNORE
        print(f"  GT: {annotated.mean() * 100:.1f}% of the crop is mortar, "
              f"aggregates = {gt[annotated].mean() * 100:.1f}% of the mortar", flush=True)
        Image.fromarray(gt).save(os.path.join(args.output_dir, f'{dom}_gt_crop.png'))

        for exp in args.experiments:
            ecfg = EXPERIMENTS[exp]
            for src_name, src in ecfg['sources'].items():
                if args.sources and src_name not in args.sources:
                    continue
                pred, info = predictions_in_crop(meta, (x0, y0, w, h), src, ecfg['root'], dom)
                if pred is None:
                    print(f"  [skip] {exp}/{src_name}: missing {info}", flush=True)
                    continue
                pred = green_to_aggregate(pred, src['green_class'], ecfg['labels_swapped'])
                s = scores(gt, pred)
                save_error_map(gt, pred, os.path.join(args.output_dir, f'{dom}_{exp}_{src_name}_errors.png'))
                rows.append({'domain': dom, 'experiment': exp, 'source': src_name, 'patches_used': info, **s})
                print(f"  {exp:11s} {src_name:6s} mIoU={s['miou']:6.2f}  IoU binder={s['iou_binder']:6.2f}  "
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


def load_gray_shape(path):
    """(height, width) of an image without decoding the pixels."""
    with Image.open(path) as img:
        return img.size[1], img.size[0]


if __name__ == '__main__':
    sys.exit(main())
