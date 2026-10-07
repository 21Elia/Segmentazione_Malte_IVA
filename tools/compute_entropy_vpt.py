#!/usr/bin/env python3
"""
Shannon Entropy (SE) & Test-Time Augmentation Variance (VPT) Computation Tool.
Bachelor's Thesis in Computer Engineering — Elia Awad, University of Florence (UNIFI).

Computes probability-based metrics under domain shift:
1. SE (Shannon Entropy per pixel):
   H(x, y) = - sum_c (p_c * log2(p_c))
   Measures internal model uncertainty/ambiguity in a single forward pass.
   Processes .npy (float16) softmax probability maps generated with `infer_mm.py --save-softmax`
   or `tools/run_all_inferences_softmax.py`.

2. VPT (Geometric C4 Test-Time Augmentation Variance):
   Computes pixel-wise variance of prediction across K=8 geometric transforms.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gc
from semseg.patch_geometry import paste_content
import json
import glob
import argparse
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tabulate import tabulate
import csv
from datetime import datetime

Image.MAX_IMAGE_PIXELS = None

DEFAULT_DOMAINS = {
    '1_SCALA': {
        'meta': 'data/nuove_patches/1_SCALA/grid_metadata.json',
        'validity_mask': 'output/inspection/1_SCALA_validity_mask.tif',
        'fallback_mask': 'output/inspection/1_SCALA_validity_mask.png',
        'softmax_pattern': 'output/inference_nuove/inference_nuove_{exp}/1_SCALA/MMSFormer-B3/{softmax_subdir}',
        'vpt_pattern': 'output/inference_nuove/inference_nuove_{exp}/1_SCALA/MMSFormer-B3/vpt'
    },
    'UNITO_B': {
        'meta': 'data/nuove_patches/UNITO_B/grid_metadata.json',
        'validity_mask': 'output/inspection/UNITO_B_validity_mask.tif',
        'fallback_mask': 'output/inspection/UNITO_B_validity_mask.png',
        'softmax_pattern': 'output/inference_nuove/inference_nuove_{exp}/UNITO_B/MMSFormer-B3/{softmax_subdir}',
        'vpt_pattern': 'output/inference_nuove/inference_nuove_{exp}/UNITO_B/MMSFormer-B3/vpt'
    },
    'ARCHEO_02': {
        'meta': 'data/patches_5x_scale=057/ARCHEO_02/grid_metadata.json',
        'validity_mask': 'output/inspection/ARCHEO_02_validity_mask.tif',
        'fallback_mask': 'output/inspection/ARCHEO_02_validity_mask.png',
        'softmax_pattern': 'output/inference_5x/inference_5x_scale=057_{exp}/ARCHEO_02/MMSFormer-B3/{softmax_subdir}',
        'vpt_pattern': 'output/inference_5x/inference_5x_scale=057_{exp}/ARCHEO_02/MMSFormer-B3/vpt'
    }
}

EXPERIMENTS = ['baseline', 'exp1', 'exp2', 'exp3', 'exp4a']
V2_EXPERIMENTS = ['v2_baseline', 'v2_exp1', 'v2_exp2', 'v2_exp4b']

# In-domain reference (historical test split, written by run_all_inferences_softmax.py
# --domains INDOMAIN_TEST). First-campaign models are scored against the old labels, whose
# swapped classes match their outputs; v2 models against the corrected labels.
INDOMAIN_PATTERN = 'output/inference_indomain/{exp}/MMSFormer-B3'
LABEL_MAPPING_2CLASS = np.array([0, 0, 1, 3] + [3] * 252)  # porosity -> binder, 255 -> ignore


def label_root_for(exp):
    return 'data/mortars_v2/label' if exp.startswith('v2_') else 'data/mortars/label'


def auroc(score, positive):
    """Rank-based AUROC of `score` for detecting `positive` (no sklearn needed)."""
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    n_pos = positive.sum(); n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def indomain_reference(experiments, softmax_subdir, output_dir, stride=7):
    """SE / VPT on the historical test split, SE on correct vs wrong pixels and AUROC(SE -> error)."""
    rows = []
    for exp in experiments:
        base = INDOMAIN_PATTERN.format(exp=exp)
        sm_dir = os.path.join(base, softmax_subdir)
        files = sorted(glob.glob(os.path.join(sm_dir, '*.npy')))
        if not files:
            print(f"  [Skip] no in-domain softmax for {exp}: {sm_dir}")
            continue
        se_all, err_all, vpt_sum, vpt_n = [], [], 0.0, 0
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            lbl_path = os.path.join(label_root_for(exp), stem + '.tif')
            if not os.path.exists(lbl_path):
                continue
            label = LABEL_MAPPING_2CLASS[np.array(Image.open(lbl_path))]
            probs = np.load(f).astype(np.float32)
            keep = label != 3
            se = compute_shannon_entropy_from_probs(probs)
            wrong = probs.argmax(0) != label
            se_all.append(se[keep][::stride]); err_all.append(wrong[keep][::stride])
            vpt_path = os.path.join(base, 'vpt', stem + '.npy')
            if os.path.exists(vpt_path):
                v = np.load(vpt_path)[keep]
                vpt_sum += float(v.sum()); vpt_n += v.size
        se_all, err_all = np.concatenate(se_all), np.concatenate(err_all)
        row = {
            'Model': exp.upper(), 'Patches': len(files),
            'Mean SE (bit)': round(float(se_all.mean()), 4),
            'SE | correct': round(float(se_all[~err_all].mean()), 4),
            'SE | wrong': round(float(se_all[err_all].mean()), 4),
            'AUROC SE->error': round(auroc(se_all, err_all), 4),
            'Error rate %': round(float(err_all.mean() * 100), 2),
            'Mean VPT': round(vpt_sum / vpt_n, 5) if vpt_n else '',
        }
        rows.append(row)
        print(f"  {exp:12s} " + "  ".join(f"{k}={v}" for k, v in row.items() if k != 'Model'))
    if rows:
        out = os.path.join(output_dir, f'indomain_reference_{softmax_subdir}.csv')
        with open(out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(f"[OK] In-domain reference saved to: {out}")


def load_indomain_reference(output_dir, softmax_subdir):
    """{MODEL: {'se': mean SE, 'vpt': mean VPT}} from a previous --indomain run, if any."""
    path = os.path.join(output_dir, f'indomain_reference_{softmax_subdir}.csv')
    ref = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                ref[r['Model']] = {'se': float(r['Mean SE (bit)']),
                                   'vpt': float(r['Mean VPT']) if r['Mean VPT'] else None}
    return ref


def find_file(primary_path: str, fallback_path: str = None) -> str:
    """Returns primary path if exists, otherwise fallback path or primary."""
    if os.path.exists(primary_path):
        return primary_path
    if fallback_path and os.path.exists(fallback_path):
        return fallback_path
    return primary_path


def compute_shannon_entropy_from_probs(prob_array: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Computes pixel-wise Shannon entropy in bits (base-2 logarithm)."""
    p = np.clip(prob_array.astype(np.float32), eps, 1.0)
    axis = 0 if prob_array.ndim == 3 and prob_array.shape[0] == 2 else -1
    entropy_hw = -np.sum(p * np.log2(p), axis=axis)
    return entropy_hw


def reconstruct_map(metadata_path: str, patch_dir: str, is_softmax: bool = True) -> tuple:
    """Stitches global map (entropy or VPT) from patch .npy files using grid_metadata.json."""
    if not os.path.exists(metadata_path) or not os.path.exists(patch_dir):
        return None, 0, 0

    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    img_h = meta['image_height']
    img_w = meta['image_width']
    patch_size = meta['patch_size']
    patches = meta['patches']

    canvas = np.full((img_h, img_w), np.nan, dtype=np.float32)
    reconstructed_count = 0
    missing_count = 0

    for p in patches:
        filename_npy = p['filename'].replace('.tif', '.npy')
        npy_path = os.path.join(patch_dir, filename_npy)

        if not os.path.exists(npy_path):
            missing_count += 1
            continue

        arr = np.load(npy_path)
        if is_softmax:
            patch_data = compute_shannon_entropy_from_probs(arr)
        else:
            patch_data = arr  # VPT map is already (H, W)

        # Real content pasted at its NP-frame position (semseg/patch_geometry.py)
        if paste_content(canvas, patch_data, p, patch_size):
            reconstructed_count += 1

    return canvas, reconstructed_count, missing_count


def save_heatmap(data_map: np.ndarray, valid_mask: np.ndarray, save_path: str, max_val: float = 1.0):
    """Generates and saves a colored heatmap."""
    clean_map = np.nan_to_num(data_map, nan=0.0)
    norm = np.clip(clean_map / max_val, 0.0, 1.0)
    uint8_img = (norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(uint8_img, cv2.COLORMAP_JET)
    heatmap[~valid_mask] = [0, 0, 0]
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    cv2.imwrite(save_path, heatmap)
    print(f"  [Heatmap] Saved heatmap to: {save_path}")


def evaluate_single_domain_model(meta_path, mask_path, data_dir, is_softmax=True):
    """Computes mean, median, std, p90 for a stitched map."""
    data_map, rec, miss = reconstruct_map(meta_path, data_dir, is_softmax=is_softmax)
    if data_map is None or rec == 0:
        return None

    pil_vm = Image.open(mask_path)
    vm_arr = np.array(pil_vm)
    if vm_arr.ndim == 3:
        vm_arr = vm_arr[:, :, 0]

    target_h, target_w = data_map.shape
    if vm_arr.shape[:2] != (target_h, target_w):
        vm_arr = cv2.resize(vm_arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    valid_mask = (vm_arr < 128) & (~np.isnan(data_map))
    valid_vals = data_map[valid_mask]
    if len(valid_vals) == 0:
        return None

    return {
        'mean': float(np.mean(valid_vals)),
        'median': float(np.median(valid_vals)),
        'std': float(np.std(valid_vals)),
        'p90': float(np.percentile(valid_vals, 90)),
        'valid_pixels': int(np.count_nonzero(valid_mask)),
        'data_map': data_map,
        'valid_mask': valid_mask
    }


def main():
    parser = argparse.ArgumentParser(description="Shannon Entropy (SE) & TTA Variance (VPT) Computation Tool")
    parser.add_argument('--all', action='store_true',
                        help="Run evaluation on all available models and domains automatically")
    parser.add_argument('--metric', choices=['se', 'vpt'], default='se',
                        help="Metric to compute: 'se' (Shannon Entropy) or 'vpt' (TTA Variance)")
    parser.add_argument('--domains', nargs='+', default=list(DEFAULT_DOMAINS.keys()),
                        help="Domains to evaluate")
    parser.add_argument('--experiments', nargs='+', default=EXPERIMENTS,
                        help="Experiments to evaluate")
    parser.add_argument('--output-dir', type=str, default='output/metrics',
                        help="Output directory for reports")
    parser.add_argument('--save-heatmaps', action='store_true',
                        help="Save colored heatmap PNGs for each model and domain")
    parser.add_argument('--variant', choices=['single', 'tta'], default='tta',
                        help="Softmax maps to use: single forward pass (softmax/) or TTA average (softmax_tta/)")
    parser.add_argument('--softmax-subdir', type=str, default=None,
                        help="Overrides the folder name. First-campaign outputs: 'softmax' (it holds the TTA average)")
    parser.add_argument('--indomain', action='store_true',
                        help="Compute the in-domain reference on the historical test split instead of the OOD domains")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    is_softmax = (args.metric == 'se')
    softmax_subdir = args.softmax_subdir or ('softmax' if args.variant == 'single' else 'softmax_tta')

    if args.indomain:
        print(f"IN-DOMAIN REFERENCE (historical test split), softmax folder '{softmax_subdir}'")
        indomain_reference(args.experiments, softmax_subdir, args.output_dir)
        return
    reference = load_indomain_reference(args.output_dir, softmax_subdir)

    print("=" * 80)
    print(f"EVALUATION OF PROBABILISTIC METRICS: {args.metric.upper()}")
    print(f"Domains Selected    : {args.domains}")
    print(f"Experiments Selected: {args.experiments}")
    print(f"Output Directory    : {args.output_dir}")
    print("=" * 80)

    summary_rows = []
    full_results = {}

    for dom_name in args.domains:
        if dom_name not in DEFAULT_DOMAINS:
            continue

        dom_info = DEFAULT_DOMAINS[dom_name]
        meta_path = dom_info['meta']
        mask_path = find_file(dom_info['validity_mask'], dom_info.get('fallback_mask'))
        dir_pattern = dom_info['softmax_pattern'] if is_softmax else dom_info['vpt_pattern']
        dir_pattern = dir_pattern.replace('{softmax_subdir}', softmax_subdir)

        print(f"\n--- Processing Domain: {dom_name} ---")
        full_results[dom_name] = {}

        for exp in args.experiments:
            data_dir = dir_pattern.format(exp=exp)
            if not os.path.exists(data_dir):
                print(f"  [Skip] Data folder missing for {exp}: {data_dir}")
                continue

            print(f"  Evaluating {exp.upper()} ...", end="", flush=True)
            res = evaluate_single_domain_model(meta_path, mask_path, data_dir, is_softmax=is_softmax)
            if res is None:
                print(" No valid files found.")
                continue

            full_results[dom_name][exp] = {
                'mean': round(res['mean'], 4),
                'median': round(res['median'], 4),
                'std': round(res['std'], 4),
                'p90': round(res['p90'], 4),
                'valid_pixels': res['valid_pixels']
            }

            unit = "bit" if is_softmax else ""
            ref_value = reference.get(exp.upper(), {}).get('se' if is_softmax else 'vpt')
            delta = f"{res['mean'] - ref_value:+.4f}" if ref_value is not None else "n/a"
            summary_rows.append([
                dom_name,
                exp.upper(),
                f"{res['mean']:.4f} {unit}".strip(),
                delta,
                f"{res['median']:.4f} {unit}".strip(),
                f"{res['std']:.4f} {unit}".strip(),
                f"{res['p90']:.4f} {unit}".strip(),
                f"{res['valid_pixels']:,}"
            ])
            print(" Done.")

            if args.save_heatmaps:
                heatmap_name = f"{dom_name}_{exp}_{args.metric}_heatmap.png"
                save_heatmap(res['data_map'], res['valid_mask'],
                             os.path.join(args.output_dir, heatmap_name),
                             max_val=1.0 if is_softmax else 0.25)

            del res
            gc.collect()

    if summary_rows:
        headers = ["Domain", "Model", f"Mean {args.metric.upper()}", "Δ vs in-domain", "Median", "Std Dev", "90th Pct", "Valid Pixels"]
        print("\n" + "=" * 80)
        print(f"SUMMARY TABLE: {args.metric.upper()} ACROSS ALL DOMAINS AND MODELS")
        print("=" * 80)
        print(tabulate(summary_rows, headers=headers, tablefmt="github"))

        # Save JSON
        json_out = os.path.join(args.output_dir, f"{args.metric}_summary_results.json")
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2)
        print(f"\n[OK] JSON summary saved to: {json_out}")

        # Save CSV
        csv_out = os.path.join(args.output_dir, f"{args.metric}_summary_results.csv")
        with open(csv_out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(summary_rows)
        print(f"[OK] CSV summary saved to: {csv_out}")

        # Save Markdown Report
        md_out = os.path.join(args.output_dir, f"{args.metric}_summary_results.md")
        with open(md_out, 'w', encoding='utf-8') as f:
            f.write(f"# Summary Report: {args.metric.upper()} Across All Domains and Models\n\n")
            f.write(f"- **Metric Evaluated:** {args.metric.upper()}\n")
            f.write(f"- **Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(tabulate(summary_rows, headers=headers, tablefmt="github"))
            f.write("\n")
        print(f"[OK] Markdown summary saved to: {md_out}")
    else:
        print("\nNo probability (.npy) files found yet. Run `tools/run_all_inferences_softmax.py` first.")


if __name__ == '__main__':
    main()
