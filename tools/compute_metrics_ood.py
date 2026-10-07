#!/usr/bin/env python3
"""
OOD Surrogate Metrics Computation Tool for Mortar Semantic Segmentation.
Bachelor's Thesis in Computer Engineering — Elia Awad, University of Florence (UNIFI).

Computes surrogate metrics without Ground Truth as defined in the thesis evaluation framework:
1. ISN (Speckle Noise Index): fraction of pixels in connected components < tau px (physical continuity).
2. FAA (Fraction of Aggregate Area Phi): aggregate area / valid mortar area (petrographic sanity check).
3. TTD (Test-Time Disagreement): pixel-wise disagreement percentage between pairs of models.

Operates directly on full-resolution reconstructed segmentation maps (PNG).
Automatically handles validity mask resizing for ARCHEO_02 (calibrated 0.57x resolution).
"""

import os
import sys
import gc
import json
import argparse
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tabulate import tabulate
import csv
from datetime import datetime

# Disable PIL image size limit for gigapixel stitched sections (e.g., ARCHEO_02)
Image.MAX_IMAGE_PIXELS = None


# Default path registry across the project
DEFAULT_PATHS = {
    'domains': {
        '1_SCALA': {
            'validity_mask': 'output/inspection/1_SCALA_validity_mask.tif',
            'fallback_mask': 'output/inspection/1_SCALA_validity_mask.png',
            'recon_pattern': 'output/inference_nuove/inference_nuove_{exp}/1_SCALA_reconstructed.png'
        },
        'UNITO_B': {
            'validity_mask': 'output/inspection/UNITO_B_validity_mask.tif',
            'fallback_mask': 'output/inspection/UNITO_B_validity_mask.png',
            'recon_pattern': 'output/inference_nuove/inference_nuove_{exp}/UNITO_B_reconstructed.png'
        },
        'ARCHEO_02': {
            'validity_mask': 'output/inspection/ARCHEO_02_validity_mask.tif',
            'fallback_mask': 'output/inspection/ARCHEO_02_validity_mask.png',
            'recon_pattern': 'output/inference_5x/inference_5x_scale=057_{exp}/ARCHEO_02/ARCHEO_02_calibrated_reconstructed_segmap.png'
        }
    },
    'experiments': ['baseline', 'exp1', 'exp2', 'exp3', 'exp4a']
}


def find_file(primary_path: str, fallback_path: str = None) -> str:
    """Returns primary path if exists, otherwise fallback path or raises FileNotFoundError."""
    if os.path.exists(primary_path):
        return primary_path
    if fallback_path and os.path.exists(fallback_path):
        return fallback_path
    raise FileNotFoundError(f"File not found: '{primary_path}' (fallback: '{fallback_path}')")


def load_validity_mask(mask_path: str, target_shape_hw: tuple) -> np.ndarray:
    """
    Loads validity mask according to Notari / inspect_and_mask convention:
      0: Valid Mortar Section (Black)
    255: Ignored Background / Resin / Scan Padding (White)
    If mask dimensions do not match target_shape_hw (H, W), resizes using
    NEAREST-neighbor interpolation to ensure coordinate alignment (e.g., ARCHEO_02).
    """
    print(f"  [Mask] Loading validity mask: {mask_path}")
    pil_mask = Image.open(mask_path)
    mask_arr = np.array(pil_mask)

    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]

    target_h, target_w = target_shape_hw
    orig_h, orig_w = mask_arr.shape[:2]

    if (orig_h, orig_w) != (target_h, target_w):
        print(f"  [Mask] Resizing mask from ({orig_h}, {orig_w}) to ({target_h}, {target_w}) [NEAREST]")
        mask_resized = cv2.resize(mask_arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        binary_mask = mask_resized < 128
    else:
        binary_mask = mask_arr < 128

    valid_px = int(np.count_nonzero(binary_mask))
    total_px = target_h * target_w
    pct = (valid_px / total_px) * 100 if total_px > 0 else 0
    print(f"  [Mask] Valid mortar pixels (value 0): {valid_px:,} ({pct:.2f}% of canvas)")
    return binary_mask


def load_reconstructed_prediction(pred_path: str) -> np.ndarray:
    """
    Loads reconstructed PNG segmentation map and converts to uint8 binary map:
    0 = Binder Matrix (black pixel [0, 0, 0])
    1 = Aggregates (green pixel [0, 255, 0])
    """
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction map not found: {pred_path}")

    print(f"  [Pred] Loading prediction map: {pred_path}")
    pil_img = Image.open(pred_path).convert('RGB')
    arr = np.array(pil_img, dtype=np.uint8)

    # Inverted palette assigns green [0, 255, 0] to aggregates.
    # Green channel > 127 indicates aggregate; 0 indicates binder or unclassified background.
    pred_binary = (arr[:, :, 1] > 127).astype(np.uint8)
    del arr, pil_img
    return pred_binary


def compute_isn_and_faa(pred_bin: np.ndarray, valid_mask: np.ndarray, tau: int = 100) -> dict:
    """
    Computes:
    - FAA (Fraction of Aggregate Area Phi) = count(Aggregate & Valid) / total_valid_px
    - ISN (Speckle Noise Index) = pixels in connected components < tau / total_valid_px
    - Per-class ISN breakdown (Binder vs Aggregates)
    """
    total_valid_px = int(np.count_nonzero(valid_mask))
    if total_valid_px == 0:
        return {
            'total_valid_px': 0,
            'faa_ratio': 0.0,
            'faa_percent': 0.0,
            'isn_total_ratio': 0.0,
            'isn_total_percent': 0.0,
            'speckle_px_total': 0,
            'speckle_components_total': 0,
            'isn_binder_percent': 0.0,
            'isn_aggregate_percent': 0.0,
            'isn_binder_ratio': 0.0,
            'isn_aggregate_ratio': 0.0
        }

    # 1. FAA: Fraction of Aggregate Area
    agg_valid_px = int(np.count_nonzero((pred_bin == 1) & valid_mask))
    faa_ratio = agg_valid_px / total_valid_px
    faa_percent = faa_ratio * 100.0

    # 2. ISN: Connected component analysis for each class
    speckle_px_total = 0
    speckle_comp_total = 0
    speckle_px_by_class = {0: 0, 1: 0}
    speckle_comp_by_class = {0: 0, 1: 0}

    # 8-connectivity for natural 2D physical spatial continuity
    for cls_id in [0, 1]:
        cls_mask = ((pred_bin == cls_id) & valid_mask).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)

        # Label 0 represents background (0-valued pixels)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < tau:
                speckle_px_total += area
                speckle_comp_total += 1
                speckle_px_by_class[cls_id] += area
                speckle_comp_by_class[cls_id] += 1

        del cls_mask, labels, stats
        gc.collect()

    isn_total_ratio = speckle_px_total / total_valid_px
    isn_total_percent = isn_total_ratio * 100.0
    isn_binder_ratio = speckle_px_by_class[0] / total_valid_px
    isn_aggregate_ratio = speckle_px_by_class[1] / total_valid_px

    return {
        'total_valid_px': total_valid_px,
        'faa_ratio': round(faa_ratio, 6),
        'faa_percent': round(faa_percent, 2),
        'isn_total_ratio': round(isn_total_ratio, 6),
        'isn_total_percent': round(isn_total_percent, 3),
        'speckle_px_total': int(speckle_px_total),
        'speckle_components_total': int(speckle_comp_total),
        'isn_binder_percent': round(isn_binder_ratio * 100.0, 3),
        'isn_aggregate_percent': round(isn_aggregate_ratio * 100.0, 3),
    }


def compute_ttd(pred_a: np.ndarray, pred_b: np.ndarray, valid_mask: np.ndarray) -> dict:
    """
    Computes Test-Time Disagreement (TTD) pixel-wise between two models:
    TTD(A, B) = count(pred_A != pred_B & valid) / count(valid)
    """
    total_valid_px = int(np.count_nonzero(valid_mask))
    if total_valid_px == 0:
        return {'ttd_ratio': 0.0, 'ttd_percent': 0.0, 'disagree_px': 0}

    disagree_mask = (pred_a != pred_b) & valid_mask
    disagree_px = int(np.count_nonzero(disagree_mask))
    ttd_ratio = disagree_px / total_valid_px
    ttd_percent = ttd_ratio * 100.0

    return {
        'ttd_ratio': round(ttd_ratio, 6),
        'ttd_percent': round(ttd_percent, 2),
        'disagree_px': disagree_px
    }


def main():
    parser = argparse.ArgumentParser(description="OOD Surrogate Metrics Computation Tool (ISN, FAA, TTD)")
    parser.add_argument('--domains', nargs='+', default=list(DEFAULT_PATHS['domains'].keys()),
                        help="Target domains to evaluate (e.g., 1_SCALA UNITO_B ARCHEO_02)")
    parser.add_argument('--experiments', nargs='+', default=DEFAULT_PATHS['experiments'],
                        help="Experiments to compare (e.g., baseline exp1 exp2 exp3 exp4a)")
    parser.add_argument('--tau', type=int, default=100,
                        help="Speckle noise threshold in pixels (default: 100)")
    parser.add_argument('--output-dir', type=str, default='output/metrics',
                        help="Output directory for JSON, CSV, and Markdown reports")
    parser.add_argument('--export-md', action='store_true', default=True,
                        help="Export formatted Markdown report for thesis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("OOD SURROGATE METRICS EVALUATION FRAMEWORK — ELIA AWAD THESIS (UNIFI)")
    print(f"Selected Domains    : {args.domains}")
    print(f"Selected Experiments: {args.experiments}")
    print(f"Speckle Threshold (tau): < {args.tau} px")
    print(f"Output Directory    : {args.output_dir}")
    print("=" * 80)

    all_results = {
        'metadata': {
            'tau': args.tau,
            'domains': args.domains,
            'experiments': args.experiments
        },
        'domains': {}
    }

    summary_rows = []
    ttd_summary_rows = []

    for dom_name in args.domains:
        if dom_name not in DEFAULT_PATHS['domains']:
            print(f"Warning: Unknown domain '{dom_name}', skipping.")
            continue

        print(f"\n" + "#" * 70)
        print(f"DOMAIN ANALYSIS: {dom_name}")
        print("#" * 70)

        dom_cfg = DEFAULT_PATHS['domains'][dom_name]
        mask_path = find_file(dom_cfg['validity_mask'], dom_cfg.get('fallback_mask'))

        # 1. Load predictions for all available experiments
        loaded_preds = {}
        target_shape = None

        for exp in args.experiments:
            recon_path = dom_cfg['recon_pattern'].format(exp=exp)
            if not os.path.exists(recon_path):
                print(f"  [Skip] Prediction missing for {exp}: {recon_path}")
                continue

            try:
                pred_bin = load_reconstructed_prediction(recon_path)
                loaded_preds[exp] = pred_bin
                if target_shape is None:
                    target_shape = pred_bin.shape
                else:
                    assert target_shape == pred_bin.shape, f"Shape mismatch for {exp} in {dom_name}"
            except Exception as e:
                print(f"  [Error] Failed to load {recon_path}: {e}")

        if not loaded_preds:
            print(f"No predictions loaded for {dom_name}.")
            continue

        # 2. Load and align validity mask
        valid_mask = load_validity_mask(mask_path, target_shape)

        domain_metrics = {
            'target_shape': list(target_shape),
            'valid_pixels': int(np.count_nonzero(valid_mask)),
            'experiments': {},
            'ttd_matrix': {}
        }

        # 3. Compute ISN and FAA for each experiment
        print(f"\n--- Computing ISN and FAA ({dom_name}) ---")
        for exp, pred_bin in loaded_preds.items():
            print(f"  Processing experiment: {exp} ...", end="", flush=True)
            res = compute_isn_and_faa(pred_bin, valid_mask, tau=args.tau)
            domain_metrics['experiments'][exp] = res
            print(" Done.")

        # Determine baseline speckle for delta calculation (if baseline was evaluated in this domain)
        baseline_isn = None
        if 'baseline' in domain_metrics['experiments']:
            baseline_isn = domain_metrics['experiments']['baseline']['speckle_px_total']

        for exp in loaded_preds.keys():
            res = domain_metrics['experiments'][exp]
            delta_baseline_str = "-"
            if baseline_isn is not None and baseline_isn > 0 and exp != 'baseline':
                pct_diff = ((res['speckle_px_total'] - baseline_isn) / baseline_isn) * 100.0
                delta_baseline_str = f"{pct_diff:+.1f}%"

            summary_rows.append([
                dom_name,
                exp.upper(),
                f"{res['faa_percent']:.2f}%",
                f"{res['isn_total_percent']:.3f}%",
                f"{res['speckle_px_total']:,}",
                delta_baseline_str,
                f"{res['isn_binder_percent']:.3f}%",
                f"{res['isn_aggregate_percent']:.3f}%"
            ])

        # 4. Compute TTD (Test-Time Disagreement) between model pairs
        print(f"\n--- Computing TTD across model pairs ({dom_name}) ---")
        exp_keys = list(loaded_preds.keys())
        for i in range(len(exp_keys)):
            exp_a = exp_keys[i]
            for j in range(i + 1, len(exp_keys)):
                exp_b = exp_keys[j]
                ttd_res = compute_ttd(loaded_preds[exp_a], loaded_preds[exp_b], valid_mask)
                pair_key = f"{exp_a}_vs_{exp_b}"
                domain_metrics['ttd_matrix'][pair_key] = ttd_res

                note = ""
                if (exp_a == 'exp4a' and exp_b == 'exp1') or (exp_a == 'exp1' and exp_b == 'exp4a'):
                    note = "★ Exp4A vs Exp1 (Preservation/Drift)"
                elif (exp_a == 'exp4a' and exp_b == 'baseline') or (exp_a == 'baseline' and exp_b == 'exp4a'):
                    note = "★ Exp4A vs Baseline (Adaptation Impact)"
                elif (exp_a == 'exp2' and exp_b == 'baseline') or (exp_a == 'baseline' and exp_b == 'exp2'):
                    note = "Exp2 vs Baseline (Heavy Photo)"

                ttd_summary_rows.append([
                    dom_name,
                    exp_a.upper(),
                    exp_b.upper(),
                    f"{ttd_res['ttd_percent']:.2f}%",
                    f"{ttd_res['disagree_px']:,}",
                    note
                ])

        all_results['domains'][dom_name] = domain_metrics

        del loaded_preds, valid_mask
        gc.collect()

    # =========================================================================
    # PRINT SUMMARY TABLES AND EXPORT REPORTS
    # =========================================================================
    headers_summary = [
        "Domain", "Model", "FAA (Aggregate %)", "ISN Total %",
        "Speckle Pixels", "Δ Baseline", "ISN Binder %", "ISN Aggregate %"
    ]
    print("\n" + "=" * 80)
    print("SUMMARY TABLE 1: ISN (SPECKLE NOISE) & FAA (FRACTION OF AGGREGATE AREA)")
    print("=" * 80)
    print(tabulate(summary_rows, headers=headers_summary, tablefmt="github"))

    headers_ttd = ["Domain", "Model A", "Model B", "TTD (%)", "Discordant Pixels", "Relevance"]
    print("\n" + "=" * 80)
    print("SUMMARY TABLE 2: TEST-TIME DISAGREEMENT (TTD)")
    print("=" * 80)
    print(tabulate(ttd_summary_rows, headers=headers_ttd, tablefmt="github"))

    # Save JSON
    json_path = os.path.join(args.output_dir, 'ood_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Detailed metrics saved to JSON: {json_path}")

    # Save CSV Summary
    csv_summary_path = os.path.join(args.output_dir, 'ood_metrics_summary.csv')
    with open(csv_summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers_summary)
        writer.writerows(summary_rows)
    print(f"[OK] ISN/FAA summary table saved to CSV: {csv_summary_path}")

    # Save CSV TTD
    csv_ttd_path = os.path.join(args.output_dir, 'ood_ttd_matrix.csv')
    with open(csv_ttd_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers_ttd)
        writer.writerows(ttd_summary_rows)
    print(f"[OK] TTD matrix table saved to CSV: {csv_ttd_path}")

    # Save Markdown Report
    if args.export_md:
        md_path = os.path.join(args.output_dir, 'report_metriche_ood_calcolate.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Quantitative OOD Surrogate Metrics Report\n\n")
            f.write(f"- **Speckle Noise Threshold ($\\tau$):** {args.tau} px\n")
            f.write(f"- **Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 1. Speckle Noise Index (ISN) and Fraction of Aggregate Area (FAA)\n\n")
            f.write(tabulate(summary_rows, headers=headers_summary, tablefmt="github"))
            f.write("\n\n## 2. Test-Time Disagreement (TTD)\n\n")
            f.write(tabulate(ttd_summary_rows, headers=headers_ttd, tablefmt="github"))
            f.write("\n")
        print(f"[OK] Markdown report generated for thesis: {md_path}")

    print("\nExecution successfully completed!")


if __name__ == '__main__':
    main()
