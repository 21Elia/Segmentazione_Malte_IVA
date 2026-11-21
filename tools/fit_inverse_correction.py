#!/usr/bin/env python3
"""
Inverse photometric correction (target -> Source): fit and closure test.

The data-driven pipeline fits gamma, contrast and saturation that bring the Source onto each target.
Here the roles are swapped: the same fit (tools/quantify_domain_shift.fit_tone, same training functions,
same order gamma -> contrast -> saturation, same number of iterations) is run on the patches of each
target, so that their median V, median gray standard deviation and median S match those of the Source.
Each target and modality (NP, NX) gets its own parameters. Blur and noise are not corrected.

Inverse closure test: the normalized W1 (W1 / max intra-Source W1, the metric of closure_test.csv)
between the target and the Source is measured on the 9 channels, before and after the correction.

Inputs (from tools/quantify_domain_shift.py, default output/domain_shift_analysis_v2/):
  domain_shift_stats.npz   Source histograms and per-patch medians (reference)
  intra_source_w1.csv      normalization of W1

Outputs (default output/domain_shift_analysis_v2/inverse_correction/):
  inverse_correction_params.json   {target: {NP: {gamma, contrast, saturation}, NX: {...}}}
                                   input of tools/run_all_inferences_softmax.py --inverse-correction
  inverse_correction_fit.csv       fitted parameters, reciprocals of the forward fit, medians reached
  inverse_closure_test.csv         normalized W1 per channel, before and after the correction
  inverse_correction_preview.png   one patch per target and modality, before and after

Usage (from the repository root, CPU only):
    python tools/fit_inverse_correction.py
"""

import os
import sys
import csv
import json
import argparse
import warnings

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.quantify_domain_shift import (DOMAINS, TARGETS, MODALITIES, CHANNELS, HUE_BINS, sample_patches,
                                         load_pair, to_tensor, apply_tone, fit_tone, w1, w1_circular)

warnings.filterwarnings("ignore", category=RuntimeWarning)
IDENTITY = (1.0, 1.0, 1.0)


def read_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def channel_hists(tensors, params):
    """Histograms of the 9 channels on the valid pixels of the transformed patches."""
    H = {c: np.zeros(256) for c in CHANNELS}
    for t, valid in tensors:
        rgb = np.ascontiguousarray(apply_tone(t, *params).permute(1, 2, 0).numpy())
        spaces = {'rgb': rgb, 'hsv': cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), 'lab': cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)}
        for c, (space, i) in CHANNELS.items():
            H[c] += np.bincount(spaces[space][:, :, i][valid].ravel(), minlength=256)[:256]
    return H


def normalized_w1(H, source_hist, intra_max):
    out = {}
    for k, c in enumerate(CHANNELS):
        d = w1_circular(H[c][:HUE_BINS], source_hist[k][:HUE_BINS]) if c == 'H' else w1(H[c], source_hist[k])
        out[c] = d / max(intra_max[c], 1e-6)
    return out


def main():
    parser = argparse.ArgumentParser(description="Inverse photometric correction (target -> Source): fit and closure test")
    parser.add_argument('--analysis-dir', default=os.path.join('output', 'domain_shift_analysis_v2'))
    parser.add_argument('--output-dir', default=None, help="Default: <analysis-dir>/inverse_correction")
    parser.add_argument('--n-samples', type=int, default=400, help="Must match quantify_domain_shift.py")
    parser.add_argument('--seed', type=int, default=42, help="Must match quantify_domain_shift.py")
    args = parser.parse_args()
    out_dir = args.output_dir or os.path.join(args.analysis_dir, 'inverse_correction')
    os.makedirs(out_dir, exist_ok=True)

    stats = dict(np.load(os.path.join(args.analysis_dir, 'domain_shift_stats.npz')))
    intra = {(r['modality'], r['channel']): float(r['max'])
             for r in read_csv(os.path.join(args.analysis_dir, 'intra_source_w1.csv'))}
    forward = {(r['modality'], r['operation']): r for r in read_csv(os.path.join(args.analysis_dir, 'exp4b_parameters.csv'))}

    params, fit_rows, closure_rows, previews = {}, [], [], {}
    for t in TARGETS:
        files = sample_patches(DOMAINS[t], args.n_samples, args.seed, stratified=False)
        print(f"\n=== {t}: {len(files)} patches ===", flush=True)
        params[t] = {}
        for m, mod in MODALITIES.items():
            tensors = []
            for f in files:
                img, valid = load_pair(f, mod)
                if img is not None and valid.any():
                    tensors.append((to_tensor(img), valid))

            # Source medians: same definition as the target medians of the forward fit
            reference = (float(np.nanmedian(stats[f'{m}|Source|v_med'])),
                         float(np.nanmedian(stats[f'{m}|Source|gray_std'])),
                         float(np.nanmedian(stats[f'{m}|Source|s_med'])))
            g, c, s, reached = fit_tone(tensors, reference)
            params[t][m] = {'gamma': round(g, 4), 'contrast': round(c, 4), 'saturation': round(s, 4)}

            fwd = {op: float(forward[(m, op)][t]) for op in ('gamma', 'contrast', 'saturation')}
            fit_rows.append({'target': t, 'modality': m, 'patches': len(tensors),
                             'gamma': round(g, 3), 'contrast': round(c, 3), 'saturation': round(s, 3),
                             '1/gamma_forward': round(1 / fwd['gamma'], 3),
                             '1/contrast_forward': round(1 / fwd['contrast'], 3),
                             '1/saturation_forward': round(1 / fwd['saturation'], 3),
                             'source_V': round(reference[0], 1), 'source_std': round(reference[1], 1), 'source_S': round(reference[2], 1),
                             'reached_V': round(reached[0], 1), 'reached_std': round(reached[1], 1), 'reached_S': round(reached[2], 1)})
            print(f"  {m}: gamma {g:.3f} (1/fwd {1 / fwd['gamma']:.3f}), contrast {c:.3f} (1/fwd {1 / fwd['contrast']:.3f}), "
                  f"saturation {s:.3f} (1/fwd {1 / fwd['saturation']:.3f}) | Source V/std/S "
                  f"{reference[0]:.1f}/{reference[1]:.1f}/{reference[2]:.1f}, reached {reached[0]:.1f}/{reached[1]:.1f}/{reached[2]:.1f}",
                  flush=True)

            # Inverse closure test: corrected target vs Source
            intra_max = {ch: intra[(m, ch)] for ch in CHANNELS}
            src_hist = stats[f'{m}|Source|hist']
            for label, p in (('before', IDENTITY), ('after', (g, c, s))):
                res = normalized_w1(channel_hists(tensors, p), src_hist, intra_max)
                closure_rows.append({'modality': m, 'target': t, 'correction': label,
                                     **{ch: round(x, 3) for ch, x in res.items()},
                                     'mean': round(float(np.mean(list(res.values()))), 3)})
            before, after = closure_rows[-2], closure_rows[-1]
            print("  normalized W1 before -> after: " + ", ".join(f"{ch} {before[ch]:.2f}->{after[ch]:.2f}" for ch in CHANNELS),
                  flush=True)

            # Preview: the patch with the most valid pixels
            idx = int(np.argmax([v.sum() for _, v in tensors]))
            previews[(t, m)] = (tensors[idx][0], apply_tone(tensors[idx][0], g, c, s))

    with open(os.path.join(out_dir, 'inverse_correction_params.json'), 'w', encoding='utf-8') as f:
        json.dump({'description': "Inverse photometric correction target -> Source (tools/fit_inverse_correction.py). "
                                  "Order: gamma -> contrast -> saturation, torchvision functions on uint8 RGB.",
                   'params': params}, f, indent=2)
    for name, rows in (('inverse_correction_fit.csv', fit_rows), ('inverse_closure_test.csv', closure_rows)):
        with open(os.path.join(out_dir, name), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    fig, axes = plt.subplots(len(TARGETS), 4, figsize=(16, 4 * len(TARGETS)))
    for i, t in enumerate(TARGETS):
        for j, m in enumerate(MODALITIES):
            for k, (img, label) in enumerate(zip(previews[(t, m)], ('originale', 'corretta'))):
                ax = axes[i, 2 * j + k]
                ax.imshow(img.permute(1, 2, 0).numpy()); ax.set_title(f"{t} {m} {label}"); ax.axis('off')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'inverse_correction_preview.png'), dpi=110); plt.close(fig)
    print(f"\nResults saved to {out_dir}")


if __name__ == '__main__':
    main()
