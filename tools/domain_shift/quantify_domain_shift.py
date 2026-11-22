#!/usr/bin/env python3
"""
Domain shift quantification (thesis, Chapter 3) between the historical sections (Source) and the three
new sections (1_SCALA, UNITO_B, ARCHEO_02 rescaled by S = 0.57), and derivation of the parameters of the
data-driven photometric augmentation (thesis, Section 4.4). The statistics are in semseg/domain_shift.py.

  - every statistic uses only VALID pixels (alignment padding excluded through valid/<patch>.png);
  - per-channel distributions (RGB, HSV, CIELAB) are compared with the Wasserstein distance W1, on the
    circle for the hue, normalized by the largest W1 between one historical section and the others;
  - per-patch statistics: median V, median S, gray standard deviation (contrast);
  - sharpness is contrast-normalized, var(Laplacian) / var(gray), and translated into the equivalent
    Gaussian blur sigma (softer targets) or Gaussian noise sigma (sharper targets) through calibration
    curves measured on the Source;
  - texture: GLCM correlation on the content rectangle of every patch;
  - Source patches are sampled stratified by section.

Data-driven parameters (data_driven_parameters.csv):
  - gamma, contrast and saturation are fitted in sequence on Source calibration patches with the same
    torchvision functions used in training, so that the median V, the median gray standard deviation
    and the median S of the transformed Source match those of each target;
  - blur and noise sigmas come from the calibration curves;
  - every range includes the identity and widens the measured extremes by 5% (rule_low, rule_high).
  The ranges used in training (configs/ablation/data_driven/) start from this rule, with the manual
  changes described in the thesis (Table 4.5).

Outputs (default output/domain_shift_analysis/):
  domain_shift_report.csv         long table: domain, modality, metric, key, value
  augmentation_suggestions.csv    per modality: median ratios per domain and a suggested range
  data_driven_parameters.csv      fitted values per target and the range given by the 5% rule
  intra_source_w1.csv             W1 of each historical section against the others
  blur_calibration.csv            sigma -> residual normalized sharpness on the Source
  noise_calibration.csv           noise sigma -> normalized sharpness gain on the Source
  white_point.csv                 NP colour of the brightest pixels (illumination check)
  hue_by_section.csv              circular hue statistics per Source section and per target
  domain_shift_stats.npz          histograms and per-patch statistics (input of plot_domain_shift.py)
  w1_normalized_heatmap.png, sharpness.png, glcm_correlation.png

Usage (from the repository root, CPU only):
    python tools/preprocessing/make_valid_masks.py          # once, writes the valid/ masks
    python tools/domain_shift/quantify_domain_shift.py
    python tools/domain_shift/plot_domain_shift.py         # figures of the analysis
"""

import os
import sys
import csv
import json
import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from semseg.domain_shift import (BLUR_SIGMAS, CHANNELS, DOMAINS, GLCM_DISTANCES, HUE_BINS, MODALITIES,
                                  NOISE_SIGMAS, RANGE_MARGIN, TARGETS, accumulate, blur_calibration,
                                  calibration_set, circular_concentration, circular_mean_deg,
                                  equivalent_noise_sigma, equivalent_sigma, fit_tone, hist_quantile,
                                  load_calibration, noise_calibration, rule_range, sample_patches, section_of,
                                  signed_deg, to_tensor, w1, w1_circular)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    parser = argparse.ArgumentParser(description="Domain shift quantification (valid pixels only)")
    parser.add_argument('--n-samples', type=int, default=400, help="Patches per domain (Source: stratified by section)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default=os.path.join('output', 'domain_shift_analysis'))
    parser.add_argument('--skip-glcm', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    samples = {d: sample_patches(root, args.n_samples, args.seed, stratified=(d == 'Source'))
               for d, root in DOMAINS.items()}
    for d, s in samples.items():
        has_valid = os.path.isdir(os.path.join(DOMAINS[d], 'valid'))
        print(f"{d:10s}: {len(s)} patches sampled" + ("" if has_valid else "  WARNING: no valid/ masks, run tools/preprocessing/make_valid_masks.py"))
    cal_files = calibration_set(samples['Source'])
    print(f"calibration / fitting: {len(cal_files)} Source patches from {len({section_of(f) for f in cal_files})} sections")

    rows, suggestions, intra_rows, dd_rows, hue_rows = [], [], [], [], []
    add = lambda dom, mod, metric, key, value: rows.append(
        {'domain': dom, 'modality': mod, 'metric': metric, 'key': key, 'value': value})
    stats, calib, noise_calib = {}, {}, {}

    for mod_name, mod in MODALITIES.items():
        print(f"\n=== modality {mod_name} ===")
        stats[mod_name] = {d: accumulate(s, mod, not args.skip_glcm) for d, s in samples.items()}
        cal = load_calibration(cal_files, mod)
        calib[mod_name] = blur_calibration(cal)
        noise_calib[mod_name] = noise_calibration(cal, args.seed)
        src = stats[mod_name]['Source']

        # --- intra-source reference: each section vs the others
        sections = src['hist_by_section']
        intra = {c: {} for c in CHANNELS}
        for sec, hists in sections.items():
            for c in CHANNELS:
                rest = sum(sections[o][c] for o in sections if o != sec)
                intra[c][sec] = (w1_circular if c == 'H' else w1)(rest[:HUE_BINS] if c == 'H' else rest,
                                                                   hists[c][:HUE_BINS] if c == 'H' else hists[c])
        for c in CHANNELS:
            vals = intra[c]
            worst = max(vals, key=vals.get)
            intra_rows.append({'modality': mod_name, 'channel': c, 'median': round(float(np.median(list(vals.values()))), 2),
                               'max': round(vals[worst], 2), 'max_section': worst,
                               **{s: round(v, 2) for s, v in sorted(vals.items())}})

        # --- per-domain statistics
        for d in DOMAINS:
            st = stats[mod_name][d]
            for c in CHANNELS:
                h = st['hist'][c]
                add(d, mod_name, 'median', c, hist_quantile(h, 0.5))
                add(d, mod_name, 'iqr', c, hist_quantile(h, 0.75) - hist_quantile(h, 0.25))
            add(d, mod_name, 'hue_circular_mean_deg', 'H(S>=30)', round(circular_mean_deg(st['hue_sat']), 1))
            add(d, mod_name, 'hue_concentration_R', 'H(S>=30)', round(circular_concentration(st['hue_sat']), 3))
            norm_sharp = np.array(st['lap']) / np.array(st['grayvar'])
            add(d, mod_name, 'laplacian_var_valid', 'median', round(float(np.nanmedian(st['lap'])), 2))
            add(d, mod_name, 'gray_std', 'median', round(float(np.nanmedian(st['gray_std'])), 2))
            add(d, mod_name, 'norm_sharpness', 'median', round(float(np.nanmedian(norm_sharp)), 5))
            add(d, mod_name, 'patch_median_V', 'median', round(float(np.nanmedian(st['v_med'])), 2))
            add(d, mod_name, 'patch_median_S', 'median', round(float(np.nanmedian(st['s_med'])), 2))
            if st['glcm_corr']:
                corr, cont = np.mean(st['glcm_corr'], 0), np.mean(st['glcm_contrast'], 0)
                for i, dist in enumerate(GLCM_DISTANCES):
                    add(d, mod_name, 'glcm_correlation', f'd={dist}', round(float(corr[i]), 4))
                    add(d, mod_name, 'glcm_contrast', f'd={dist}', round(float(cont[i]), 3))
                add(d, mod_name, 'glcm_patches', 'n', len(st['glcm_corr']))

        # --- hue per Source section and per target (saturated pixels only)
        ang = signed_deg(np.arange(HUE_BINS) * 2.0)
        for d in DOMAINS:
            st = stats[mod_name][d]
            total_sat = max(sum(st['sat_by_section'].values()), 1)
            for g in sorted(st['hue_by_section'], key=lambda s: int(s[3:]) if s[3:].isdigit() else 0):
                h = st['hue_by_section'][g]
                n = max(h.sum(), 1)
                hue_rows.append({
                    'modality': mod_name, 'domain': d, 'group': g if d == 'Source' else d,
                    'patches': st['section'].count(g),
                    'saturated_fraction': round(st['sat_by_section'][g] / max(st['valid_by_section'][g], 1), 3),
                    'share_of_domain_saturated': round(st['sat_by_section'][g] / total_sat, 3),
                    'circular_mean_deg': round(circular_mean_deg(h), 1),
                    'concentration_R': round(circular_concentration(h), 3),
                    'mode_deg': round(float(ang[int(np.argmax(h))]), 1),
                    'mass_-90_-20': round(float(h[(ang >= -90) & (ang <= -20)].sum() / n), 3),
                    'mass_0_60': round(float(h[(ang >= 0) & (ang <= 60)].sum() / n), 3)})

        # --- shift of every target vs Source
        src_ns = float(np.nanmedian(np.array(src['lap']) / np.array(src['grayvar'])))
        ratios = {'brightness (V)': {}, 'saturation (S)': {}, 'contrast (gray std)': {}, 'blur sigma': {}, 'hue shift deg': {}}
        noise_eq = {}
        for d in TARGETS:
            st = stats[mod_name][d]
            for c in CHANNELS:
                hs, ht = src['hist'][c], st['hist'][c]
                dist = w1_circular(hs[:HUE_BINS], ht[:HUE_BINS]) if c == 'H' else w1(hs, ht)
                iqr = hist_quantile(hs, 0.75) - hist_quantile(hs, 0.25)
                add(d, mod_name, 'w1', c, round(dist, 2))
                add(d, mod_name, 'w1_over_source_iqr', c, round(dist / max(iqr, 1), 3))
                add(d, mod_name, 'w1_over_intra_source_max', c, round(dist / max(max(intra[c].values()), 1e-6), 3))
            ratios['brightness (V)'][d] = hist_quantile(st['hist']['V'], 0.5) / max(hist_quantile(src['hist']['V'], 0.5), 1)
            ratios['saturation (S)'][d] = hist_quantile(st['hist']['S'], 0.5) / max(hist_quantile(src['hist']['S'], 0.5), 1)
            ratios['contrast (gray std)'][d] = float(np.nanmedian(st['gray_std']) / np.nanmedian(src['gray_std']))
            tgt_ns = float(np.nanmedian(np.array(st['lap']) / np.array(st['grayvar'])))
            add(d, mod_name, 'norm_sharpness_ratio', 'target/source', round(tgt_ns / src_ns, 3))
            ratios['blur sigma'][d] = equivalent_sigma(tgt_ns / src_ns, calib[mod_name])
            noise_eq[d] = equivalent_noise_sigma(tgt_ns / src_ns, noise_calib[mod_name])
            add(d, mod_name, 'noise_sigma_equivalent', 'sigma', round(noise_eq[d], 2))
            dh = (circular_mean_deg(st['hue_sat']) - circular_mean_deg(src['hue_sat']) + 180) % 360 - 180
            ratios['hue shift deg'][d] = dh

        for quantity, per_dom in ratios.items():
            vals = np.array(list(per_dom.values()), dtype=float)
            if not np.isfinite(vals).any():
                suggested = 'n/a'
            elif quantity == 'blur sigma':
                suggested = [0.0, round(float(np.nanmax(vals)), 2)]
            elif quantity == 'hue shift deg':
                suggested = [round(min(0.0, float(np.nanmin(vals))), 1), round(max(0.0, float(np.nanmax(vals))), 1)]
            else:
                suggested = [round(min(1.0, float(np.nanmin(vals))), 3), round(max(1.0, float(np.nanmax(vals))), 3)]
            suggestions.append({'modality': mod_name, 'quantity': quantity,
                                **{d: round(v, 3) for d, v in per_dom.items()},
                                'suggested_range': suggested})
        print("  " + " | ".join(f"{q}: " + ", ".join(f"{d}={v:.2f}" for d, v in pd.items()) for q, pd in ratios.items()))

        # --- data-driven parameters: gamma / contrast / saturation fitted on the calibration patches
        tensors = [(to_tensor(img), valid) for img, valid, _ in cal]
        fitted = {}
        for d in TARGETS:
            st = stats[mod_name][d]
            target = (float(np.nanmedian(st['v_med'])), float(np.nanmedian(st['gray_std'])), float(np.nanmedian(st['s_med'])))
            g, c, s, reached = fit_tone(tensors, target)
            fitted[d] = {'gamma': g, 'contrast': c, 'saturation': s}
            print(f"  fit {d}: gamma {g:.3f}, contrast {c:.3f}, saturation {s:.3f} | "
                  f"target V/std/S {target[0]:.1f}/{target[1]:.1f}/{target[2]:.1f}, reached {reached[0]:.1f}/{reached[1]:.1f}/{reached[2]:.1f}")
        for op, identity, decimals, per_dom in (
                ('gamma', 1.0, 2, {d: fitted[d]['gamma'] for d in TARGETS}),
                ('contrast', 1.0, 2, {d: fitted[d]['contrast'] for d in TARGETS}),
                ('saturation', 1.0, 2, {d: fitted[d]['saturation'] for d in TARGETS}),
                ('blur sigma', 0.0, 2, ratios['blur sigma']),
                ('noise sigma', 0.0, 1, noise_eq)):
            lo, hi = rule_range(list(per_dom.values()), identity, decimals)
            dd_rows.append({'modality': mod_name, 'operation': op, 'identity': identity,
                               **{d: round(v, 3) for d, v in per_dom.items()}, 'rule_low': lo, 'rule_high': hi})

    # ------------------------------------------------------------------ outputs
    def write_csv(name, data):
        if not data:
            return
        keys = []
        for r in data:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(args.output_dir, name), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(data)
        print(f"Saved {os.path.join(args.output_dir, name)}")

    write_csv('domain_shift_report.csv', rows)
    write_csv('augmentation_suggestions.csv', suggestions)
    write_csv('data_driven_parameters.csv', dd_rows)
    write_csv('intra_source_w1.csv', intra_rows)
    write_csv('blur_calibration.csv', [{'modality': m, **{f'sigma={s}': round(v, 4) for s, v in zip(BLUR_SIGMAS, c)}}
                                       for m, c in calib.items()])
    write_csv('noise_calibration.csv', [{'modality': m, **{f'sigma={s}': round(v, 4) for s, v in zip(NOISE_SIGMAS, c)}}
                                        for m, c in noise_calib.items()])
    white_rows = []
    for d in DOMAINS:
        wr = np.array(stats['NP'][d]['white_rgb'])
        if len(wr):
            r, g, b = np.median(wr, 0)
            white_rows.append({'domain': d, 'patches': len(wr), 'R': round(float(r), 1), 'G': round(float(g), 1),
                               'B': round(float(b), 1), 'B/R': round(float(b / r), 3)})
    write_csv('white_point.csv', white_rows)
    write_csv('hue_by_section.csv', hue_rows)

    # statistics for tools/domain_shift/plot_domain_shift.py
    arrays = {'channels': np.array(list(CHANNELS)), 'glcm_distances': np.array(GLCM_DISTANCES)}
    for m in MODALITIES:
        for d in DOMAINS:
            st, key = stats[m][d], f'{m}|{d}|'
            arrays[key + 'hist'] = np.array([st['hist'][c] for c in CHANNELS])
            arrays[key + 'hue'] = st['hue_sat']
            for k in ('lap', 'grayvar', 'gray_std', 'v_med', 's_med'):
                arrays[key + k] = np.array(st[k], dtype=float)
            arrays[key + 'section'] = np.array(st['section'])
            arrays[key + 'file'] = np.array(st['file'])
            arrays[key + 'glcm_corr'] = np.array(st['glcm_corr']).reshape(-1, len(GLCM_DISTANCES))
            arrays[key + 'glcm_contrast'] = np.array(st['glcm_contrast']).reshape(-1, len(GLCM_DISTANCES))
            names = sorted(st['hist_by_section'])
            arrays[key + 'sec_names'] = np.array(names)
            arrays[key + 'sec_hist'] = np.array([[st['hist_by_section'][s][c] for c in CHANNELS] for s in names])
            arrays[key + 'sec_hue'] = np.array([st['hue_by_section'][s] for s in names])
    np.savez_compressed(os.path.join(args.output_dir, 'domain_shift_stats.npz'), **arrays)
    print(f"Saved {os.path.join(args.output_dir, 'domain_shift_stats.npz')}")

    # normalized W1 heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, mod_name in zip(axes, MODALITIES):
        mat = np.array([[next(r['value'] for r in rows if r['domain'] == d and r['modality'] == mod_name
                              and r['metric'] == 'w1_over_intra_source_max' and r['key'] == c)
                         for d in TARGETS] for c in CHANNELS])
        im = ax.imshow(mat, cmap='RdYlGn_r', vmin=0, vmax=max(2.0, mat.max()), aspect='auto')
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha='center', va='center', fontsize=9)
        ax.set_xticks(range(len(TARGETS))); ax.set_xticklabels(TARGETS)
        ax.set_yticks(range(len(CHANNELS))); ax.set_yticklabels(list(CHANNELS))
        ax.set_title(f"{mod_name}: W1 / max intra-source W1 (>1 = beyond section variability)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(args.output_dir, 'w1_normalized_heatmap.png'), dpi=140); plt.close(fig)

    # sharpness
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod_name in zip(axes, MODALITIES):
        data = [np.array(stats[mod_name][d]['lap']) / np.array(stats[mod_name][d]['grayvar']) for d in DOMAINS]
        ax.boxplot([x[np.isfinite(x)] for x in data], showfliers=False)
        ax.set_xticks(range(1, len(DOMAINS) + 1)); ax.set_xticklabels(list(DOMAINS))
        ax.set_title(f"{mod_name}: normalized sharpness var(Lap)/var(gray), valid pixels")
    fig.tight_layout(); fig.savefig(os.path.join(args.output_dir, 'sharpness.png'), dpi=140); plt.close(fig)

    # GLCM correlation
    if not args.skip_glcm:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, mod_name in zip(axes, MODALITIES):
            for d in DOMAINS:
                c = stats[mod_name][d]['glcm_corr']
                if c:
                    ax.plot(GLCM_DISTANCES, np.mean(c, 0), marker='o', label=d, linestyle='--' if d == 'Source' else '-')
            ax.set_xscale('log'); ax.set_xticks(GLCM_DISTANCES); ax.set_xticklabels(GLCM_DISTANCES)
            ax.set_title(f"{mod_name}: GLCM correlation (content only, padding cropped)"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.output_dir, 'glcm_correlation.png'), dpi=140); plt.close(fig)

    with open(os.path.join(args.output_dir, 'run_info.json'), 'w', encoding='utf-8') as f:
        json.dump({'n_samples': args.n_samples, 'seed': args.seed, 'domains': DOMAINS,
                   'patches': {d: len(s) for d, s in samples.items()},
                   'calibration_patches': len(cal_files), 'range_margin': RANGE_MARGIN}, f, indent=2)
    print("\nDone.")


if __name__ == '__main__':
    main()
