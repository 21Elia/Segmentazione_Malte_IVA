#!/usr/bin/env python3
"""
Domain shift quantification (v2) between the historical sections (Source) and the three
out-of-domain sections (1_SCALA, UNITO_B, ARCHEO_02 at S=0.57).

What changed with respect to the first version (see report_verifica_domain_shift_e_analisi_exp4a):
  - every statistic uses only VALID pixels: the zero padding of the NP/NX alignment is excluded
    through the per-patch masks valid/<patch>.png (tools/make_valid_masks.py). The padding step
    used to dominate the Laplacian variance (Source NP 601 -> 218 without it);
  - hue is an angle: its Wasserstein distance is computed on the circle (OpenCV H in [0, 180));
  - W1 is reported raw and normalized by the Source inter-quartile range of the channel, and an
    intra-source reference is added (each historical section against the others): a shift is
    meaningful only if it exceeds the variability between sections of the same microscope;
  - directions and magnitudes come from MEDIAN RATIOS target / source (brightness, saturation,
    contrast), which map directly to multiplicative augmentation factors;
  - sharpness is contrast-normalized, var(Laplacian) / var(gray), and translated into the
    equivalent Gaussian blur sigma through a calibration curve measured on the Source;
  - GLCM descriptors are computed only on patches with at least 98% valid pixels;
  - Source patches are sampled stratified by section.

Outputs (default output/domain_shift_analysis_v2/):
  domain_shift_report_v2.csv      long table: domain, modality, metric, key, value
  augmentation_suggestions.csv    per modality: ratios per domain and a suggested factor range
  intra_source_w1.csv             W1 of each historical section against the others
  blur_calibration.csv            sigma -> residual normalized sharpness on the Source
  w1_normalized_heatmap.png, sharpness.png, glcm_correlation.png

Usage (from the repository root, CPU only):
    python tools/make_valid_masks.py          # once, writes the valid/ masks
    python tools/quantify_domain_shift.py
    python tools/quantify_domain_shift.py --n-samples 600 --skip-glcm
"""

import os
import re
import csv
import glob
import json
import random
import argparse
import warnings

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

warnings.filterwarnings("ignore", category=RuntimeWarning)

DOMAINS = {
    'Source': 'data/mortars_v2',
    '1_SCALA': 'data/nuove_patches/1_SCALA',
    'UNITO_B': 'data/nuove_patches/UNITO_B',
    'ARCHEO_02': 'data/patches_5x_scale=057/ARCHEO_02',
}
TARGETS = ['1_SCALA', 'UNITO_B', 'ARCHEO_02']
MODALITIES = {'NP': 'paralleli', 'NX': 'incrociati'}
CHANNELS = {"R": ("rgb", 0), "G": ("rgb", 1), "B": ("rgb", 2), "H": ("hsv", 0), "S": ("hsv", 1),
            "V": ("hsv", 2), "L*": ("lab", 0), "a*": ("lab", 1), "b*": ("lab", 2)}
HUE_BINS = 180          # OpenCV 8-bit hue range, 2 degrees per level
HUE_MIN_SATURATION = 30  # hue is undefined on grey pixels: circular statistics use S >= this
GLCM_DISTANCES = [1, 2, 5, 10, 20]
BLUR_SIGMAS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]


# ---------------------------------------------------------------------------- sampling

def section_of(path):
    m = re.match(r'sec(\d+)_', os.path.basename(path))
    return f"sec{m.group(1)}" if m else 'single'


def list_patches(root):
    files = sorted(glob.glob(os.path.join(root, 'paralleli', '*.tif')))
    return [f for f in files if not os.path.basename(f).startswith('sec5_')]


def sample_patches(root, n, seed, stratified):
    """Random sample of NP patch paths; stratified by section (proportional) for the Source."""
    files = list_patches(root)
    rng = random.Random(seed)
    if not stratified:
        rng.shuffle(files)
        return files[:n]
    groups = {}
    for f in files:
        groups.setdefault(section_of(f), []).append(f)
    out = []
    for sec, group in sorted(groups.items()):
        rng.shuffle(group)
        out += group[:max(1, round(n * len(group) / len(files)))]
    return out


def load_pair(np_path, modality):
    path = np_path if modality == 'paralleli' else 'incrociati'.join(np_path.rsplit('paralleli', 1))
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    valid_path = os.path.join(os.path.dirname(os.path.dirname(np_path)), 'valid',
                              os.path.basename(np_path).replace('.tif', '.png'))
    if os.path.exists(valid_path):
        valid = cv2.imread(valid_path, cv2.IMREAD_GRAYSCALE) > 127
    else:
        valid = img.sum(axis=2) > 0   # fallback: drop exact-black pixels (alignment padding)
    return img, valid


# ---------------------------------------------------------------------------- statistics

def w1(h1, h2):
    """1-D Wasserstein distance between two histograms (unit = channel levels)."""
    return float(np.abs(np.cumsum(h1 / h1.sum()) - np.cumsum(h2 / h2.sum())).sum())


def w1_circular(h1, h2):
    """W1 on the circle: min over the origin of sum |F - G - c|, attained at c = median(F - G)."""
    d = np.cumsum(h1 / h1.sum()) - np.cumsum(h2 / h2.sum())
    return float(np.abs(d - np.median(d)).sum())


def hist_quantile(h, q):
    return int(np.searchsorted(np.cumsum(h) / h.sum(), q))


def circular_mean_deg(h):
    """Circular mean of an OpenCV hue histogram, in degrees."""
    ang = np.deg2rad(np.arange(HUE_BINS) * 2.0)
    return float(np.rad2deg(np.arctan2((h * np.sin(ang)).sum(), (h * np.cos(ang)).sum())) % 360)


def circular_concentration(h):
    """Mean resultant length R in [0, 1]: near 0 the hue is spread and its mean is not meaningful."""
    ang = np.deg2rad(np.arange(HUE_BINS) * 2.0)
    n = max(h.sum(), 1)
    return float(np.hypot((h * np.sin(ang)).sum(), (h * np.cos(ang)).sum()) / n)


def lap_stats(gray, valid):
    """(var Laplacian, var gray) on valid pixels whose 3x3 neighbourhood is valid."""
    v = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if v.sum() < 1000:
        return np.nan, np.nan
    lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
    return float(lap[v].var()), float(gray[v].astype(np.float64).var())


def glcm_curves(gray):
    q = np.clip((gray.astype(np.float32) / 256 * 64).astype(np.uint8), 0, 63)
    g = graycomatrix(q, GLCM_DISTANCES, [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=64, symmetric=True, normed=True)
    return graycoprops(g, 'correlation').mean(1), graycoprops(g, 'contrast').mean(1)


def accumulate(files, modality, do_glcm):
    """Per-domain accumulators for one modality."""
    acc = {
        'hist': {c: np.zeros(256) for c in CHANNELS},
        'hue_sat': np.zeros(HUE_BINS),                 # hue of saturated pixels only
        'hist_by_section': {},
        'lap': [], 'grayvar': [], 'gray_std': [], 'glcm_corr': [], 'glcm_contrast': [],
    }
    for f in files:
        img, valid = load_pair(f, modality)
        if img is None:
            continue
        spaces = {'rgb': cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'hsv': cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
                  'lab': cv2.cvtColor(img, cv2.COLOR_BGR2Lab)}
        sec = acc['hist_by_section'].setdefault(section_of(f), {c: np.zeros(256) for c in CHANNELS})
        for c, (space, i) in CHANNELS.items():
            h = np.bincount(spaces[space][:, :, i][valid].ravel(), minlength=256)[:256].astype(float)
            acc['hist'][c] += h
            sec[c] += h
        hsv = spaces['hsv']
        sat_ok = valid & (hsv[:, :, 1] >= HUE_MIN_SATURATION)
        acc['hue_sat'] += np.bincount(hsv[:, :, 0][sat_ok].ravel(), minlength=HUE_BINS)[:HUE_BINS]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lv, gv = lap_stats(gray, valid)
        acc['lap'].append(lv); acc['grayvar'].append(gv)
        acc['gray_std'].append(float(gray[valid].std()) if valid.any() else np.nan)
        if do_glcm and valid.mean() >= 0.98:
            corr, contrast = glcm_curves(gray)
            acc['glcm_corr'].append(corr); acc['glcm_contrast'].append(contrast)
    return acc


def blur_calibration(files, modality, per_section=10):
    """Median residual normalized sharpness of Source patches blurred with each sigma.

    Uses up to `per_section` patches from every historical section, so the curve reflects all
    of them. The blur spreads the zero padding into nearby pixels, so the statistics are taken
    on the valid mask eroded by the kernel radius of the largest sigma, the same for every sigma.
    """
    radius = int(np.ceil(3 * max(BLUR_SIGMAS)))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    by_section = {}
    for f in files:   # files are already shuffled within each section by sample_patches
        if len(by_section.setdefault(section_of(f), [])) < per_section:
            by_section[section_of(f)].append(f)
    grays = []
    for sec_files in by_section.values():
        for f in sec_files:
            img, valid = load_pair(f, modality)
            if img is None:
                continue
            core = cv2.erode(valid.astype(np.uint8), kernel).astype(bool)
            grays.append((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), core))
    print(f"  blur calibration: {len(grays)} Source patches from {len(by_section)} sections")
    curve = []
    for s in BLUR_SIGMAS:
        ratios = []
        for g, v in grays:
            lv0, gv0 = lap_stats(g, v)
            gb = cv2.GaussianBlur(g, (0, 0), s) if s > 0 else g
            lv, gv = lap_stats(gb, v)
            ratios.append((lv / gv) / (lv0 / gv0))
        curve.append(float(np.nanmedian(ratios)))
    return curve


def equivalent_sigma(ratio, curve):
    """Blur sigma whose residual sharpness equals `ratio` (0 if the target is not softer)."""
    if not np.isfinite(ratio) or not np.all(np.isfinite(curve)):
        return float('nan')
    if ratio >= 1.0:
        return 0.0
    # curve decreases with sigma: interpolate on the reversed arrays
    return float(np.interp(ratio, curve[::-1], BLUR_SIGMAS[::-1]))


# ---------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Domain shift quantification v2 (valid pixels only)")
    parser.add_argument('--n-samples', type=int, default=400, help="Patches per domain (Source: stratified by section)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default=os.path.join('output', 'domain_shift_analysis_v2'))
    parser.add_argument('--skip-glcm', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    samples = {d: sample_patches(root, args.n_samples, args.seed, stratified=(d == 'Source'))
               for d, root in DOMAINS.items()}
    for d, s in samples.items():
        has_valid = os.path.isdir(os.path.join(DOMAINS[d], 'valid'))
        print(f"{d:10s}: {len(s)} patches sampled" + ("" if has_valid else "  WARNING: no valid/ masks, run tools/make_valid_masks.py"))

    rows, suggestions, intra_rows = [], [], []
    add = lambda dom, mod, metric, key, value: rows.append(
        {'domain': dom, 'modality': mod, 'metric': metric, 'key': key, 'value': value})
    stats, calib = {}, {}

    for mod_name, mod in MODALITIES.items():
        print(f"\n=== modality {mod_name} ===")
        stats[mod_name] = {d: accumulate(s, mod, not args.skip_glcm) for d, s in samples.items()}
        calib[mod_name] = blur_calibration(samples['Source'], mod)
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
            if st['glcm_corr']:
                corr, cont = np.mean(st['glcm_corr'], 0), np.mean(st['glcm_contrast'], 0)
                for i, dist in enumerate(GLCM_DISTANCES):
                    add(d, mod_name, 'glcm_correlation', f'd={dist}', round(float(corr[i]), 4))
                    add(d, mod_name, 'glcm_contrast', f'd={dist}', round(float(cont[i]), 3))
                add(d, mod_name, 'glcm_patches', 'n', len(st['glcm_corr']))

        # --- shift of every target vs Source
        src_ns = float(np.nanmedian(np.array(src['lap']) / np.array(src['grayvar'])))
        ratios = {'brightness (V)': {}, 'saturation (S)': {}, 'contrast (gray std)': {}, 'blur sigma': {}, 'hue shift deg': {}}
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

    write_csv('domain_shift_report_v2.csv', rows)
    write_csv('augmentation_suggestions.csv', suggestions)
    write_csv('intra_source_w1.csv', intra_rows)
    write_csv('blur_calibration.csv', [{'modality': m, **{f'sigma={s}': round(v, 4) for s, v in zip(BLUR_SIGMAS, c)}}
                                       for m, c in calib.items()])

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
            ax.set_title(f"{mod_name}: GLCM correlation (patches >= 98% valid)"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.output_dir, 'glcm_correlation.png'), dpi=140); plt.close(fig)

    with open(os.path.join(args.output_dir, 'run_info.json'), 'w', encoding='utf-8') as f:
        json.dump({'n_samples': args.n_samples, 'seed': args.seed, 'domains': DOMAINS,
                   'patches': {d: len(s) for d, s in samples.items()}}, f, indent=2)
    print("\nDone.")


if __name__ == '__main__':
    main()
