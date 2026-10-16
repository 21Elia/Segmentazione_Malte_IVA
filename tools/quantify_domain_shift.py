#!/usr/bin/env python3
"""
Domain shift quantification (v2) between the historical sections (Source) and the three
out-of-domain sections (1_SCALA, UNITO_B, ARCHEO_02 at S=0.57), and derivation of the
photometric augmentation parameters of Exp 4B.

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
    equivalent Gaussian blur sigma (softer targets) or Gaussian noise sigma (sharper targets)
    through calibration curves measured on the Source;
  - GLCM descriptors are computed on the content rectangle of every patch (padding cropped);
  - Source patches are sampled stratified by section.

Exp 4B parameters (exp4b_parameters.csv):
  - gamma, contrast and saturation are fitted in sequence on Source patches with the same
    torchvision functions used in training, so that the median V, the median gray standard
    deviation and the median S of the transformed Source match those of each target;
  - blur and noise sigmas come from the calibration curves;
  - every range includes the identity and widens the measured extremes by 5% (rule_low, rule_high).

Outputs (default output/domain_shift_analysis_v2/):
  domain_shift_report_v2.csv      long table: domain, modality, metric, key, value
  augmentation_suggestions.csv    per modality: median ratios per domain and a suggested range
  exp4b_parameters.csv            fitted values per target and the range given by the 5% rule
  intra_source_w1.csv             W1 of each historical section against the others
  blur_calibration.csv            sigma -> residual normalized sharpness on the Source
  noise_calibration.csv           noise sigma -> normalized sharpness gain on the Source
  white_point.csv                 NP colour of the brightest pixels (illumination check)
  hue_by_section.csv              circular hue statistics per Source section and per target
  domain_shift_stats.npz          histograms and per-patch statistics (input of plot_domain_shift.py)
  w1_normalized_heatmap.png, sharpness.png, glcm_correlation.png

Usage (from the repository root, CPU only):
    python tools/make_valid_masks.py          # once, writes the valid/ masks
    python tools/quantify_domain_shift.py
    python tools/plot_domain_shift.py         # all the figures of the analysis note
    python tools/quantify_domain_shift.py --n-samples 600 --skip-glcm
"""

import os
import re
import csv
import glob
import json
import math
import random
import argparse
import warnings

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
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
NOISE_SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0, 15.0]
CALIBRATION_PER_SECTION = 10   # Source patches per section used for calibration and fitting
WHITE_PERCENTILE = 99          # white point: brightest 1% of the valid pixels of a patch ...
WHITE_MIN_LEVEL = 150          # ... only if that percentile is a bright area (pores, resin)
RANGE_MARGIN = 1.05            # measured extremes are widened by 5%
FIT_ITERATIONS = 4


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


def calibration_set(files, per_section=CALIBRATION_PER_SECTION):
    """Up to `per_section` Source patches from every section (files are shuffled within a section)."""
    by_section = {}
    for f in files:
        if len(by_section.setdefault(section_of(f), [])) < per_section:
            by_section[section_of(f)].append(f)
    return [f for group in by_section.values() for f in group]


def load_pair(np_path, modality):
    path = np_path if modality == 'paralleli' else 'incrociati'.join(np_path.rsplit('paralleli', 1))
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
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


def signed_deg(deg):
    """Angle in degrees mapped to [-180, 180), red at 0."""
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


def lap_stats(gray, valid):
    """(var Laplacian, var gray) on valid pixels whose 3x3 neighbourhood is valid."""
    v = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if v.sum() < 1000:
        return np.nan, np.nan
    lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
    return float(lap[v].var()), float(gray[v].astype(np.float64).var())


def content_box(valid):
    """Slices of the bounding rectangle of the valid pixels (the alignment padding is a border)."""
    rows, cols = np.flatnonzero(valid.any(1)), np.flatnonzero(valid.any(0))
    if len(rows) == 0:
        return None
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def glcm_curves(gray):
    q = np.clip((gray.astype(np.float32) / 256 * 64).astype(np.uint8), 0, 63)
    g = graycomatrix(q, GLCM_DISTANCES, [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=64, symmetric=True, normed=True)
    return graycoprops(g, 'correlation').mean(1), graycoprops(g, 'contrast').mean(1)


def accumulate(files, modality, do_glcm):
    """Per-domain accumulators for one modality."""
    acc = {
        'hist': {c: np.zeros(256) for c in CHANNELS},
        'hue_sat': np.zeros(HUE_BINS),                 # hue of saturated pixels only
        'hist_by_section': {}, 'hue_by_section': {}, 'valid_by_section': {}, 'sat_by_section': {},
        'lap': [], 'grayvar': [], 'gray_std': [], 'v_med': [], 's_med': [], 'section': [], 'file': [],
        'glcm_corr': [], 'glcm_contrast': [], 'white_rgb': [],
    }
    for f in files:
        img, valid = load_pair(f, modality)
        if img is None:
            continue
        spaces = {'rgb': cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'hsv': cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
                  'lab': cv2.cvtColor(img, cv2.COLOR_BGR2Lab)}
        name = section_of(f)
        sec = acc['hist_by_section'].setdefault(name, {c: np.zeros(256) for c in CHANNELS})
        for c, (space, i) in CHANNELS.items():
            h = np.bincount(spaces[space][:, :, i][valid].ravel(), minlength=256)[:256].astype(float)
            acc['hist'][c] += h
            sec[c] += h
        hsv = spaces['hsv']
        sat_ok = valid & (hsv[:, :, 1] >= HUE_MIN_SATURATION)
        hue_h = np.bincount(hsv[:, :, 0][sat_ok].ravel(), minlength=HUE_BINS)[:HUE_BINS].astype(float)
        acc['hue_sat'] += hue_h
        acc['hue_by_section'][name] = acc['hue_by_section'].get(name, np.zeros(HUE_BINS)) + hue_h
        acc['valid_by_section'][name] = acc['valid_by_section'].get(name, 0) + int(valid.sum())
        acc['sat_by_section'][name] = acc['sat_by_section'].get(name, 0) + int(sat_ok.sum())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lv, gv = lap_stats(gray, valid)
        acc['lap'].append(lv); acc['grayvar'].append(gv)
        acc['gray_std'].append(float(gray[valid].std()) if valid.any() else np.nan)
        acc['v_med'].append(float(np.median(hsv[:, :, 2][valid])) if valid.any() else np.nan)
        acc['s_med'].append(float(np.median(hsv[:, :, 1][valid])) if valid.any() else np.nan)
        acc['section'].append(name); acc['file'].append(os.path.basename(f))
        if do_glcm:
            box = content_box(valid)
            if box is not None and gray[box].shape[0] >= 64 and gray[box].shape[1] >= 64:
                corr, contrast = glcm_curves(gray[box])
                acc['glcm_corr'].append(corr); acc['glcm_contrast'].append(contrast)
        if modality == 'paralleli' and valid.any():
            v = hsv[:, :, 2]
            thr = np.percentile(v[valid], WHITE_PERCENTILE)
            bright = valid & (v >= thr)
            if thr >= WHITE_MIN_LEVEL and bright.sum() > 50:
                acc['white_rgb'].append(spaces['rgb'][bright].astype(float).mean(0))
    return acc


# ---------------------------------------------------------------------------- calibration and fitting

def load_calibration(files, modality):
    """Source calibration patches: (BGR image, valid mask, valid mask eroded by the largest blur radius)."""
    radius = int(np.ceil(3 * max(BLUR_SIGMAS)))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    out = []
    for f in files:
        img, valid = load_pair(f, modality)
        if img is None:
            continue
        # The blur spreads the zero padding into nearby pixels: statistics on the eroded mask
        out.append((img, valid, cv2.erode(valid.astype(np.uint8), kernel).astype(bool)))
    return out


def blur_calibration(cal):
    """Median residual normalized sharpness of the calibration patches blurred with each sigma."""
    curve = []
    for s in BLUR_SIGMAS:
        ratios = []
        for img, _, core in cal:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lv0, gv0 = lap_stats(g, core)
            gb = cv2.GaussianBlur(g, (0, 0), s) if s > 0 else g
            lv, gv = lap_stats(gb, core)
            ratios.append((lv / gv) / (lv0 / gv0))
        curve.append(float(np.nanmedian(ratios)))
    return curve


def noise_calibration(cal, seed):
    """Median normalized sharpness gain of the calibration patches with Gaussian noise of each sigma.

    Same formula as the 'noise' augmentation: float + N(0, sigma^2) per pixel and channel,
    clamp to [0, 255], cast to uint8 (truncation).
    """
    rng = np.random.default_rng(seed)
    curve = []
    for s in NOISE_SIGMAS:
        ratios = []
        for img, _, core in cal:
            g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lv0, gv0 = lap_stats(g0, core)
            if s > 0:
                noisy = np.clip(img.astype(np.float32) + rng.standard_normal(img.shape).astype(np.float32) * s, 0, 255)
                g = cv2.cvtColor(noisy.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                g = g0
            lv, gv = lap_stats(g, core)
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


def equivalent_noise_sigma(ratio, curve):
    """Noise sigma whose sharpness gain equals `ratio` (0 if the target is not sharper)."""
    if not np.isfinite(ratio) or not np.all(np.isfinite(curve)):
        return float('nan')
    if ratio <= 1.0:
        return 0.0
    return float(np.interp(ratio, curve, NOISE_SIGMAS))   # the curve increases with sigma


def to_tensor(img_bgr):
    return torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).contiguous()


def apply_tone(t, gamma, contrast, saturation):
    """gamma -> contrast -> saturation, the order of the Exp 4B config, with the training functions."""
    if gamma != 1.0:
        t = TF.adjust_gamma(t, gamma)
    if contrast != 1.0:
        t = TF.adjust_contrast(t, contrast)
    if saturation != 1.0:
        t = TF.adjust_saturation(t, saturation)
    return t


def tone_stats(tensors, params):
    """Median over patches of (median V, gray std, median S) on valid pixels, after the transform."""
    v, sd, s = [], [], []
    for t, valid in tensors:
        rgb = np.ascontiguousarray(apply_tone(t, *params).permute(1, 2, 0).numpy())
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        v.append(np.median(hsv[:, :, 2][valid])); s.append(np.median(hsv[:, :, 1][valid]))
        sd.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[valid].std())
    return float(np.median(v)), float(np.median(sd)), float(np.median(s))


def fit_tone(tensors, target):
    """Gamma, contrast and saturation that bring the calibration patches onto the target medians.

    target = (median V, median gray std, median S) of the target patches. Coordinate updates,
    repeated FIT_ITERATIONS times because each operation slightly moves the other medians:
      gamma      *= ln(V_t/255) / ln(V/255)   (a gamma maps the median onto the median: monotone)
      contrast   *= std_t / std               (adjust_contrast scales the deviations from the mean)
      saturation *= S_t / S
    """
    tv, tsd, ts = target
    gamma = contrast = saturation = 1.0
    for _ in range(FIT_ITERATIONS):
        v, _, _ = tone_stats(tensors, (gamma, contrast, saturation))
        if 0 < v < 255 and 0 < tv < 255:
            gamma *= math.log(tv / 255) / math.log(v / 255)
        _, sd, _ = tone_stats(tensors, (gamma, contrast, saturation))
        contrast *= tsd / max(sd, 1e-6)
        _, _, s = tone_stats(tensors, (gamma, contrast, saturation))
        saturation *= ts / max(s, 1e-6)
    return gamma, contrast, saturation, tone_stats(tensors, (gamma, contrast, saturation))


def rule_range(values, identity, decimals):
    """Range including the identity, with the measured extremes widened by RANGE_MARGIN, rounded outwards."""
    vals = [v for v in values if np.isfinite(v)]
    lo, hi = min([identity] + vals), max([identity] + vals)
    if identity == 0:                       # sigmas: only the upper end comes from a measurement
        hi = hi * RANGE_MARGIN if hi > 0 else 0.0
    else:
        lo = lo / RANGE_MARGIN if lo < identity else lo
        hi = hi * RANGE_MARGIN if hi > identity else hi
    f = 10 ** decimals
    return math.floor(lo * f + 1e-9) / f, math.ceil(hi * f - 1e-9) / f


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
    cal_files = calibration_set(samples['Source'])
    print(f"calibration / fitting: {len(cal_files)} Source patches from {len({section_of(f) for f in cal_files})} sections")

    rows, suggestions, intra_rows, exp4b_rows, hue_rows = [], [], [], [], []
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

        # --- Exp 4B parameters: gamma / contrast / saturation fitted on the calibration patches
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
            exp4b_rows.append({'modality': mod_name, 'operation': op, 'identity': identity,
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

    write_csv('domain_shift_report_v2.csv', rows)
    write_csv('augmentation_suggestions.csv', suggestions)
    write_csv('exp4b_parameters.csv', exp4b_rows)
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

    # statistics for tools/plot_domain_shift.py
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
