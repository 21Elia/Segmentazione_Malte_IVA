"""
Image statistics of the domain shift analysis (thesis, Chapter 3) and the fit of the data-driven
augmentation parameters (thesis, Section 4.4), shared by the tools of the analysis:

  tools/quantify_domain_shift.py      statistics, distances and parameters (writes the analysis folder)
  tools/plot_domain_shift.py          figures of the analysis
  tools/check_augmentation_params.py  closure test and coverage of a training config
  tools/fit_inverse_correction.py     inverse correction target -> Source

Every statistic uses only VALID pixels: the zero padding of the NP/NX alignment is excluded
through the per-patch masks valid/<patch>.png written by tools/make_valid_masks.py.
"""

import os
import re
import glob
import math
import random
import warnings

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Patch folders of each domain (tools/preprocess.py, tools/preprocess_new_images.py, tools/preprocess_5x.py)
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
    """gamma -> contrast -> saturation, the order of the data-driven config, with the training functions."""
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
