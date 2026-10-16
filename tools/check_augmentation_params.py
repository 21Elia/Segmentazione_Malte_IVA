#!/usr/bin/env python3
"""
Verification of the photometric augmentation parameters derived by tools/quantify_domain_shift.py.

1. Closure test. Source calibration patches (10 per section) are transformed with the values
   estimated for each target, and the normalized W1 (W1 / max intra-Source W1, the metric of the
   W1 heatmap) between the transformed Source and the target is measured on the 9 channels.
   Variants:
     none          the Source as it is
     hsv           brightness, contrast, saturation, hue, blur with the values of augmentation_suggestions.csv
     hsv_no_hue    the same without hue                                    (= model A)
     rgb_gains     one gain per channel (median ratio) + contrast
     gamma         gamma, contrast, saturation of exp4b_parameters.csv, + blur (= model B)
     temperature   colour temperature (gains 1, sqrt(t), t) + refitted gamma, contrast, saturation (= model C)
   Outputs closure_test.csv, ds2_test_chiusura.png (none / hsv / hsv_no_hue / rgb_gains),
   ds2_test_chiusura_modelli.png (none / A / B / C) and the previews ds2_anteprima_{NP,NX}.png,
   ds2_anteprima_modelli_{NP,NX}.png.

2. Config check (--cfg). The PHOTOMETRIC block of a training config is applied with the training
   class (semseg.augmentations_mm.PhotometricAugmentation) to the calibration patches, several random
   draws each. For median V, gray std, median S and normalized sharpness it reports where the median
   of every target falls in the distribution of the augmented Source (percentile, 2.5-97.5 = covered).
   Outputs config_coverage.csv, ds2_copertura_config.png and ds2_anteprima_config.png.

Usage (from the repository root, after tools/quantify_domain_shift.py):
    python tools/check_augmentation_params.py
    python tools/check_augmentation_params.py --cfg configs/mortars_v2_exp4b.yaml
"""

import os
import sys
import csv
import json
import math
import random
import argparse
import warnings

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import cv2
import numpy as np
import torch
import yaml
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.quantify_domain_shift import (DOMAINS, TARGETS, MODALITIES, CHANNELS, HUE_BINS, sample_patches,
                                         calibration_set, load_calibration, load_pair, lap_stats, w1, w1_circular,
                                         hist_quantile, to_tensor, fit_tone, section_of)
from semseg.augmentations_mm import PhotometricAugmentation

warnings.filterwarnings("ignore", category=RuntimeWarning)
VARIANTS_ROUND1 = ['none', 'hsv', 'hsv_no_hue', 'rgb_gains']
VARIANTS_ROUND2 = ['none', 'hsv_no_hue', 'gamma', 'temperature']
LABELS = {'none': 'nessuna trasformazione', 'hsv': 'operazioni HSV (b, c, s, h, blur)', 'hsv_no_hue': 'A: HSV senza tinta (brightness, contrast, saturation)',
          'rgb_gains': 'guadagni RGB + contrasto', 'gamma': 'B: gamma, contrast, saturation',
          'temperature': 'C: temperatura, gamma, contrast, saturation'}
SHORT = {'hsv': 'HSV', 'hsv_no_hue': 'A', 'rgb_gains': 'guadagni', 'gamma': 'B', 'temperature': 'C'}
BAR_COLORS = {'none': '0.6', 'hsv': '#9467bd', 'hsv_no_hue': '#c5b0d5', 'rgb_gains': '#17becf', 'gamma': '#e377c2', 'temperature': '#17becf'}


def read_csv(folder, name):
    with open(os.path.join(folder, name), encoding='utf-8') as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------- transforms

def blur(t, sigma):
    if sigma <= 0:
        return t
    k = max(3, 2 * math.ceil(3.0 * sigma) + 1)
    return TF.gaussian_blur(t, kernel_size=k, sigma=[sigma, sigma])


def gains(t, g):
    return torch.clamp(t.float() * torch.tensor(g, dtype=torch.float32).view(3, 1, 1), 0, 255).to(torch.uint8)


def chain(t, ops):
    """Applies a list of (name, value) operations in order, with the training functions."""
    for name, v in ops:
        if name == 'brightness': t = TF.adjust_brightness(t, v)
        elif name == 'contrast': t = TF.adjust_contrast(t, v)
        elif name == 'saturation': t = TF.adjust_saturation(t, v)
        elif name == 'hue': t = TF.adjust_hue(t, max(-0.5, min(0.5, v)))
        elif name == 'gamma': t = TF.adjust_gamma(t, v)
        elif name == 'blur': t = blur(t, v)
        elif name == 'gains': t = gains(t, v)
        elif name == 'temperature': t = gains(t, [1.0, math.sqrt(v), v])
    return t


def channel_hists(tensors, ops):
    """Histograms of the 9 channels on valid pixels of the transformed patches."""
    H = {c: np.zeros(256) for c in CHANNELS}
    for t, valid in tensors:
        rgb = np.ascontiguousarray(chain(t, ops).permute(1, 2, 0).numpy())
        spaces = {'rgb': rgb, 'hsv': cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), 'lab': cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)}
        for c, (space, i) in CHANNELS.items():
            H[c] += np.bincount(spaces[space][:, :, i][valid].ravel(), minlength=256)[:256]
    return H


def normalized_w1(H, target_hist, intra_max):
    out = {}
    for k, c in enumerate(CHANNELS):
        d = w1_circular(H[c][:HUE_BINS], target_hist[k][:HUE_BINS]) if c == 'H' else w1(H[c], target_hist[k])
        out[c] = d / max(intra_max[c], 1e-6)
    return out


def median_std(tensors, ops):
    return float(np.median([cv2.cvtColor(np.ascontiguousarray(chain(t, ops).permute(1, 2, 0).numpy()),
                                         cv2.COLOR_RGB2GRAY)[v].std() for t, v in tensors]))


def build_variants(m, t, tensors, stats, sugg, params):
    """Operation lists of every variant for modality m and target t."""
    s = {r['quantity']: float(r[t]) for r in sugg if r['modality'] == m}
    p = {r['operation']: float(r[t]) for r in params if r['modality'] == m}
    blur_ops = [('blur', p['blur sigma'])] if p['blur sigma'] > 0 else []
    src_h, tgt_h = stats[f'{m}|Source|hist'], stats[f'{m}|{t}|hist']
    tgt_std = float(np.nanmedian(stats[f'{m}|{t}|gray_std']))
    hsv = [('brightness', s['brightness (V)']), ('contrast', s['contrast (gray std)']), ('saturation', s['saturation (S)'])]
    variants = {'none': [],
                'hsv': hsv + [('hue', s['hue shift deg'] / 360.0)] + blur_ops,
                'hsv_no_hue': hsv + blur_ops,
                'gamma': [('gamma', p['gamma']), ('contrast', p['contrast']), ('saturation', p['saturation'])] + blur_ops}
    g = [hist_quantile(tgt_h[i], 0.5) / max(hist_quantile(src_h[i], 0.5), 1) for i in range(3)]   # R, G, B medians
    variants['rgb_gains'] = [('gains', g), ('contrast', tgt_std / median_std(tensors, [('gains', g)]))]
    k = (hist_quantile(tgt_h[2], 0.5) / max(hist_quantile(tgt_h[0], 0.5), 1)) / (hist_quantile(src_h[2], 0.5) / max(hist_quantile(src_h[0], 0.5), 1))
    warm = [(chain(x, [('temperature', k)]), v) for x, v in tensors]
    target = (float(np.nanmedian(stats[f'{m}|{t}|v_med'])), tgt_std, float(np.nanmedian(stats[f'{m}|{t}|s_med'])))
    gm, c, sat, _ = fit_tone(warm, target)
    variants['temperature'] = [('temperature', k), ('gamma', gm), ('contrast', c), ('saturation', sat)] + blur_ops
    return variants


def ops_title(ops):
    short = {'brightness': 'b', 'contrast': 'c', 'saturation': 's', 'hue': 'h', 'gamma': 'gamma', 'blur': 'blur',
             'temperature': 't', 'gains': 'RGB'}
    parts = []
    for name, v in ops:
        if name == 'gains':
            parts.append('RGB ' + ', '.join(f'{x:.2f}' for x in v))
        elif name == 'hue':
            parts.append(f'h {v * 360:+.0f} gradi')
        else:
            parts.append(f'{short[name]} {v:.2f}')
    return ', '.join(parts) if parts else 'originale'


# ---------------------------------------------------------------------------- figures

def preview_index(cal_np):
    """First calibration patch without padding that contains bright areas (pores, resin) in NP,
    so that the previews show whether a transform keeps white areas white."""
    fallback = None
    for i, (img, valid, _) in enumerate(cal_np):
        if valid.mean() <= 0.98:
            continue
        fallback = i if fallback is None else fallback
        if (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2][valid] >= 230).mean() >= 0.02:
            return i
    return fallback if fallback is not None else 0


def closure_figure(results, variants, path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
    chans = list(CHANNELS); x = np.arange(len(chans)); w = 0.8 / len(variants)
    for r, m in enumerate(MODALITIES):
        for c, t in enumerate(TARGETS):
            ax = axes[r, c]
            for i, v in enumerate(variants):
                ax.bar(x + (i - (len(variants) - 1) / 2) * w, [results[(m, t, v)][ch] for ch in chans], w,
                       color=BAR_COLORS[v], label=LABELS[v])
            ax.axhline(1, color='k', ls='--', lw=1); ax.set_xticks(x); ax.set_xticklabels(chans); ax.grid(axis='y', alpha=0.3)
            ax.set_title(f"{m}: Source trasformato vs {t}")
            if c == 0:
                ax.set_ylabel('W1 / max W1 intra-Source')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def preview_figure(m, mod, src_tensor, src_name, all_variants, columns, stats, path, seed):
    fig, axes = plt.subplots(3, len(columns) + 1, figsize=(3.5 * (len(columns) + 1), 10.8))
    rng = random.Random(seed)
    for r, t in enumerate(TARGETS):
        for k, v in enumerate(columns):
            ops = all_variants[(m, t)][v]
            ax = axes[r, k]
            ax.imshow(chain(src_tensor, ops).permute(1, 2, 0).numpy()); ax.set_xticks([]); ax.set_yticks([])
            title = f'Source originale ({src_name})' if v == 'none' else f'{SHORT[v]}: {ops_title(ops)}'
            ax.set_title(title, fontsize=8)
        files = list(stats[f'NP|{t}|file']); rng.shuffle(files)
        for name in files:          # a real target patch without padding
            img, valid = load_pair(os.path.join(DOMAINS[t], 'paralleli', name), 'paralleli')
            if img is not None and valid.mean() > 0.98:
                img, _ = load_pair(os.path.join(DOMAINS[t], 'paralleli', name), mod)
                break
        ax = axes[r, -1]; ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{t} reale', fontsize=8); axes[r, 0].set_ylabel(f'verso {t}', fontsize=11)
    fig.suptitle(f"{m}: la stessa patch del Source trasformata con i parametri stimati per ogni target", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


# ---------------------------------------------------------------------------- config check

def patch_stats(img_rgb, valid):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lv, gv = lap_stats(gray, valid)
    return {'V': float(np.median(hsv[:, :, 2][valid])), 'std': float(gray[valid].std()),
            'S': float(np.median(hsv[:, :, 1][valid])), 'sharp': lv / gv}


def config_check(cfg_path, cal, stats, draws, seed, out_dir, fig_dir):
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    spec = (cfg.get('TRAIN', {}).get('AUGMENTATION') or {}).get('PHOTOMETRIC')
    if not spec:
        print(f"  {cfg_path}: no TRAIN.AUGMENTATION.PHOTOMETRIC block, config check skipped")
        return
    aug = PhotometricAugmentation(spec)
    random.seed(seed); torch.manual_seed(seed)
    keys = ('V', 'std', 'S', 'sharp')
    orig = {m: {k: [] for k in keys} for m in MODALITIES}
    augd = {m: {k: [] for k in keys} for m in MODALITIES}
    for (np_img, np_valid, _), (nx_img, nx_valid, _) in zip(cal['NP'], cal['NX']):
        base = {'paralleli': to_tensor(np_img), 'incrociati': to_tensor(nx_img)}
        valid = {'paralleli': np_valid, 'incrociati': nx_valid}
        for m, mod in MODALITIES.items():
            for k, v in patch_stats(np.ascontiguousarray(base[mod].permute(1, 2, 0).numpy()), valid[mod]).items():
                orig[m][k].append(v)
        for _ in range(draws):
            sample = aug({k: v.clone() for k, v in base.items()})
            for m, mod in MODALITIES.items():
                for k, v in patch_stats(np.ascontiguousarray(sample[mod].permute(1, 2, 0).numpy()), valid[mod]).items():
                    augd[m][k].append(v)
    rows = []
    for m in MODALITIES:
        src_sharp = stats[f'{m}|Source|lap'] / stats[f'{m}|Source|grayvar']
        for t in TARGETS:
            tgt = {'V': np.nanmedian(stats[f'{m}|{t}|v_med']), 'std': np.nanmedian(stats[f'{m}|{t}|gray_std']),
                   'S': np.nanmedian(stats[f'{m}|{t}|s_med']),
                   'sharp': np.nanmedian(stats[f'{m}|{t}|lap'] / stats[f'{m}|{t}|grayvar']) / np.nanmedian(src_sharp)
                   * np.nanmedian(orig[m]['sharp'])}   # sharpness: target/Source ratio applied to these patches
            for k in keys:
                a = np.array(augd[m][k]); a = a[np.isfinite(a)]
                pct = float((a < tgt[k]).mean() * 100)
                rows.append({'modality': m, 'target': t, 'statistic': k, 'target_median': round(float(tgt[k]), 4),
                             'source_median': round(float(np.nanmedian(orig[m][k])), 4),
                             'augmented_p2.5': round(float(np.percentile(a, 2.5)), 4), 'augmented_p97.5': round(float(np.percentile(a, 97.5)), 4),
                             'target_percentile': round(pct, 1), 'covered': 2.5 <= pct <= 97.5})
    with open(os.path.join(out_dir, 'config_coverage.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"  {r['modality']} {r['target']:9s} {r['statistic']:5s} target {r['target_median']:8.3f} -> "
              f"percentile {r['target_percentile']:5.1f} of the augmented Source {'(covered)' if r['covered'] else '(NOT covered)'}")

    labels = {'V': 'mediana di V', 'std': 'dev. std. dei grigi', 'S': 'mediana di S', 'sharp': 'nitidezza normalizzata'}
    colors = {'1_SCALA': '#ff7f0e', 'UNITO_B': '#2ca02c', 'ARCHEO_02': '#d62728'}
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for r_, m in enumerate(MODALITIES):
        for c, k in enumerate(keys):
            ax = axes[r_, c]
            ax.boxplot([np.array(orig[m][k])[np.isfinite(orig[m][k])], np.array(augd[m][k])[np.isfinite(augd[m][k])]],
                       whis=(2.5, 97.5), showfliers=False)
            ax.set_xticks([1, 2]); ax.set_xticklabels(['Source', 'Source\naugmentato'])
            for row in rows:
                if row['modality'] == m and row['statistic'] == k:
                    ax.axhline(row['target_median'], color=colors[row['target']], lw=2, label=row['target'])
            ax.set_title(f"{m}: {labels[k]} per patch"); ax.grid(axis='y', alpha=0.3)
        axes[r_, 0].legend(fontsize=8)
    fig.suptitle(f"Copertura dei target con l'augmentation di {os.path.basename(cfg_path)} (baffi: 2.5-97.5 percentile)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, 'ds2_copertura_config.png'), dpi=110); plt.close(fig)

    random.seed(seed + 1)
    i = preview_index(cal['NP'])
    np_img, _, _ = cal['NP'][i]; nx_img, _, _ = cal['NX'][i]
    base = {'paralleli': to_tensor(np_img), 'incrociati': to_tensor(nx_img)}
    fig, axes = plt.subplots(2, 8, figsize=(24, 7.4))
    for j in range(8):
        sample = {k: v.clone() for k, v in base.items()} if j == 0 else aug({k: v.clone() for k, v in base.items()})
        for r_, mod in enumerate(('paralleli', 'incrociati')):
            ax = axes[r_, j]; ax.imshow(sample[mod].permute(1, 2, 0).numpy()); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(('originale' if j == 0 else f'estrazione {j}') + (' NP' if r_ == 0 else ' NX'), fontsize=9)
    fig.suptitle(f"Sette estrazioni casuali dell'augmentation di {os.path.basename(cfg_path)} sulla stessa patch", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, 'ds2_anteprima_config.png'), dpi=90); plt.close(fig)


# ---------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Closure test and config check of the photometric augmentation parameters")
    parser.add_argument('--analysis-dir', default=os.path.join('output', 'domain_shift_analysis_v2'))
    parser.add_argument('--cfg', default=None, help="Training config whose TRAIN.AUGMENTATION.PHOTOMETRIC is checked")
    parser.add_argument('--n-samples', type=int, default=400, help="Must match quantify_domain_shift.py")
    parser.add_argument('--seed', type=int, default=42, help="Must match quantify_domain_shift.py")
    parser.add_argument('--draws', type=int, default=5, help="Random augmentations per patch in the config check")
    args = parser.parse_args()
    folder = args.analysis_dir
    fig_dir = os.path.join(folder, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    stats = dict(np.load(os.path.join(folder, 'domain_shift_stats.npz')))
    sugg = read_csv(folder, 'augmentation_suggestions.csv')
    params = read_csv(folder, 'exp4b_parameters.csv')
    intra = {(r['modality'], r['channel']): float(r['max']) for r in read_csv(folder, 'intra_source_w1.csv')}
    with open(os.path.join(folder, 'run_info.json'), encoding='utf-8') as f:
        info = json.load(f)
    if info.get('n_samples') != args.n_samples or info.get('seed') != args.seed:
        print(f"WARNING: the analysis used n_samples={info.get('n_samples')}, seed={info.get('seed')}")

    cal_files = calibration_set(sample_patches(DOMAINS['Source'], args.n_samples, args.seed, stratified=True))
    print(f"calibration patches: {len(cal_files)}")
    cal = {m: load_calibration(cal_files, mod) for m, mod in MODALITIES.items()}

    # ---- 1. closure test
    results, all_variants, rows = {}, {}, []
    for m in MODALITIES:
        tensors = [(to_tensor(img), valid) for img, valid, _ in cal[m]]
        intra_max = {c: intra[(m, c)] for c in CHANNELS}
        for t in TARGETS:
            variants = build_variants(m, t, tensors, stats, sugg, params)
            all_variants[(m, t)] = variants
            for v, ops in variants.items():
                res = normalized_w1(channel_hists(tensors, ops), stats[f'{m}|{t}|hist'], intra_max)
                results[(m, t, v)] = res
                rows.append({'modality': m, 'target': t, 'variant': v, 'operations': ops_title(ops),
                             **{c: round(x, 3) for c, x in res.items()}, 'mean': round(float(np.mean(list(res.values()))), 3)})
            print(f"  {m} {t:9s} mean normalized W1: " + ', '.join(f"{v} {np.mean(list(results[(m, t, v)].values())):.2f}" for v in variants))
    with open(os.path.join(folder, 'closure_test.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    closure_figure(results, VARIANTS_ROUND1, os.path.join(fig_dir, 'ds2_test_chiusura.png'))
    closure_figure(results, VARIANTS_ROUND2, os.path.join(fig_dir, 'ds2_test_chiusura_modelli.png'))

    for m, mod in MODALITIES.items():
        idx = preview_index(cal['NP'])   # the same physical patch for NP and NX
        src_tensor, src_name = to_tensor(cal[m][idx][0]), section_of(cal_files[idx])
        preview_figure(m, mod, src_tensor, src_name, all_variants, ['none', 'hsv', 'rgb_gains'], stats,
                       os.path.join(fig_dir, f'ds2_anteprima_{m}.png'), args.seed)
        preview_figure(m, mod, src_tensor, src_name, all_variants, ['none', 'hsv_no_hue', 'gamma', 'temperature'], stats,
                       os.path.join(fig_dir, f'ds2_anteprima_modelli_{m}.png'), args.seed)
    print(f"Closure test saved to {os.path.join(folder, 'closure_test.csv')}")

    # ---- 2. config check
    if args.cfg:
        print(f"\nConfig check: {args.cfg}")
        config_check(args.cfg, cal, stats, args.draws, args.seed, folder, fig_dir)
    print(f"Figures saved to {fig_dir}")


if __name__ == '__main__':
    main()
