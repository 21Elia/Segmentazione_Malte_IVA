#!/usr/bin/env python3
"""
Figures of the domain shift analysis (v2), from the outputs of tools/quantify_domain_shift.py.

Reads domain_shift_stats.npz and the CSV files of the analysis folder; only the example-patch
figure opens images. Writes every figure to <analysis folder>/figures/ with the ds2_ prefix used
in the analysis note (analisi_domain_shift_v2_e_parametri_exp4b):

  ds2_esempi_patch.png             two patches per domain, NP and NX side by side
  ds2_cdf_NP.png, ds2_cdf_NX.png   CDFs of R, G, B, V, S, L* with the single Source sections
  ds2_punto_di_bianco_NP.png       NP colour of the brightest pixels (illumination check)
  ds2_w1_intra_vs_target.png       W1 of each Source section vs the rest, and of each target
  ds2_statistiche_per_sezione.png  per-patch median V, median S, gray std by Source section
  ds2_tinta_polare.png             hue of saturated pixels: polar and unrolled
  ds2_calibrazione_blur.png        blur sigma -> residual sharpness, with the targets
  ds2_calibrazione_rumore.png      noise sigma -> sharpness gain, with the targets
  ds2_glcm_contrasto.png           GLCM contrast vs distance
  ds2_rapporti_mediane.png         median ratios target / Source and suggested ranges
  ds2_w1_normalized_heatmap.png, ds2_sharpness.png, ds2_glcm_correlation.png
                                   copies of the figures written by quantify_domain_shift.py

Usage (from the repository root, after tools/quantify_domain_shift.py):
    python tools/plot_domain_shift.py
    python tools/plot_domain_shift.py --analysis-dir output/domain_shift_analysis_v2
"""

import os
import sys
import csv
import json
import shutil
import random
import argparse

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.quantify_domain_shift import (DOMAINS, TARGETS, MODALITIES, GLCM_DISTANCES, HUE_BINS,
                                         circular_mean_deg, circular_concentration, signed_deg, load_pair)

COLORS = {'Source': '#1f77b4', '1_SCALA': '#ff7f0e', 'UNITO_B': '#2ca02c', 'ARCHEO_02': '#d62728'}
plt.rcParams.update({'font.size': 10})


def read_csv(folder, name):
    with open(os.path.join(folder, name), encoding='utf-8') as f:
        return list(csv.DictReader(f))


def report_value(report, domain, modality, metric, key):
    return float(next(r['value'] for r in report if r['domain'] == domain and r['modality'] == modality
                      and r['metric'] == metric and r['key'] == key))


def sigma_columns(row):
    """{sigma: value} from a calibration CSV row with 'sigma=<s>' columns."""
    return {float(k.split('=')[1]): float(v) for k, v in row.items() if k.startswith('sigma=')}


def section_order(names):
    return sorted(names, key=lambda s: int(s[3:]) if s[3:].isdigit() else 0)


# ---------------------------------------------------------------------------- figures

def fig_example_patches(stats, out, seed):
    rng = random.Random(seed)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12.4))
    for i, d in enumerate(DOMAINS):
        files = list(stats[f'NP|{d}|file'])
        rng.shuffle(files)
        picks = []
        for name in files:           # two patches without alignment padding
            path = os.path.join(DOMAINS[d], 'paralleli', name)
            img, valid = load_pair(path, 'paralleli')
            if img is not None and valid.mean() > 0.98:
                picks.append(path)
            if len(picks) == 2:
                break
        for j, path in enumerate(picks):
            for k, mod in enumerate(('paralleli', 'incrociati')):
                img, _ = load_pair(path, mod)
                ax = axes[i, j * 2 + k]
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.set_xticks([]); ax.set_yticks([])
                sec = os.path.basename(path).split('_')[0]
                ax.set_title(f"{d} {'NP' if k == 0 else 'NX'}" + (f" ({sec})" if d == 'Source' else ''), fontsize=10)
    fig.suptitle("Esempi di patch 512x512 (2 per dominio, NP e NX affiancate)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'ds2_esempi_patch.png'), dpi=110); plt.close(fig)


def fig_cdf(stats, out):
    channels = list(stats['channels'])
    for m in MODALITIES:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        for ax, c in zip(axes.ravel(), ['R', 'G', 'B', 'V', 'S', 'L*']):
            ci = channels.index(c)
            for h in stats[f'{m}|Source|sec_hist']:
                ax.plot(np.cumsum(h[ci]) / h[ci].sum(), color='0.75', lw=0.9)
            for d in DOMAINS:
                h = stats[f'{m}|{d}|hist'][ci]
                ax.plot(np.cumsum(h) / h.sum(), color=COLORS[d], lw=2.6 if d == 'Source' else 2.2,
                        ls='--' if d == 'Source' else '-', label=d)
            ax.plot([], [], color='0.75', label='singole sezioni Source')
            ax.set_title(f"{m} - CDF canale {c}"); ax.set_xlim(0, 255); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
            ax.set_xlabel('livello (0-255)'); ax.set_ylabel('F(livello)')
        axes[0, 0].legend(fontsize=8, loc='lower right')
        fig.tight_layout(); fig.savefig(os.path.join(out, f'ds2_cdf_{m}.png'), dpi=110); plt.close(fig)


def fig_white_point(white, out):
    names = [r['domain'] for r in white]
    rgb = np.array([[float(r['R']), float(r['G']), float(r['B'])] for r in white])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={'width_ratios': [3, 2]})
    x, w = np.arange(len(names)), 0.25
    for j, (c, col) in enumerate(zip('RGB', ['#d62728', '#2ca02c', '#1f77b4'])):
        axes[0].bar(x + (j - 1) * w, rgb[:, j], w, color=col, label=c)
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"{r['domain']}\n({r['patches']} patch, B/R {float(r['B/R']):.2f})" for r in white])
    axes[0].set_ylim(0, 300); axes[0].grid(axis='y', alpha=0.3); axes[0].legend(ncol=3, loc='upper center')
    axes[0].set_title("NP: colore dei pixel piu chiari (1% piu luminoso di ogni patch, mediana tra le patch)")
    for i, r in enumerate(rgb):
        axes[1].add_patch(plt.Rectangle((i, 0), 0.9, 1, color=r / 255, ec='0.6'))
        axes[1].text(i + 0.45, -0.12, names[i], ha='center', fontsize=9)
    axes[1].set_xlim(-0.1, len(names)); axes[1].set_ylim(-0.25, 1.05); axes[1].axis('off')
    axes[1].set_title("il 'bianco' di ogni dominio")
    fig.tight_layout(); fig.savefig(os.path.join(out, 'ds2_punto_di_bianco_NP.png'), dpi=110); plt.close(fig)


def fig_intra_vs_target(intra, report, out):
    by_key = {(r['modality'], r['channel']): r for r in intra}
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for r, m in enumerate(MODALITIES):
        for c, ch in enumerate(['V', 'S', 'L*', 'B']):
            ax, row = axes[r, c], by_key[(m, ch)]
            secs = section_order([k for k in row if k.startswith('sec')])
            ax.bar(secs, [float(row[s]) for s in secs], color='#9ecae1', label='sezione vs resto del Source')
            for t in TARGETS:
                ax.axhline(report_value(report, t, m, 'w1', ch), color=COLORS[t], lw=2, label=f'{t} vs Source')
            ax.set_title(f"{m} canale {ch}: W1 (livelli)"); ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.grid(axis='y', alpha=0.3)
        axes[r, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'ds2_w1_intra_vs_target.png'), dpi=110); plt.close(fig)


def fig_per_section(stats, out):
    secs = section_order(set(stats['NP|Source|section']))
    groups = secs + TARGETS
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    for r, m in enumerate(MODALITIES):
        src_sec = stats[f'{m}|Source|section']
        for c, (key, label) in enumerate([('v_med', 'mediana di V per patch'), ('s_med', 'mediana di S per patch'),
                                          ('gray_std', 'dev. std. dei grigi per patch')]):
            src = stats[f'{m}|Source|{key}']
            data = [src[src_sec == s] for s in secs] + [stats[f'{m}|{t}|{key}'] for t in TARGETS]
            data = [x[np.isfinite(x)] for x in data]
            ax = axes[r, c]
            bp = ax.boxplot(data, showfliers=False, patch_artist=True)
            for i, b in enumerate(bp['boxes']):
                b.set_facecolor('#c6dbef' if i < len(secs) else COLORS[groups[i]]); b.set_alpha(0.8)
            ax.axhline(np.nanmedian(src), color=COLORS['Source'], ls='--', lw=1.2, label='mediana Source')
            ax.set_xticks(range(1, len(groups) + 1)); ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=8)
            ax.set_title(f"{m}: {label}"); ax.grid(axis='y', alpha=0.3)
        axes[r, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'ds2_statistiche_per_sezione.png'), dpi=110); plt.close(fig)


def fig_hue(stats, out):
    fig = plt.figure(figsize=(15, 12))
    ang = np.deg2rad(np.arange(HUE_BINS) * 2.0 + 1.0)
    unrolled = signed_deg(np.arange(HUE_BINS) * 2.0)
    order = np.argsort(unrolled)
    for i, m in enumerate(MODALITIES):
        ax = fig.add_subplot(2, 2, i + 1, projection='polar')
        hists = {d: stats[f'{m}|{d}|hue'] for d in DOMAINS}
        ymax = max((h / h.sum()).max() for h in hists.values())
        for d, h in hists.items():
            hn = h / h.sum()
            ax.plot(np.append(ang, ang[0]), np.append(hn, hn[0]), color=COLORS[d], lw=1.8,
                    ls='--' if d == 'Source' else '-', label=d)
            ax.annotate('', xy=(np.deg2rad(circular_mean_deg(h)), circular_concentration(h) * ymax), xytext=(0, 0),
                        arrowprops=dict(color=COLORS[d], lw=3, arrowstyle='-|>'))
        ax.set_yticklabels([])
        ax.set_title(f"{m}: distribuzione della tinta (pixel con S >= 30)\nfrecce: media circolare, lunghezza = R x massimo",
                     pad=24, fontsize=11)
        if i == 0:
            ax.legend(loc='lower left', bbox_to_anchor=(-0.25, -0.12), fontsize=9)
        ax2 = fig.add_subplot(2, 2, i + 3)
        for d, h in hists.items():
            hn = h / h.sum()
            smooth = np.convolve(np.r_[hn[-1:], hn, hn[:1]], np.ones(3) / 3, mode='valid')   # light circular smoothing
            mu, R = circular_mean_deg(h), circular_concentration(h)
            ax2.plot(unrolled[order], smooth[order], color=COLORS[d], lw=2, ls='--' if d == 'Source' else '-',
                     label=f"{d}: media {mu:.0f} gradi, R = {R:.2f}")
            ax2.axvline(float(signed_deg(mu)), color=COLORS[d], lw=1, ls=':')
        ax2.set_xlim(-180, 180); ax2.set_xticks(range(-180, 181, 30)); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
        ax2.set_xlabel('tinta (gradi, 0 = rosso, 60 = giallo, -120 = blu, -60 = magenta)'); ax2.set_ylabel('frazione dei pixel saturi')
        ax2.set_title(f"{m}: la stessa distribuzione 'srotolata'")
    fig.tight_layout(); fig.savefig(os.path.join(out, 'ds2_tinta_polare.png'), dpi=110); plt.close(fig)


def fig_blur(blur, report, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, m in zip(axes, MODALITIES):
        curve = sigma_columns(next(r for r in blur if r['modality'] == m))
        sig, val = list(curve), list(curve.values())
        ax.plot(sig, val, 'k-o', label='curva di calibrazione (Source sfocato)')
        for t in TARGETS:
            r = report_value(report, t, m, 'norm_sharpness_ratio', 'target/source')
            ax.axhline(r, color=COLORS[t], lw=2, label=f'{t}: rapporto {r:.2f}')
            if r < 1:
                s_eq = float(np.interp(r, val[::-1], sig[::-1]))
                ax.plot([s_eq], [r], 'o', color=COLORS[t], ms=10)
                ax.annotate(f'sigma eq. = {s_eq:.2f}', (s_eq, r), textcoords='offset points', xytext=(10, 10), color=COLORS[t])
        ax.set_xlabel('sigma del blur gaussiano (px)'); ax.set_ylabel('nitidezza normalizzata residua'); ax.set_ylim(0, 2.1)
        ax.set_title(f"{m}: calibrazione blur -> nitidezza"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'ds2_calibrazione_blur.png'), dpi=110); plt.close(fig)


def fig_noise(noise, report, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, m in zip(axes, MODALITIES):
        curve = sigma_columns(next(r for r in noise if r['modality'] == m))
        sig, val = list(curve), list(curve.values())
        ax.plot(sig, val, 'k-o', label='curva di calibrazione (Source + rumore)')
        ratios = {t: report_value(report, t, m, 'norm_sharpness_ratio', 'target/source') for t in TARGETS}
        for t, r in ratios.items():
            ax.axhline(r, color=COLORS[t], lw=1.8, label=f'{t}: rapporto {r:.2f}' + (' (piu morbida: niente rumore)' if r < 1 else ''))
        ax.plot([sig[-1]], [val[-1]], 's', color='0.3', ms=9)
        ax.annotate(f'sigma {sig[-1]:.0f}: nitidezza x{val[-1]:.1f}\n(il rumore del vecchio Exp 2)', (sig[-1], val[-1]),
                    textcoords='offset points', xytext=(-150, -40), fontsize=9, color='0.25', arrowprops=dict(arrowstyle='->', color='0.4'))
        ax.set_yscale('log'); ax.set_ylim(0.45, max(14, val[-1] * 1.3))
        ax.set_yticks([0.5, 1, 1.5, 2, 3, 5, 10]); ax.set_yticklabels(['0.5', '1', '1.5', '2', '3', '5', '10'])
        ax.set_xlabel("sigma dell'operazione noise (livelli su 255, per canale)")
        ax.set_ylabel('nitidezza normalizzata / originale (scala log)')
        ax.set_title(f'{m}: calibrazione rumore -> nitidezza'); ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8, loc='upper left')
        ins = ax.inset_axes([0.47, 0.1, 0.5, 0.42])
        small = [i for i, s in enumerate(sig) if s <= 6]
        ins.plot([sig[i] for i in small], [val[i] for i in small], 'k-o', ms=3)
        for t, r in ratios.items():
            s_eq = report_value(report, t, m, 'noise_sigma_equivalent', 'sigma')
            if r > 1:
                ins.axhline(r, color=COLORS[t], lw=1.4); ins.plot([s_eq], [r], 'o', color=COLORS[t], ms=7)
                ins.annotate(f'{s_eq:.1f}', (s_eq, r), textcoords='offset points', xytext=(5, 4), color=COLORS[t],
                             fontsize=9, fontweight='bold')
        ins.set_xlim(0, 5.2); ins.set_ylim(0.95, 2.15); ins.tick_params(labelsize=7); ins.grid(alpha=0.3)
        ins.set_title('zoom: sigma 0-5, lettura del sigma equivalente', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'ds2_calibrazione_rumore.png'), dpi=110); plt.close(fig)


def fig_glcm_contrast(report, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, m in zip(axes, MODALITIES):
        for d in DOMAINS:
            ax.plot(GLCM_DISTANCES, [report_value(report, d, m, 'glcm_contrast', f'd={k}') for k in GLCM_DISTANCES],
                    marker='o', color=COLORS[d], ls='--' if d == 'Source' else '-', label=d)
        ax.set_xscale('log'); ax.set_xticks(GLCM_DISTANCES); ax.set_xticklabels(GLCM_DISTANCES)
        ax.set_xlabel('distanza d (px)'); ax.set_ylabel('contrasto GLCM (livelli quantizzati a 64)')
        ax.set_title(f"{m}: contrasto GLCM (contenuto delle patch, padding escluso)"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, 'ds2_glcm_contrasto.png'), dpi=110); plt.close(fig)


def fig_ratios(suggestions, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for r, m in enumerate(MODALITIES):
        for c, quantity in enumerate(['brightness (V)', 'saturation (S)', 'contrast (gray std)']):
            row = next(x for x in suggestions if x['modality'] == m and x['quantity'] == quantity)
            ax, vals = axes[r, c], [float(row[t]) for t in TARGETS]
            ax.bar(TARGETS, vals, color=[COLORS[t] for t in TARGETS]); ax.axhline(1.0, color='k', ls='--', lw=1)
            lo, hi = json.loads(row['suggested_range'])
            ax.axhspan(lo, hi, color='0.85', zorder=0, label=f'intervallo suggerito [{lo}, {hi}]')
            for i, v in enumerate(vals):
                ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
            ax.set_title(f"{m}: {quantity}, rapporto target/Source"); ax.legend(fontsize=8)
            ax.set_ylim(0, max(1.6, max(vals) * 1.15))
    fig.tight_layout(); fig.savefig(os.path.join(out, 'ds2_rapporti_mediane.png'), dpi=110); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Figures of the domain shift analysis v2")
    parser.add_argument('--analysis-dir', default=os.path.join('output', 'domain_shift_analysis_v2'))
    parser.add_argument('--seed', type=int, default=7, help="Seed for the choice of the example patches")
    args = parser.parse_args()
    folder = args.analysis_dir
    out = os.path.join(folder, 'figures')
    os.makedirs(out, exist_ok=True)

    stats = dict(np.load(os.path.join(folder, 'domain_shift_stats.npz')))
    report = read_csv(folder, 'domain_shift_report_v2.csv')

    fig_example_patches(stats, out, args.seed)
    fig_cdf(stats, out)
    fig_white_point(read_csv(folder, 'white_point.csv'), out)
    fig_intra_vs_target(read_csv(folder, 'intra_source_w1.csv'), report, out)
    fig_per_section(stats, out)
    fig_hue(stats, out)
    fig_blur(read_csv(folder, 'blur_calibration.csv'), report, out)
    fig_noise(read_csv(folder, 'noise_calibration.csv'), report, out)
    fig_glcm_contrast(report, out)
    fig_ratios(read_csv(folder, 'augmentation_suggestions.csv'), out)
    for name in ('w1_normalized_heatmap.png', 'sharpness.png', 'glcm_correlation.png'):
        if os.path.exists(os.path.join(folder, name)):
            shutil.copy(os.path.join(folder, name), os.path.join(out, 'ds2_' + name))
    print(f"Figures saved to {out}")


if __name__ == '__main__':
    main()
