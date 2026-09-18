#!/usr/bin/env python3
"""
Domain Shift Quantification Script for MMSFormer Mortar Segmentation.

Computes a rigorous mathematical quantification of the feature/color distribution
gap between the historical source dataset (12 sections, Notari) and the three
out-of-domain target datasets (1_SCALA, UNITO_B, ARCHEO_02 at S=0.57).

Metrics computed:
  1. Wasserstein Distance W1 (Earth Mover's Distance) on 9 color channels
     (R, G, B from RGB; H, S, V from HSV; L*, a*, b* from CIE Lab)
     computed separately for NP (paralleli) and NX (incrociati) modalities.

  2. Laplacian Variance: measures optical sharpness / spatial frequency content.
     Quantifies whether new images are blurred/noisier than the training set.

  3. GLCM (Gray-Level Co-occurrence Matrix) Haralick Descriptors:
     Contrast, Homogeneity, Energy, Correlation — computed at 5 spatial
     distances (d=1,2,5,10,20 px) to capture the spatial correlation function.
     The multi-distance Correlation curve is the key metric for validating the
     metrological calibration factor S=0.57 for ARCHEO_02.

Outputs (saved to output/domain_shift_analysis/):
  - domain_shift_report.csv       : full numerical table for thesis
  - wasserstein_heatmap.png       : W1 heatmap (channels x domains, NP and NX)
  - laplacian_boxplot.png         : box plot of per-patch Laplacian variance
  - glcm_correlation_curve.png    : spatial correlation vs distance d
  - haralick_overview_d1.png      : bar chart of all 4 Haralick descriptors at d=1
  - cdf_plots/<channel>_<mod>.png : CDF comparison plots per channel/modality

Usage (run on Windows machine via SSH):
    cd Segmentazione_Malte_IVA-main
    python tools/quantify_domain_shift.py
    python tools/quantify_domain_shift.py --n-samples 500 --skip-glcm
    python tools/quantify_domain_shift.py --help

Arguments:
    --n-samples INT    Patches to sample per domain per modality (default: 300)
    --seed      INT    Random seed for reproducibility (default: 42)
    --output-dir STR   Output directory (default: output/domain_shift_analysis)
    --skip-glcm        Skip GLCM computation (faster, omits texture analysis)

Notes:
    - sec5_* patches are EXCLUDED from the source dataset to match the exact
      training subset used for MMSFormer-B3 fine-tuning experiments.
    - Script is CPU-only. Expected runtime: ~10-20 min (300 samples/domain).
      With --skip-glcm: ~2-5 min.
    - All paths are relative to the repository root (run from there).
"""

import os
import sys
import glob
import random
import argparse
import warnings
import csv

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "-8"

import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_OFF)
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for SSH usage
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from tqdm import tqdm

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    # older scikit-image
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GLCM_DISTANCES = [1, 2, 5, 10, 20]
GLCM_LEVELS = 64
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

# (color_space_key, channel_index)
COLOR_CHANNELS = {
    "R":  ("rgb", 0),
    "G":  ("rgb", 1),
    "B":  ("rgb", 2),
    "H":  ("hsv", 0),
    "S":  ("hsv", 1),
    "V":  ("hsv", 2),
    "L*": ("lab", 0),
    "a*": ("lab", 1),
    "b*": ("lab", 2),
}

HARALICK_PROPS = ["contrast", "homogeneity", "energy", "correlation"]

DATASETS = {
    "Source (Storico)": (
        os.path.join("data", "mortars", "paralleli"),
        os.path.join("data", "mortars", "incrociati"),
    ),
    "1_SCALA": (
        os.path.join("data", "nuove_patches", "1_SCALA", "paralleli"),
        os.path.join("data", "nuove_patches", "1_SCALA", "incrociati"),
    ),
    "UNITO_B": (
        os.path.join("data", "nuove_patches", "UNITO_B", "paralleli"),
        os.path.join("data", "nuove_patches", "UNITO_B", "incrociati"),
    ),
    "ARCHEO_02 (S=0.57)": (
        os.path.join("data", "patches_5x_scale=057", "ARCHEO_02", "paralleli"),
        os.path.join("data", "patches_5x_scale=057", "ARCHEO_02", "incrociati"),
    ),
}

DOMAIN_ORDER = ["Source (Storico)", "1_SCALA", "UNITO_B", "ARCHEO_02 (S=0.57)"]

DOMAIN_COLORS = {
    "Source (Storico)":    "#2563EB",
    "1_SCALA":             "#DC2626",
    "UNITO_B":             "#16A34A",
    "ARCHEO_02 (S=0.57)":  "#D97706",
}

MODALITIES = ["NP", "NX"]


# ---------------------------------------------------------------------------
# 1. Patch Sampling
# ---------------------------------------------------------------------------

def sample_patches(patch_dir: str, n_samples: int, seed: int) -> list:
    """
    Returns a reproducible random sample of up to n_samples .tif patch files
    from patch_dir. Files whose basename starts with 'sec5_' are excluded to
    match the exact training subset (Sezione 5 was discarded from training).
    """
    if not os.path.isdir(patch_dir):
        print(f"  [WARNING] Directory not found: {patch_dir}", file=sys.stderr)
        return []

    all_files = sorted(glob.glob(os.path.join(patch_dir, "*.tif")))
    # Exclude Sezione 5 (not used in training — excluded for coerenza col training set)
    all_files = [f for f in all_files if not os.path.basename(f).startswith("sec5_")]
    # Exclude empty files
    all_files = [f for f in all_files if os.path.getsize(f) > 0]

    if not all_files:
        print(f"  [WARNING] No valid .tif files in: {patch_dir}", file=sys.stderr)
        return []

    rng = random.Random(seed)
    rng.shuffle(all_files)
    sampled = all_files[:n_samples]
    print(f"    Sampled {len(sampled):4d} / {len(all_files):5d}  from  {patch_dir}")
    return sampled


# ---------------------------------------------------------------------------
# 2. Wasserstein Distance W1
# ---------------------------------------------------------------------------

def _load_color_spaces(bgr_img: np.ndarray) -> dict:
    """Converts BGR uint8 image to RGB, HSV and CIE Lab color spaces."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2Lab)
    return {"rgb": rgb, "hsv": hsv, "lab": lab}


def accumulate_histograms(patch_paths: list, n_bins: int = 256) -> dict:
    """
    Loads each patch, converts to all color spaces, and accumulates
    per-channel histograms (unnormalized integer counts).

    Returns:
        Dict mapping channel_name -> np.ndarray of shape (n_bins,).
    """
    accum = {ch: np.zeros(n_bins, dtype=np.float64) for ch in COLOR_CHANNELS}

    for path in tqdm(patch_paths, leave=False, desc="      hist"):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        spaces = _load_color_spaces(bgr)
        for ch_name, (space, idx) in COLOR_CHANNELS.items():
            channel_data = spaces[space][:, :, idx].ravel().astype(np.float32)
            hist, _ = np.histogram(channel_data, bins=n_bins, range=(0, 256))
            accum[ch_name] += hist.astype(np.float64)

    return accum


def wasserstein_from_hists(src_hist: np.ndarray, tgt_hist: np.ndarray) -> float:
    """
    Computes W1 between two unnormalized histograms via the closed-form
    CDF integral:  W1 = sum_k |CDF_P(k) - CDF_Q(k)|  (step = 1 intensity level).
    Result is in intensity levels [0, 255].
    """
    src_prob = src_hist / src_hist.sum() if src_hist.sum() > 0 else src_hist
    tgt_prob = tgt_hist / tgt_hist.sum() if tgt_hist.sum() > 0 else tgt_hist
    cdf_src = np.cumsum(src_prob)
    cdf_tgt = np.cumsum(tgt_prob)
    return float(np.sum(np.abs(cdf_src - cdf_tgt)))


# ---------------------------------------------------------------------------
# 3. Laplacian Variance
# ---------------------------------------------------------------------------

def compute_laplacian_var(patch_paths: list) -> dict:
    """
    Computes the Laplacian Variance for each patch as a measure of optical
    sharpness.  sigma^2_Delta = Var(Laplacian(gray)).
    High = sharp/noisy; Low = blurry.
    """
    variances = []
    for path in tqdm(patch_paths, leave=False, desc="      lap"):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        variances.append(float(np.var(lap)))

    if not variances:
        return {}
    arr = np.array(variances)
    return {
        "variances": variances,
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr)),
        "p10":    float(np.percentile(arr, 10)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
        "p90":    float(np.percentile(arr, 90)),
    }


# ---------------------------------------------------------------------------
# 4. GLCM Haralick Descriptors
# ---------------------------------------------------------------------------

def compute_glcm_haralick(patch_paths: list) -> dict:
    """
    Computes GLCM Haralick descriptors at multiple spatial distances.

    Parameters:
        64 gray levels, distances={1,2,5,10,20} px,
        angles={0,45,90,135} deg (averaged for rotation invariance),
        symmetric=True, normed=True.

    Returns:
        Dict: {distance_int: {property_str: {mean, median, std, p25, p75}}}
    """
    per_patch = {d: {prop: [] for prop in HARALICK_PROPS} for d in GLCM_DISTANCES}

    for path in tqdm(patch_paths, leave=False, desc="      glcm"):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Quantize [0,255] -> [0, GLCM_LEVELS-1]
        gray_q = np.clip(
            (gray.astype(np.float32) / 256.0 * GLCM_LEVELS).astype(np.uint8),
            0, GLCM_LEVELS - 1,
        )

        glcm = graycomatrix(
            gray_q,
            distances=GLCM_DISTANCES,
            angles=GLCM_ANGLES,
            levels=GLCM_LEVELS,
            symmetric=True,
            normed=True,
        )
        # glcm: (levels, levels, n_distances, n_angles)

        for d_idx, d in enumerate(GLCM_DISTANCES):
            glcm_d = glcm[:, :, d_idx : d_idx + 1, :]
            for prop in HARALICK_PROPS:
                vals = graycoprops(glcm_d, prop)  # shape (1, n_angles)
                per_patch[d][prop].append(float(np.mean(vals)))

    results = {}
    for d in GLCM_DISTANCES:
        results[d] = {}
        for prop in HARALICK_PROPS:
            arr = np.array(per_patch[d][prop])
            if len(arr) == 0:
                results[d][prop] = {}
                continue
            results[d][prop] = {
                "mean":   float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std":    float(np.std(arr)),
                "p25":    float(np.percentile(arr, 25)),
                "p75":    float(np.percentile(arr, 75)),
            }
    return results


# ---------------------------------------------------------------------------
# 5. Plotting Functions
# ---------------------------------------------------------------------------

def _save_fig(fig, output_dir: str, filename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def plot_wasserstein_heatmap(wasserstein_results: dict, output_dir: str) -> None:
    """
    Heatmap: rows = color channels, cols = target domains.
    Separate subplots for NP and NX.
    Color: green (low W1) to red (high W1).
    Values annotated in each cell.
    """
    target_domains = [d for d in DOMAIN_ORDER if d != "Source (Storico)"]
    channels = list(COLOR_CHANNELS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Wasserstein Distance $W_1$ (Earth Mover's Distance)\n"
        "Sorgente: Dataset Storico Notari (12 Sezioni) — valori in livelli di intensità [0–255]",
        fontsize=12, fontweight="bold",
    )

    for ax_idx, modality in enumerate(MODALITIES):
        ax = axes[ax_idx]
        matrix = np.zeros((len(channels), len(target_domains)))

        for col_idx, domain in enumerate(target_domains):
            for row_idx, ch in enumerate(channels):
                w1 = wasserstein_results.get(domain, {}).get(modality, {}).get(ch, 0.0)
                matrix[row_idx, col_idx] = w1

        vmax = max(float(matrix.max()), 1.0)
        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=vmax)

        for r in range(len(channels)):
            for c in range(len(target_domains)):
                val = matrix[r, c]
                txt_color = "white" if val > vmax * 0.6 else "black"
                ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, color=txt_color, fontweight="bold")

        ax.set_xticks(range(len(target_domains)))
        ax.set_xticklabels(target_domains, rotation=20, ha="right", fontsize=10)
        ax.set_yticks(range(len(channels)))
        ax.set_yticklabels(channels, fontsize=10)
        mod_label = "Nicols Paralleli / PPL" if modality == "NP" else "Nicols Incrociati / XPL"
        ax.set_title(f"Modalità {modality} ({mod_label})", fontsize=11, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("$W_1$ [livelli di intensità]", fontsize=9)

        # Separator lines between RGB/HSV/Lab channel groups
        for sep in [3, 6]:
            ax.axhline(sep - 0.5, color="black", linewidth=1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save_fig(fig, output_dir, "wasserstein_heatmap.png")


def plot_laplacian_boxplot(laplacian_results: dict, output_dir: str) -> None:
    """
    Box plot of per-patch Laplacian Variance distributions,
    one subplot per modality.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    fig.suptitle(
        "Varianza Laplaciana $\\sigma^2_\\Delta$ — Misura di Nitidezza Ottica\n"
        "(valori più alti = immagini più nitide; valori più bassi = sfocatura)",
        fontsize=12, fontweight="bold",
    )

    for ax_idx, modality in enumerate(MODALITIES):
        ax = axes[ax_idx]
        data_list, labels, colors_list = [], [], []

        for domain in DOMAIN_ORDER:
            entry = laplacian_results.get(domain, {}).get(modality, {})
            variances = entry.get("variances", [])
            if variances:
                data_list.append(variances)
                labels.append(domain.replace(" (S=0.57)", "\n(S=0.57)"))
                colors_list.append(DOMAIN_COLORS.get(domain, "#6B7280"))

        bp = ax.boxplot(data_list, patch_artist=True, notch=False,
                        showfliers=False, widths=0.6)
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        for ml in bp["medians"]:
            ml.set_color("black")
            ml.set_linewidth(2)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("$\\sigma^2_\\Delta$ [$\\mathrm{px}^2$]", fontsize=10)
        mod_label = "Nicols Paralleli" if modality == "NP" else "Nicols Incrociati"
        ax.set_title(f"Modalità {modality} ({mod_label})", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.35, linestyle="--")

        for i, variances in enumerate(data_list):
            med = float(np.median(variances))
            ax.text(i + 1, med, f" {med:.0f}", va="bottom",
                    ha="center", fontsize=8, color="black")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save_fig(fig, output_dir, "laplacian_boxplot.png")


def plot_glcm_correlation_curve(glcm_results: dict, output_dir: str) -> None:
    """
    GLCM Correlation vs spatial distance d (semi-log x axis).
    Key plot: curves of ARCHEO_02 should overlap with Source if S=0.57 is correct.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        "Funzione di Correlazione Spaziale GLCM vs Distanza $d$\n"
        "Grafico chiave: se ARCHEO_02 si sovrappone a Source → $S=0.57$ correttamente calibrato",
        fontsize=12, fontweight="bold",
    )

    for ax_idx, modality in enumerate(MODALITIES):
        ax = axes[ax_idx]

        for domain in DOMAIN_ORDER:
            x_vals, y_vals, y_err = [], [], []
            for d in GLCM_DISTANCES:
                stats = glcm_results.get(domain, {}).get(modality, {}).get(d, {}).get("correlation", {})
                if stats:
                    x_vals.append(d)
                    y_vals.append(stats["mean"])
                    y_err.append(stats["std"])

            if not x_vals:
                continue
            color = DOMAIN_COLORS.get(domain, "#6B7280")
            ls = "--" if domain == "Source (Storico)" else "-"
            lw = 2.5 if domain == "Source (Storico)" else 1.8
            ax.errorbar(x_vals, y_vals, yerr=y_err, label=domain, color=color,
                        linestyle=ls, linewidth=lw, marker="o", markersize=5,
                        capsize=4, alpha=0.88)

        ax.set_xscale("log")
        ax.set_xticks(GLCM_DISTANCES)
        ax.set_xticklabels([str(d) for d in GLCM_DISTANCES])
        ax.set_xlabel("Distanza spaziale $d$ [pixel]", fontsize=10)
        ax.set_ylabel("Correlazione GLCM (media ± std)", fontsize=10)
        mod_label = "Nicols Paralleli" if modality == "NP" else "Nicols Incrociati"
        ax.set_title(f"Modalità {modality} ({mod_label})", fontsize=11, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, loc="upper right")

        # 1/e reference line for correlation length estimation
        threshold = 1.0 / np.e
        ax.axhline(threshold, color="gray", linestyle=":", linewidth=1.5)
        ax.text(GLCM_DISTANCES[-1] * 1.05, threshold + 0.01,
                f"$1/e \\approx {threshold:.2f}$", fontsize=8, color="gray",
                va="bottom", ha="left")

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    _save_fig(fig, output_dir, "glcm_correlation_curve.png")


def plot_haralick_overview(glcm_results: dict, output_dir: str) -> None:
    """
    2x2 bar chart grid of all 4 Haralick descriptors at d=1 for all domains.
    Solid bars = NP; hatched bars = NX.
    """
    props_labels = {
        "contrast":    "Contrasto",
        "homogeneity": "Omogeneità",
        "energy":      "Energia",
        "correlation": "Correlazione",
    }
    d_ref = 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Descrittori di Haralick — GLCM a $d={d_ref}$ pixel\n"
        "Confronto tra domini (barre piene = NP; tratteggiate = NX)",
        fontsize=12, fontweight="bold",
    )
    axes_flat = axes.flatten()

    for prop_idx, (prop, prop_label) in enumerate(props_labels.items()):
        ax = axes_flat[prop_idx]
        n_domains = len(DOMAIN_ORDER)
        n_mod = len(MODALITIES)
        bar_width = 0.35
        x_pos = np.arange(n_domains)

        for mod_idx, modality in enumerate(MODALITIES):
            means, stds = [], []
            for domain in DOMAIN_ORDER:
                stats = (glcm_results.get(domain, {})
                         .get(modality, {})
                         .get(d_ref, {})
                         .get(prop, {}))
                means.append(stats.get("mean", 0.0))
                stds.append(stats.get("std", 0.0))

            offset = (mod_idx - n_mod / 2 + 0.5) * bar_width
            x_shifted = x_pos + offset
            hatch = "" if modality == "NP" else "///"
            alpha = 0.85 if modality == "NP" else 0.55

            for b_idx, (xpos, mean, std) in enumerate(zip(x_shifted, means, stds)):
                domain = DOMAIN_ORDER[b_idx]
                ax.bar(xpos, mean, width=bar_width * 0.9,
                       color=DOMAIN_COLORS.get(domain, "#6B7280"),
                       alpha=alpha, hatch=hatch, edgecolor="black", linewidth=0.5)
                ax.errorbar(xpos, mean, yerr=std, fmt="none",
                            color="black", linewidth=1, capsize=2)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [d.replace(" (S=0.57)", "\n(S=0.57)") for d in DOMAIN_ORDER],
            fontsize=8,
        )
        ax.set_ylabel(prop_label, fontsize=10)
        ax.set_title(prop_label, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    legend_elements = [
        Patch(facecolor="gray", alpha=0.85, label="NP (Nicols Paralleli)"),
        Patch(facecolor="gray", alpha=0.55, hatch="///", label="NX (Nicols Incrociati)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.06, 1, 0.91])
    _save_fig(fig, output_dir, "haralick_overview_d1.png")


def plot_cdf_comparisons(
    source_hists: dict,
    target_hists_per_domain: dict,
    output_dir: str,
) -> None:
    """
    Per each color channel and modality: plots CDFs of source (blue dashed)
    and all target domains with shaded area = W1.
    """
    cdf_dir = os.path.join(output_dir, "cdf_plots")
    os.makedirs(cdf_dir, exist_ok=True)
    x = np.arange(256)

    for modality in MODALITIES:
        for ch_name in COLOR_CHANNELS:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.set_title(
                f"CDF Comparativa — Canale {ch_name} | Modalità {modality}",
                fontsize=11, fontweight="bold",
            )

            src_hist = source_hists.get(modality, {}).get(ch_name, np.zeros(256))
            src_prob = src_hist / src_hist.sum() if src_hist.sum() > 0 else src_hist
            cdf_src = np.cumsum(src_prob)
            ax.plot(x, cdf_src, color=DOMAIN_COLORS["Source (Storico)"],
                    linewidth=2.5, linestyle="--", label="Source (Storico)", zorder=5)

            for domain, domain_hists in target_hists_per_domain.items():
                tgt_hist = domain_hists.get(modality, {}).get(ch_name, np.zeros(256))
                tgt_prob = tgt_hist / tgt_hist.sum() if tgt_hist.sum() > 0 else tgt_hist
                cdf_tgt = np.cumsum(tgt_prob)
                color = DOMAIN_COLORS.get(domain, "#6B7280")
                w1 = float(np.sum(np.abs(cdf_src - cdf_tgt)))
                ax.fill_between(x, cdf_src, cdf_tgt, alpha=0.15, color=color)
                ax.plot(x, cdf_tgt, color=color, linewidth=1.8,
                        label=f"{domain}  ($W_1$={w1:.1f})")

            ax.set_xlabel("Livello di intensità", fontsize=10)
            ax.set_ylabel("CDF cumulata $F(x)$", fontsize=10)
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 1.02)
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(alpha=0.3, linestyle="--")

            safe_ch = ch_name.replace("*", "star").replace(" ", "_")
            _save_fig(fig, cdf_dir, f"{safe_ch}_{modality}.png")


# ---------------------------------------------------------------------------
# 6. CSV Report
# ---------------------------------------------------------------------------

def save_csv_report(
    wasserstein_results: dict,
    laplacian_results: dict,
    glcm_results: dict,
    output_dir: str,
) -> None:
    """Saves a flat CSV with all numerical results for direct use in thesis."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "domain_shift_report.csv")
    rows = []

    # Wasserstein
    for domain in DOMAIN_ORDER:
        if domain not in wasserstein_results:
            continue
        for modality in MODALITIES:
            for ch_name, w1 in wasserstein_results[domain].get(modality, {}).items():
                rows.append({"domain": domain, "modality": modality,
                             "metric": "wasserstein_W1", "channel_or_stat": ch_name,
                             "distance_px": "", "value": f"{w1:.4f}",
                             "unit": "intensity_levels"})

    # Laplacian Variance
    for domain in DOMAIN_ORDER:
        for modality in MODALITIES:
            entry = laplacian_results.get(domain, {}).get(modality, {})
            for stat in ["mean", "median", "std", "p10", "p25", "p75", "p90"]:
                val = entry.get(stat)
                if val is None:
                    continue
                rows.append({"domain": domain, "modality": modality,
                             "metric": "laplacian_variance", "channel_or_stat": stat,
                             "distance_px": "", "value": f"{val:.4f}", "unit": "px^2"})

    # GLCM Haralick
    for domain in DOMAIN_ORDER:
        for modality in MODALITIES:
            for d in GLCM_DISTANCES:
                for prop in HARALICK_PROPS:
                    stats = (glcm_results.get(domain, {})
                             .get(modality, {})
                             .get(d, {})
                             .get(prop, {}))
                    for stat in ["mean", "median", "std"]:
                        val = stats.get(stat)
                        if val is None:
                            continue
                        rows.append({"domain": domain, "modality": modality,
                                     "metric": f"glcm_{prop}", "channel_or_stat": stat,
                                     "distance_px": str(d), "value": f"{val:.6f}",
                                     "unit": "—"})

    fieldnames = ["domain", "modality", "metric", "channel_or_stat",
                  "distance_px", "value", "unit"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"    Saved: {csv_path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantify Domain Shift for MMSFormer Mortar Segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-samples", type=int, default=300,
                        help="Patches to sample per domain per modality.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join("output", "domain_shift_analysis"),
                        help="Output directory for CSV and PNG files.")
    parser.add_argument("--skip-glcm", action="store_true", default=False,
                        help="Skip GLCM computation (much faster; omits texture analysis).")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("  Domain Shift Quantification — MMSFormer Mortar Segmentation")
    print("=" * 72)
    print(f"  n_samples / domain / modality : {args.n_samples}")
    print(f"  Random seed                   : {args.seed}")
    print(f"  Output directory              : {args.output_dir}")
    print(f"  Skip GLCM                     : {args.skip_glcm}")
    print()

    # ------------------------------------------------------------------
    # 1. Sample patches
    # ------------------------------------------------------------------
    print("[1/5] Sampling patches ...")
    sampled_patches = {}
    for domain, (np_dir, nx_dir) in DATASETS.items():
        print(f"  {domain}")
        sampled_patches[domain] = {
            "NP": sample_patches(np_dir, args.n_samples, args.seed),
            "NX": sample_patches(nx_dir, args.n_samples, args.seed),
        }
    print()

    source_domain = "Source (Storico)"
    target_domains = [d for d in DOMAIN_ORDER if d != source_domain]

    # ------------------------------------------------------------------
    # 2. Wasserstein W1
    # ------------------------------------------------------------------
    print("[2/5] Computing Wasserstein W1 distances ...")

    print(f"  Building source histograms ({source_domain}) ...")
    src_hists = {
        "NP": accumulate_histograms(sampled_patches[source_domain]["NP"]),
        "NX": accumulate_histograms(sampled_patches[source_domain]["NX"]),
    }

    wasserstein_results = {}
    target_hists_for_cdf = {}

    for domain in target_domains:
        print(f"  Target: {domain}")
        tgt_hists = {
            "NP": accumulate_histograms(sampled_patches[domain]["NP"]),
            "NX": accumulate_histograms(sampled_patches[domain]["NX"]),
        }
        target_hists_for_cdf[domain] = tgt_hists

        w_results = {}
        for modality in MODALITIES:
            mod_w = {}
            for ch_name in COLOR_CHANNELS:
                mod_w[ch_name] = wasserstein_from_hists(
                    src_hists[modality][ch_name],
                    tgt_hists[modality][ch_name],
                )
            w_results[modality] = mod_w
        wasserstein_results[domain] = w_results

        np_v = w_results["NP"].get("V", float("nan"))
        nx_v = w_results["NX"].get("V", float("nan"))
        np_b = w_results["NP"].get("b*", float("nan"))
        print(f"    W1 (NP | V)={np_v:.2f}   (NX | V)={nx_v:.2f}   (NP | b*)={np_b:.2f}")

    print()

    # ------------------------------------------------------------------
    # 3. Laplacian Variance
    # ------------------------------------------------------------------
    print("[3/5] Computing Laplacian Variance ...")
    laplacian_results = {}
    for domain in DOMAIN_ORDER:
        print(f"  {domain}")
        laplacian_results[domain] = {}
        for modality in MODALITIES:
            result = compute_laplacian_var(sampled_patches[domain][modality])
            laplacian_results[domain][modality] = result
            med = result.get("median", float("nan"))
            print(f"    [{modality}] median σ²_Δ = {med:.1f}")
    print()

    # ------------------------------------------------------------------
    # 4. GLCM Haralick
    # ------------------------------------------------------------------
    if not args.skip_glcm:
        print("[4/5] Computing GLCM Haralick descriptors (slowest step) ...")
        glcm_results = {}
        for domain in DOMAIN_ORDER:
            print(f"  {domain}")
            glcm_results[domain] = {}
            for modality in MODALITIES:
                result = compute_glcm_haralick(sampled_patches[domain][modality])
                glcm_results[domain][modality] = result
                corr_d1 = (result.get(1, {})
                               .get("correlation", {})
                               .get("mean", float("nan")))
                print(f"    [{modality}] GLCM Correlation (d=1) = {corr_d1:.4f}")
    else:
        print("[4/5] GLCM skipped (--skip-glcm).")
        glcm_results = {}
    print()

    # ------------------------------------------------------------------
    # 5. Plots + CSV
    # ------------------------------------------------------------------
    print("[5/5] Generating plots and CSV ...")

    print("  → wasserstein_heatmap.png")
    plot_wasserstein_heatmap(wasserstein_results, args.output_dir)

    print("  → laplacian_boxplot.png")
    plot_laplacian_boxplot(laplacian_results, args.output_dir)

    print("  → CDF comparison plots (cdf_plots/)")
    plot_cdf_comparisons(src_hists, target_hists_for_cdf, args.output_dir)

    if glcm_results:
        print("  → glcm_correlation_curve.png")
        plot_glcm_correlation_curve(glcm_results, args.output_dir)

        print("  → haralick_overview_d1.png")
        plot_haralick_overview(glcm_results, args.output_dir)

    print("  → domain_shift_report.csv")
    save_csv_report(wasserstein_results, laplacian_results, glcm_results, args.output_dir)

    print()
    print("=" * 72)
    print("  DONE. All outputs saved to:", args.output_dir)
    print("=" * 72)
    print()
    print("  Next steps:")
    print("  1. Open domain_shift_report.csv and review the W1 values.")
    print("  2. Consult the Diagnostic Matrix in:")
    print("     docs/domain_shift_quantification_theory.md")
    print("     to classify the type of shift for each domain.")
    print("  3. Use the classification to calibrate the augmentation parameters")
    print("     for the next data-driven fine-tuning experiment.")


if __name__ == "__main__":
    main()
