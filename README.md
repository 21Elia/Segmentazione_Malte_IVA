# Multimodal segmentation of historical mortars

Semantic segmentation of thin sections of historical mortars into **binder** and **aggregates**, from
plane-polarized (NP, *Nicols Paralleli*) and cross-polarized (NX, *Nicols Incrociati*) images, with
[MMSFormer](https://github.com/csiplab/MMSFormer)-B3. The code accompanies the bachelor's thesis
*MSegmentazione Semantica Multimodale di Malte Storiche: Mitigazione del Domain Shift tramite Vision
Transformer e Data-Driven Augmentation* (Elia Awad, Università degli Studi di Firenze, 2026). It starts from the
project of L. Notari for the Image and Video Analysis course, itself based on the original MMSFormer code
(see `README_ORIGINALE.md`).

The thesis studies the **domain shift** between the ten annotated historical sections (Source) and three
new sections (`1_SCALA`, `UNITO_B`, `ARCHEO_02`): it measures it, compares a generic and a data-driven
photometric augmentation through an ablation with several seeds, evaluates the models on the annotated
crops of `1_SCALA` and `UNITO_B`, and tests an inverse photometric correction of the new sections at
inference time.

## Repository layout

```
configs/
  baseline/                 Baseline (geometric augmentation only) and its replica on a second machine
  ablation/data_driven/     single data-driven operations at P_op = 0.3 / 0.6 / 0.9, full data-driven pipeline
  ablation/generic/         generic pipeline and its two halves (high / low frequency)
semseg/                     library
  datasets/mortars.py       dataset, train / val / test split, porosity merged into binder
  augmentations_mm.py       geometric and photometric augmentation of the NP/NX pairs
  models/                   MMSFormer (backbone, fusion, decoder)
  losses.py, schedulers.py, optimizers.py, metrics.py
  normalization.py          input normalization (ImageNet), shared by training and inference
  checkpoints.py            checkpoint sidecar files (.meta.json), best checkpoint of a run folder
  patch_geometry.py         geometry of the aligned patches (content, padding, position in the section)
  preprocessing.py          helpers of the preprocessing tools (padding, NP/NX alignment, previews)
  domain_shift.py           statistics of the domain shift analysis and fit of the data-driven parameters
tools/                      scripts, run from the repository root (see "Reproducing the thesis")
  reproduce_thesis.py       runs the whole pipeline with the arguments used in the thesis
  preprocessing/            NP/NX alignment, labels, validity masks and patches of every section
  domain_shift/             domain shift analysis, data-driven parameters, inverse correction fit
  training/                 training and test-set evaluation
  evaluation/               predictions on the new sections, out-of-domain metrics, result tables
```

## Installation

Python 3.12 or later is required. Install PyTorch for the CUDA version of the machine first
([instructions](https://pytorch.org/get-started/locally/)), then the other dependencies:

```bash
pip install -r requirements.txt
```

Download the ImageNet weights of the MiT-B3 backbone (`mit_b3.pth`, from the
[SegFormer weights](https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5)) into
`checkpoints/pretrained/segformer/mit_b3.pth`. Training needs a GPU with about 10 GB of memory
(batch 4, FP32); one run takes 7-8 hours on an RTX 5070 Ti and about 17 hours on an RTX 2080 Ti.

## Data

The images are not part of the repository. The tools expect the raw data in a folder `DIA_FIRENZE`
**next to** the repository (paths are relative to the repository root; every tool also accepts `--src`):

```
DIA_FIRENZE/
  1/ ... 12/                NICOLS PARALLELI.tif, NICOLS INCROCIATI.tif, AGGREGATE.tif, POROSITY.tif, TOTAL.tif
                            (section 4 has no NX image and is skipped; section 5 is excluded when loading)
  Nuove Immagini/
    1NP_SCALA.tif, 1NX_SCALA.tif
    1/                      annotated crop of 1_SCALA: 1NP_SCALA_RITAGLIO.tif, AGGREGATE/POROSITY/TOTAL.tif
    2/                      UNITO_B: NP_UNITO_BW_B.tif, NX_UNITO_B.tif, crop RITAGLIO.tif and its masks
  Prova immagini 5X/
    ARCHEO_02 Nicols paralleli_5x.jpg, ARCHEO_02 Nicols incrociati_5x.jpg
```

In every mask, black marks the class (aggregate, pore) or the valid mortar (TOTAL).

Generated data (`data/`) and results (`output/`) are written inside the repository and ignored by git:

| Folder | Content | Written by |
|---|---|---|
| `data/mortars/` | 3866 patches 512 x 512 of the 11 historical sections with NP/NX/label (3305 without section 5) | `preprocess.py` |
| `data/nuove_patches/{1_SCALA,UNITO_B}/` | 236 and 72 patches, no labels | `preprocess_new_images.py` |
| `data/patches_5x_scale=057/ARCHEO_02/` | 539 patches after rescaling by S = 0.57 | `preprocess_5x.py` |
| `*/valid/` | per-patch masks of the alignment padding | `make_valid_masks.py` |
| `output/alignment_samples/`, `output/inspection/` | NP/NX alignment previews, validity masks of the new sections | preprocessing tools |
| `output/domain_shift_analysis/` | statistics, distances, data-driven parameters, inverse correction | domain shift tools |
| `output/MMSFormer/<run>/` | checkpoints, `train.log`, `train_summary.json`, `eval_*.txt` | `train_mm.py`, `val_mm.py` |
| `output/results/`, `output/confusion/` | test-set patches with a high error rate, confusion matrices | `val_mm.py` |
| `output/inference_nuove/` | predicted masks on the new sections | `run_all_inferences_softmax.py` |
| `output/inference_custom/` | predicted masks on a folder given with `--patches-dir` | `run_all_inferences_softmax.py` |
| `output/gt_ood/`, `gt_ood_corr/` | out-of-domain metrics and error maps | `eval_ood_gt.py` |
| `output/summary/` | tables of the ablation | `summarize_results.py` |

The `MMSF-V3-` prefix of the run folders (and the `v3_` prefix of the model keys derived from it) is the one of
the results of the thesis and is kept so that the existing outputs stay readable.

## Reproducing the thesis

`tools/reproduce_thesis.py` runs every step below in order, with the arguments used in the thesis.
`--dry-run` prints the commands; `--stages` selects some steps; `--only` restricts training and test to some
runs. A training run whose folder already has `train_summary.json` is skipped.

```bash
python tools/reproduce_thesis.py --dry-run
python tools/reproduce_thesis.py --stages train --only gen_060
```

The single steps:

| Step | Command | Thesis |
|---|---|---|
| Historical patches, NP/NX alignment, labels | `python tools/preprocessing/preprocess.py` | 2.1, 2.3, 2.4 |
| Validity masks of the new sections | `python tools/preprocessing/inspect_and_mask_new_images.py`, `python tools/preprocessing/inspect_and_mask_5x.py` | 2.4 |
| Patches of the new sections (ARCHEO_02 rescaled) | `python tools/preprocessing/preprocess_new_images.py`, `python tools/preprocessing/preprocess_5x.py` | 2.2, 2.5 |
| Padding masks | `python tools/preprocessing/make_valid_masks.py` | 2.4 |
| Domain shift statistics and data-driven parameters | `python tools/domain_shift/quantify_domain_shift.py` | 3, 4.4 |
| Figures of the analysis | `python tools/domain_shift/plot_domain_shift.py` | 3 |
| Closure test, coverage of a config | `python tools/domain_shift/check_augmentation_params.py --cfg configs/ablation/data_driven/full/mortars_v3_dd-060.yaml` | 4.4.4 |
| Inverse correction parameters | `python tools/domain_shift/fit_inverse_correction.py` | 6.9 |
| Training (one run) | `python tools/training/train_mm.py --cfg <config> [--seed 42]` | 5.1, 5.3 |
| Test set | `python tools/training/val_mm.py --cfg <config> --run-dir <run folder>` | 6 |
| Predictions on the new sections | `python tools/evaluation/run_all_inferences_softmax.py --domains 1_SCALA UNITO_B [--tta]` | 6, 6.8 |
| Predictions with inverse correction | `python tools/evaluation/run_all_inferences_softmax.py --models <baselines> --domains 1_SCALA UNITO_B --inverse-correction output/domain_shift_analysis/inverse_correction/inverse_correction_params.json` | 6.9 |
| Out-of-domain metrics | `python tools/evaluation/eval_ood_gt.py --sources single tta` | 2.6, 5.5, 6 |
| Tables of the ablation | `python tools/evaluation/summarize_results.py --corr-csv output/gt_ood_corr/gt_ood_metrics.csv` | 6 |
| Full-section prediction map | `python tools/evaluation/reconstruct_section.py --meta <grid_metadata.json> --pred-dir <masks folder> --output <png>` | 2.6 |

### Runs of the ablation

Every config sets `TRAIN.SEED: 3407`; `--seed N` trains another seed and adds `-seedN` to the run folder
(the folder name also gets `-p080` from `P_IMAGE = 0.8`). The 37 runs of the thesis:

| Config | Seeds |
|---|---|
| `configs/baseline/mortars_v3_baseline-median.yaml` | 3407, 42, 3408 |
| `configs/ablation/data_driven/{gamma,saturation,noise,blur}/*-{030,060,090}.yaml` | 3407 (gamma 090 also 42) |
| `configs/ablation/data_driven/contrast/*-{030,060,090}.yaml` | 3407, 42, 3408 |
| `configs/ablation/data_driven/full/mortars_v3_dd-060.yaml` | 3407, 42, 3408 |
| `configs/ablation/generic/mortars_v3_gen-060.yaml`, `-tex-060`, `-col-060` | 3407, 42, 3408 |

`configs/baseline/mortars_v3_baseline-median-micc.yaml` is the Baseline replica with seed 3407 trained on a
second machine (Section 6.1); it is not part of the means.

### Expected results

Mean +- standard deviation across seeds (mIoU, %), as printed by `summarize_results.py`:

| Configuration | Runs | Test (in-domain) | 1_SCALA | UNITO_B | Mean OOD |
|---|---|---|---|---|---|
| Baseline | 3 | 78.77 +- 0.45 | 50.9 +- 5.3 | 52.5 +- 18.1 | 51.7 +- 11.2 |
| Single data-driven operations | 22 | 78.29 +- 0.72 | 51.1 +- 8.9 | 50.7 +- 10.3 | 50.9 +- 8.5 |
| Data-driven pipeline | 3 | 77.69 +- 0.28 | 62.7 +- 4.1 | 65.3 +- 4.3 | 64.0 +- 4.2 |
| Generic pipeline | 3 | 76.96 +- 0.93 | 63.7 +- 7.1 | 72.3 +- 1.4 | 68.0 +- 4.2 |
| Generic, high frequency | 3 | 77.62 +- 1.08 | 48.3 +- 9.0 | 45.0 +- 7.5 | 46.6 +- 7.5 |
| Generic, low frequency | 3 | 76.88 +- 0.23 | 66.9 +- 3.3 | 66.5 +- 1.9 | 66.7 +- 2.4 |
| Baseline + inverse correction | 3 | - | 54.7 +- 3.5 | 59.3 +- 5.4 | 57.0 +- 1.9 |

**Training is not deterministic.** `semseg/utils/utils.py` sets `cudnn.benchmark = True` and
`cudnn.deterministic = False`, and GPUs differ numerically: retraining with the same config and seed gives a
different model. In-domain results move by a few tenths of a point, out-of-domain results by up to about
10 points per section (Section 6.1 of the thesis). Compare means over seeds, not single runs.

## Credits and license

MMSFormer: M. K. Reza, A. Prater-Bennette, M. S. Asif, *MMSFormer: Multimodal Transformer for Material and
Semantic Segmentation*, IEEE OJSP 2024. SegFormer: E. Xie et al., NeurIPS 2021. The dataset loader, the first
preprocessing and the adaptation of MMSFormer to the mortars are by L. Notari. Apache-2.0 license (see `LICENSE`).
