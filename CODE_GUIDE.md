# Guide to the Python files

What every `.py` file of the repository does, what it reads and what it writes. The scripts in `tools/` are
run from the repository root (`python tools/<folder>/<script>.py --help` lists all the options); the modules
in `semseg/` are a library: they are imported by the scripts and never run on their own.

The pipeline, in order:

```
DIA_FIRENZE (raw images)
  -> tools/preprocessing/   patches 512 x 512, NP/NX aligned            -> data/
  -> tools/domain_shift/    statistics, data-driven parameters          -> output/domain_shift_analysis/
  -> tools/training/        training and test-set evaluation            -> output/MMSFormer/<run>/
  -> tools/evaluation/      predictions and metrics on the new sections -> output/inference_nuove/, gt_ood/, summary/
```

`tools/reproduce_thesis.py` runs the whole chain with the arguments of the thesis.

## tools/ (scripts)

### tools/reproduce_thesis.py

Runs every step of the thesis in order, by calling the other scripts with the arguments used for the results.

- **Input:** the stages to run (`--stages preprocess domain_shift train replica test infer evaluate summary`),
  optionally a subset of runs (`--only gen_060 dd_060`).
- **Output:** whatever the called scripts write. `--dry-run` only prints the commands. A training run whose
  folder already has `train_summary.json` is skipped.

### tools/preprocessing/

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `preprocess.py` | Patches of the historical sections: grid of 512 x 512 cells inside the valid mortar (TOTAL), NP/NX alignment by phase correlation, 4-class label (0 binder, 1 porosity, 2 aggregates, 3 ignore = alignment padding) | `DIA_FIRENZE/<1..12>/` (NP, NX, AGGREGATE, POROSITY, TOTAL) | `data/mortars/{paralleli,incrociati,label}/`, `data/mortars/<section>/grid_metadata.json`, alignment previews in `output/alignment_samples/<section>/` |
| `inspect_and_mask_new_images.py` | Validity mask of 1_SCALA and UNITO_B (mortar vs resin/background), by Otsu thresholding and morphology | `DIA_FIRENZE/Nuove Immagini/` | `output/inspection/<SECTION>_validity_mask.tif` (0 = mortar, 255 = background) + previews |
| `inspect_and_mask_5x.py` | The same for ARCHEO_02 (5X objective), computed on a downscaled copy | `DIA_FIRENZE/Prova immagini 5X/` | `output/inspection/ARCHEO_02_validity_mask.tif` + previews |
| `preprocess_new_images.py` | Patches of 1_SCALA and UNITO_B inside their validity mask, NP/NX alignment; no labels | images + `output/inspection/*_validity_mask.tif` | `data/nuove_patches/<SECTION>/{paralleli,incrociati}/`, `grid_metadata.json`, previews in `output/alignment_samples/` |
| `preprocess_5x.py` | Rescales ARCHEO_02 by S = 0.57 to the resolution of the historical sections, then the same patching and alignment | 5X images + validity mask | `data/patches_5x_scale=057/ARCHEO_02/...`, `grid_metadata.json`, previews |
| `make_valid_masks.py` | One mask per patch marking the real content (255) and the zero padding added by the alignment (0), computed from `grid_metadata.json` | `grid_metadata.json` of each dataset | `<dataset>/valid/<patch>.png` |

### tools/domain_shift/

| Script | What it does | Reads | Writes (in `output/domain_shift_analysis/`) |
|---|---|---|---|
| `quantify_domain_shift.py` | Measures the domain shift between the Source and the three targets (per-channel histograms and W1 distance, median V/S, contrast, sharpness, GLCM texture) and fits the parameters of the data-driven augmentation | patches + `valid/` masks of the four domains | `domain_shift_stats.npz`, `domain_shift_report.csv`, `data_driven_parameters.csv`, `augmentation_suggestions.csv`, `intra_source_w1.csv`, blur/noise calibration, `white_point.csv`, `hue_by_section.csv`, `run_info.json`, three figures |
| `plot_domain_shift.py` | Figures of Chapter 3 from the outputs above | the analysis folder (+ a few patches for the example figure) | `figures/ds2_*.png` |
| `check_augmentation_params.py` | Closure test: applies the fitted parameters to Source patches and checks that the W1 distance to each target drops (Section 4.4.4). With `--cfg`, checks that the photometric block of a training config covers the targets | analysis folder, Source patches, optionally a config | `closure_test.csv`, `config_coverage.csv`, `figures/ds2_test_chiusura*.png`, previews |
| `fit_inverse_correction.py` | Fits the inverse correction (target -> Source): gamma, contrast and saturation per target and modality that bring the new sections towards the Source, and its closure test | analysis folder, patches of the targets | `inverse_correction/inverse_correction_params.json` (+ fit and closure CSVs, preview) |

### tools/training/

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `train_mm.py` | Trains MMSFormer-B3 on the historical sections with the augmentation of a config; early stopping on the moving median of the validation mIoU | `--cfg <config>`, optional `--seed`, `data/mortars/`, `checkpoints/pretrained/segformer/mit_b3.pth` | `output/MMSFormer/<run>/`: best checkpoint + `.meta.json`, `_last.pth`, `train.log`, `train_summary.json`, TensorBoard events |
| `val_mm.py` | Evaluates a checkpoint on the test split (IoU, F1, accuracy per class). Its `evaluate()` function is also the validation loop of `train_mm.py` | `--cfg`, `--run-dir` (best checkpoint) or `--model-path` | `eval_<timestamp>.txt` next to the checkpoint, `output/confusion/<run>/`; test patches in `output/results/` only with `EVAL.SAVE_PREDICTIONS: true` |

### tools/evaluation/

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `run_all_inferences_softmax.py` | Predictions of one or more trained models on patch folders: the new sections (`--domains`) or any folder (`--patches-dir`, `--pattern`). Options: TTA over 8 flips/rotations (`--tta`), inverse correction (`--inverse-correction`), softmax maps (`--save-softmax`) | `output/MMSFormer/<run>/` (best checkpoint), patches | one PNG mask per patch (green = aggregates) in `output/inference_nuove/...`, `inference_5x/...` or `inference_custom/...`, sub-folder `masks/`, `masks_tta/`, `masks_corr/` |
| `eval_ood_gt.py` | Out-of-domain metrics: maps the predicted patches back onto the annotated crop of 1_SCALA and UNITO_B (padding removed) and compares them with the ground truth | the masks above, `grid_metadata.json`, the GT crops in `DIA_FIRENZE/Nuove Immagini/{1,2}/` | `output/gt_ood/` (or `gt_ood_corr/`): `gt_ood_metrics.csv/.json`, `crop_offsets.json`, error maps in `<model>/<domain>_<source>_errors.png` |
| `summarize_results.py` | Tables of Chapter 6: one row per run and mean +- std across seeds per configuration | `train_summary.json` and `eval_*.txt` of every run, `gt_ood_metrics.csv` (+ the one of the correction) | `output/summary/runs_summary.csv`, `configs_summary.csv`, `correction_summary.csv` |
| `reconstruct_section.py` | Stitches the predicted patches of one section into a full-resolution map, optionally overlaid on the photo | `--meta grid_metadata.json`, `--pred-dir <masks folder>` | the PNG given with `--output` (+ `--overlay-output`) |

## semseg/ (library)

Nothing in `semseg/` reads or writes files on its own initiative: the functions receive arrays, tensors or paths
from the scripts and return results. Every module is imported by at least one script.

### Data, augmentation, normalization

| Module | Content | Used by |
|---|---|---|
| `datasets/mortars.py` | `MORTARS` dataset: lists the patches of `data/mortars` (section 5 excluded), splits them into train / val / test (60/20/20, fixed seed), returns `([NP, NX], label, file name)`; merges porosity into binder | `train_mm.py`, `val_mm.py` |
| `augmentations_mm.py` | Augmentation of the NP/NX pairs: geometric (flips, 90-degree rotations, rotation, resized crop) applied identically to both modalities and to the label, photometric (`PhotometricAugmentation`: gamma, contrast, saturation, brightness, hue, blur, noise, per modality or synchronous), built from the `AUGMENTATION` block of a config | `train_mm.py`, `val_mm.py`, `check_augmentation_params.py` |
| `normalization.py` | The input normalization (`imagenet` or `scale01`): `normalize_tensor()` applies it, `resolve_normalization()` picks it from the config or from the checkpoint sidecar, so training and inference always match | training, validation, inference |

### Model and training

| Module | Content |
|---|---|
| `models/mmsformer.py` | `MMSFormer`: backbone + decoder head, takes the list `[NP, NX]` and returns the class logits |
| `models/backbones/mmsformer.py` | MiT-B3 encoder (one per modality) and the fusion blocks between NP and NX |
| `models/heads/segformer.py` | SegFormer decoder (all-MLP head) |
| `models/base.py`, `models/layers/` | base class (loading of the pretrained backbone), shared layers, weight initialization |
| `losses.py` | cross-entropy (plain and OHEM) and Dice, with the ignore label |
| `optimizers.py`, `schedulers.py` | AdamW/SGD and the learning-rate schedules (warmup + poly / cosine / plateau) |
| `metrics.py` | `Metrics`: confusion histogram that skips the ignore label; IoU, F1 and accuracy per class and their means |
| `utils/utils.py` | seeds, cuDNN setup, logger, FLOPs and other helpers |
| `checkpoints.py` | the `.meta.json` sidecar of every checkpoint (normalization, seed, epoch, ...), `find_best_checkpoint(run_dir)` and `load_state_dict(path)` |

### Preprocessing and analysis helpers

| Module | Content | Used by |
|---|---|---|
| `preprocessing.py` | centered padding, check that a grid cell lies inside the valid mask, NP/NX alignment of a patch pair by phase correlation (`align_patch_pair`), false-colour previews, lookup of the image files | the preprocessing scripts |
| `patch_geometry.py` | where the real content of an aligned patch is (`content_slices`, `valid_mask`) and where it falls in the full section (`np_rect`, `paste_content`), from the shifts in `grid_metadata.json` | `make_valid_masks.py`, `eval_ood_gt.py`, `reconstruct_section.py` |
| `domain_shift.py` | the statistics of the domain shift analysis and the fit of the photometric parameters (see below) | the four `tools/domain_shift/` scripts |

`domain_shift.py` holds the computations shared by the domain shift scripts, so that they are written once:

- **patch sampling:** `list_patches`, `sample_patches` (stratified by section for the Source), `calibration_set`
  (10 Source patches per section), `load_pair` (image + validity mask);
- **distances and histograms:** `w1` (Wasserstein distance between two histograms), `w1_circular` (the same
  on the hue circle), quantiles and circular statistics of the hue;
- **per-domain accumulation:** `accumulate` collects histograms, per-patch medians, sharpness and GLCM texture
  of a list of patches, on valid pixels only;
- **sharpness calibration:** `blur_calibration`, `noise_calibration` and `equivalent_sigma` /
  `equivalent_noise_sigma` translate a sharpness ratio into the blur or noise sigma that produces it on the Source;
- **photometric fit:** `apply_tone` applies gamma -> contrast -> saturation with the same torchvision functions
  used in training, `tone_stats` measures median V, gray std and median S, `fit_tone` finds the three values
  that bring a set of patches onto target medians. `quantify_domain_shift.py` uses it Source -> target (the
  data-driven parameters), `fit_inverse_correction.py` target -> Source (the inverse correction);
- **ranges:** `rule_range` widens the measured extremes into the augmentation range of the config.
