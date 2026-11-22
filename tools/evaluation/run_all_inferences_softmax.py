#!/usr/bin/env python3
"""
Batch inference of the trained models on the patches of the new sections (out-of-domain).

Every run folder under --runs-dir (default output/campaign3/MMSFormer, one folder per config and seed,
written by tools/training/train_mm.py) is registered automatically: MMSF-V3-GEN-060-p080-seed42 becomes the model
key 'v3_gen_060_p080_seed42'. For each model the best checkpoint of its folder is loaded, with the input
normalization recorded in its .meta.json sidecar.

For every patch pair (NP, NX) the predicted class map is saved as a PNG (green = aggregates, black = binder):
  - single forward pass                       -> masks/
  - --tta: mean of the softmax over 8 flips/rotations (the dihedral group of the square) -> masks_tta/
  - --inverse-correction PARAMS_JSON: the patches of each target are first brought towards the Source with
    the gamma, contrast and saturation fitted by tools/domain_shift/fit_inverse_correction.py -> masks_corr/ (masks_tta_corr/)

Outputs: <output-root>/inference_nuove/inference_nuove_<model>/<SECTION>/MMSFormer-B3/<masks dir>/<patch>.png
for 1_SCALA and UNITO_B, <output-root>/inference_5x/inference_5x_scale=057_<model>/ARCHEO_02/... for ARCHEO_02.

Any other patch folder with paralleli/ and incrociati/ sub-folders (e.g. the historical sections in data/mortars)
can be given with --patches-dir instead of --domains; --pattern keeps only the patches whose file name matches
(e.g. 'sec10_*' for one section). Masks go to <output-root>/inference_custom/inference_custom_<model>/<NAME>/...
Patches of data/mortars include the training set: use these maps to inspect the predictions, and
tools/training/val_mm.py for the test-set metrics.
They are read by tools/evaluation/eval_ood_gt.py and tools/evaluation/reconstruct_section.py.

Usage (from the repository root):
    python tools/evaluation/run_all_inferences_softmax.py --domains 1_SCALA UNITO_B
    python tools/evaluation/run_all_inferences_softmax.py --models v3_baseline_med v3_gen_060_p080 --domains 1_SCALA UNITO_B --tta
    python tools/evaluation/run_all_inferences_softmax.py --models v3_baseline_med v3_baseline_med_seed42 v3_baseline_med_seed3408 \
        --domains 1_SCALA UNITO_B \
        --inverse-correction output/domain_shift_analysis_v2/inverse_correction/inverse_correction_params.json
    python tools/evaluation/run_all_inferences_softmax.py --models v3_baseline_med --patches-dir data/mortars --pattern 'sec10_*'
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import glob
import json
import time
import argparse
import numpy as np
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable

from semseg.models import MMSFormer
from semseg.normalization import normalize_tensor
from semseg.checkpoints import find_best_checkpoint, read_checkpoint_meta, load_state_dict

Image.MAX_IMAGE_PIXELS = None

DEFAULT_RUNS_DIR = os.path.join('output', 'campaign3', 'MMSFormer')
DEFAULT_OUTPUT_ROOT = os.path.join('output', 'campaign3')

# Patch folders (tools/preprocessing/preprocess_new_images.py, tools/preprocessing/preprocess_5x.py) and output sub-folder of each domain
DOMAINS_REGISTRY = {
    '1_SCALA': {
        'patches_dir': 'data/nuove_patches/1_SCALA',
        'vis_save_dir': 'inference_nuove/inference_nuove_{exp}/1_SCALA'
    },
    'UNITO_B': {
        'patches_dir': 'data/nuove_patches/UNITO_B',
        'vis_save_dir': 'inference_nuove/inference_nuove_{exp}/UNITO_B'
    },
    'ARCHEO_02': {
        'patches_dir': 'data/patches_5x_scale=057/ARCHEO_02',
        'vis_save_dir': 'inference_5x/inference_5x_scale=057_{exp}/ARCHEO_02'
    },
}

# Test-time augmentation: (name, forward transform, inverse transform) on [B, C, H, W] tensors
TTA_TENSOR_TRANSFORMS = [
    ('identity', lambda x: x, lambda x: x),
    ('hflip', lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
    ('vflip', lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[2])),
    ('rot180', lambda x: torch.rot90(x, k=2, dims=[2, 3]), lambda x: torch.rot90(x, k=2, dims=[2, 3])),
    ('rot90', lambda x: torch.rot90(x, k=1, dims=[2, 3]), lambda x: torch.rot90(x, k=-1, dims=[2, 3])),
    ('rot270', lambda x: torch.rot90(x, k=3, dims=[2, 3]), lambda x: torch.rot90(x, k=-3, dims=[2, 3])),
    ('hflip_rot90', lambda x: torch.rot90(torch.flip(x, dims=[3]), k=1, dims=[2, 3]),
                    lambda x: torch.flip(torch.rot90(x, k=-1, dims=[2, 3]), dims=[3])),
    ('vflip_rot90', lambda x: torch.rot90(torch.flip(x, dims=[2]), k=1, dims=[2, 3]),
                    lambda x: torch.flip(torch.rot90(x, k=-1, dims=[2, 3]), dims=[2])),
]


def model_key(run_dir):
    """Model key of a run folder: MMSF-V3-GEN-060-p080-seed42 -> 'v3_gen_060_p080_seed42'."""
    return os.path.basename(run_dir).lower().replace('mmsf-', '', 1).replace('-', '_')


def discover_models(runs_dir):
    """{model key: run folder} for every run folder under runs_dir."""
    return {model_key(d): d for d in sorted(glob.glob(os.path.join(runs_dir, 'MMSF-*'))) if os.path.isdir(d)}


def get_preprocessing(normalization, size=(512, 512)):
    """Resize (a no-op on 512 x 512 patches) + normalization + batch dimension."""
    return T.Compose([
        T.Resize(size),
        T.Lambda(lambda x: normalize_tensor(x, normalization)),
        T.Lambda(lambda x: x.unsqueeze(0))
    ])


def domain_files(dom_info, pattern='*'):
    """Sorted list of NP patch paths of a domain whose file name matches pattern."""
    return sorted(glob.glob(os.path.join(dom_info['patches_dir'], 'paralleli', '**', pattern + '.tif'), recursive=True))


def load_model(checkpoint_path, device):
    """Loads MMSFormer-B3 with the weights of a checkpoint, in evaluation mode."""
    model = MMSFormer('MMSFormer-B3', num_classes=2, modals=['paralleli', 'incrociati'])
    model.load_state_dict(load_state_dict(checkpoint_path))
    model = model.to(device)
    model.eval()
    return model


def open_img(file_path):
    """Opens a TIFF patch as a uint8 tensor [3, H, W]."""
    pil_img = Image.open(file_path).convert('RGB')
    return (TF.to_tensor(pil_img) * 255).to(torch.uint8)


def inverse_correction(img, p):
    """Inverse photometric correction target -> Source of tools/domain_shift/fit_inverse_correction.py.

    Same functions and order as semseg.domain_shift.apply_tone: gamma -> contrast -> saturation
    on the uint8 RGB patch, before resizing and normalization.
    """
    if p['gamma'] != 1.0:
        img = TF.adjust_gamma(img, p['gamma'])
    if p['contrast'] != 1.0:
        img = TF.adjust_contrast(img, p['contrast'])
    if p['saturation'] != 1.0:
        img = TF.adjust_saturation(img, p['saturation'])
    return img


def main():
    parser = argparse.ArgumentParser(description="Batch inference on the patches of the new sections")
    parser.add_argument('--runs-dir', default=DEFAULT_RUNS_DIR, help="Folder with one sub-folder per training run")
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT, help="Root of the prediction folders")
    parser.add_argument('--models', nargs='+', default=None,
                        help="Model keys (e.g. v3_baseline_med v3_gen_060_p080_seed42); default: every run in --runs-dir")
    parser.add_argument('--domains', nargs='+', default=list(DOMAINS_REGISTRY), choices=list(DOMAINS_REGISTRY))
    parser.add_argument('--patches-dir', default=None,
                        help="Any patch folder with paralleli/ and incrociati/ sub-folders (e.g. data/mortars); "
                             "replaces --domains")
    parser.add_argument('--name', default=None,
                        help="Output sub-folder of --patches-dir (default: the folder name, plus the pattern if given)")
    parser.add_argument('--pattern', default='*',
                        help="Keep only the patches whose file name (without .tif) matches, e.g. 'sec10_*'")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-softmax', action='store_true',
                        help="Also save the float16 softmax maps (.npy, several GB)")
    parser.add_argument('--tta', action='store_true',
                        help="Average the softmax over the 8 flips/rotations; masks go to masks_tta/")
    parser.add_argument('--inverse-correction', default=None, metavar='PARAMS_JSON',
                        help="Apply the inverse photometric correction of tools/domain_shift/fit_inverse_correction.py "
                             "(inverse_correction_params.json) to every NP/NX patch; masks go to masks_corr/ "
                             "(masks_tta_corr/ with --tta), so the uncorrected predictions are kept")
    args = parser.parse_args()

    # Domains to process: the registered new sections, or a single folder given with --patches-dir
    domains = {k: DOMAINS_REGISTRY[k] for k in args.domains}
    if args.patches_dir:
        name = args.name or os.path.basename(os.path.normpath(args.patches_dir))
        if not args.name and args.pattern != '*':
            name += '_' + args.pattern.replace('*', '').replace('?', '').strip('_')
        domains = {name: {'patches_dir': args.patches_dir,
                          'vis_save_dir': f'inference_custom/inference_custom_{{exp}}/{name}'}}

    registry = discover_models(args.runs_dir)
    models = args.models or list(registry)
    correction = None
    if args.inverse_correction:
        with open(args.inverse_correction, encoding='utf-8') as f:
            correction = json.load(f)['params']

    device = torch.device(args.device)
    palette = torch.tensor([[0, 0, 0], [0, 255, 0]])  # class 0: binder (black), class 1: aggregates (green)

    print("=" * 80)
    print(f"Models          : {models}")
    print(f"Domains         : {list(domains)}" + (f" (pattern {args.pattern})" if args.pattern != '*' else ''))
    print(f"Device          : {device}")
    print(f"TTA             : {'8 transforms' if args.tta else 'off (single forward pass)'}")
    print(f"Inverse correct.: {args.inverse_correction or 'none'}")
    print("=" * 80)

    for key in models:
        if key not in registry:
            print(f"Warning: no run folder for model '{key}' in {args.runs_dir}, skipping.")
            continue
        ckpt_path = find_best_checkpoint(registry[key])
        if ckpt_path is None:
            print(f"Warning: no checkpoint with a .meta.json sidecar in {registry[key]}, skipping.")
            continue
        normalization = read_checkpoint_meta(ckpt_path)['normalization']

        print("\n" + "#" * 70)
        print(f"MODEL {key} ({ckpt_path}), input normalization: {normalization}")
        print("#" * 70)
        model = load_model(ckpt_path, device)
        preprocess = get_preprocessing(normalization)

        for dom_key, dom_info in domains.items():
            if correction is not None and dom_key not in correction:
                print(f"  [Skip] No inverse-correction parameters for domain {dom_key}")
                continue
            vis_dir = Path(args.output_root) / dom_info['vis_save_dir'].format(exp=key) / 'MMSFormer-B3'

            # Single-pass, TTA and corrected predictions go to separate folders and never overwrite each other
            suffix = ('_tta' if args.tta else '') + ('_corr' if correction is not None else '')
            masks_dir = vis_dir / f'masks{suffix}'
            softmax_dir = vis_dir / f'softmax{suffix}'
            os.makedirs(masks_dir, exist_ok=True)
            if args.save_softmax:
                os.makedirs(softmax_dir, exist_ok=True)

            files = domain_files(dom_info, args.pattern)
            if not files:
                print(f"  [Skip] No patch files found for domain {dom_key} in {dom_info['patches_dir']}")
                continue
            print(f"\n--- {key} on {dom_key}: {len(files)} patch pairs ---")

            start_t = time.time()
            with torch.inference_mode():
                for file_p in tqdm(files, desc=f"{key} [{dom_key}]"):
                    file_x = 'incrociati'.join(file_p.rsplit('paralleli', 1))
                    if not os.path.exists(file_x):
                        print(f"  Warning: matching NX patch not found: {file_x}, skipping.")
                        continue

                    img1_raw = open_img(file_p)
                    img2_raw = open_img(file_x)
                    if correction is not None:
                        img1_raw = inverse_correction(img1_raw, correction[dom_key]['NP'])
                        img2_raw = inverse_correction(img2_raw, correction[dom_key]['NX'])
                    img1_t = preprocess(img1_raw).to(device)
                    img2_t = preprocess(img2_raw).to(device)
                    stem = Path(file_p).stem

                    if not args.tta:
                        probs = model([img1_t, img2_t]).softmax(dim=1)  # [1, 2, 512, 512]
                    else:
                        k_probs = []
                        for _, fwd_fn, inv_fn in TTA_TENSOR_TRANSFORMS:
                            t_prob = model([fwd_fn(img1_t), fwd_fn(img2_t)]).softmax(dim=1)
                            k_probs.append(inv_fn(t_prob))  # back to the original orientation
                        probs = torch.stack(k_probs, dim=0).mean(dim=0)

                    if args.save_softmax:
                        np.save(softmax_dir / f"{stem}.npy", probs.squeeze(0).cpu().half().numpy())

                    pred_labels = probs.argmax(dim=1).squeeze(0).cpu().to(int)
                    Image.fromarray(palette[pred_labels].numpy().astype(np.uint8)).save(masks_dir / f"{stem}.png")

            elapsed = time.time() - start_t
            print(f"  Completed in {elapsed:.1f}s" + (f" ({len(files) / elapsed:.1f} patch/s)" if elapsed > 0 else ""))
            print(f"  Masks saved to: {masks_dir}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
