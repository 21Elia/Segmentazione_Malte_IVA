#!/usr/bin/env python3
"""
Automated Batch Inference & Softmax / TTA Probability Extraction Runner.
Bachelor's Thesis in Computer Engineering — Elia Awad, University of Florence (UNIFI).

Executes inference across all 5 models and all 3 target OOD domains with:
1. Standard forward pass with float16 Softmax probability saving (.npy) for Shannon Entropy (SE).
2. Optional C4 Test-Time Augmentation (TTA) with 8 geometric transforms to compute
   Prediction Variance (VPT) per patch.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable

from semseg.models import *
from semseg.datasets import *
from semseg.normalization import normalize_tensor, NORMALIZATIONS, LEGACY_NORMALIZATION, read_checkpoint_meta, checkpoint_meta_path

Image.MAX_IMAGE_PIXELS = None

# Pre-defined registry of models and checkpoints
MODELS_REGISTRY = {
    'baseline': {
        'checkpoint': 'output/MMSFormer/MMSF-MORTARS-CONFIG/MMSFormer_MMSFormer-B3_MORTARS_epoch81_79.51.pth',
        'pipeline': 'v1'
    },
    'exp1': {
        'checkpoint': 'output/MMSFormer/MMSF-EXP1-GEO-AUG/MMSFormer_MMSFormer-B3_MORTARS_epoch48_80.61.pth',
        'pipeline': 'v1'
    },
    'exp2': {
        'checkpoint': 'output/MMSFormer/MMSF-EXP2-PHOTO-AUG/MMSFormer_MMSFormer-B3_MORTARS_epoch50_78.41.pth',
        'pipeline': 'exp2'
    },
    'exp3': {
        'checkpoint': 'output/MMSFormer/MMSF-EXP3-SOFT-AUG/MMSFormer_MMSFormer-B3_MORTARS_epoch39_79.41.pth',
        'pipeline': 'exp3'
    },
    'exp4a': {
        'checkpoint': 'output/MMSFormer/MMSF-EXP4A-ASYM-PHOTO/MMSFormer_MMSFormer-B3_MORTARS_epoch40_78.78.pth',
        'pipeline': 'exp4'
    },
    # Second campaign: best checkpoint found automatically in the run folder, normalization
    # read from its .meta.json sidecar (see semseg/normalization.py)
    'v2_baseline': {'run_dir': 'output/MMSFormer/MMSF-V2-BASELINE'},
    'v2_exp1':     {'run_dir': 'output/MMSFormer/MMSF-V2-EXP1'},
    'v2_exp2':     {'run_dir': 'output/MMSFormer/MMSF-V2-EXP2'},
    'v2_exp4b':    {'run_dir': 'output/MMSFormer/MMSF-V2-EXP4B'},
}
FIRST_CAMPAIGN = ['baseline', 'exp1', 'exp2', 'exp3', 'exp4a']

DOMAINS_REGISTRY = {
    '1_SCALA': {
        'patches_dir': 'data/nuove_patches/1_SCALA',
        'vis_save_dir': 'output/inference_nuove/inference_nuove_{exp}/1_SCALA'
    },
    'UNITO_B': {
        'patches_dir': 'data/nuove_patches/UNITO_B',
        'vis_save_dir': 'output/inference_nuove/inference_nuove_{exp}/UNITO_B'
    },
    'ARCHEO_02': {
        'patches_dir': 'data/patches_5x_scale=057/ARCHEO_02',
        'vis_save_dir': 'output/inference_5x/inference_5x_scale=057_{exp}/ARCHEO_02'
    },
    # In-domain reference: the historical test split (same images in data/mortars and
    # data/mortars_v2; only the labels differ). Used by compute_entropy_vpt.py --indomain.
    'INDOMAIN_TEST': {
        'test_split_of': 'data/mortars_v2',
        'vis_save_dir': 'output/inference_indomain/{exp}'
    }
}

# C4 TTA Transformations (Forward and Inverse for PyTorch Tensors [B, C, H, W])
TTA_TENSOR_TRANSFORMS = [
    # (name, fwd_fn, inv_fn)
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


def get_preprocessing(pipeline_name: str, size: tuple = (512, 512)):
    """Resize + normalization. pipeline_name is a normalization ('imagenet' / 'scale01')
    or a legacy first-campaign augmentation name (mapped by LEGACY_NORMALIZATION)."""
    normalization = pipeline_name if pipeline_name in NORMALIZATIONS else LEGACY_NORMALIZATION[pipeline_name]
    return T.Compose([
        T.Resize(size),
        T.Lambda(lambda x: normalize_tensor(x, normalization)),
        T.Lambda(lambda x: x.unsqueeze(0))
    ])


def find_best_checkpoint(run_dir: str):
    """Best weights of a v2 run: the .pth with a .meta.json sidecar that is not '_last' / '_checkpoint'."""
    candidates = [p for p in glob.glob(os.path.join(run_dir, '*.pth'))
                  if not p.endswith(('_last.pth', '_checkpoint.pth')) and os.path.isfile(checkpoint_meta_path(p))]
    if not candidates:
        return None
    return max(candidates, key=lambda p: read_checkpoint_meta(p).get('val_miou', 0))


def resolve_model(model_key: str):
    """(checkpoint_path, normalization) for a registry entry."""
    info = MODELS_REGISTRY[model_key]
    if 'checkpoint' in info:
        return info['checkpoint'], info['pipeline']
    ckpt = find_best_checkpoint(info['run_dir'])
    if ckpt is None:
        return None, None
    return ckpt, read_checkpoint_meta(ckpt)['normalization']


def domain_files(dom_info: dict):
    """Sorted list of NP patch paths of a domain."""
    if 'test_split_of' in dom_info:
        return [str(f) for f in MORTARS(dom_info['test_split_of'], 'test', None, ['paralleli', 'incrociati']).files]
    return sorted(glob.glob(os.path.join(dom_info['patches_dir'], 'paralleli', '**', '*.tif'), recursive=True))


def load_model(checkpoint_path: str, device: torch.device):
    """Loads MMSFormer-B3 model and checkpoint weights."""
    model = MMSFormer('MMSFormer-B3', num_classes=2, modals=['paralleli', 'incrociati'])
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def open_img(file_path: str):
    """Opens TIFF image and converts to uint8 tensor [3, H, W]."""
    pil_img = Image.open(file_path).convert('RGB')
    return (TF.to_tensor(pil_img) * 255).to(torch.uint8)


def main():
    parser = argparse.ArgumentParser(description="Batch Runner for Softmax and TTA Probability Extraction")
    parser.add_argument('--models', nargs='+', default=FIRST_CAMPAIGN,
                        help="Models to execute (e.g. baseline exp1 exp2 exp3 exp4a)")
    parser.add_argument('--domains', nargs='+', default=list(DOMAINS_REGISTRY.keys()),
                        help="Domains to evaluate (e.g. 1_SCALA UNITO_B ARCHEO_02)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Execution device (cuda or cpu)")
    parser.add_argument('--tta', action='store_true',
                        help="Enable the 8-transform D4 test-time augmentation (VPT + averaged softmax, saved to masks_tta/ and softmax_tta/)")
    args = parser.parse_args()

    device = torch.device(args.device)
    palette = torch.tensor([[0, 0, 0], [0, 255, 0]])  # Class 0: Binder (black), Class 1: Aggregates (green)

    print("=" * 80)
    print("BATCH INFERENCE & PROBABILITY EXTRACTION RUNNER (SE & VPT)")
    print(f"Models Selected : {args.models}")
    print(f"Domains Selected: {args.domains}")
    print(f"Execution Device: {device}")
    print(f"TTA C4 Mode     : {'ENABLED (K=8 passes per patch for VPT)' if args.tta else 'STANDARD (1 pass with softmax)'}")
    print("=" * 80)

    for model_key in args.models:
        if model_key not in MODELS_REGISTRY:
            print(f"Warning: Unknown model '{model_key}', skipping.")
            continue

        ckpt_path, normalization = resolve_model(model_key)
        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found: {ckpt_path}, skipping.")
            continue

        print(f"\n" + "#" * 70)
        print(f"LOADING MODEL: {model_key.upper()} ({ckpt_path})")
        print("#" * 70)

        model = load_model(ckpt_path, device)
        preprocess = get_preprocessing(normalization)
        print(f"  Input normalization: {normalization}")

        for dom_key in args.domains:
            if dom_key not in DOMAINS_REGISTRY:
                continue

            dom_info = DOMAINS_REGISTRY[dom_key]
            vis_dir = Path(dom_info['vis_save_dir'].format(exp=model_key)) / 'MMSFormer-B3'

            # Single-pass and TTA outputs go to separate folders, so a --tta run never
            # overwrites the single-pass predictions (first-campaign masks/ and softmax/
            # folders were overwritten by the TTA run).
            suffix = '_tta' if args.tta else ''
            masks_dir = vis_dir / f'masks{suffix}'
            softmax_dir = vis_dir / f'softmax{suffix}'
            vpt_dir = vis_dir / 'vpt'

            os.makedirs(masks_dir, exist_ok=True)
            os.makedirs(softmax_dir, exist_ok=True)
            if args.tta:
                os.makedirs(vpt_dir, exist_ok=True)

            files = domain_files(dom_info)
            if not files:
                print(f"  [Skip] No patch files found for domain {dom_key}")
                continue

            print(f"\n--- Model {model_key.upper()} on Domain {dom_key}: {len(files)} patch pairs ---")

            start_t = time.time()
            with torch.inference_mode():
                for file_p in tqdm(files, desc=f"{model_key.upper()} [{dom_key}]"):
                    file_x = 'incrociati'.join(file_p.rsplit('paralleli', 1))
                    if not os.path.exists(file_x):
                        print(f"  Warning: Matching crossed patch not found: {file_x}, skipping.")
                        continue

                    img1_raw = open_img(file_p)
                    img2_raw = open_img(file_x)

                    img1_t = preprocess(img1_raw).to(device)
                    img2_t = preprocess(img2_raw).to(device)

                    stem = Path(file_p).stem

                    if not args.tta:
                        # Single standard forward pass
                        logits = model([img1_t, img2_t])
                        probs = logits.softmax(dim=1)  # [1, 2, 512, 512]
                    else:
                        # K=8 C4 Test-Time Augmentation
                        k_probs = []
                        for _, fwd_fn, inv_fn in TTA_TENSOR_TRANSFORMS:
                            t_img1 = fwd_fn(img1_t)
                            t_img2 = fwd_fn(img2_t)
                            t_logits = model([t_img1, t_img2])
                            t_prob = t_logits.softmax(dim=1)
                            # Invert spatial transformation to return to base coordinates
                            inv_prob = inv_fn(t_prob)
                            k_probs.append(inv_prob)

                        stacked_k = torch.stack(k_probs, dim=0)  # [8, 1, 2, 512, 512]
                        probs = stacked_k.mean(dim=0)            # Mean probability [1, 2, 512, 512]

                        # Compute VPT: Variance of predicted aggregate class (index 1) across the K=8 transforms
                        agg_k = stacked_k[:, 0, 1, :, :]         # [8, 512, 512]
                        vpt_patch = agg_k.var(dim=0, unbiased=False).cpu().numpy().astype(np.float32)
                        np.save(vpt_dir / f"{stem}.npy", vpt_patch)

                    # Save Softmax Probability Map (.npy float16)
                    softmax_np = probs.squeeze(0).cpu().half().numpy()
                    np.save(softmax_dir / f"{stem}.npy", softmax_np)

                    # Update / save clean binary mask
                    pred_labels = probs.argmax(dim=1).squeeze(0).cpu().to(int)
                    palette_mask = palette[pred_labels].numpy().astype(np.uint8)
                    Image.fromarray(palette_mask).save(masks_dir / f"{stem}.png")

            elapsed = time.time() - start_t
            speed_str = f"({len(files)/elapsed:.1f} patch/s)" if elapsed > 0 else ""
            print(f"  Completed in {elapsed:.1f}s {speed_str}".strip())
            print(f"  Saved Softmax to: {softmax_dir}")
            if args.tta:
                print(f"  Saved VPT to    : {vpt_dir}")

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("ALL BATCH INFERENCES AND PROBABILITY EXTRACTIONS COMPLETED!")
    print("=" * 80)


if __name__ == '__main__':
    main()
