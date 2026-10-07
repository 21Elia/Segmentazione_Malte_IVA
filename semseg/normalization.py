"""
Single source of truth for the input normalization of MMSFormer.

Two normalizations exist:
  - 'imagenet': x / 255, then (x - mean) / std with the ImageNet statistics
  - 'scale01' : x / 255 only (how the first-campaign Baseline and Exp1 were trained)

Every tool (training, validation, inference) resolves the normalization with
resolve_normalization() and applies it with normalize_tensor() / the sample-dict
transforms in semseg.augmentations_mm, so training and inference cannot diverge.

Training writes a sidecar file '<checkpoint>.meta.json' next to every saved
checkpoint, recording the normalization and the other settings needed to use it.
"""

import json
import os

import torch
import torchvision.transforms.functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NORMALIZATIONS = ('imagenet', 'scale01')

# First-campaign configs only declared TRAIN.AUGMENTATION as a string; this is the
# normalization each of those checkpoints was actually trained with (verified on the
# test split by evaluating every checkpoint with both normalizations).
LEGACY_NORMALIZATION = {
    'v1': 'scale01',
    'exp1': 'scale01',
    'exp2': 'imagenet',
    'exp3': 'imagenet',
    'v2_soft': 'imagenet',
    'exp4': 'imagenet',
}


def normalize_tensor(x, name):
    """Normalizes a CxHxW (or BxCxHxW) image tensor with values in [0, 255]."""
    if name not in NORMALIZATIONS:
        raise ValueError(f"Unknown normalization '{name}' (expected one of {NORMALIZATIONS})")
    x = x.float() / 255.0
    if name == 'imagenet':
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    return x


def checkpoint_meta_path(checkpoint_path):
    return f"{checkpoint_path}.meta.json"


def write_checkpoint_meta(checkpoint_path, meta):
    with open(checkpoint_meta_path(checkpoint_path), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


def read_checkpoint_meta(checkpoint_path):
    path = checkpoint_meta_path(checkpoint_path)
    if checkpoint_path and os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return None


def _legacy_augmentation_name(cfg):
    """Returns the first-campaign TRAIN.AUGMENTATION string, if the config has one."""
    for section in (cfg.get('TRAIN', {}), cfg, cfg.get('TEST', {})):
        value = section.get('AUGMENTATION') if isinstance(section, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def resolve_normalization(cfg, checkpoint_path=None):
    """
    Resolves the input normalization, in order of priority:
      1. DATASET.NORMALIZATION in the config (required for all second-campaign configs)
      2. the '<checkpoint>.meta.json' sidecar written at training time
      3. the legacy TRAIN.AUGMENTATION string of first-campaign configs
    Raises an error if none is available, instead of guessing from file names.
    If both the config and the sidecar declare it, they must agree.
    """
    declared = cfg.get('DATASET', {}).get('NORMALIZATION')
    meta = read_checkpoint_meta(checkpoint_path) if checkpoint_path else None
    from_meta = meta.get('normalization') if meta else None

    if declared is not None:
        declared = str(declared).lower()
        if declared not in NORMALIZATIONS:
            raise ValueError(f"DATASET.NORMALIZATION must be one of {NORMALIZATIONS}, got '{declared}'")
        if from_meta is not None and from_meta != declared:
            raise ValueError(
                f"Normalization mismatch: config declares '{declared}' but checkpoint "
                f"{checkpoint_path} was trained with '{from_meta}'")
        return declared
    if from_meta is not None:
        return from_meta

    legacy = _legacy_augmentation_name(cfg)
    if legacy in LEGACY_NORMALIZATION:
        return LEGACY_NORMALIZATION[legacy]

    raise ValueError(
        "Cannot determine the input normalization: add DATASET.NORMALIZATION "
        "('imagenet' or 'scale01') to the config.")
