"""
Single source of truth for the input normalization of MMSFormer.

Two normalizations exist:
  - 'imagenet': x / 255, then (x - mean) / std with the ImageNet statistics (used by every model of the thesis)
  - 'scale01' : x / 255 only

Every tool (training, validation, inference) resolves the normalization with
resolve_normalization() and applies it with normalize_tensor() / the sample-dict
transforms in semseg.augmentations_mm, so training and inference cannot diverge.
Training records it in the '<checkpoint>.meta.json' sidecar (semseg/checkpoints.py).
"""

import torchvision.transforms.functional as TF

from semseg.checkpoints import read_checkpoint_meta

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NORMALIZATIONS = ('imagenet', 'scale01')


def normalize_tensor(x, name):
    """Normalizes a CxHxW (or BxCxHxW) image tensor with values in [0, 255]."""
    if name not in NORMALIZATIONS:
        raise ValueError(f"Unknown normalization '{name}' (expected one of {NORMALIZATIONS})")
    x = x.float() / 255.0
    if name == 'imagenet':
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    return x


def resolve_normalization(cfg, checkpoint_path=None):
    """
    Resolves the input normalization, in order of priority:
      1. DATASET.NORMALIZATION in the config
      2. the '<checkpoint>.meta.json' sidecar written at training time
    Raises an error if neither is available, instead of guessing.
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

    raise ValueError(
        "Cannot determine the input normalization: add DATASET.NORMALIZATION "
        "('imagenet' or 'scale01') to the config.")
