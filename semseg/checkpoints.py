"""
Checkpoints written by tools/train_mm.py and how to find and load them.

Every saved checkpoint '<name>.pth' has a sidecar '<name>.pth.meta.json' with the settings needed
to use it consistently (normalization, dataset, augmentation, seed, epoch, validation mIoU).
A run folder holds the best checkpoint (by validation mIoU), its '_checkpoint.pth' twin with the
optimizer state, and '_last.pth' with the weights of the last epoch.
"""

import glob
import json
import os

import torch


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


def find_best_checkpoint(run_dir):
    """Best weights of a run folder: the .pth with a .meta.json sidecar that is not '_last' / '_checkpoint'.

    Training keeps a single best checkpoint per run; if there were several, the one with the
    highest validation mIoU is returned. Returns None if the folder has none.
    """
    candidates = [p for p in glob.glob(os.path.join(run_dir, '*.pth'))
                  if not p.endswith(('_last.pth', '_checkpoint.pth')) and os.path.isfile(checkpoint_meta_path(p))]
    if not candidates:
        return None
    return max(candidates, key=lambda p: read_checkpoint_meta(p).get('val_miou', 0))


def load_state_dict(checkpoint_path):
    """Model weights of a checkpoint: plain state dict or '_checkpoint.pth' dict, 'module.' prefix removed."""
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    return state_dict
