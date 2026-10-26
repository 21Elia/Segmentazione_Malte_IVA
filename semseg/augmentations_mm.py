import torchvision.transforms.functional as TF 
import random
import math
import torch
from torch import Tensor
from typing import Tuple, List, Union
from semseg.normalization import IMAGENET_MEAN, IMAGENET_STD, NORMALIZATIONS


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        first_img_key = [k for k in sample.keys() if k != 'mask'][0]
        img, mask = sample[first_img_key], sample['mask']
        if mask.ndim == 2:
            assert img.shape[1:] == mask.shape
        else:
            assert img.shape[1:] == mask.shape[1:]

        for transform in self.transforms:
            sample = transform(sample)

        return sample


class Normalize:
    def __init__(self, mean: list = (0.485, 0.456, 0.406), std: list = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict) -> dict:
        for k, v in sample.items():
            if k == 'mask':
                continue
            tensor = v.float() / 255.0
            sample[k] = TF.normalize(tensor, self.mean, self.std)
        
        return sample


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            for k, v in sample.items():
                sample[k] = TF.hflip(v)
        return sample


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            for k, v in sample.items():
                sample[k] = TF.vflip(v)
        return sample


class RandomRotation90:
    """Rotate the sample by a random multiple of 90 degrees (C4 cyclic group). Lossless."""
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            k_rot = random.choice([1, 2, 3])  # 90°, 180°, 270°
            for key, v in sample.items():
                sample[key] = torch.rot90(v, k_rot, dims=[1, 2])
        return sample


class RandomRotation:
    def __init__(self, degrees: float = 10.0, p: float = 0.2, seg_fill: int = 0, expand: bool = False) -> None:
        self.p = p
        self.angle = degrees
        self.expand = expand
        self.seg_fill = seg_fill

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            random_angle = random.random() * 2 * self.angle - self.angle
            for k, v in sample.items():
                if k == 'mask':                
                    sample[k] = TF.rotate(v, random_angle, TF.InterpolationMode.NEAREST, self.expand, fill=self.seg_fill)
                else:
                    sample[k] = TF.rotate(v, random_angle, TF.InterpolationMode.BILINEAR, self.expand, fill=0)
        return sample


class Resize:
    def __init__(self, size: Union[int, Tuple[int], List[int]]) -> None:
        self.size = size

    def __call__(self, sample: dict) -> dict:
        first_img_key = [k for k in sample.keys() if k != 'mask'][0]
        H, W = sample[first_img_key].shape[1:]

        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        for k, v in sample.items():
            if k == 'mask':                
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)

        alignH, alignW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        for k, v in sample.items():
            if k == 'mask':                
                sample[k] = TF.resize(v, (alignH, alignW), TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, (alignH, alignW), TF.InterpolationMode.BILINEAR)
        return sample


class RandomResizedCrop:
    def __init__(self, size: Union[int, Tuple[int], List[int]], scale: Tuple[float, float] = (0.5, 2.0), seg_fill: int = 0) -> None:
        self.size = size
        self.scale = scale
        self.seg_fill = seg_fill

    def __call__(self, sample: dict) -> dict:
        first_img_key = [k for k in sample.keys() if k != 'mask'][0]
        H, W = sample[first_img_key].shape[1:]
        tH, tW = self.size

        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        scale = int(tH*ratio), int(tW*ratio)
        scale_factor = min(max(scale)/max(H, W), min(scale)/min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)

        for k, v in sample.items():
            if k == 'mask':                
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)

        margin_h = max(sample[first_img_key].shape[1] - tH, 0)
        margin_w = max(sample[first_img_key].shape[2] - tW, 0)
        # randint is inclusive: margin+1 could start the crop 1 px past the image and pad it
        y1 = random.randint(0, margin_h)
        x1 = random.randint(0, margin_w)
        y2 = y1 + tH
        x2 = x1 + tW
        for k, v in sample.items():
            sample[k] = v[:, y1:y2, x1:x2]

        if sample[first_img_key].shape[1:] != self.size:
            padding = [0, 0, tW - sample[first_img_key].shape[2], tH - sample[first_img_key].shape[1]]
            for k, v in sample.items():
                if k == 'mask':                
                    sample[k] = TF.pad(v, padding, fill=self.seg_fill)
                else:
                    sample[k] = TF.pad(v, padding, fill=0)

        return sample


class Scale01:
    def __call__(self, sample: dict) -> dict:
        for k, v in sample.items():
            if k == 'mask':
                continue
            sample[k] = v.float() / 255.0
        return sample


def get_normalization_transform(normalization: str):
    """Final sample-dict transform for a normalization name (see semseg/normalization.py)."""
    if normalization == 'imagenet':
        return Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if normalization == 'scale01':
        return Scale01()
    raise ValueError(f"Unknown normalization '{normalization}' (expected one of {NORMALIZATIONS})")


def get_val_augmentation(size: Union[int, Tuple[int], List[int]], normalization: str):
    """Validation / test pipeline: resize to a multiple of 32 + normalization (see semseg/normalization.py)."""
    return Compose([
        Resize(size),
        get_normalization_transform(normalization)
    ])


# =======================================================================================
# Training pipelines, built from the TRAIN.AUGMENTATION dictionary of the config
# =======================================================================================
#
# TRAIN.AUGMENTATION, for example:
#
#   AUGMENTATION:
#     GEOMETRIC:
#       HFLIP: 0.5                         # probability
#       VFLIP: 0.5
#       ROT90: 0.5
#       ROTATE: {DEGREES: 15, P: 0.2}
#       RESIZED_CROP: {SCALE: [0.5, 2.0]}
#     PHOTOMETRIC:                         # optional
#       SYNC: false                        # true: same random parameters for every modality
#       paralleli:                         # per-modality op lists (SYNC: false) ...
#         - {OP: brightness, RANGE: [0.6, 1.45], P: 0.7}
#       incrociati:
#         - {OP: blur, SIGMA: [0.2, 0.5], P: 0.3}
#       # SHARED: [ ... ]                  # ... or a single shared list (SYNC: true)
#       P_IMAGE: 0.5                       # optional, see below
#       AT_LEAST_ONE: true                 # optional, only with P_IMAGE (default true)
#
# P_IMAGE (optional): probability that an image goes through the photometric pipeline. Without
# P_IMAGE every op is drawn independently with its own P and an image may come out untouched.
# AT_LEAST_ONE (with P_IMAGE): if true, a selected image always receives at least one op: the
# per-op draws are repeated until one succeeds, so P_IMAGE is exactly the fraction of perturbed
# images, but the effective per-op probabilities are inflated (P / (1 - prod(1 - P_i)), e.g. 0.3
# becomes 0.39 with four ops at 0.3). If false, a selected image draws every op once with its own P
# and may come out untouched: each op fires with probability P_IMAGE * P, and all P = 0 is allowed
# (no photometric op at all, as in the Baseline).
#
# Pipeline order: photometric -> geometric -> normalization (DATASET.NORMALIZATION), so that
# photometric ops never act on the zero padding introduced by rotations and crops.
#
# Without GEOMETRIC.RESIZED_CROP the images are only resized (as in validation).
#
# Photometric ops (applied in list order, each with its own probability P):
#   brightness / contrast / saturation : multiplicative factor in RANGE (LOG: true samples log-uniformly)
#   hue                                 : shift in RANGE, torchvision units ([-0.5, 0.5] = full circle)
#   gamma                               : exponent in RANGE (> 1 darkens, < 1 brightens)
#   jitter                              : torchvision-style colour jitter with BRIGHTNESS / CONTRAST /
#                                         SATURATION / HUE ranges, applied together with one P
#   blur                                : Gaussian blur, sigma in SIGMA (px); KERNEL fixes the kernel size,
#                                         otherwise it is 2*ceil(3*sigma)+1
#   noise                               : additive Gaussian noise, std in SIGMA (intensity levels)


def _sample_range(rng, log=False):
    lo, hi = float(rng[0]), float(rng[1])
    if log:
        return math.exp(random.uniform(math.log(lo), math.log(hi)))
    return random.uniform(lo, hi)


class _PhotometricOp:
    """One photometric operation: sample() draws its random parameters, apply() uses them."""

    def __init__(self, spec: dict) -> None:
        self.name = str(spec['OP']).lower()
        self.p = float(spec.get('P', 1.0))
        self.spec = spec
        if self.name not in ('brightness', 'contrast', 'saturation', 'hue', 'gamma', 'jitter', 'blur', 'noise'):
            raise ValueError(f"Unknown photometric op '{self.name}'")

    def sample(self):
        s = self.spec
        if self.name in ('brightness', 'contrast', 'saturation', 'gamma'):
            return _sample_range(s['RANGE'], s.get('LOG', False))
        if self.name == 'hue':
            return _sample_range(s['RANGE'])
        if self.name == 'jitter':
            return {k: _sample_range(s[k]) for k in ('BRIGHTNESS', 'CONTRAST', 'SATURATION', 'HUE') if k in s}
        return _sample_range(s['SIGMA'])  # blur, noise

    def apply(self, v: Tensor, param) -> Tensor:
        if self.name == 'brightness':
            return TF.adjust_brightness(v, param)
        if self.name == 'contrast':
            return TF.adjust_contrast(v, param)
        if self.name == 'saturation':
            return TF.adjust_saturation(v, param)
        if self.name == 'hue':
            return TF.adjust_hue(v, param)
        if self.name == 'gamma':
            return TF.adjust_gamma(v, param)
        if self.name == 'jitter':
            if 'BRIGHTNESS' in param: v = TF.adjust_brightness(v, param['BRIGHTNESS'])
            if 'CONTRAST' in param: v = TF.adjust_contrast(v, param['CONTRAST'])
            if 'SATURATION' in param: v = TF.adjust_saturation(v, param['SATURATION'])
            if 'HUE' in param: v = TF.adjust_hue(v, param['HUE'])
            return v
        if self.name == 'blur':
            if param <= 0:
                return v
            k = int(self.spec.get('KERNEL', 0)) or max(3, 2 * math.ceil(3.0 * param) + 1)
            k += 1 - k % 2  # odd kernel
            return TF.gaussian_blur(v, kernel_size=k, sigma=[param, param])
        # noise
        noisy = v.float() + torch.randn_like(v.float()) * param
        return torch.clamp(noisy, 0, 255).to(v.dtype)


class PhotometricAugmentation:
    """
    Per-modality (SYNC: false) or synchronous (SYNC: true) photometric augmentation.
    Never touches the 'mask' key. In synchronous mode every op draws its parameters once
    and applies them to all modalities (noise uses the same std but independent noise).
    """

    def __init__(self, spec: dict) -> None:
        self.sync = bool(spec.get('SYNC', False))
        if self.sync:
            self.ops = {'*': [_PhotometricOp(o) for o in spec.get('SHARED', [])]}
        else:
            self.ops = {k: [_PhotometricOp(o) for o in v] for k, v in spec.items() if k not in ('SYNC', 'SHARED', 'P_IMAGE', 'AT_LEAST_ONE')}
        self.p_image = spec.get('P_IMAGE')
        self.at_least_one = spec.get('AT_LEAST_ONE', True)
        if not isinstance(self.at_least_one, bool):
            raise ValueError(f"AT_LEAST_ONE must be true or false, got {self.at_least_one!r}")
        if self.p_image is not None:
            self.p_image = float(self.p_image)
            if not 0.0 <= self.p_image <= 1.0:
                raise ValueError(f"P_IMAGE must be in [0, 1], got {self.p_image}")
            # with resampling, all P = 0 would loop forever
            if self.at_least_one and self.p_image > 0 and not any(op.p > 0 for ops in self.ops.values() for op in ops):
                raise ValueError("P_IMAGE > 0 with AT_LEAST_ONE needs at least one photometric op with P > 0")

    def __repr__(self) -> str:
        ops = {k: [op.name for op in v] for k, v in self.ops.items()}
        return f"PhotometricAugmentation(sync={self.sync}, p_image={self.p_image}, at_least_one={self.at_least_one}, ops={ops})"

    def _draw_at_least_one(self) -> dict:
        """Which ops fire, {modality: [bool per op]}, conditioned on at least one op firing."""
        while True:
            fire = {k: [random.random() < op.p for op in ops] for k, ops in self.ops.items()}
            if any(any(f) for f in fire.values()):
                return fire

    def __call__(self, sample: dict) -> dict:
        keys = [k for k in sample if k != 'mask']
        if self.p_image is not None:
            return self._call_p_image(sample, keys)
        if self.sync:
            for op in self.ops['*']:
                if random.random() < op.p:
                    param = op.sample()
                    for k in keys:
                        sample[k] = op.apply(sample[k], param)
            return sample
        for k, ops in self.ops.items():
            if k not in sample:
                raise KeyError(f"Photometric spec refers to modality '{k}', not present in the sample {keys}")
            for op in ops:
                if random.random() < op.p:
                    sample[k] = op.apply(sample[k], op.sample())
        return sample

    def _call_p_image(self, sample: dict, keys: list) -> dict:
        # Kept apart from the per-op path above so that configs without P_IMAGE draw the same random sequence
        if random.random() >= self.p_image:
            return sample
        if self.at_least_one:
            fire = self._draw_at_least_one()
        else:
            fire = {k: [random.random() < op.p for op in ops] for k, ops in self.ops.items()}
        if self.sync:
            for op, f in zip(self.ops['*'], fire['*']):
                if f:
                    param = op.sample()
                    for k in keys:
                        sample[k] = op.apply(sample[k], param)
            return sample
        for k, ops in self.ops.items():
            if k not in sample:
                raise KeyError(f"Photometric spec refers to modality '{k}', not present in the sample {keys}")
            for op, f in zip(ops, fire[k]):
                if f:
                    sample[k] = op.apply(sample[k], op.sample())
        return sample


def build_geometric_transforms(spec: dict, size, seg_fill: int = 0) -> list:
    """Geometric transforms in a fixed order: HFLIP, VFLIP, ROT90, ROTATE, RESIZED_CROP."""
    ops = []
    if spec.get('HFLIP'):
        ops.append(RandomHorizontalFlip(p=float(spec['HFLIP'])))
    if spec.get('VFLIP'):
        ops.append(RandomVerticalFlip(p=float(spec['VFLIP'])))
    if spec.get('ROT90'):
        ops.append(RandomRotation90(p=float(spec['ROT90'])))
    if spec.get('ROTATE'):
        r = spec['ROTATE']
        ops.append(RandomRotation(degrees=float(r['DEGREES']), p=float(r.get('P', 1.0)), seg_fill=seg_fill))
    if spec.get('RESIZED_CROP'):
        ops.append(RandomResizedCrop(size, scale=tuple(spec['RESIZED_CROP'].get('SCALE', (0.5, 2.0))), seg_fill=seg_fill))
    else:
        ops.append(Resize(size))
    return ops


def build_train_augmentation(aug_spec: dict, size, seg_fill: int, normalization: str) -> Compose:
    """Builds the training pipeline: photometric -> geometric -> normalization."""
    transforms = []
    if aug_spec.get('PHOTOMETRIC'):
        transforms.append(PhotometricAugmentation(aug_spec['PHOTOMETRIC']))
    transforms += build_geometric_transforms(aug_spec.get('GEOMETRIC', {}), size, seg_fill)
    transforms.append(get_normalization_transform(normalization))
    return Compose(transforms)
