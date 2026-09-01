import torchvision.transforms.functional as TF 
import random
import math
import torch
from torch import Tensor
from typing import Tuple, List, Union, Optional


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


class RandomColorJitter:
    """Legacy color jitter operating on all non-mask image modalities in sample dict."""
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            brightness = random.uniform(0.5, 1.5)
            contrast = random.uniform(0.5, 1.5)
            saturation = random.uniform(0.5, 1.5)
            for k, v in sample.items():
                if k == 'mask':
                    continue
                v_adj = TF.adjust_brightness(v, brightness)
                v_adj = TF.adjust_contrast(v_adj, contrast)
                v_adj = TF.adjust_saturation(v_adj, saturation)
                sample[k] = v_adj
        return sample


class RandomMultiModalColorJitter:
    """Trasformazione fotometrica sincrona multimodale per NP e NX.
    Applica brightness, contrast, saturation e un piccolissimo hue shift (<=0.05)
    in modo sincrono su tutte le modalità d'immagine (escludendo la maschera GT).
    """
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5) -> None:
        self.brightness_factor = brightness
        self.contrast_factor = contrast
        self.saturation_factor = saturation
        self.hue_factor = hue
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            b = random.uniform(1 - self.brightness_factor, 1 + self.brightness_factor)
            c = random.uniform(1 - self.contrast_factor, 1 + self.contrast_factor)
            s = random.uniform(1 - self.saturation_factor, 1 + self.saturation_factor)
            h = random.uniform(-self.hue_factor, self.hue_factor)

            for k, v in sample.items():
                if k == 'mask':
                    continue
                v_adj = TF.adjust_brightness(v, b)
                v_adj = TF.adjust_contrast(v_adj, c)
                v_adj = TF.adjust_saturation(v_adj, s)
                v_adj = TF.adjust_hue(v_adj, h)
                sample[k] = v_adj
        return sample


class RandomGammaCorrection:
    """Applica una correzione gamma casuale I_out = I_in^gamma su tutte le modalità d'immagine."""
    def __init__(self, gamma_range=(0.7, 1.5), p=0.3) -> None:
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            for k, v in sample.items():
                if k == 'mask':
                    continue
                sample[k] = TF.adjust_gamma(v, gamma)
        return sample


class RandomGaussianNoise:
    """Aggiunge rumore gaussiano N(0, sigma^2) a tutte le modalità d'immagine del sample dict."""
    def __init__(self, sigma_range=(0, 15), p=0.3) -> None:
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
            for k, v in sample.items():
                if k == 'mask':
                    continue
                noise = torch.randn_like(v.float()) * sigma
                noisy_v = torch.clamp(v.float() + noise, 0, 255).to(v.dtype)
                sample[k] = noisy_v
        return sample


class RandomMultiModalGaussianBlur:
    """Applica Gaussian Blur con kernel e sigma casuali a tutte le modalità d'immagine."""
    def __init__(self, kernel_size=(3, 3), sigma_range=(0.1, 2.0), p=0.2) -> None:
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
            for k, v in sample.items():
                if k == 'mask':
                    continue
                sample[k] = TF.gaussian_blur(v, self.kernel_size, [sigma, sigma])
        return sample


class RandomGaussianBlur:
    """Legacy Gaussian Blur operating on all non-mask image modalities in sample dict."""
    def __init__(self, kernel_size: int = 3, p: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            for k, v in sample.items():
                if k == 'mask':
                    continue
                sample[k] = TF.gaussian_blur(v, self.kernel_size)
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
        y1 = random.randint(0, margin_h+1)
        x1 = random.randint(0, margin_w+1)
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


def get_train_augmentation(size: Union[int, Tuple[int], List[int]], seg_fill: int = 0):
    return Compose([
        RandomColorJitter(p=0.2),
        RandomHorizontalFlip(p=0.5),
        RandomGaussianBlur(3, p=0.2),
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


class Scale01:
    def __call__(self, sample: dict) -> dict:
        for k, v in sample.items():
            if k == 'mask':
                continue
            sample[k] = v.float() / 255.0
        return sample


def get_val_augmentation(size: Union[int, Tuple[int], List[int]], aug_version: str = 'v1'):
    if aug_version in ['exp2', 'exp3', 'v2_soft']:
        return Compose([
            Resize(size),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        return Compose([
            Resize(size),
            Scale01()
        ])


def get_train_augmentation_exp1(size: Union[int, Tuple[int], List[int]], seg_fill: int = 0):
    """Esperimento 1: pipeline con augmentation geometrica avanzata."""
    return Compose([
        RandomColorJitter(p=0.2),
        RandomGaussianBlur(3, p=0.2),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(p=0.5),
        RandomRotation(degrees=15, p=0.2, seg_fill=seg_fill),
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def get_train_augmentation_exp2(size: Union[int, Tuple[int], List[int]], seg_fill: int = 0):
    """Esperimento 2: Data Augmentation Fotometrica/Radiometrica Sincrona + Geometrica Exp.1."""
    return Compose([
        # --- Fase 1: Trasformazioni Fotometriche/Radiometriche Sincrone ---
        RandomMultiModalColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5),
        RandomGammaCorrection(gamma_range=(0.7, 1.5), p=0.3),
        RandomGaussianNoise(sigma_range=(0, 15), p=0.3),
        RandomMultiModalGaussianBlur(kernel_size=(3, 3), sigma_range=(0.1, 2.0), p=0.2),

        # --- Fase 2: Trasformazioni Geometriche Avanzate (dall'Esperimento 1 Record) ---
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(p=0.5),
        RandomRotation(degrees=15, p=0.2, seg_fill=seg_fill),
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),

        # --- Fase 3: Normalizzazione ImageNet ---
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def get_train_augmentation_exp3(size: Union[int, Tuple[int], List[int]], seg_fill: int = 0):
    """Esperimento 3: Data Augmentation Fotometrica/Radiometrica Soft Sincrona + Geometrica Exp.1.
    Progettata per evitare l'over-regularization e ottimizzare l'inferenza zero-shot sulle Nuove Immagini (1_SCALA, UNITO_B).
    """
    return Compose([
        # --- Fase 1: Trasformazioni Fotometriche/Radiometriche Soft Sincrone ---
        RandomMultiModalColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02, p=0.3),
        RandomGammaCorrection(gamma_range=(0.9, 1.1), p=0.2),
        RandomMultiModalGaussianBlur(kernel_size=(3, 3), sigma_range=(0.1, 1.0), p=0.1),

        # --- Fase 2: Trasformazioni Geometriche Avanzate (dall'Esperimento 1 Record) ---
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(p=0.5),
        RandomRotation(degrees=15, p=0.2, seg_fill=seg_fill),
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),

        # --- Fase 3: Normalizzazione ImageNet ---
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


get_train_augmentation_v2_soft = get_train_augmentation_exp3