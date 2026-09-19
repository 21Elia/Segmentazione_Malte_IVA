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
    if aug_version in ['exp2', 'exp3', 'v2_soft', 'exp4']:
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


class AsymmetricMultiModalPhotoAugmentation:
    """Augmentation fotometrica disaccoppiata NP vs NX, calibrata sui W₁ misurati (Milestone 6).

    Applica pipeline fotometriche DIVERSE e INDIPENDENTI alle due modalità ottiche:

    - 'paralleli' (NP / PPL): pipeline FORTE, calibrata sulle distanze di Wasserstein
      misurate tra Source Storico e i tre domini target. Brightness ±0.40, Saturation ±0.45,
      Contrast ±0.25, Gamma [0.75,1.40], GaussianBlur σ∈[0.3,1.74] px, Hue ±0.02.

    - 'incrociati' (NX / XPL): pipeline CONSERVATIVA, per proteggere i colori di
      birifrangenza di Michel-Lévy che sono un segnale fisico diagnostico intrinseco e stabile.
      Brightness ±0.15, Saturation ±0.10, Gamma [0.90,1.10]. NO HueJitter (W₁(a*,NX)<1.3),
      NO GaussianBlur (ARCHEO_02 NX validato, UNITO_B NX non critico).

    Le trasformazioni geometriche che precedono questa classe nella Compose pipeline
    vengono applicate con lo stesso seed casuale a NP e NX (co-registrazione spaziale):
    ogni RandomHorizontalFlip, RandomVerticalFlip, RandomRotation90 ecc. prende UNA
    decisione casuale e la applica a TUTTE le chiavi del sample dict, garantendo che
    NP e NX rappresentino sempre lo stesso campo visivo con la stessa geometria.

    Le trasformazioni fotometriche QUI DENTRO invece usano seed INDIPENDENTI per NP e NX:
    un brightness=1.3 su NP e brightness=0.85 su NX contemporaneamente è fisicamente
    plausibile (intensità lampada vs calibrazione polarizzatori sono indipendenti).

    Derivazione parametri NP:
        brightness ±0.40  ← W₁(V,NP) medio ≈ 44 → 44/128 ≈ 0.34, +margine → 0.40
        saturation ±0.45  ← W₁(S,NP) max  = 59 → 59/128 ≈ 0.46 → 0.45
        contrast   ±0.25  ← GLCM Contrasto NP: +20% su 1_SCALA, −63% su UNITO_B
        gamma    [0.75,1.40] ← copre sia sottoesposizione (1_SCALA/UNITO_B) sia sovraesposizione (ARCHEO_02)
        blur σ ≤ 1.74    ← σ_max = 2√(1 − 145/601) da Laplaciano mediano UNITO_B NP
        hue      ±0.02   ← W₁(b*,NP) max = 8.1 → 8.1/128 ≈ 0.06; usato 1/3 per sicurezza

    Derivazione parametri NX:
        brightness ±0.15  ← W₁(V,NX) medio ≈ 24.7 → 24.7/128 ≈ 0.19; valore conservativo
        saturation ±0.10  ← W₁(S,NX) < 25 ovunque; ultra-conservativo per Michel-Lévy
        gamma  [0.90,1.10] ← micro-correzione esposizione XPL, range stretto

    Riferimenti:
        domain_shift_outputs_analysis.md — sezioni OUTPUT 2 (Wasserstein), OUTPUT 3 (Laplaciano)
        cdf_analysis_complete.md — analisi canali V, S, b* per NP e NX
    """

    def __init__(
        self,
        # ── NP parameters (calibrated on measured W₁) ──────────────────────────
        np_brightness: float = 0.40,
        np_saturation: float = 0.45,
        np_contrast: float = 0.25,
        np_gamma_range: tuple = (0.75, 1.40),
        np_blur_sigma_range: tuple = (0.3, 1.74),
        np_hue: float = 0.02,
        np_p_brightness: float = 0.70,
        np_p_saturation: float = 0.50,
        np_p_contrast: float = 0.40,
        np_p_gamma: float = 0.40,
        np_p_blur: float = 0.30,
        np_p_hue: float = 0.30,
        # ── NX parameters (conservative, protects Michel-Lévy colours) ──────────
        nx_brightness: float = 0.15,
        nx_saturation: float = 0.10,
        nx_gamma_range: tuple = (0.90, 1.10),
        nx_p_brightness: float = 0.50,
        nx_p_saturation: float = 0.30,
        nx_p_gamma: float = 0.30,
        # ── Modal keys ──────────────────────────────────────────────────────────
        np_key: str = 'paralleli',
        nx_key: str = 'incrociati',
    ) -> None:
        # NP
        self.np_brightness = np_brightness
        self.np_saturation = np_saturation
        self.np_contrast = np_contrast
        self.np_gamma_range = np_gamma_range
        self.np_blur_sigma_range = np_blur_sigma_range
        self.np_hue = np_hue
        self.np_p_brightness = np_p_brightness
        self.np_p_saturation = np_p_saturation
        self.np_p_contrast = np_p_contrast
        self.np_p_gamma = np_p_gamma
        self.np_p_blur = np_p_blur
        self.np_p_hue = np_p_hue
        # NX
        self.nx_brightness = nx_brightness
        self.nx_saturation = nx_saturation
        self.nx_gamma_range = nx_gamma_range
        self.nx_p_brightness = nx_p_brightness
        self.nx_p_saturation = nx_p_saturation
        self.nx_p_gamma = nx_p_gamma
        # Keys
        self.np_key = np_key
        self.nx_key = nx_key

    # ──────────────────────────────────────────────────────────────────────────
    def _apply_np(self, v: Tensor) -> Tensor:
        """Pipeline fotometrica forte su NP (Nicols Paralleli / luce bianca)."""
        # Brightness: simula variazioni di intensità della lampada tra microscopi
        if random.random() < self.np_p_brightness:
            factor = random.uniform(1.0 - self.np_brightness, 1.0 + self.np_brightness)
            v = TF.adjust_brightness(v, factor)

        # Saturation: simula ipersaturazione in-camera (1_SCALA) e desaturazione (ARCHEO_02)
        if random.random() < self.np_p_saturation:
            factor = random.uniform(1.0 - self.np_saturation, 1.0 + self.np_saturation)
            v = TF.adjust_saturation(v, factor)

        # Contrast: copre contrasto alto (1_SCALA GLCM=20.7) e basso (UNITO_B GLCM=6.4)
        if random.random() < self.np_p_contrast:
            factor = random.uniform(1.0 - self.np_contrast, 1.0 + self.np_contrast)
            v = TF.adjust_contrast(v, factor)

        # Gamma: simula sotto/sovraesposizione non lineare della lampada
        if random.random() < self.np_p_gamma:
            gamma = random.uniform(self.np_gamma_range[0], self.np_gamma_range[1])
            v = TF.adjust_gamma(v, gamma)

        # Gaussian Blur: simula sfocatura ottica (UNITO_B: Laplaciano −76%)
        # kernel_size adattivo: almeno 2*ceil(3*sigma)+1, minimo 3, sempre dispari
        if random.random() < self.np_p_blur:
            sigma = random.uniform(self.np_blur_sigma_range[0], self.np_blur_sigma_range[1])
            k = max(3, 2 * math.ceil(3.0 * sigma) + 1)
            if k % 2 == 0:
                k += 1  # garantisce dispari (safety check)
            v = TF.gaussian_blur(v, kernel_size=k, sigma=[sigma, sigma])

        # Hue: simula lieve dominante calda di 1_SCALA (W₁(B,NP)=62 → deficit di blu)
        # Range ±0.02 vincolato da W₁(b*,NP) max=8.1 → 8.1/128≈0.06; usato 1/3
        if random.random() < self.np_p_hue:
            h = random.uniform(-self.np_hue, self.np_hue)
            v = TF.adjust_hue(v, h)

        return v

    # ──────────────────────────────────────────────────────────────────────────
    def _apply_nx(self, v: Tensor) -> Tensor:
        """Pipeline fotometrica conservativa su NX (Nicols Incrociati / XPL).

        Protegge i colori di birifrangenza di Michel-Lévy: NO HueJitter
        (W₁(a*,NX) < 1.3 = invariante fisico), NO GaussianBlur (struttura
        spaziale XPL di ARCHEO_02 già validata da GLCM, UNITO_B NX non critico).
        """
        # Brightness: lieve variazione di luminosità XPL (W₁(V,NX) medio ≈ 24.7)
        if random.random() < self.nx_p_brightness:
            factor = random.uniform(1.0 - self.nx_brightness, 1.0 + self.nx_brightness)
            v = TF.adjust_brightness(v, factor)

        # Saturation: minima variazione vivacità XPL (W₁(S,NX) < 25 ovunque)
        if random.random() < self.nx_p_saturation:
            factor = random.uniform(1.0 - self.nx_saturation, 1.0 + self.nx_saturation)
            v = TF.adjust_saturation(v, factor)

        # Gamma: micro-correzione esposizione XPL (range stretto [0.90, 1.10])
        if random.random() < self.nx_p_gamma:
            gamma = random.uniform(self.nx_gamma_range[0], self.nx_gamma_range[1])
            v = TF.adjust_gamma(v, gamma)

        return v

    # ──────────────────────────────────────────────────────────────────────────
    def __call__(self, sample: dict) -> dict:
        """Applica le pipeline NP e NX con seed casuali INDIPENDENTI."""
        if self.np_key in sample:
            sample[self.np_key] = self._apply_np(sample[self.np_key])
        if self.nx_key in sample:
            sample[self.nx_key] = self._apply_nx(sample[self.nx_key])
        return sample


def get_train_augmentation_exp4(size: Union[int, Tuple[int], List[int]], seg_fill: int = 0):
    """Esperimento 4A: Augmentation Geometrica C₄ + Fotometrica Disaccoppiata NP/NX.

    Strategia a due fasi:

    FASE 1 — Geometrica sincrona (identica a Exp 1, già dimostratamente ottimale):
        I transform geometrici (flip, rotazioni, crop) vengono applicati a TUTTE
        le chiavi del sample dict con la STESSA decisione casuale → NP e NX ricevono
        sempre la stessa trasformazione spaziale (co-registrazione garantita).

    FASE 2 — Fotometrica disaccoppiata (AsymmetricMultiModalPhotoAugmentation):
        NP riceve augmentation forte calibrata sui W₁ misurati (Milestone 6).
        NX riceve augmentation conservativa per proteggere i colori di Michel-Lévy.
        I seed casuali delle due pipeline sono INDIPENDENTI (fotometria fisicamente
        indipendente tra sorgente lampada bianca e sistema di polarizzatori).

    Checkpoint di partenza: Exp 1 (epoch 48, Val mIoU 80.61%) + LR=1e-5 + 40 epoche.
    """
    return Compose([
        # ── Fase 1: Trasformazioni Geometriche Sincrone (co-registrazione NP↔NX) ──
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(p=0.5),
        RandomRotation(degrees=15, p=0.2, seg_fill=seg_fill),
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),

        # ── Fase 2: Augmentation Fotometrica Disaccoppiata (NP forte, NX conservativa) ──
        AsymmetricMultiModalPhotoAugmentation(),

        # ── Fase 3: Normalizzazione ImageNet ──
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])