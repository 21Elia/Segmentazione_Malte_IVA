import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import argparse
import yaml
from pathlib import Path
from torchvision import io, transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from semseg.utils.utils import timer
from semseg.datasets import *
from semseg.models import *
from torch import Tensor
import glob

class SemSeg:
    def __init__(self, cfg) -> None:
        # inference device
        device_str = cfg.get('DEVICE', 'cpu')
        if not torch.cuda.is_available() and 'cuda' in str(device_str).lower():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_str)

        # get dataset classes' colors and labels
        dataset_class = eval(cfg['DATASET']['NAME'])
        dataset = dataset_class(
            root=cfg['DATASET']['ROOT'],
            split='train',  # o un split dummy
            modals=cfg['DATASET']['MODALS'],
            num_classes=cfg['DATASET']['NUM_CLASSES']
        )
        if cfg.get('TEST', {}).get('INVERT_PALETTE', False):
            self.palette = dataset.PALETTE.flip(0)
        elif 'PALETTE' in cfg.get('TEST', {}):
            self.palette = torch.tensor(cfg['TEST']['PALETTE'])
        else:
            self.palette = dataset.PALETTE
        self.labels = dataset.CLASSES

        # initialize the model and load weights
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette), cfg['DATASET']['MODALS'])
        checkpoint = torch.load(cfg['EVAL']['MODEL_PATH'], map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        msg = self.model.load_state_dict(state_dict)
        print(msg)
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocessing
        self.size = cfg['TEST']['IMAGE_SIZE']
        aug_version = cfg['TRAIN'].get('AUGMENTATION', 'v1') if 'TRAIN' in cfg else cfg.get('AUGMENTATION', 'v1')
        if aug_version in ['exp2', 'exp3', 'v2_soft']:
            self.tf_pipeline_modal = T.Compose([
                T.Resize(self.size),
                T.Lambda(lambda x: x / 255),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                T.Lambda(lambda x: x.unsqueeze(0))
            ])
        else:
            self.tf_pipeline_modal = T.Compose([
                T.Resize(self.size),
                T.Lambda(lambda x: x / 255),
                T.Lambda(lambda x: x.unsqueeze(0))
            ])

    '''def _open_img(self, file):
        # legge immagini e gestisce canali
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img'''
    def _open_img(self, file):
        # legge immagini TIFF con PIL e converte in tensor CxHxW
        pil_img = Image.open(file).convert('RGB')  # forza 3 canali
        img = TF.to_tensor(pil_img) * 255          # torchvision tensors aspettano [0,255]
        return img.to(torch.uint8)

    def postprocess(self, orig_img: Tensor, seg_map: Tensor) -> tuple:
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)
        palette_seg = self.palette[seg_map].squeeze()
        
        # 1. Pure palette mask (Green=Aggregates, Black=Binder)
        mask_np = palette_seg.to(torch.uint8).numpy()
        mask_pil = Image.fromarray(mask_np)

        # 2. Blended patch overlay (60% parallel image + 40% prediction palette)
        orig_hwc = orig_img.permute(1, 2, 0).float()
        overlay_np = (orig_hwc * 0.6 + palette_seg.float() * 0.4).to(torch.uint8).numpy()
        overlay_pil = Image.fromarray(overlay_np)

        return mask_pil, overlay_pil

    @torch.inference_mode()
    @timer
    def model_forward(self, imgs):
        return self.model(imgs)

    def predict(self, img_fname: str, overlay: bool = False) -> tuple:
        # due modali: paralleli + incrociati
        x1_path = img_fname
        x2_path = img_fname.replace('paralleli', 'incrociati')

        img1 = self.tf_pipeline_modal(self._open_img(x1_path)).to(self.device)
        img2 = self.tf_pipeline_modal(self._open_img(x2_path)).to(self.device)

        sample = [img1, img2]
        seg_map = self.model_forward(sample)
        orig_img = self._open_img(img_fname)
        return self.postprocess(orig_img, seg_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str)
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    test_file = Path(cfg['TEST']['FILE'])
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    modals_name = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    save_dir = Path(cfg['TEST']['VIS_SAVE_DIR'])/(cfg['MODEL']['BACKBONE'])
    
    masks_dir = save_dir / 'masks'
    overlays_dir = save_dir / 'overlays'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    semseg = SemSeg(cfg)

    if test_file.is_file():
        mask_img, overlay_img = semseg.predict(str(test_file))
        mask_img.save(masks_dir / f"{test_file.stem}.png")
        overlay_img.save(overlays_dir / f"{test_file.stem}.png")
        print(f"Saved mask to: {masks_dir / f'{test_file.stem}.png'}")
        print(f"Saved overlay patch to: {overlays_dir / f'{test_file.stem}.png'}")
    else:
        if cfg['DATASET']['NAME'] == 'MORTARS':
            # cerca tutte le TIFF in paralleli/ (sottocartelle opzionali)
            files = sorted(glob.glob(os.path.join(str(test_file), 'paralleli', '**', '*.tif'), recursive=True))
        else:
            raise NotImplementedError()

        print(f"Running inference on {len(files)} patch pairs from: {test_file}")
        for file in files:
            mask_img, overlay_img = semseg.predict(file)
            filename = os.path.basename(file).replace('.tif', '.png')
            
            # Save mask inside masks/ folder only
            mask_img.save(masks_dir / filename)
            
            # Save semi-transparent overlay inside overlays/ folder only
            overlay_img.save(overlays_dir / filename)

        print(f"\nInference complete!")
        print(f" - Clean Masks saved to    : {masks_dir}")
        print(f" - Patch Overlays saved to : {overlays_dir}")

 
            