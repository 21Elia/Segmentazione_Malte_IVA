import torch
import argparse
import yaml
from pathlib import Path
from torchvision import io, transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import os
from semseg.utils.utils import timer
from semseg.datasets import *
from semseg.models import *
from torch import Tensor
import glob

class SemSeg:
    def __init__(self, cfg) -> None:
        # inference device
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        dataset_class = eval(cfg['DATASET']['NAME'])
        dataset = dataset_class(
            root=cfg['DATASET']['ROOT'],
            split='train',  # o un split dummy
            modals=cfg['DATASET']['MODALS'],
            num_classes=cfg['DATASET']['NUM_CLASSES']
        )
        self.palette = dataset.PALETTE
        self.labels = dataset.CLASSES

        # initialize the model and load weights
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette), cfg['DATASET']['MODALS'])
        msg = self.model.load_state_dict(torch.load(cfg['EVAL']['MODEL_PATH'], map_location='cpu'))
        print(msg)
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocessing
        self.size = cfg['TEST']['IMAGE_SIZE']
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

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)
        seg_image = self.palette[seg_map].squeeze()
        if overlay: 
            seg_image = (orig_img.permute(1, 2, 0) * 0.6) + (seg_image * 0.4)
        image = seg_image.to(torch.uint8)
        return Image.fromarray(image.numpy())

    @torch.inference_mode()
    @timer
    def model_forward(self, imgs):
        return self.model(imgs)

    def predict(self, img_fname: str, overlay: bool) -> Tensor:
        # due modali: paralleli + incrociati
        x1_path = img_fname
        x2_path = img_fname.replace('paralleli', 'incrociati')

        img1 = self.tf_pipeline_modal(self._open_img(x1_path)).to(self.device)
        img2 = self.tf_pipeline_modal(self._open_img(x2_path)).to(self.device)

        sample = [img1, img2]
        seg_map = self.model_forward(sample)
        seg_map = self.postprocess(self._open_img(img_fname), seg_map, overlay)
        return seg_map


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
    os.makedirs(save_dir, exist_ok=True)

    semseg = SemSeg(cfg)

    if test_file.is_file():
        segmap = semseg.predict(str(test_file), cfg['TEST']['OVERLAY'])
        segmap.save(save_dir / f"{test_file.stem}.png")
    else:
        if cfg['DATASET']['NAME'] == 'MORTARS':
            # cerca tutte le TIFF in paralleli/ (sottocartelle opzionali)
            files = sorted(glob.glob(os.path.join(str(test_file), 'paralleli', '**', '*.tif'), recursive=True))
        else:
            raise NotImplementedError()

        for file in files:
            segmap = semseg.predict(file, cfg['TEST']['OVERLAY'])
            filename = os.path.basename(file).replace('.tif', '.png')
            save_path = save_dir / filename
            segmap.save(save_path) 
            