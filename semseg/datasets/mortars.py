import os
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from pathlib import Path
from typing import Tuple
import glob
from torch.utils.data import DataLoader
from semseg.augmentations_mm import build_train_augmentation
from PIL import Image
import numpy as np
import random
import argparse
import yaml



class MORTARS(Dataset):
    """
    Mortar thin sections: two RGB modalities (NP, NX) and labels with 3 classes + ignore (porosity merged into
    binder when num_classes = 2). Section 5 is excluded; train / val / test split with a fixed seed.
    """

    def __init__(
        self,
        root: str = 'data/mortars',
        split: str = 'train',
        transform=None,
        modals=['paralleli', 'incrociati'],
        case=None,
        val_ratio=0.20,
        test_ratio=0.20,
        seed=42,
        num_classes = 2
    ):
        super().__init__()

        # Class names are kept in Italian: they appear in the eval_*.txt tables read by summarize_results.py
        if num_classes == 2 :
            self.CLASSES = ["Legante", "Aggregati"]
            self.PALETTE = torch.tensor([
                [0, 0, 0],    # binder, black (class 0)
                [0, 255, 0]   # aggregates, green (class 1)
            ])
        else:
            self.CLASSES = ["Legante", "Porosità", "Aggregati"]
            self.PALETTE = torch.tensor([
                [0, 0, 0],  # binder, black
                [255, 0, 0], # porosity, red
                [0, 255, 0]  # aggregates, green
            ])

        # 'all' = every patch, no split (e.g. out-of-domain patches with ground truth)
        assert split in ['train', 'val', 'test', 'all']
        self.transform = transform
        self.modals = modals
        self.n_classes = len(self.CLASSES)  

        if self.n_classes == 2:
            self.label_mapping = torch.tensor([0, 0, 1, 3]) # merges porosity into binder

        self.ignore_label = 3

        # --------------------------------------------------
        # 1. all the patches of the dataset (section 5 excluded)
        # --------------------------------------------------
        all_files = sorted(
            glob.glob(os.path.join(root, modals[0], '*.tif'))
        )
        # Section 5 is excluded (as in the 2classes_no5 dataset of Notari)
        all_files = [
            f for f in all_files
            if not (os.path.basename(f).startswith(('sec5_', 'sec05_')) or '_sec5_' in os.path.basename(f))
        ]

        if not all_files:
            raise RuntimeError(f"No images found in {os.path.join(root, modals[0])}")

        # --------------------------------------------------
        # 2. reproducible split
        # --------------------------------------------------
        if split == 'all':
            self.files = all_files
            print(f"[MORTARS] all: {len(self.files)} samples")
            return

        random.seed(seed)
        random.shuffle(all_files)

        n = len(all_files)
        n_test = int(test_ratio * n)
        n_val  = int(val_ratio * n)
        n_train = n - n_val - n_test

        if split == 'train':
            self.files = all_files[:n_train]
        elif split == 'val':
            self.files = all_files[n_train:n_train + n_val]
        else:  # test
            self.files = all_files[n_train + n_val:]

        print(f"[MORTARS] {split}: {len(self.files)} samples")
    
    
    def _open_img(self, file):
        img = Image.open(file).convert('RGB')  # forces RGB
        img = TF.to_tensor(img)  
        img = (img * 255).byte() 
        return img
    
    def _open_label(self, file):
        lbl = Image.open(file)
        lbl = torch.from_numpy(np.array(lbl)).long()  # [H,W]
        return lbl.unsqueeze(0)  # [1,H,W]

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[list, Tensor]:
        sample = {}

        # images of the modalities
        for modal in self.modals:
            img_path = self.files[index].replace(self.modals[0], modal)
            img = self._open_img(img_path)
            sample[modal] = img

        # label
        lbl_path = self.files[index].replace(self.modals[0], 'label')
        label = self._open_label(lbl_path)
        label[label == 255] = self.ignore_label

        sample['mask'] = label

        if self.transform:
            sample = self.transform(sample)

        label = sample.pop('mask')
        label = label.squeeze(0)  
        sample_list = [sample[m] for m in self.modals]
        if self.n_classes == 2:
            label = self.label_mapping[label] # remaps the classes without porosity


        return sample_list, label.long(), os.path.basename(img_path)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    num_classes = cfg['DATASET']['NUM_CLASSES']

    traintransform = build_train_augmentation(cfg['TRAIN']['AUGMENTATION'], cfg['TRAIN']['IMAGE_SIZE'],
                                              cfg['DATASET']['IGNORE_LABEL'], cfg['DATASET']['NORMALIZATION'])
    trainset = MORTARS(num_classes = num_classes, transform=traintransform, split='train')
    trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=False, pin_memory=False)

    for i, (sample, lbl, fname) in enumerate(trainloader):
        print(torch.unique(lbl))
        if i >= 2:
            break
