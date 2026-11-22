"""
Evaluation on the historical sections (test split by default) and the validation loop used during training.

Writes eval_<timestamp>.txt (IoU, F1 and accuracy per class) next to the evaluated checkpoint and, if
EVAL.SAVE_PREDICTIONS / EVAL.SAVE_CONFUSION are true, the prediction maps and the confusion matrix.

Usage (from the repository root):
    python tools/training/val_mm.py --cfg configs/ablation/generic/mortars_v3_gen-060.yaml \
        --run-dir output/MMSFormer/MMSF-V3-GEN-060-p080-seed42 --vis-dir output/results/v3_gen_060_seed42
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import argparse
import yaml
import time
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_val_augmentation
from semseg.normalization import resolve_normalization
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from semseg.checkpoints import find_best_checkpoint, load_state_dict
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2









@torch.no_grad()
def save_confusion_matrix(): # Generates and saves confusion matrix plot
    cm_file = Path(cfg['EVAL']['CONFUSION_DIR'] + "/confusion_matrix.npy")
    cm = np.load(cm_file)
    
    if cfg['DATASET']['NUM_CLASSES'] == 3:
        class_names = ["Legante", "Porosità", "Aggregati"] # If porosity is considered
    else:
        class_names = ["Legante", "Aggregati"] 

    save_path = cfg['EVAL']['CONFUSION_DIR'] + "/confusion_matrix.png"

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.nan_to_num(cm)  # Avoid division by zero

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)

    # Annotate values inside matrix cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, f"{cm[i, j]:.2f}", ha='center', va='center', color='red')

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)





@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn=None, cfg=None):
    print('Evaluating...')
    model.eval()
    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    test_loss = 0.0
    iter = 0
    for images, labels, fname in tqdm(dataloader):
        images = [x.to(device) for x in images]
        labels = labels.to(device)
        preds = model(images)
        
        pred_labels = preds.argmax(dim=1)  # (B, H, W)
        gt = labels.view(-1).cpu().numpy()
        pd = pred_labels.view(-1).cpu().numpy()
        mask = gt != dataloader.dataset.ignore_label
        conf_matrix += confusion_matrix(
            gt[mask],
            pd[mask],
            labels=list(range(n_classes))
        )

        metrics.update(preds.softmax(dim=1), labels)
        if cfg is not None and cfg['EVAL']['VIS_SAVE_DIR'] and cfg['EVAL']['SAVE_PREDICTIONS']: # Save qualitative visualizations
            mask_valid = labels != dataloader.dataset.ignore_label
            wrong = (pred_labels != labels) & mask_valid
            error_rate = wrong.sum().float() / mask_valid.sum().float()

            # Compute absolute difference of aggregate-classified pixels between prediction and ground truth label
            num_ones_pred = ((pred_labels == 1) & mask_valid).sum().item()
            num_ones_label = ((labels == 1) & mask_valid).sum().item()
            diff = num_ones_label - num_ones_pred
            # Normalized difference
            if diff < 0:
                diff = (-1) * diff
            diff = diff / (512 * 512)

            if error_rate > cfg['EVAL']['ERROR_THRESHOLD'] and diff < 1:

                vis_dir = Path(cfg['EVAL']['VIS_SAVE_DIR'])
                vis_dir.mkdir(parents=True, exist_ok=True)

                pred_mask = pred_labels[0]

                name, _ = os.path.splitext(fname[0])
                save_path = vis_dir / f"{name}_4prediction_{error_rate:.3f}_{diff:.3f}.png"

                save_pred_map(pred_mask, save_path, dataloader.dataset.PALETTE)


        if loss_fn is not None:
            loss = loss_fn(preds, labels)
            test_loss += loss.item()
        iter += 1
    
    test_loss /= iter
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()

    if cfg is not None and cfg['EVAL'].get('SAVE_CONFUSION', False):
        save_dir = Path(cfg['EVAL']['CONFUSION_DIR'])
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'confusion_matrix.npy', conf_matrix)
        save_confusion_matrix()
    
    return acc, macc, f1, mf1, ious, miou, test_loss





@torch.no_grad() # Save prediction maps to disk
def save_pred_map(pred, save_path, palette):
    """
    pred: tensor of shape (H, W) with class values [0..num_classes]
    palette: list/tensor of RGB tuples for each class
    """
    pred_np = pred.cpu().numpy().astype(np.uint8)
    color_map = np.zeros((pred_np.shape[0], pred_np.shape[1], 3), dtype=np.uint8)

    for cls_id, color in enumerate(palette):
        color_map[pred_np == cls_id] = color

    cv2.imwrite(save_path, cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))





@torch.no_grad()
def complete_output_dir(): # Associates predictions with corresponding input modalities and labels
    predictions_dir = cfg['EVAL']['VIS_SAVE_DIR']
    labels_dir = cfg['DATASET']['ROOT'] + "/label"
    incrociati_dir = cfg['DATASET']['ROOT'] + "/incrociati"
    paralleli_dir = cfg['DATASET']['ROOT'] + "/paralleli"

    est = (".png", ".tif")

    predictions = {f.split("_4prediction_")[0] for f in os.listdir(predictions_dir) if f.lower().endswith(est)} 
    labels = {f.split(".")[0] for f in os.listdir(labels_dir) if f.lower().endswith(est)}
    incrociati = {f.split(".")[0] for f in os.listdir(incrociati_dir) if f.lower().endswith(est)}
    paralleli = {f.split(".")[0] for f in os.listdir(paralleli_dir) if f.lower().endswith(est)}
    common_file = predictions & labels & incrociati & paralleli
    print(f"Found {len(common_file)} complete set.")

    # Instantiate dataset to obtain exact palette and label mapping
    dataset_dummy = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', None, cfg['DATASET']['MODALS'], num_classes=cfg['DATASET']['NUM_CLASSES'])
    palette = dataset_dummy.PALETTE.cpu().numpy()

    for name in common_file:
        label_path = os.path.join(labels_dir, name+".tif")
        label_img = Image.open(label_path).convert("L")  
        label_array = np.array(label_img)
        label_tensor = torch.from_numpy(label_array).long()
        label_mapped = dataset_dummy.label_mapping[label_tensor].numpy()

        color_label = np.zeros((label_mapped.shape[0], label_mapped.shape[1], 3), dtype=np.uint8)
        
        for cls_id, color in enumerate(palette):
            mask = (label_mapped == cls_id)
            color_label[mask] = color

        color_label_img = Image.fromarray(color_label)
        color_label_img.save(os.path.join(predictions_dir, f"{name}_3label.tif"))
        
        # Save image files for both modalities
        shutil.copy(os.path.join(incrociati_dir, name+".tif"),
                    os.path.join(predictions_dir, f"{name}_1incrociati.tif"))
        shutil.copy(os.path.join(paralleli_dir, name+".tif"),
                    os.path.join(predictions_dir, f"{name}_2paralleli.tif"))

    print("output completed!")





def main(cfg):
    device = torch.device(cfg['DEVICE'])
    eval_cfg = cfg['EVAL']
    normalization = resolve_normalization(cfg, eval_cfg['MODEL_PATH'])
    print(f"Input normalization: {normalization}")
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'], normalization=normalization)
    cases = [None] 
    
    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists(): 
        raise FileNotFoundError(f"EVAL.MODEL_PATH not found: {model_path}")
    print(f"Evaluating {model_path}...")

    exp_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    eval_path = os.path.join(os.path.dirname(eval_cfg['MODEL_PATH']), 'eval_{}.txt'.format(exp_time))

    for case in cases:
        #dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform, cfg['DATASET']['MODALS'], case)
   
        dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], cfg['EVAL'].get('SPLIT', 'test'), transform, cfg['DATASET']['MODALS'], case, num_classes=cfg['DATASET']['NUM_CLASSES'])

        model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes, cfg['DATASET']['MODALS'])
        msg = model.load_state_dict(load_state_dict(model_path))
        print(msg)
        model = model.to(device)
        sampler_val = None
        dataloader = DataLoader(dataset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=eval_cfg['BATCH_SIZE'], pin_memory=False, sampler=sampler_val)
        acc, macc, f1, mf1, ious, miou, _ = evaluate(model, dataloader, device, cfg=cfg)

        table = {
            'Class': list(dataset.CLASSES) + ['Mean'],
            'IoU': ious + [miou],
            'F1': f1 + [mf1],
            'Acc': acc + [macc]
        }
        print("mIoU : {}".format(miou))
        print("Results saved in {}".format(eval_path))

        with open(eval_path, 'a+') as f:
            f.writelines(eval_cfg['MODEL_PATH'])
            f.write(f"\nDataset: {cfg['DATASET']['ROOT']} | split: {cfg['EVAL'].get('SPLIT', 'test')} | normalization: {normalization}")
            f.write("\n============== Eval on {} {} images =================\n".format(case, len(dataset)))
            f.write("\n")
            print(tabulate(table, headers='keys'), file=f)
    
    if cfg is not None and cfg['EVAL']['VIS_SAVE_DIR'] and cfg['EVAL']['SAVE_PREDICTIONS']:
        complete_output_dir()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Training config of the run to evaluate')
    parser.add_argument('--root', type=str, default=None, help='Overrides DATASET.ROOT')
    parser.add_argument('--split', type=str, default=None, choices=['train', 'val', 'test', 'all'],
                        help="Split to evaluate (default: test); 'all' evaluates every patch of DATASET.ROOT")
    parser.add_argument('--model-path', type=str, default=None, help='Overrides EVAL.MODEL_PATH')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Run folder: evaluates its best checkpoint (overrides EVAL.MODEL_PATH)')
    parser.add_argument('--vis-dir', type=str, default=None, help='Overrides EVAL.VIS_SAVE_DIR / CONFUSION_DIR prefix')
    args = parser.parse_args()

    with open(args.cfg, encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # Command-line overrides are written into cfg, so every function sees the same values
    if args.root:
        cfg['DATASET']['ROOT'] = args.root
    if args.split:
        cfg['EVAL']['SPLIT'] = args.split
    if args.model_path:
        cfg['EVAL']['MODEL_PATH'] = args.model_path
    if args.run_dir:
        best = find_best_checkpoint(args.run_dir)
        if best is None:
            parser.error(f'No checkpoint with a .meta.json sidecar in {args.run_dir}')
        cfg['EVAL']['MODEL_PATH'] = best
    if args.vis_dir:
        cfg['EVAL']['VIS_SAVE_DIR'] = args.vis_dir
        cfg['EVAL']['CONFUSION_DIR'] = args.vis_dir + '_confusion'

    setup_cudnn()
    main(cfg)



