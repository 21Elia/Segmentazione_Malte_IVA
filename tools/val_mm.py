import torch
import argparse
import yaml
import math
import os
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
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from math import ceil
import numpy as np
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from sklearn.metrics import confusion_matrix
import cv2





def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img




@torch.no_grad()
def sliding_predict(model, image, num_classes, flip=True):
    image_size = image[0].shape
    tile_size = (int(ceil(image_size[2]*1)), int(ceil(image_size[3]*1)))
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = torch.zeros((num_classes, image_size[2], image_size[3]), device=torch.device('cuda'))
    count_predictions = torch.zeros((image_size[2], image_size[3]), device=torch.device('cuda'))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = [modal[:, :, y_min:y_max, x_min:x_max] for modal in image]
            padded_img = [pad_image(modal, tile_size) for modal in img]
            tile_counter += 1
            padded_prediction = model(padded_img)
            if flip:
                fliped_img = [padded_modal.flip(-1) for padded_modal in padded_img]
                fliped_predictions = model(fliped_img)
                padded_prediction += fliped_predictions.flip(-1)
            predictions = padded_prediction[:, :, :img[0].shape[2], :img[0].shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.squeeze(0)

    return total_predictions.unsqueeze(0)





@torch.no_grad()
def save_confusion_matrix(): #crea immagine confusion matrix
    cm_file = Path(cfg['EVAL']['CONFUSION_DIR'] + "/confusion_matrix.npy")
    cm = np.load(cm_file)
    
    if cfg['DATASET']['NUM_CLASSES'] == 3:
        class_names = ["Legante", "Porosità", "Aggregati"] # se si considera la porosità
    else:
        class_names = ["Legante", "Aggregati"] 

    save_path= cfg['EVAL']['CONFUSION_DIR'] + "/confusion_matrix.png"

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.nan_to_num(cm)  # evita divisioni per zero

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)

    # Scrivi i valori all'interno delle celle
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
    sliding = False
    test_loss = 0.0
    iter = 0
    for images, labels, fname in tqdm(dataloader):
        images = [x.to(device) for x in images]
        labels = labels.to(device)
        if sliding:
            # preds = sliding_predict(model, images, num_classes=n_classes).softmax(dim=1)
            preds = sliding_predict(model, images, num_classes=n_classes)
        else:
            # preds = model(images).softmax(dim=1)
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
        if cfg is not None and cfg['EVAL']['VIS_SAVE_DIR'] and cfg['EVAL']['SAVE_PREDICTIONS']:#per salvare i risultati
            mask_valid = labels != dataloader.dataset.ignore_label
            wrong = (pred_labels != labels) & mask_valid
            error_rate = wrong.sum().float() / mask_valid.sum().float()

            # Per calcolare la differenza assoluta di pixel classificati come aggregati tra predizione e label
            num_ones_pred = ((pred_labels == 1) & mask_valid).sum().item()
            num_ones_label = ((labels == 1) & mask_valid).sum().item()
            diff = num_ones_label - num_ones_pred
            # normalizzata
            if diff<0:
                diff = (-1)*diff
            diff = diff/(512*512)

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





@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = [F.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=True) for img in images]
            scaled_images = [scaled_img.to(device) for scaled_img in scaled_images]
            logits = model(scaled_images)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = [torch.flip(scaled_img, dims=(3,)) for scaled_img in scaled_images]
                logits = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)
    
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou





@torch.no_grad() #salva le mappe in output
def save_pred_map(pred, save_path, palette):
    """
    pred: tensor shape (H, W) con valori [0..num_classes]
    palette: lista di tuple RGB per ogni classe
    """
    pred_np = pred.cpu().numpy().astype(np.uint8)
    color_map = np.zeros((pred_np.shape[0], pred_np.shape[1], 3), dtype=np.uint8)

    for cls_id, color in enumerate(palette):
        color_map[pred_np == cls_id] = color

    cv2.imwrite(save_path, cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))





@torch.no_grad()
def complete_output_dir(): #associa ogni predizione alle relative immagini e label
    predictions_dir = cfg['EVAL']['VIS_SAVE_DIR']
    labels_dir = cfg['DATASET']['ROOT'] + "/label"
    incrociati_dir = cfg['DATASET']['ROOT'] + "/incrociati"
    paralleli_dir = cfg['DATASET']['ROOT'] + "/paralleli"

    est = (".png", ".tif")

    # FUNZIONA CON I NOMI ASSEGNATI DAL MODELLO 
    #predictions = {f[:-16] for f in os.listdir(predictions_dir) if f.lower().endswith(est)} 
    #labels = {f[:-4] for f in os.listdir(labels_dir) if f.lower().endswith(est)}
    #incrociati = {f[:-4] for f in os.listdir(incrociati_dir) if f.lower().endswith(est)}
    #paralleli = {f[:-4] for f in os.listdir(paralleli_dir) if f.lower().endswith(est)}
    predictions = {f.split("_4prediction_")[0] for f in os.listdir(predictions_dir) if f.lower().endswith(est)} 
    labels = {f.split(".")[0] for f in os.listdir(labels_dir) if f.lower().endswith(est)}
    incrociati = {f.split(".")[0] for f in os.listdir(incrociati_dir) if f.lower().endswith(est)}
    paralleli = {f.split(".")[0] for f in os.listdir(paralleli_dir) if f.lower().endswith(est)}
    common_file = predictions & labels & incrociati & paralleli
    print(f"Found {len(common_file)} complete set.")

    # Mappa colori
    color_map = {
        0: (0, 0, 0),   #legante nero
        1: (255, 0, 0), #porosità rosso
        2: (0, 255, 0), #aggregati verdi
        3: (0, 0, 255), #ignore blu
    }

    for name in common_file:
        #label
        label_path = os.path.join(labels_dir, name+".tif")
        label_img = Image.open(label_path).convert("L")  
        label_array = np.array(label_img)
        if cfg['DATASET']['NUM_CLASSES'] == 2:
            mapping = np.array([0, 0, 2, 3])  #rimappo la porosità
            label_array = mapping[label_array]


        color_label = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)
        
        for val, color in color_map.items():
            mask = label_array == val
            color_label[mask, 0] = color[0]
            color_label[mask, 1] = color[1]
            color_label[mask, 2] = color[2]

        color_label_img = Image.fromarray(color_label)
        color_label_img.save(os.path.join(predictions_dir, f"{name}_3label.tif"))
        
        # immagini nelle due modalità
        shutil.copy(os.path.join(incrociati_dir, name+".tif"),
                    os.path.join(predictions_dir, f"{name}_1incrociati.tif"))
        shutil.copy(os.path.join(paralleli_dir, name+".tif"),
                    os.path.join(predictions_dir, f"{name}_2paralleli.tif"))

    print("output completed!")





def main(cfg):
    device = torch.device(cfg['DEVICE'])
    eval_cfg = cfg['EVAL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    cases = [None] 
    
    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists(): 
        raise FileNotFoundError
    print(f"Evaluating {model_path}...")

    exp_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    eval_path = os.path.join(os.path.dirname(eval_cfg['MODEL_PATH']), 'eval_{}.txt'.format(exp_time))

    for case in cases:
        #dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform, cfg['DATASET']['MODALS'], case)
   
        dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'test', transform, cfg['DATASET']['MODALS'], case, num_classes=cfg['DATASET']['NUM_CLASSES'])

        model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes, cfg['DATASET']['MODALS'])
        msg = model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        print(msg)
        model = model.to(device)
        sampler_val = None
        dataloader = DataLoader(dataset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=eval_cfg['BATCH_SIZE'], pin_memory=False, sampler=sampler_val)
        if True:
            if eval_cfg['MSF']['ENABLE']:
                acc, macc, f1, mf1, ious, miou = evaluate_msf(model, dataloader, device, eval_cfg['MSF']['SCALES'], eval_cfg['MSF']['FLIP'])
            else:
                acc, macc, f1, mf1, ious, miou, _ = evaluate(model, dataloader, device, cfg=cfg)

            table = {
                'Class': list(dataset.CLASSES) + ['Mean'],
                'IoU': ious + [miou],
                'F1': f1 + [mf1],
                'Acc': acc + [macc]
            }
            print("mIoU : {}".format(miou))
            print("Results saved in {}".format(eval_cfg['MODEL_PATH']))

        with open(eval_path, 'a+') as f:
            f.writelines(eval_cfg['MODEL_PATH'])
            f.write("\n============== Eval on {} {} images =================\n".format(case, len(dataset)))
            f.write("\n")
            print(tabulate(table, headers='keys'), file=f)
    
    if cfg is not None and cfg['EVAL']['VIS_SAVE_DIR'] and cfg['EVAL']['SAVE_PREDICTIONS']:
        complete_output_dir()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/mcubes_rgbadn.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    # gpu = setup_ddp()
    # main(cfg, gpu)
    main(cfg)



