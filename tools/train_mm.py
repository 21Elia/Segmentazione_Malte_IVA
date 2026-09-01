import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch 
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler, WeightedRandomSampler
from torch import distributed as dist
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation, get_train_augmentation_exp1, get_train_augmentation_exp2, get_train_augmentation_exp3
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from tools.val_mm import evaluate
import wandb
from semseg.metrics import Metrics
import gc
import json




def compute_sample_weights(dataset, ignore_label):
    #Calcola pesi per WeightedRandomSampler (livello immagine) robusto a classi molto rare.

    max_class_weight=10.0 #massimo peso per classe per evitare pesi enormi
    eps=1e-6 # per evitare divisioni per zero
    verbose=True 

    n_classes = dataset.n_classes
    class_pixel_counts = torch.zeros(n_classes, dtype=torch.float) # tensore di zeri per contare i pixel di ogni classe in tutto il dataset
    sample_classes = [] # lista per salvare quali classi compaiono in ciascun campione

    iterator = dataset
    if verbose:
        iterator = tqdm(dataset, desc="Computing sample weights")

    for _, lbl, _ in iterator:
        classes = torch.unique(lbl) # trova i valori di classe (0, 1, 2, 3) unici presenti nella mappa
        if ignore_label is not None:
            classes = classes[classes != ignore_label] # rimuove la classe ignore_label ovvero la classe 3 che rappresenta il padding o zone prive di informazione

        sample_classes.append(classes)

        for c in classes:
            class_pixel_counts[c] += (lbl == c).sum() # conta quanti pixel appartengono a ciascuna classe

    # calcola il peso di ogni classe invertendo la frequenza dei pixel (sotto sqrt per ammorbidire il peso)
    # => classi rare avranno pesi alti, classi frequenti pesi bassi
    class_weights = torch.sqrt(1.0 / (class_pixel_counts + eps))

    class_weights = torch.clamp(class_weights, max=max_class_weight) # clampa i pesi al valore massimo 10

    # assegna a ciascuna immagine il peso della classe più rara presente in essa
    # => classi rare verranno estratte con maggiore frequenza dal data loader
    sample_weights = []
    for classes in sample_classes: 
        if len(classes) == 0:
            sample_weights.append(1.0)  
        else:
            w = class_weights[classes].max().item()
            sample_weights.append(w)

    return sample_weights




def main(cfg, save_dir):
    # inizializzazione delle variabile leggendo lo .yaml
    start = time.time()
    best_mIoU = 0.0
    best_epoch = 0
    num_workers = 2
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    resume_path = cfg['MODEL']['RESUME']
    gpus = cfg['GPUs']
    use_wandb = cfg['USE_WANDB']
    wandb_name = cfg['WANDB_NAME']
    # gpus = int(os.environ['WORLD_SIZE'])

    # crea le pipeine di data augmentation (crop, flip, rotazioni, normalizzazione) per il training e validation
    aug_version = train_cfg.get('AUGMENTATION', 'v1')
    if aug_version in ['exp3', 'v2_soft']:
        traintransform = get_train_augmentation_exp3(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
        logger.info(f'Using augmentation pipeline: exp3 / v2_soft (soft synchronous photometric + geometric augmentation)')
    elif aug_version == 'exp2':
        traintransform = get_train_augmentation_exp2(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
        logger.info(f'Using augmentation pipeline: exp2 (synchronous photometric + geometric augmentation)')
    elif aug_version == 'exp1':
        traintransform = get_train_augmentation_exp1(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
        logger.info(f'Using augmentation pipeline: exp1 (geometric augmentation)')
    else:
        traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
        logger.info(f'Using augmentation pipeline: v1 (baseline)')
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'], aug_version=aug_version)

    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform, dataset_cfg['MODALS'], num_classes=cfg['DATASET']['NUM_CLASSES'])
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'], num_classes=cfg['DATASET']['NUM_CLASSES'])
    class_names = trainset.CLASSES

    # --- inizializzazione modello e checkpoint ---
    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes, dataset_cfg['MODALS'])
    resume_checkpoint = None
    if os.path.isfile(resume_path):
        resume_checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        state_dict = resume_checkpoint['model_state_dict']
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict)
        logger.info(msg)
    else:
        # Carica i pesi pre-trained. Determina se il file contiene solo i pesi
        # del backbone (es. mit_b3.pth) o l'intero modello (es. epoch81_79.51.pth)
        pretrained_path = model_cfg['PRETRAINED']
        if os.path.isfile(pretrained_path):
            pretrained_state = torch.load(pretrained_path, map_location=torch.device('cpu'))
            if 'state_dict' in pretrained_state:
                pretrained_state = pretrained_state['state_dict']
            if 'model' in pretrained_state:
                pretrained_state = pretrained_state['model']
            # Verifica se contiene chiavi del decode_head (= modello intero)
            has_decode_head = any('decode_head' in k for k in pretrained_state.keys())
            if has_decode_head:
                # Caricamento modello intero (fine-tuning da checkpoint completo)
                msg = model.load_state_dict(pretrained_state, strict=False)
                logger.info(f'Loaded FULL model from {pretrained_path}')
                logger.info(msg)
            else:
                # Caricamento solo backbone (pre-training da ImageNet)
                model.init_pretrained(pretrained_path)
                logger.info(f'Loaded backbone from {pretrained_path}')
            del pretrained_state
        else:
            logger.warning(f'Pretrained file not found: {pretrained_path}. Training from scratch.')
    
    model = torch.nn.DataParallel(model, device_ids=cfg['GPU_IDs'])
    model = model.to(device)
    
    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE'] # calcola quanti batch in un'epoca di addestramento

    # --- inizializza la loss function, ottimizzatore e scheduler ---
    #loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    cls_weights = None
    if loss_cfg['CLS_WEIGHTS']:
        cls_weights = torch.tensor(loss_cfg['CLASS_WEIGHTS'], dtype=torch.float).to(device)
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, cls_weights=cls_weights)

    start_epoch = 0
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, int((epochs+1)*iters_per_epoch), sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])

    # --- definisce come campionare i dati per il training---
    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        sampler_val = None
        model = DDP(model, device_ids=[gpu], output_device=0, find_unused_parameters=True)
    else:
        if train_cfg['WEIGHTED_RANDOM_SAMPLER']:
            sample_weights = compute_sample_weights(trainset, ignore_label=trainset.ignore_label)
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        else:   
            sampler = RandomSampler(trainset)
        sampler_val = None

    # Se si riprende un addestramento, ripristina lo stato dell'epoca,
    # dell'ottimizzatore e dello scheduler e libera memoria dal checkpoint
    if resume_checkpoint:
        start_epoch = resume_checkpoint['epoch'] - 1
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        loss = resume_checkpoint['loss']        
        best_mIoU = resume_checkpoint['best_miou']
        del resume_checkpoint

    # istanzia i DataLoader per il training set e validation set
    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=False, sampler=sampler)
    valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=False, sampler=sampler_val)

    scaler = GradScaler(enabled=train_cfg['AMP'])
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer = SummaryWriter(str(save_dir))
        logger.info('================== model complexity =====================')
        cal_flops(model, dataset_cfg['MODALS'], logger)
        logger.info('================== model structure =====================')
        logger.info(model)
        logger.info('================== training config =====================')
        logger.info(cfg)
        logger.info('================== parameter count =====================')
        logger.info(sum(p.numel() for p in model.parameters() if p.requires_grad))


    # --- Training Loop principale ---
    for epoch in range(start_epoch, epochs):
        # Clean Memory
        torch.cuda.empty_cache() # ad ogni epoca svuota la cache cuda
        gc.collect()             # e invoca il garbage collector

        model.train()
        if train_cfg['DDP']: sampler.set_epoch(epoch)

        train_loss = 0.0        
        lr = scheduler.get_lr()
        lr = sum(lr) / len(lr)
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")
        metrics = Metrics(trainset.n_classes, trainloader.dataset.ignore_label, device)

        for iter, (sample, lbl, _) in pbar: # ciclo sui singoli batch di immagini
            optimizer.zero_grad(set_to_none=True) # azzera i gradienti degli step precedenti
            sample = [x.to(device) for x in sample] # sposta l'input
            lbl = lbl.to(device)                    # e le maschere lbl sulla GPU

            # --- forward pass ---
            with autocast(enabled=train_cfg['AMP']):    # autocast (precisione mista float16/float32).
                logits = model(sample)                  # La rete produce i logits
                loss = loss_fn(logits, lbl)             # e viene calcola la loss

            metrics.update(logits.softmax(dim=1), lbl)  # applica la softmax ai logits e aggiorna la confusion matrix
                                                        # per calcolare le metriche del training

            # --- backward pass --- 
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN or Inf loss detected at Epoch {epoch+1}, Iter {iter+1}. Skipping backward step.")
            else:
                scaler.scale(loss).backward() # calcola i gradienti riscalati per evitare sottoflusso di memoria
                scaler.step(optimizer)        # applica l'aggiornamento dei pesi
                scaler.update()               
                train_loss += loss.item()     # aggiorna la loss accumulata del training

            scheduler.step()                  # aggiorna il learning rate a livello di iterazione
            if torch.cuda.is_available():
                torch.cuda.synchronize()      # attende che le operazioni su GPU abbiano terminato l'esecuzione

            lr = scheduler.get_lr()               
            lr = sum(lr) / len(lr)
            if lr <= 1e-8:
                lr = 1e-8 # minimum of lr


            # Clean Memory
            torch.cuda.empty_cache()        # pulisce la vram
            gc.collect()

            # aggiorna il messaggio sulla progress bar
            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")
        
       
        epoch_train_loss = train_loss / iters_per_epoch # calcola la loss media dell'epoca appena conclusa
        train_losses.append(epoch_train_loss) # la salva in train_losses e la invia a tensor board

        train_loss /= iter+1
        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            writer.add_scalar('train/loss', train_loss, epoch)

        # calcola le metriche complessive dell'epoca per il training set
        ious, miou = metrics.compute_iou() # IoU di ciascuna classe, mIoU media
        acc, macc = metrics.compute_pixel_acc() # pixel accuracy
        f1, mf1 = metrics.compute_f1() # F1-score

        # if use_wandb:
        train_log_data = {
            "Epoch": epoch+1,
            "Train Loss": train_loss,
            "Train mIoU": miou,
            "Train Pixel Acc": macc,
            "Train F1": mf1,
        }
        if use_wandb:
            wandb.log(train_log_data)

        if ((epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch+1)>train_cfg['EVAL_START']) or (epoch+1) == epochs:
            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                acc, macc, f1, mf1, ious, miou, test_loss = evaluate(model, valloader, device, loss_fn=loss_fn)
                writer.add_scalar('val/mIoU', miou, epoch)

                # if use wandb
                log_data = {
                    "Validation Loss": test_loss,
                    "Validation mIoU": miou,
                    "Validation Pixel Acc": macc,
                    "Validation F1": mf1,
                }
                log_data.update(train_log_data)
                print(log_data)
                if use_wandb:
                    wandb.log(log_data)
                # se la miou ottenuta supera la precedente best_mIoU => elimina i vecchi file di pesi salvati e aggiorna best_mIoU e best_epoch
                if miou > best_mIoU:
                    prev_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    prev_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    if os.path.isfile(prev_best): os.remove(prev_best)
                    if os.path.isfile(prev_best_ckp): os.remove(prev_best_ckp)
                    best_mIoU = miou
                    best_epoch = epoch+1
                    cur_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    cur_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    # torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), cur_best)

                    # salva su disco 2 file: 
                    # cur_best: contiene solo i pesi del modello (state_dict()), utile per fare inferenza rapida
                    # cur_best_ckp: contiene il checkpoint completo (inclusi stati dell'optimizer, scheduler ed epoca attuale),
                    # in modo tale da poter riprendere l'addestramento in futuro

                    torch.save(model.module.state_dict(), cur_best)
                    # --- 
                    torch.save({'epoch': best_epoch,
                                'model_state_dict': model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss,
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_miou': best_mIoU,
                                }, cur_best_ckp)
                    logger.info(print_iou(epoch, ious, miou, acc, macc, class_names))
                logger.info(f"Current epoch:{epoch} mIoU: {miou} Best mIoU: {best_mIoU}")

    # conclusione e pulizia 
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    # Stampa a schermo una tabella riassuntiva finale con la miglior mIoU e il tempo totale, 
    # e salva la lista di tutte le loss di training nel file train_loss.json.
    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    logger.info(tabulate(table, numalign='right'))
    with open(os.path.join(save_dir, "train_loss.json"), "w") as f:
        json.dump(train_losses, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/mcubes_rgbadn.yaml', help='Configuration file to use')
    args = parser.parse_args()

    train_losses = [] # lista globale per l'andamento della loss a ogni epoca

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()

    """legge il .yaml e crea cartelle varie, file di log ecc."""
    # gpu = setup_ddp()
    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    model = cfg['MODEL']['BACKBONE']
    # exp_name = '_'.join([cfg['DATASET']['NAME'], model, modals])
    exp_name = cfg['WANDB_NAME']
    if cfg.get('USE_WANDB', False):
        try:
            wandb.init(project="MMSF-Mortars", name=exp_name)
        except Exception as e:
            print(f"Warning: wandb.init failed ({e}). Proceeding without wandb logging.", flush=True)

    save_dir = Path(cfg['SAVE_DIR'], exp_name)
    if os.path.isfile(cfg['MODEL']['RESUME']):
        save_dir =  Path(os.path.dirname(cfg['MODEL']['RESUME']))
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(save_dir / 'train.log')
    main(cfg, save_dir)
    cleanup_ddp()