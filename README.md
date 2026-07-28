**MMSFormer - Segmentazione delle malte**

Questo progetto estende il modello MMSFormer per affrontare il problema della segmentazione delle malte, con immagini acquisite in due modalità: paralleli e incrociati.

Per poter addestrare il modello, come viene spiegato nel README originale di MMSFormer presente nella directory del progetto, è necessario scaricare i pesi preaddestrati del backbone SegFormere (disponibili al link: https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5) e inserirli nella seguente struttura:

```
checkpoints/pretrained/segformer
├── mit_b0.pth
├── mit_b1.pth
├── mit_b2.pth
├── mit_b3.pth
├── mit_b4.pth
└── mit_b5.pth
```

Nel file di configurazione mortars.yaml si sceglie quale variante del backbone utilizzare e il relativo file di pesi:

```
BACKBONE : MMSFormer-B3                   # variante del modello
PRETRAINED: 'checkpoints/pretrained/segformer/mit_b3.pth'   # pesi del backbone
```

---

## Modifiche principali rispetto al modello originale

1. **Dataset personalizzato**

   * È stato creato un nuovo file `dataset.py` nella cartella `semseg/dataset`.
   * Permette il caricamento del dataset delle malte con le modalità: paralleli e incrociati.

2. **Gestione della distribuzione non omogenea delle classi**

   * Possibilità di usare **Weighted Random Sampler** durante il training.

     * I pesi vengono calcolati **automaticamente** in base alla frequenza delle classi.
     * Attivabile nel file di configurazione con:

       ```
       WEIGHTED_RANDOM_SAMPLER: true
       ```
   * In alternativa, è possibile assegnare pesi personalizzati alle classi nel calcolo della loss.

     * Attivabile con:

       ```
       CLS_WEIGHTS: true
       CLASS_WEIGHTS: [w1, w2, w3]  
       ```

       I pesi delle classi sono da impostare manualmente tenendo conto che le classi rappresentano rispettivamente:

       * Legante: 70.46%
       * Porosità: 3.62%
       * Aggregati: 25.92%

   * Infine, è possibile addestrare il modello focalizzandosi su legante e aggregati senza considerare la porosità impostando nel file di configurazione:

       ```
       NUM_CLASSES : 2  
       ```
     
3. **Funzionalità di testing**

   * Generazione di un file con **mappe delle predizioni** sul dataset di test.
   * Salvataggio delle immagini originali, label associate alle predizioni in un’unica cartella.
   * Possibilità di generare la **confusion matrix**.
   * Attivabili tramite configurazione:

     ```
     SAVE_PREDICTIONS: true
     SAVE_CONFUSION: true
     ```
   * Cartelle di salvataggio:

     ```
     VIS_SAVE_DIR: 'data/mortars_results'
     CONFUSION_DIR: 'data/confusion_matrix'
     ```

---

## Comandi principali

### Training

Per allenare il modello con la configurazione `mortars.yaml`:

```
python -m tools.train_mm --cfg configs/mortars.yaml
```

* Il modello verrà salvato in:

```
output/MMSFormer/MMSF-MORTARS-CONFIG
```

* Nome del modello:

```
MMSFormer_MMSFormer-B3_MORTARS_epochXX_XX.XX.pth
```

### Testing

Per testare il modello su un dataset di test:

```
python -m tools.val_mm --cfg configs/mortars.yaml
```

* Assicurarsi di aver aggiornato nel file `mortars.yaml` il percorso del modello salvato nella sezione `EVAL`:

```
MODEL_PATH: 'output/MMSFormer/MMSF-MORTARS-CONFIG/MMSFormer_MMSFormer-B3_MORTARS_epochXX_XX.XX.pth'
```

### Inference

Per fare inferenza con il modello su nuovi dati:

```
python -m tools.infer_mm --cfg configs/mortars.yaml
```

* Assicurarsi di aver aggiornato nel file `mortars.yaml` il percorso del modello salvato, nella sezione `TEST` :

```
MODEL_PATH: 'output/MMSFormer/MMSF-MORTARS-CONFIG/MMSFormer_MMSFormer-B3_MORTARS_epochXX_XX.XX.pth'
```

### Modelli addestrati

Sono presenti dei modelli addestrati per la segmentazione delle malte al link: https://drive.google.com/drive/folders/1RFcC_c-t8dWocq6t473JKHmrPFLEwPcz?usp=sharing .
C'è un modello addestrato utilizzando tre classi ( porosità, legante, aggregati ) e due modelli addestrati sul dataset al quale è stata tolta la classe porosità.

---

## Esempio di configurazione (`mortars.yaml`)

```yaml
DEVICE          : cuda                        
SAVE_DIR        : 'output/MMSFormer'          
#GPUs            : 2
#GPU_IDs         : [0, 1]
GPUs            : 1
GPU_IDs         : [0]
USE_WANDB       : false
WANDB_NAME      : 'MMSF-MORTARS-CONFIG'           

MODEL:
  NAME          : MMSFormer                                         # name of the model you are using
  BACKBONE      : MMSFormer-B3                                      # model variant
  PRETRAINED    : 'checkpoints/pretrained/segformer/mit_b3.pth'     # backbone model's weight 
  RESUME        : ''                                                # checkpoint file

DATASET:
  NAME          : MORTARS                                           # dataset name to be trained with (camvid, cityscapes, ade20k)
  ROOT          : '/mortars_dataset'        # dataset root path
  IGNORE_LABEL  : 3
  # MODALS        : ['img']
  MODALS        : ['paralleli', 'incrociati'] 
  NUM_CLASSES   : 3                                              

TRAIN:
  IMAGE_SIZE    : [512, 512]                                         # training image size in (h, w) 
  BATCH_SIZE    : 4                                                  # batch size used to train
  EPOCHS        : 1                                                  # number of epochs to train
  EVAL_START    : 0                                                  # evaluation interval during training
  EVAL_INTERVAL : 1                                                  # evaluation interval during training
  AMP           : true                                               # use AMP in training
  DDP           : false                                              # use DDP training
  WEIGHTED_RANDOM_SAMPLER : false                      

LOSS:
  NAME          : OhemCrossEntropy                                   # loss function name
  CLS_WEIGHTS   : false                                              # use class weights in loss calculation
  CLASS_WEIGHTS : [1.0, 8.0, 2.5] 

OPTIMIZER:
  NAME          : adamw                                              # optimizer name
  LR            : 0.0001                                             # initial learning rate used in optimizer
  WEIGHT_DECAY  : 0.1                                                # decay rate used in optimizer 

SCHEDULER:
  NAME          : warmuppolylr                                       # scheduler name
  POWER         : 0.9                                                # scheduler power
  WARMUP        : 10                                                 # warmup epochs used in scheduler
  WARMUP_RATIO  : 0.1                                                # warmup ratio
  

EVAL:
  MODEL_PATH    : 'output/MMSFormer/MMSF-MORTARS-CONFIG/MMSFormer_MMSFormer-B3_MORTARS_epoch100_77.81.pth'         # Path to your saved model
  IMAGE_SIZE    : [512, 512]                                         # evaluation image size in (h, w)                       
  BATCH_SIZE    : 1                                                  # batch size
  SAVE_PREDICTIONS : true
  ERROR_THRESHOLD: 0.25
  SAVE_CONFUSION: True
  VIS_SAVE_DIR  : 'data/mortars_results'                             # Where to save visualization
  CONFUSION_DIR : 'data/confusion_matrix'
  MSF:  
    ENABLE      : false                                              # multi-scale and flip evaluation  
    FLIP        : true                                               # use flip in evaluation  
    SCALES      : [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]                  # scales used in MSF evaluation                


TEST:
  MODEL_PATH    : 'output/MMSFormer/MMSF-MORTARS-CONFIG/MMSFormer_MMSFormer-B3_MORTARS_epoch100_77.81.pth'         # Path to your saved model
  IMAGE_SIZE    : [512, 512]                                               
  VIS_SAVE_DIR  : 'infer_results'                                    # Where to save visualization
  FILE          : '/home/leonardonotari/elabIVA/infer_dataset'
  OVERLAY       :  true

```


