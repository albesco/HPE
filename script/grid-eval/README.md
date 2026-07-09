# Tutorial: grid-evaluation locale HPE

Questa cartella contiene gli script parametrici per scegliere configurazioni promettenti su validation, senza usare il test set per selezionare iperparametri.

Script disponibili:

```text
script/grid-eval/grid_eval_Yolo26x-Detection.py
script/grid-eval/grid_eval_Yolo26x-Pose.py
script/grid-eval/grid_eval_VitPose.py
script/grid-eval/grid_eval_common.py
```

## 1. Scopo

La grid-evaluation serve a confrontare poche configurazioni controllate, di default 4 run da 5 epoche:

- YOLO26x-Detection: detector one-class `swimmer` per fornire bbox alla pipeline YOLO26x-Detection -> ViTPose++.
- YOLO26x-Pose: modello end-to-end che rileva il nuotatore e predice keypoint COCO17.
- ViTPose++ huge: modello top-down valutato nelle condizioni migliori, cioe con bbox GT in train e validation.

Regola generale:

- train e validation sono usati per la scelta iperparametri;
- test e solo opzionale come metadato, non come criterio di scelta;
- ogni run salva comando, argomenti, log, status, metriche, checkpoint best e last/latest;
- non vengono salvati checkpoint intermedi epoch-by-epoch quando il framework lo consente.

## 2. Ambiente e installazione

Gli script assumono il repository nella root:

```bash
cd /home/albertosco/HPE
```

Ambiente Conda previsto:

```bash
conda activate vitpose
```

Controlli rapidi:

```bash
python --version
yolo --help
python -c "import torch, mmcv, mmpose, ultralytics"
```

Dipendenze attese:

- `tmux` per lanciare run lunghe persistenti;
- Ultralytics YOLO per gli script YOLO;
- MMPose/MMCV/PyTorch per ViTPose++;
- `PyYAML` se si vogliono leggere griglie da file YAML.

Se `PyYAML` non e disponibile, installarlo nell'ambiente corretto:

```bash
conda activate vitpose
pip install pyyaml
```

Nota GPU:

- per run lunghe usare sempre `tmux`;
- se `nvidia-smi` non funziona sul server, verificare CUDA da PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 3. Interfaccia comune

Tutti e tre gli script espongono la stessa interfaccia generale:

| Parametro | Default | Descrizione |
|---|---:|---|
| `--train-data` | richiesto salvo `--list-grid` | split train: images dir per YOLO, COCO JSON per ViTPose++ |
| `--val-data` | richiesto salvo `--list-grid` | split validation: images dir per YOLO, COCO JSON per ViTPose++ |
| `--test-data` | opzionale | registrato come metadato; non usato per scegliere iperparametri |
| `--epochs` | `5` | epoche per configurazione |
| `--patience` | `5` | patience nativa dove supportata; in ViTPose++ e registrata per coerenza |
| `--metric` | `mAP50-95` | metrica primaria/fallback, mappata su AP keypoint o bbox secondo lo script |
| `--output-dir` | script-specifico | root delle grid-evaluation |
| `--experiment-name` | richiesto | nome esperimento sotto `--output-dir` |
| `--resume` | off | riprende da last/latest della stessa configurazione se presente |
| `--dry-run` | off | crea directory/config/report senza training |
| `--device` | `auto` | se specificato, passa device/GPU al framework |
| `--workers` | framework default | worker dataloader; se omesso resta default/reference |
| `--seed` | framework default | seed opzionale |
| `--overwrite` | off | permette rilancio di config completate |
| `--max-runs` | tutte | limita il numero di configurazioni eseguite |
| `--list-grid` | off | stampa la griglia e termina |
| `--grid-param` | nessuno | griglia da CLI, ripetibile |
| `--grid-json` | nessuno | griglia da JSON |
| `--grid-yaml` | nessuno | griglia da YAML |

Regola sulle griglie:

- usare una sola modalita tra default, `--grid-param`, `--grid-json`, `--grid-yaml`;
- se si passano piu modalita insieme, lo script termina con errore chiaro.

## 4. Griglie default

### YOLO26x-Detection

```text
cfg_01_lr0_0.00067_imgsz_640
cfg_02_lr0_0.00100_imgsz_640
cfg_03_lr0_0.00067_imgsz_768
cfg_04_lr0_0.00100_imgsz_768
```

### YOLO26x-Pose

```text
cfg_01_lr0_0.00067_imgsz_640
cfg_02_lr0_0.00100_imgsz_640
cfg_03_lr0_0.00067_imgsz_768
cfg_04_lr0_0.00100_imgsz_768
```

### ViTPose++ huge

```text
cfg_01_lr_0.00067_crop_384x128
cfg_02_lr_0.00100_crop_384x128
cfg_03_lr_0.00067_crop_512x128
cfg_04_lr_0.00100_crop_512x128
```

Per ViTPose++ la crop e sempre scritta come `width x height`. Quindi `384x128` significa:

```text
MMPose data_cfg.image_size = [384, 128]
ViT backbone img_size      = (128, 384)
heatmap_size               = [96, 32]
```

## 5. Formati griglia JSON/YAML

JSON:

```json
{
  "grid": {
    "lr0": [0.00067, 0.001],
    "imgsz": [640, 768]
  }
}
```

YAML:

```yaml
grid:
  lr0: [0.00067, 0.001]
  imgsz: [640, 768]
```

Per ViTPose++:

```yaml
grid:
  lr: [0.00067, 0.001]
  crop_size: [384x128, 512x128]
```

Esempio CLI equivalente per YOLO:

```bash
--grid-param lr0=0.00067,0.001 --grid-param imgsz=640,768
```

Esempio CLI equivalente per ViTPose++:

```bash
--grid-param lr=0.00067,0.001 --grid-param crop_size=384x128,512x128
```

## 6. Output prodotti

Ogni esperimento scrive sotto:

```text
runs/grid-eval/<model>/<experiment-name>/
```

Esempio:

```text
runs/grid-eval/yolo26x_detection/yolo26x_det_grid2x2/
  cfg_01_lr0_0.00067_imgsz_640/
    command.txt
    args.json
    status.json
    stdout_stderr.log        # solo dopo training reale
    weights/best.pt          # YOLO dopo training reale
    weights/last.pt          # YOLO dopo training reale
  summary.csv
  summary.json
  report.md
  best_config.json
  best_config_command.txt
  grid_effective.json
```

Per ViTPose++ ogni config contiene anche:

```text
effective_config.py
training_status.txt          # durante training reale
best_AP_epoch_*.pth          # dopo training reale
latest.pth                   # dopo training reale
```

## 7. Stato run e resume

Ogni configurazione ha un file:

```text
status.json
```

Stati principali:

| Stato | Significato |
|---|---|
| `pending` | config preparata o dry-run |
| `running` | training in corso |
| `completed` | training completato con exit code 0 |
| `failed` | training terminato con errore |

Comportamento:

- se una config e `completed`, lo script la salta;
- con `--overwrite`, la config puo essere rilanciata;
- con `--resume`, se esiste `weights/last.pt` per YOLO o `latest.pth` per ViTPose++, lo script prova a riprendere la stessa config.

## 8. YOLO26x-Detection

### Dati richiesti

Passare le directory immagini degli split YOLO detection:

```text
data/intermediate/SUW_frames/_Yolo26x_detection/images/train
data/intermediate/SUW_frames/_Yolo26x_detection/images/val
data/intermediate/SUW_frames/_Yolo26x_detection/images/test
```

Lo script inferisce le label corrispondenti da:

```text
data/intermediate/SUW_frames/_Yolo26x_detection/labels/<split>
```

Formato label atteso: 5 campi per riga:

```text
class x_center y_center width height
```

### Dry-run

```bash
python script/grid-eval/grid_eval_Yolo26x-Detection.py \
  --train-data data/intermediate/SUW_frames/_Yolo26x_detection/images/train \
  --val-data data/intermediate/SUW_frames/_Yolo26x_detection/images/val \
  --test-data data/intermediate/SUW_frames/_Yolo26x_detection/images/test \
  --output-dir runs/grid-eval/yolo26x_detection \
  --experiment-name yolo26x_det_grid2x2 \
  --dry-run
```

### Training in tmux

```bash
tmux new-session -d -s grid_yolo26x_det \
  'cd /home/albertosco/HPE && python script/grid-eval/grid_eval_Yolo26x-Detection.py \
    --train-data data/intermediate/SUW_frames/_Yolo26x_detection/images/train \
    --val-data data/intermediate/SUW_frames/_Yolo26x_detection/images/val \
    --test-data data/intermediate/SUW_frames/_Yolo26x_detection/images/test \
    --output-dir runs/grid-eval/yolo26x_detection \
    --experiment-name yolo26x_det_grid2x2'
```

### Default e scelta best

Default principali:

| Parametro | Default |
|---|---:|
| `--epochs` | `5` |
| `--patience` | `5` |
| modello | `models/detection/yolo26x.pt` |
| checkpoint policy | `save_period=-1`, best + last |
| split selezione | `val` |

Criterio di scelta:

1. massima recall bbox validation;
2. AP75 o IoU medio se disponibili;
3. mAP50-95 bbox;
4. preferire `imgsz=640` a parita.

AP75, IoU medio e keypoint coverage restano `null` se non calcolati da un evaluator separato.

## 9. YOLO26x-Pose

### Dati richiesti

Passare le directory immagini degli split YOLO pose:

```text
data/intermediate/SUW_frames/_Yolo26x_pose/images/train
data/intermediate/SUW_frames/_Yolo26x_pose/images/val
data/intermediate/SUW_frames/_Yolo26x_pose/images/test
```

Lo script inferisce le label corrispondenti da:

```text
data/intermediate/SUW_frames/_Yolo26x_pose/labels/<split>
```

Formato label atteso: 56 campi per riga:

```text
class bbox(4) 17*(x y visibility)
```

### Dry-run

```bash
python script/grid-eval/grid_eval_Yolo26x-Pose.py \
  --train-data data/intermediate/SUW_frames/_Yolo26x_pose/images/train \
  --val-data data/intermediate/SUW_frames/_Yolo26x_pose/images/val \
  --test-data data/intermediate/SUW_frames/_Yolo26x_pose/images/test \
  --output-dir runs/grid-eval/yolo26x_pose \
  --experiment-name yolo26x_pose_grid2x2 \
  --dry-run
```

### Training in tmux

```bash
tmux new-session -d -s grid_yolo26x_pose \
  'cd /home/albertosco/HPE && python script/grid-eval/grid_eval_Yolo26x-Pose.py \
    --train-data data/intermediate/SUW_frames/_Yolo26x_pose/images/train \
    --val-data data/intermediate/SUW_frames/_Yolo26x_pose/images/val \
    --test-data data/intermediate/SUW_frames/_Yolo26x_pose/images/test \
    --output-dir runs/grid-eval/yolo26x_pose \
    --experiment-name yolo26x_pose_grid2x2'
```

### Default e scelta best

Default principali:

| Parametro | Default |
|---|---:|
| `--epochs` | `5` |
| `--patience` | `5` |
| modello | `models/pose/yolo26x-pose.pt` |
| batch | `1` |
| optimizer | `AdamW` |
| checkpoint policy | `save_period=-1`, best + last |
| split selezione | `val` |

Nota optimizer:

- il progetto documenta che `optimizer=auto` puo ignorare o sovrascrivere `lr0`;
- lo script imposta `optimizer=AdamW`, coerente con la grid storica YOLO26x-Pose, per rendere `lr0` effettivo;
- augmentation, weight decay, scheduler, loss pose-specific, kobj, RLE loss e NMS restano default del framework/modello.

Criterio di scelta:

1. massima Keypoint AP / mAP50-95 su validation;
2. AP75 se disponibile;
3. AR se disponibile;
4. preferire `imgsz=640` a parita.

AP75, AR, errore medio per keypoint e metriche distali restano `null` se non calcolate da un evaluator separato.

## 10. ViTPose++ huge

### Dati richiesti

Passare i JSON COCO degli split VitPose++:

```text
data/intermediate/SUW_frames/_VitPosePP/annotations/person_keypoints_train.json
data/intermediate/SUW_frames/_VitPosePP/annotations/person_keypoints_val.json
data/intermediate/SUW_frames/_VitPosePP/annotations/person_keypoints_test.json
```

Lo script inferisce le immagini da:

```text
data/intermediate/SUW_frames/_train_canonical/train2017
data/intermediate/SUW_frames/_train_canonical/val2017
data/intermediate/SUW_frames/_train_canonical/test2017
```

Se si usa una struttura diversa, passare esplicitamente:

```bash
--train-images PATH --val-images PATH --test-images PATH
```

### Bbox GT

ViTPose++ usa bbox GT top-down in train e validation:

```python
use_gt_bbox=True
bbox_file=''
```

Le bbox sono quelle presenti nelle annotazioni COCO. Non vengono usate bbox predette dal detector.

### Dry-run

```bash
python script/grid-eval/grid_eval_VitPose.py \
  --train-data data/intermediate/SUW_frames/_VitPosePP/annotations/person_keypoints_train.json \
  --val-data data/intermediate/SUW_frames/_VitPosePP/annotations/person_keypoints_val.json \
  --test-data data/intermediate/SUW_frames/_VitPosePP/annotations/person_keypoints_test.json \
  --output-dir runs/grid-eval/vitpose \
  --experiment-name vitpose_grid2x2 \
  --dry-run
```

### Training in tmux

```bash
tmux new-session -d -s grid_vitpose \
  'cd /home/albertosco/HPE && python script/grid-eval/grid_eval_VitPose.py \
    --train-data data/intermediate/SUW_frames/_VitPosePP/annotations/person_keypoints_train.json \
    --val-data data/intermediate/SUW_frames/_VitPosePP/annotations/person_keypoints_val.json \
    --test-data data/intermediate/SUW_frames/_VitPosePP/annotations/person_keypoints_test.json \
    --output-dir runs/grid-eval/vitpose \
    --experiment-name vitpose_grid2x2 \
    --device 0'
```

### Default e scelta best

Default principali:

| Parametro | Default |
|---|---:|
| `--epochs` | `5` |
| `--patience` | `5` registrato, non early-stop attivo |
| checkpoint pretrained | `models/pose/wholebody.pth` |
| base config | inferita da `_VitPosePP/generated_configs/swimxyz_vitposepp_huge.py` |
| workers | `2` se non specificato |
| checkpoint policy | `max_keep_ckpts=1`, `save_best='AP'`, `latest.pth` |
| split selezione | `val` |

Criterio di scelta:

1. massima Keypoint AP@[OKS 0.50:0.95];
2. maggiore AP75;
3. minore errore medio per keypoint se disponibile;
4. preferire crop `384x128` a parita.

## 11. Monitoraggio tmux

Elencare sessioni:

```bash
tmux ls
```

Entrare nella sessione:

```bash
tmux attach -t grid_yolo26x_det
```

Staccarsi lasciando il job attivo:

```text
Ctrl-b poi d
```

Seguire log stdout/stderr di una config:

```bash
tail -n 80 runs/grid-eval/yolo26x_detection/yolo26x_det_grid2x2/cfg_01_lr0_0.00067_imgsz_640/stdout_stderr.log
```

Controllare status:

```bash
cat runs/grid-eval/yolo26x_detection/yolo26x_det_grid2x2/cfg_01_lr0_0.00067_imgsz_640/status.json
```

Controllare summary mentre la grid procede:

```bash
column -s, -t < runs/grid-eval/yolo26x_detection/yolo26x_det_grid2x2/summary.csv | less -S
```

Fermare una sessione:

```bash
tmux kill-session -t grid_yolo26x_det
```

## 12. Comandi di verifica rapida

Stampare la griglia senza dataset:

```bash
python script/grid-eval/grid_eval_Yolo26x-Detection.py --experiment-name check --list-grid
python script/grid-eval/grid_eval_Yolo26x-Pose.py --experiment-name check --list-grid
python script/grid-eval/grid_eval_VitPose.py --experiment-name check --list-grid
```

Limitare a una sola configurazione per smoke test reale:

```bash
python script/grid-eval/grid_eval_Yolo26x-Detection.py \
  --train-data data/intermediate/SUW_frames/_Yolo26x_detection/images/train \
  --val-data data/intermediate/SUW_frames/_Yolo26x_detection/images/val \
  --output-dir runs/grid-eval/yolo26x_detection \
  --experiment-name smoke_one_cfg \
  --max-runs 1
```

Riprendere una grid interrotta:

```bash
python script/grid-eval/grid_eval_VitPose.py \
  --train-data data/intermediate/SUW_frames/_VitPosePP/annotations/person_keypoints_train.json \
  --val-data data/intermediate/SUW_frames/_VitPosePP/annotations/person_keypoints_val.json \
  --output-dir runs/grid-eval/vitpose \
  --experiment-name vitpose_grid2x2 \
  --resume
```

## 13. Note metodologiche

- Non scegliere iperparametri usando il test set.
- Non confrontare questi risultati come valutazione finale tra modelli HPE.
- Dopo aver scelto la configurazione promettente, lanciare training/evaluation finale come esperimento separato.
- Se una metrica non e prodotta dal framework, lasciarla vuota/null nel report: non inventare valori.
- Per run pesanti su SSH, usare sempre `tmux`.
