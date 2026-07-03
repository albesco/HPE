# Tutorial: YOLO26x-Detection training

Questo tutorial descrive il flusso completo per addestrare un detector YOLO26x one-class (`swimmer`) usando il launcher parametrico corrente:

```bash
script/yolo26x_detection_training/train_yolo26x_detection_frame.sh
```

Lo script base storico e archiviato in:

```bash
script_old/yolo26x_Detection-Training/train_yolo_side_above_water.sh
```

Il nuovo launcher non riscrive quel workflow: ne riusa logica, convenzioni, comando `yolo detect train`, pruning dei checkpoint e monitor esterno per l'early stopping con `min_delta`.

## 1. Quando usare questo launcher

Usa `train_yolo26x_detection_frame.sh` quando vuoi:

- passare il dataset da CLI invece di dipendere da variabili d'ambiente;
- scegliere checkpoint iniziale, run name, learning rate e image size;
- applicare early stopping su validation `mAP50-95` con `min_delta`;
- esportare un CSV Val per epoca e i PNG di loss/mAP;
- valutare automaticamente il best checkpoint su Test;
- salvare bbox predette e overlay finali nel layout standard del repo;
- lanciare la run dentro `tmux` senza costruire il comando a mano.

## 2. Installazione e prerequisiti

### Ambiente Conda

Il workflow assume l'ambiente del repo:

```bash
conda activate vitpose
```

Controlli rapidi consigliati:

```bash
yolo --help
python -c "import ultralytics, cv2, numpy, matplotlib"
```

Se uno di questi controlli fallisce, l'ambiente non e allineato al workflow.

### Dataset richiesto

Il launcher si aspetta una root dataset che possa risolvere uno di questi casi:

1. una directory `_Yolo26x_detection/`;
2. una root dataset che contiene `_Yolo26x_detection/`;
3. una root `_train_canonical/` con sibling `_Yolo26x_detection/`.

Il file YAML risolto deve essere:

```text
swimxyz_side_above_water_yolo26x_detection.yaml
```

Esempio reale:

```text
data/intermediate/SUW_frames
```

oppure:

```text
data/intermediate/SUW_frames/_Yolo26x_detection
```

### Checkpoint iniziale richiesto

Serve un checkpoint iniziale esplicito, per esempio:

```text
models/detection/yolo26x.pt
```

Lo script fallisce subito se il file non esiste.

## 3. Cosa fa il launcher

Il flusso operativo e questo:

1. risolve dataset root, YAML e split `images/` / `labels/`;
2. verifica presenza di script base e checkpoint iniziale;
3. risolve tutte le directory di output;
4. stampa i parametri risolti all'inizio della run;
5. lancia `yolo detect train` nell'ambiente `vitpose`;
6. applica early stopping esterno su `metrics/mAP50-95(B)` con `patience` + `min_delta`;
7. mantiene `best.pt`, `last.pt` e solo gli ultimi `N` checkpoint `epoch*.pt`;
8. esporta un CSV Val per epoca;
9. genera PNG di validation loss e validation mAP;
10. usa il best checkpoint per la valutazione su Test;
11. salva bbox predette, metriche e overlay bbox sullo split Test.

## 4. Parametri supportati

### Parametri principali

| Parametro CLI | Default | Significato |
|---|---|---|
| `--dataset-dir` | obbligatorio | root dataset da cui risolvere `_Yolo26x_detection` |
| `--pretrained-checkpoint` | `models/detection/yolo26x.pt` | checkpoint iniziale |
| `--run-name` | timestamp UTC `YYYYMMDD_HHMM` | nome della run |
| `--start-epoch` | `1` | epoca logica iniziale della nuova fase |
| `--max-epochs` | `100` | epoca logica finale desiderata |
| `--lr` | `0.00100` | learning rate iniziale |
| `--imgsz` | `768` | image size YOLO |
| `--device` | `0` | device CUDA o `cpu` |
| `--batch-size` | `2` | batch size |
| `--num-workers` | `2` | dataloader workers |

### Early stopping

| Parametro CLI | Default | Significato |
|---|---|---|
| `--early-stop-metric` | `mAP50-95/AP@[.50:.95]` | etichetta utente della metrica di arresto |
| `--early-stop-patience` | `3` | numero di epoche senza miglioramento valido prima dello stop |
| `--early-stop-min-delta` | `0.007` | miglioramento minimo richiesto |
| `--keep-last-n-checkpoints` | `10` | quanti checkpoint `epoch*.pt` conservare |

Nota importante:

- la metrica realmente letta dal monitor e `metrics/mAP50-95(B)` da `results.csv`;
- un miglioramento e valido solo se `current_metric >= best_metric + min_delta`.

### Output e report

| Parametro CLI | Default | Significato |
|---|---|---|
| `--checkpoint-dir` | `runs/RUN_NAME/checkpoint/` | copia dei checkpoint finali della run |
| `--reports-dir` | `runs/RUN_NAME/reports/` | directory report |
| `--val-metrics-csv` | `runs/RUN_NAME/reports/val_metrics_by_epoch.csv` | CSV validation-by-epoch |
| `--test-output-dir` | `data/output/experiments/RUN_NAME/` | root output Test |
| `--test-bbox-json` | `bbox_Test.json` | JSON bbox predette su Test |
| `--test-metrics-json` | `metrics_Test.json` | JSON metriche Test |
| `--test-overlay-dir` | `data/output/experiments/RUN_NAME/overlays_Test/` | overlay bbox su Test |
| `--overlay-max-images` | `0` | `0` = renderizza tutte le immagini Test |
| `--conf` | `0.25` | confidenza minima per le predizioni Test |

### Controllo esecuzione

| Parametro CLI | Default | Significato |
|---|---|---|
| `--use-tmux` | `no` | se `yes`, crea una sessione tmux e termina subito |
| `--evaluate-test` | `yes` | se `no`, salta il blocco finale Test |
| `--overwrite` | off | permette di cancellare output run/Test gia esistenti |

## 5. Come vengono interpretati `START_EPOCH` e `MAX_EPOCHS`

Questo punto e importante.

Il launcher avvia una nuova sessione di training da `PRETRAINED_CHECKPOINT`, non fa resume dello stato ottimizzatore di una run Ultralytics precedente.

Quindi:

- `--start-epoch` e `--max-epochs` definiscono l'intervallo logico della fase;
- il numero di epoche effettivamente lanciate e:

```text
TRAIN_EPOCHS = MAX_EPOCHS - START_EPOCH + 1
```

- il CSV Val esportato usa `START_EPOCH` per numerare le epoche nel report finale.

Esempio:

- `--start-epoch 6 --max-epochs 20`
- YOLO allena `15` epoche reali
- il report CSV numerera le epoche come `6..20`.

## 6. Output prodotti

### Run directory

Per una run `RUN_NAME=my_run`:

```text
runs/my_run/
  args.yaml
  results.csv
  training_status.txt
  monitor_status.json
  weights/
  checkpoint/
  reports/
  logs/
```

### CSV validation

Il file:

```text
runs/RUN_NAME/reports/val_metrics_by_epoch.csv
```

contiene almeno queste colonne:

```text
epoch, AP, AP50, AP75, AP_M, AP_L, AR, AR50, AR75, AR_M, AR_L, val_loss
```

Nota:

- `AP` viene popolato da `metrics/mAP50-95(B)`;
- `AP50` viene popolato da `metrics/mAP50(B)`;
- le colonne `AP75`, `AP_M`, `AP_L`, `AR`, `AR50`, `AR75`, `AR_M`, `AR_L` restano vuote se non disponibili nel `results.csv` standard di Ultralytics detection;
- `val_loss` e la somma delle componenti di validation loss disponibili (`box`, `cls`, `dfl`).

### PNG prodotti

In `REPORTS_DIR` vengono generati:

```text
loss_validation_by_epoch.png
map50_95_validation_by_epoch.png
```

### Output Test

A fine training, usando `weights/best.pt`, lo script salva:

```text
TEST_OUTPUT_DIR/TEST_BBOX_JSON
TEST_OUTPUT_DIR/TEST_METRICS_JSON
TEST_OVERLAY_DIR/
```

Le bbox Test seguono la convenzione del progetto:

- una sola bbox per frame;
- bbox selezionata per confidenza massima;
- a parita di confidenza, area `xyxy` maggiore.

## 7. Esempi pratici

### A. Esecuzione minima locale

```bash
bash script/yolo26x_detection_training/train_yolo26x_detection_frame.sh \
  --dataset-dir data/intermediate/SUW_frames \
  --pretrained-checkpoint models/detection/yolo26x.pt
```

### B. Run esplicita con i default del nuovo workflow

```bash
bash script/yolo26x_detection_training/train_yolo26x_detection_frame.sh \
  --dataset-dir data/intermediate/SUW_frames \
  --pretrained-checkpoint models/detection/yolo26x.pt \
  --run-name yolo26x-detection_SUW_frames_20260702 \
  --start-epoch 1 \
  --max-epochs 100 \
  --lr 0.00100 \
  --imgsz 768 \
  --early-stop-patience 3 \
  --early-stop-min-delta 0.007 \
  --keep-last-n-checkpoints 10
```

### C. Fase incrementale numerata come continuazione logica

```bash
bash script/yolo26x_detection_training/train_yolo26x_detection_frame.sh \
  --dataset-dir data/intermediate/SUW_frames \
  --pretrained-checkpoint runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt \
  --run-name yolo26x-detection_incremental_phase2 \
  --start-epoch 6 \
  --max-epochs 20 \
  --lr 0.00067 \
  --imgsz 768
```

### D. Esecuzione senza valutazione Test finale

```bash
bash script/yolo26x_detection_training/train_yolo26x_detection_frame.sh \
  --dataset-dir data/intermediate/SUW_frames \
  --pretrained-checkpoint models/detection/yolo26x.pt \
  --evaluate-test no
```

### E. Sovrascrittura controllata di una run esistente

```bash
bash script/yolo26x_detection_training/train_yolo26x_detection_frame.sh \
  --dataset-dir data/intermediate/SUW_frames \
  --pretrained-checkpoint models/detection/yolo26x.pt \
  --run-name yolo26x-detection_SUW_frames_retry \
  --overwrite
```

## 8. Lancio con tmux

Il launcher puo creare tmux da solo.

### Avvio

```bash
bash script/yolo26x_detection_training/train_yolo26x_detection_frame.sh \
  --dataset-dir data/intermediate/SUW_frames \
  --pretrained-checkpoint models/detection/yolo26x.pt \
  --run-name yolo26x-detection_SUW_frames_tmux \
  --use-tmux yes
```

La sessione creata ha nome:

```text
train_yolo26x_detection_<RUN_NAME>
```

Esempio:

```text
train_yolo26x_detection_yolo26x-detection_SUW_frames_tmux
```

### Attach

```bash
tmux attach -t train_yolo26x_detection_yolo26x-detection_SUW_frames_tmux
```

### Detach

```text
Ctrl-b poi d
```

### Stop

```bash
tmux kill-session -t train_yolo26x_detection_yolo26x-detection_SUW_frames_tmux
```

### Monitoraggio rapido

```bash
cat runs/yolo26x-detection_SUW_frames_tmux/training_status.txt
cat runs/yolo26x-detection_SUW_frames_tmux/monitor_status.json
tail -n 80 runs/yolo26x-detection_SUW_frames_tmux/logs/train.log
```

## 9. Relazione con lo script base

Il launcher parametrico riusa queste convenzioni dello script base:

- ambiente `conda run -n vitpose`;
- comando `yolo detect train`;
- `save=True` e `save_period=1`;
- `plots=True`;
- status file di run;
- pruning dei checkpoint `epoch*.pt`;
- layout standard dei log e dei pesi;
- uso di `best.pt` come checkpoint finale per la valutazione.

Differenze principali:

- il nuovo launcher usa CLI invece di variabili d'ambiente;
- il resume automatico non viene usato;
- l'early stopping non si appoggia al `patience` nativo Ultralytics, ma al monitor esterno con `min_delta`;
- il nuovo launcher esporta report Val e output Test strutturati.

## 10. Errori comuni e criticita

### Metriche tutte a zero

Se `precision`, `recall`, `mAP` e `val_loss` sono tutti `0`, quasi sempre YOLO non sta leggendo le label.

Controlla:

- che `images/{train,val,test}` e `labels/{train,val,test}` siano allineati;
- che non ci siano symlink di directory che fanno risolvere le immagini fuori da `_Yolo26x_detection`;
- che eventuali cache stale (`*.cache`) siano state rimosse prima del rilancio.

### Output gia esistenti

Se la run directory o `TEST_OUTPUT_DIR` esistono gia, il launcher si ferma.

Per sovrascrivere in modo esplicito:

```bash
--overwrite
```

### GPU OOM

Se la memoria GPU non basta:

1. riduci `--batch-size`;
2. poi riduci `--imgsz`;
3. se necessario usa `--device cpu` solo per debug o smoke test.

### AP75 / AR mancanti nel CSV

Non e un bug del tutorial.

Per la detection Ultralytics standard queste colonne non sono fornite nel `results.csv`, quindi vengono lasciate vuote nel report Val.

## 11. Comandi utili di validazione

### Help del launcher

```bash
bash script/yolo26x_detection_training/train_yolo26x_detection_frame.sh --help
```

### Validazione shell

```bash
bash -n script/yolo26x_detection_training/train_yolo26x_detection_frame.sh
```

### Validazione helper Python

```bash
conda run -n vitpose python -m py_compile \
  script/yolo26x_detection_training/monitor_yolo_detection_patience.py \
  script/yolo26x_detection_training/export_yolo_detection_training_report.py \
  script/yolo26x_detection_training/evaluate_yolo_detection_split.py
```

## 12. Script collegati

Training:

```text
script/yolo26x_detection_training/train_yolo26x_detection_frame.sh
script_old/yolo26x_Detection-Training/train_yolo_side_above_water.sh
script/yolo26x_detection_training/monitor_yolo_detection_patience.py
script/yolo26x_detection_training/prune_yolo_epoch_checkpoints.py
script/yolo26x_detection_training/export_yolo_detection_training_report.py
script/yolo26x_detection_training/evaluate_yolo_detection_split.py
```

Prediction / visualizzazione:

```text
script/yolo26x_detection_prediction/render_yolo_detection_overlays.py
script/yolo26x_detection_prediction/export_yolo_detection_top1_labels.py
script/yolo26x_detection_prediction/yolo_detection_utils.py
```

Grid search storico:

```text
script_old/yolo26x_Detection-Training/yolo26x_detector_grid2x2.py
```
