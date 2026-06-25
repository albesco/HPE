# Tutorial: train_vitpose_frame.sh

Questo tutorial spiega come usare `script/yolo_training/train_vitpose_frame.sh` per avviare un training VitPose++ su un dataset frame+label già preparato.

## 1. Cosa fa lo script

Lo script:

1. valida il dataset in input (`train2017`, `val2017`, `test2017`, annotazioni);
2. usa un checkpoint iniziale fornito con `--pretrained-checkpoint`;
3. genera una configurazione efficace in `runs/<RUN_NAME>/effective_config.py`;
4. avvia il training con `conda run -n vitpose`;
5. monitora l’early stopping;
6. esporta metriche di validazione;
7. esegue il test sul split `test` e produce overlay e keypoints.

È pensato per il flusso SAW / SwimXYZ e si basa sul template di `script/train_vitpose_SAW_frames_EntireSwim_20260612.sh`.

---

## 2. Prerequisiti

Prima di partire, assicurati di avere:

- un ambiente Conda chiamato `vitpose`;
- un dataset canonico con struttura simile a:

```text
data/intermediate/SAW_frames/_train_canonical/
  annotations/
  train2017/
  val2017/
  test2017/
```

- un checkpoint iniziale, ad esempio:

```text
models/pose/vitpose_huge.pth
```

Puoi verificare la struttura attuale con:

```bash
ls -1 data/intermediate/SAW_frames/_train_canonical
ls -1 models/pose
```

---

## 3. Esempio minimo

Dal root del repository:

```bash
bash script/yolo_training/train_vitpose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames/_train_canonical \
  --pretrained-checkpoint models/pose/vitpose_huge.pth \
  --max-epochs 5
```

Questo è il comando più semplice per fare un test rapido.

### Cosa succede qui

- `--dataset-dir` indica la root del dataset canonico;
- `--pretrained-checkpoint` dice da dove partire;
- `--max-epochs 5` limita il training a 5 epoche, utile per un test iniziale.

---

## 4. Parametri principali e default

Lo script supporta i seguenti argomenti (verificati dal help reale dello script).

### Parametri obbligatori

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--dataset-dir PATH` | nessuno | Cartella canonica del dataset con `train2017/`, `val2017/`, `test2017/` e `annotations/`. |
| `--pretrained-checkpoint PATH` | nessuno | Checkpoint iniziale usato come `load_from`. |
| `--max-epochs N` | nessuno | Numero totale di epoche di training. |

### Parametri opzionali

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--run-name NAME` | `$(date -u +%Y%m%d_%H%M)` | Nome della run. Viene usato come `runs/<RUN_NAME>/...`. |
| `--start-epoch N` | `1` | Epoca iniziale registrata nella metainformazione di configurazione. |
| `--lr FLOAT` | `0.00100` | Learning rate. |
| `--crop-size WxH` | `384x128` | Dimensione immagine usata per il modello. La dimensione dell’heatmap è `crop/4`. |
| `--early-stop-metric NAME` | `AP` | Metrica usata per l’early stopping. Attualmente supportato `AP` (o alias `mAP`, `mAP50-95`, `AP@[.50:.95]`). |
| `--early-stop-patience N` | `3` | Quante epoche senza miglioramento aspettare prima di fermare. |
| `--early-stop-min-delta FLOAT` | `0.007` | Miglioramento minimo richiesto per considerare una epoca “migliore”. |
| `--keep-last-n-checkpoints N` | `10` | Numero massimo di checkpoint da mantenere. |
| `--checkpoint-dir PATH` | `runs/<RUN_NAME>/checkpoint` | Cartella dei checkpoint. |
| `--reports-dir PATH` | `runs/<RUN_NAME>/reports` | Cartella per metriche e plot. |
| `--val-metrics-csv PATH` | `runs/<RUN_NAME>/reports/val_metrics_by_epoch.csv` | CSV delle metriche di validazione. |
| `--test-output-dir PATH` | `data/output/experiments/<RUN_NAME>` | Cartella per il test finale. |
| `--test-kp-json NAME_OR_PATH` | `kp_Test.json` | File dei keypoint predetti sul test set. |
| `--test-metrics-json NAME_OR_PATH` | `metrics_Test.json` | File JSON delle metriche di test. |
| `--test-overlay-dir PATH` | `data/output/experiments/<RUN_NAME>/overlays_Test` | Cartella con gli overlay di test. |
| `--device VALUE` | `auto` | `auto`, `cpu`, oppure un valore GPU come `cuda:0`. |
| `--batch-size N` | nessuno | Override del batch size usato nella configurazione. |
| `--num-workers N` | nessuno | Override del numero di worker per `data` e `test_dataloader`. |
| `--overwrite` | disattivato | Permette di riusare cartelle o file già esistenti. |

---

## 5. Come leggere i default reali

I default nel file sono dichiarati all’inizio di `train_vitpose_frame.sh`:

```bash
RUN_NAME="$(date -u +%Y%m%d_%H%M)"
START_EPOCH=1
LR="0.00100"
CROP_SIZE="384x128"
EARLY_STOP_METRIC_RAW="AP"
EARLY_STOP_PATIENCE=3
EARLY_STOP_MIN_DELTA="0.007"
KEEP_LAST_N_CHECKPOINTS=10
DEVICE="auto"
```

Quindi, se non specifichi nulla, lo script usa:

- una run con nome tipo `20260615_1430`;
- LR pari a `0.00100`;
- crop `384x128`;
- early stopping su `AP` con patience `3` e min-delta `0.007`;
- 10 checkpoint da mantenere;
- device automatico (`cuda:0` se disponibile, altrimenti `cpu`).

---

## 6. Esempi pratici

### Esempio A — run di base su dataset SAW

```bash
bash script/yolo_training/train_vitpose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames/_train_canonical \
  --pretrained-checkpoint models/pose/vitpose_huge.pth \
  --max-epochs 10 \
  --run-name saw_vitpose_baseline
```

Quando usarlo:

- vuoi una prima prova semplice;
- vuoi vedere se il dataset è compatibile;
- vuoi avere un benchmark veloce.

### Esempio B — training più conservativo

```bash
bash script/yolo_training/train_vitpose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames/_train_canonical \
  --pretrained-checkpoint models/pose/vitpose_huge.pth \
  --max-epochs 25 \
  --run-name saw_vitpose_cautious \
  --lr 0.0005 \
  --early-stop-patience 5 \
  --early-stop-min-delta 0.01
```

Quando usarlo:

- vuoi un learning rate più piccolo;
- vuoi una stop condition meno aggressiva;
- vuoi ridurre il rischio di overfitting.

### Esempio C — crop più grande e più checkpoint mantenuti

```bash
bash script/yolo_training/train_vitpose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames/_train_canonical \
  --pretrained-checkpoint models/pose/vitpose_huge.pth \
  --max-epochs 30 \
  --run-name saw_vitpose_largecrop \
  --crop-size 512x256 \
  --keep-last-n-checkpoints 20
```

Quando usarlo:

- vuoi più risoluzione di input;
- vuoi conservare più checkpoint intermedi;
- vuoi confrontare una configurazione più “ricca”.

### Esempio D — forzare batch e worker

```bash
bash script/yolo_training/train_vitpose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames/_train_canonical \
  --pretrained-checkpoint models/pose/vitpose_huge.pth \
  --max-epochs 15 \
  --run-name saw_vitpose_batch2 \
  --batch-size 2 \
  --num-workers 4
```

Quando usarlo:

- hai memoria sufficiente per batch più grandi;
- vuoi ridurre il tempo di training;
- vuoi controllare meglio il caricamento dati.

### Esempio E — riutilizzare una cartella già esistente

```bash
bash script/yolo_training/train_vitpose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames/_train_canonical \
  --pretrained-checkpoint models/pose/vitpose_huge.pth \
  --max-epochs 12 \
  --run-name saw_vitpose_retry \
  --overwrite
```

Usa `--overwrite` solo quando vuoi ricalcolare o sovrascrivere output già presenti.

---

## 7. Dove finiscono i risultati

Per ogni run, lo script crea una struttura del tipo:

```text
runs/<RUN_NAME>/
  checkpoint/
  reports/
  training_status.txt
  early_stop_status.json
  effective_config.py

data/output/experiments/<RUN_NAME>/
  kp_Test.json
  metrics_Test.json
  overlays_Test/
```

In pratica:

- `checkpoint/` contiene i checkpoint migliori/ultimi;
- `reports/` contiene metriche e grafici;
- `data/output/experiments/<RUN_NAME>/` contiene output di test e overlay.

---

## 8. Suggerimenti pratici

1. Prima di una run lunga, fai una prova con `--max-epochs 5` o `10`.
2. Se l’output è instabile, abbassa `--lr` e aumenta `--early-stop-patience`.
3. Se hai poca memoria, riduci `--batch-size` e `--crop-size`.
4. Se vuoi mantenere più checkpoint, aumenta `--keep-last-n-checkpoints`.
5. Usa `--overwrite` solo quando sai di voler riscrivere i risultati.

---

## 9. Risoluzione rapida dei problemi

### Errore: `--dataset-dir is required`

Hai dimenticato di passare la cartella del dataset.

### Errore: `Missing pretrained checkpoint`

Il percorso di `--pretrained-checkpoint` non esiste oppure è errato.

### Errore: `Checkpoint dir already exists and is not empty`

La cartella di output è già piena. Usa `--overwrite` oppure scegli un altro `--run-name`.

### Errore: `Unsupported early-stop metric`

Usa solo valori supportati come `AP`, `mAP`, `mAP50-95`, `AP@[.50:.95]`.

---

## 10. Comando consigliato per iniziare

Se vuoi partire in modo semplice e affidabile, usa questo:

```bash
bash script/yolo_training/train_vitpose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames/_train_canonical \
  --pretrained-checkpoint models/pose/vitpose_huge.pth \
  --max-epochs 10 \
  --run-name saw_vitpose_first_run
```

Questo ti dà un punto di partenza realistico per capire se il dataset, il checkpoint e la configurazione funzionano bene.
