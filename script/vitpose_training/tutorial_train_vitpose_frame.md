# Tutorial: train_vitpose_frame.sh

Questo tutorial descrive il launcher unico per training, report, Test e overlay VitPose++ su dataset frame gia preparati.
Il train usa bbox GT; il Test finale passa invece da YOLO26x-Detection e usa il checkpoint detector e `imgsz` indicati nel comando.

Script principale:

```bash
script/vitpose_training/train_vitpose_frame.sh
```

## Cosa fa

Lo script esegue l'intero flusso VitPose++:

1. riceve la root del dataset, per esempio `data/intermediate/SAW_frames`;
2. trova automaticamente `DATASET/_train_canonical` e `DATASET/_VitPosePP`;
3. genera `runs/<RUN_NAME>/effective_config.py` con path e iperparametri effettivi;
4. avvia il training MMPose/VitPose++ con `conda run -n vitpose`;
5. monitora early stopping e retention checkpoint;
6. esporta CSV/plot di validazione;
7. esegue il Test finale tramite la pipeline YOLO26x-Detection -> VitPose++;
8. scrive `kp_Test.json`, `metrics_Test.json`, `summary_Test.json` e overlay.

Il Test usa di default lo stesso dataset del training. Se serve testare su un altro dataset, bisogna passare esplicitamente `--test-dataset-dir`.

## Struttura attesa del dataset

`--dataset-dir` deve puntare alla root del dataset, non a `_train_canonical`.

Esempio corretto:

```text
data/intermediate/SAW_frames/
  _train_canonical/
    annotations/
    train2017/
    val2017/
    test2017/
  _VitPosePP/
    generated_configs/
      swimxyz_vitposepp_huge.py
```

Esempio errato:

```bash
--dataset-dir data/intermediate/SAW_frames/_train_canonical
```

Lo script rifiuta quel path per evitare mismatch silenziosi tra train e test.

## Comando minimo

```bash
bash script/vitpose_training/train_vitpose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames \
  --pretrained-checkpoint models/pose/wholebody.pth \
  --max-epochs 10 \
  --run-name vitpose_SAW_frames_test
```

## Esempio con Test esplicito sullo stesso dataset

```bash
bash script/vitpose_training/train_vitpose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames \
  --test-dataset-dir data/intermediate/SAW_frames \
  --pretrained-checkpoint models/pose/wholebody.pth \
  --max-epochs 25 \
  --run-name vitpose_SAW_frames_25ep
```

## Esempio train su A e Test su B

```bash
bash script/vitpose_training/train_vitpose_frame.sh \
  --dataset-dir data/intermediate/Side_above_water_EntireSwim_A \
  --test-dataset-dir data/intermediate/Side_above_water_EntireSwim_B \
  --pretrained-checkpoint models/pose/wholebody.pth \
  --max-epochs 30 \
  --run-name vitpose_A_testB
```

In questo caso il dataset di Test diverso e visibile nel comando, quindi non puo capitare il mismatch nascosto corretto nella sessione `Verifica-Overlay`.

## Esempio Test con YOLO26x-Detection esplicito

```bash
bash script/vitpose_training/train_vitpose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames \
  --pretrained-checkpoint models/pose/wholebody.pth \
  --max-epochs 25 \
  --run-name vitpose_SAW_frames_yolo_test \
  --yolo-detector-checkpoint runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt \
  --yolo-imgsz 768 \
  --yolo-conf 0.25
```

## Esempio solo training, senza Test finale

```bash
bash script/vitpose_training/train_vitpose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames \
  --pretrained-checkpoint models/pose/wholebody.pth \
  --max-epochs 5 \
  --run-name vitpose_SAW_frames_train_only \
  --run-test no
```

## Parametri obbligatori

| Parametro | Default | Descrizione |
|---|---|---|
| `--dataset-dir PATH` | nessuno | Root del dataset. Deve contenere `_train_canonical/` e `_VitPosePP/`. |
| `--pretrained-checkpoint PATH` | nessuno | Checkpoint iniziale usato come `load_from`. |
| `--max-epochs N` | nessuno | Numero totale massimo di epoche. |

## Parametri principali

| Parametro | Default | Descrizione |
|---|---|---|
| `--test-dataset-dir PATH` | uguale a `--dataset-dir` | Root dataset usata per il Test finale. |
| `--base-config PATH` | `DATASET/_VitPosePP/generated_configs/swimxyz_vitposepp_huge.py` | Config base VitPose++ da estendere. |
| `--run-name NAME` | timestamp UTC `YYYYMMDD_HHMM` | Nome run e root degli output. |
| `--start-epoch N` | `1` | Metadato registrato nella config effettiva. |
| `--lr FLOAT` | `0.00100` | Learning rate. |
| `--crop-size WxH` | `384x128` | Crop MMPose in formato larghezza x altezza. |
| `--early-stop-metric NAME` | `AP` | Metrica monitorata. Alias supportati: `AP`, `mAP`, `mAP50-95`. |
| `--early-stop-patience N` | `3` | Epoche senza miglioramento prima dello stop. |
| `--early-stop-min-delta FLOAT` | `0.007` | Miglioramento minimo per aggiornare il best. |
| `--keep-last-n-checkpoints N` | `10` | Numero massimo di checkpoint periodici da tenere. |
| `--device VALUE` | `auto` | `auto`, `cpu`, `cuda:0`, ecc. |
| `--batch-size N` | `4` | Override batch size per config e dataloader. |
| `--num-workers N` | `2` | Override worker per dataloader. |
| `--run-test yes\|no` | `yes` | Abilita il Test finale. |
| `--render-overlays yes\|no` | `yes` | Genera overlay dal `kp_Test.json`. |
| `--overwrite` | off | Permette di riusare cartelle/file output esistenti. |


## Lancio con tmux

Per run lunghe su SSH, usa `tmux` in modo che il training continui anche se la connessione cade.

### Avvio sessione

```bash
tmux new-session -d -s vitpose_saw_frames_10ep \
  "cd /home/albertosco/HPE && bash script/vitpose_training/train_vitpose_frame.sh \
    --dataset-dir data/intermediate/SAW_frames \
    --pretrained-checkpoint models/pose/wholebody.pth \
    --max-epochs 10 \
    --run-name vitpose_SAW_frames_10ep \
    2>&1 | tee logs/vitpose_SAW_frames_10ep.log"
```

### Gestione sessione

| Azione | Comando |
|---|---|
| Elencare sessioni | `tmux ls` |
| Entrare nella sessione | `tmux attach -t vitpose_saw_frames_10ep` |
| Uscire lasciando il job attivo | `Ctrl-b`, poi `d` |
| Vedere il log senza entrare | `tail -n 80 logs/vitpose_SAW_frames_10ep.log` |
| Vedere lo stato training | `tail -n 40 runs/vitpose_SAW_frames_10ep/training_status.txt` |
| Fermare la sessione tmux | `tmux kill-session -t vitpose_saw_frames_10ep` |

### Output attesi dopo il lancio tmux

| Output | Percorso esempio |
|---|---|
| Log top-level | `logs/vitpose_SAW_frames_10ep.log` |
| Config effettiva | `runs/vitpose_SAW_frames_10ep/effective_config.py` |
| Checkpoint | `runs/vitpose_SAW_frames_10ep/checkpoint/` |
| Report Val | `runs/vitpose_SAW_frames_10ep/reports/` |
| JSON Test | `data/output/experiments/vitpose_SAW_frames_10ep/kp_Test.json` |
| Metriche Test | `data/output/experiments/vitpose_SAW_frames_10ep/metrics_Test.json` |
| Summary Test | `data/output/experiments/vitpose_SAW_frames_10ep/summary_Test.json` |
| Overlay Test | `data/output/experiments/vitpose_SAW_frames_10ep/overlays_Test/` |

### Riprendere il controllo dopo disconnessione

```bash
tmux ls
tmux attach -t vitpose_saw_frames_10ep
```

Se la sessione non esiste piu, controlla prima i file prodotti:

```bash
ls -lh runs/vitpose_SAW_frames_10ep/checkpoint
ls -lh data/output/experiments/vitpose_SAW_frames_10ep
```

## Output prodotti

Con `--run-name vitpose_SAW_frames_test`, lo script produce:

```text
runs/vitpose_SAW_frames_test/
  checkpoint/
  reports/
  effective_config.py
  training_status.txt
  early_stop_status.json
  train_stdout.log

data/output/experiments/vitpose_SAW_frames_test/
  result_keypoints.json
  kp_Test.json
  metrics_Test.json
  overlays_Test/
```

## Script nel modulo

La cartella `script/vitpose_training/` contiene tutti i pezzi usati dal launcher:

| File | Uso |
|---|---|
| `train_vitpose_frame.sh` | Launcher unico parametrico. |
| `monitor_vitpose_patience.py` | Early stopping e retention checkpoint. |
| `export_vitpose_val_metrics.py` | Esporta metriche Val da log JSON MMPose. |
| `plot_vitpose_training_log.py` | Genera plot di loss/AP. |
| `vitpose_generate_test_overlays_from_json.py` | Render overlay dal JSON predizioni. |
| `pose_overlay_utils.py` | Helper per overlay e dataset info. |

## Note operative

- Per confronto con YOLO26x-Pose su `SAW_frames`, usare `--dataset-dir data/intermediate/SAW_frames` e lasciare il default del Test, oppure specificare un checkpoint detector con `--yolo-detector-checkpoint`.
- Usare `--test-dataset-dir` solo quando il Test deve essere intenzionalmente diverso dal training.
- Se vuoi allineare il detector a un esperimento specifico, imposta anche `--yolo-imgsz` e `--yolo-conf` in modo esplicito nel comando.
- Se una run esiste gia, scegliere un nuovo `--run-name` oppure passare `--overwrite`.
- Il file `effective_config.py` e la riga iniziale dello script stampano i path effettivi usati per train e test.
