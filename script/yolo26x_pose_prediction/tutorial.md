# Tutorial: predict_yolo26x_pose_frame.sh

Questo tutorial descrive lo script parametrico per predire keypoint con YOLO26x-Pose, calcolare metriche sul Test split e generare overlay frame/KP.

Script principale:

```bash
script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh
```

## Cosa fa

Lo script:

1. riceve un checkpoint YOLO26x-Pose;
2. riceve la root principale del dataset, per esempio `data/intermediate/SAW_frames`;
3. trova automaticamente `DATASET/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml`;
4. opzionalmente crea una vista Test campionata con symlink;
5. calcola le metriche YOLO Pose sullo split richiesto;
6. esporta `kp_Test.json`, `metrics_Test.json` e `metrics_Test.csv`;
7. genera overlay con bbox e keypoint predetti.

## Struttura attesa del dataset

`--dataset-dir` deve puntare alla root principale del dataset, non a `_train_canonical` e non a `_Yolo26x_pose`.

Esempio corretto:

```text
data/intermediate/SAW_frames/
  _train_canonical/
  _Yolo26x_pose/
    swimxyz_side_above_water_yolo26x_pose.yaml
    images/
      test/
    labels/
      test/
```

Esempi errati:

```bash
--dataset-dir data/intermediate/SAW_frames/_train_canonical
--dataset-dir data/intermediate/SAW_frames/_Yolo26x_pose
--dataset-dir data/intermediate/SAW_frames/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml
```

Lo script rifiuta questi path per evitare mismatch tra dataset e split.

## Comando minimo

```bash
bash script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh \
  --checkpoint runs/yolo26x-pose_SAW_frames_20260614/checkpoint/best.pt \
  --dataset-dir data/intermediate/SAW_frames
```

Con questo comando lo script usa tutti gli elementi del Test split e scrive gli output in:

```text
data/output/experiments/yolo26x-pose_prediction_best/
```

## Parametri

| Parametro | Default | Descrizione |
|---|---|---|
| `--checkpoint PATH` | nessuno | Checkpoint YOLO26x-Pose `.pt` da usare per predizione e metriche. |
| `--dataset-dir PATH` | nessuno | Root principale del dataset. Deve contenere `_Yolo26x_pose/`. |
| `--max-test-items N` | `0` | Numero di elementi Test random da valutare. `0` significa tutto il Test split. |
| `--seed N` | opzionale; `0` quando si campiona | Seed per campionare gli elementi Test. |
| `--conf FLOAT` | `0` | Threshold confidence YOLO. `0` significa accettare tutte le predizioni candidate. |
| `--max-detections-per-image N` | `1` | Numero massimo di predizioni esportate/disegnate per immagine. `0` significa tutte. |
| `--output-dir PATH` | `data/output/experiments/yolo26x-pose_prediction_<checkpoint-name>` | Directory per `kp_Test.json`, `metrics_Test.json`, `metrics_Test.csv` e `predict.log`. |
| `--overlays-dir PATH` | `OUTPUT_DIR/overlays_Test` | Directory dove salvare gli overlay frame/KP. |
| `--imgsz N` | `768` | Image size YOLO per validazione e predizione. |
| `--batch N` | `1` | Batch size usato da `yolo val`. |
| `--device VALUE` | `0` | Device YOLO, per esempio `0`, `1` o `cpu`. |
| `--workers N` | `2` | Numero di worker dataloader. |
| `--split NAME` | `test` | Split da valutare. Normalmente lasciare `test`. |
| `--help` | off | Mostra l'help dello script. |

## Esempi d'uso

### Test completo

```bash
bash script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh \
  --checkpoint runs/yolo26x-pose_SAW_frames_20260614/checkpoint/best.pt \
  --dataset-dir data/intermediate/SAW_frames \
  --output-dir data/output/experiments/pred_yolo_saw_full \
  --overlays-dir data/output/experiments/pred_yolo_saw_full/overlays_Test
```

Questo valuta tutto il Test split e genera overlay per tutti i frame.

### Campione random riproducibile

```bash
bash script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh \
  --checkpoint runs/yolo26x-pose_SAW_frames_20260614/checkpoint/best.pt \
  --dataset-dir data/intermediate/SAW_frames \
  --max-test-items 50 \
  --seed 42 \
  --output-dir data/output/experiments/pred_yolo_saw_sample50 \
  --overlays-dir data/output/experiments/pred_yolo_saw_sample50/overlays_Test
```

Quando `--max-test-items` e maggiore di `0`, lo script crea una vista dataset sotto `--output-dir`:

```text
dataset_view_test_n50_seed42/
  images/test/
  labels/test/
  yolo_pose_sample.yaml
  sample_report.json
```

Metriche, JSON keypoint e overlay usano tutti lo stesso campione.

### Threshold confidence esplicito

```bash
bash script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh \
  --checkpoint runs/yolo26x-pose_SAW_frames_20260614/checkpoint/best.pt \
  --dataset-dir data/intermediate/SAW_frames \
  --conf 0.25 \
  --output-dir data/output/experiments/pred_yolo_saw_conf025
```

Usa questa forma quando vuoi escludere predizioni sotto una certa confidenza. Il limite `--max-detections-per-image` resta a `1`, quindi viene esportata/disegnata solo la predizione con score piu alto per frame.


### Debug con tutte le predizioni candidate

```bash
bash script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh \
  --checkpoint runs/yolo26x-pose_SAW_frames_20260614/checkpoint/best.pt \
  --dataset-dir data/intermediate/SAW_frames \
  --conf 0 \
  --max-detections-per-image 0 \
  --max-test-items 5 \
  --output-dir data/output/experiments/pred_yolo_saw_all_candidates_debug
```

Questa modalita serve solo per diagnosi. Con `--conf 0` e `--max-detections-per-image 0` gli overlay possono contenere molte bbox e risultare illeggibili.

### Esecuzione su CPU

```bash
bash script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh \
  --checkpoint runs/yolo26x-pose_SAW_frames_20260614/checkpoint/best.pt \
  --dataset-dir data/intermediate/SAW_frames \
  --max-test-items 10 \
  --device cpu \
  --output-dir data/output/experiments/pred_yolo_saw_cpu_smoke
```

Utile per smoke test piccoli quando la GPU non e disponibile.

## Output prodotti

Con `--output-dir data/output/experiments/pred_yolo_saw_sample50`, lo script produce:

```text
data/output/experiments/pred_yolo_saw_sample50/
  kp_Test.json
  metrics_Test.json
  metrics_Test.csv
  predict.log
  dataset_view_test_n50_seed42/   # solo se --max-test-items > 0
  overlays_Test/
```

`metrics_Test.json` e `metrics_Test.csv` contengono metriche Box e Pose:

```text
box_precision
box_recall
box_map50
box_map50_95
pose_precision
pose_recall
pose_map50
pose_map50_95
```

## Lancio con tmux

Lo script non crea una sessione tmux automaticamente. Per run lunghe su SSH, lancialo dentro una sessione tmux manuale.

### Avvio sessione

```bash
tmux new-session -d -s pred_yolo_saw_sample50 \
  "cd /home/albertosco/HPE && bash script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh \
    --checkpoint runs/yolo26x-pose_SAW_frames_20260614/checkpoint/best.pt \
    --dataset-dir data/intermediate/SAW_frames \
    --max-test-items 50 \
    --seed 42 \
    --conf 0 \
    --output-dir data/output/experiments/pred_yolo_saw_sample50 \
    --overlays-dir data/output/experiments/pred_yolo_saw_sample50/overlays_Test \
    2>&1 | tee logs/pred_yolo_saw_sample50.log"
```

### Gestione sessione

| Azione | Comando |
|---|---|
| Elencare sessioni | `tmux ls` |
| Entrare nella sessione | `tmux attach -t pred_yolo_saw_sample50` |
| Uscire lasciando il job attivo | `Ctrl-b`, poi `d` |
| Vedere il log senza entrare | `tail -n 80 logs/pred_yolo_saw_sample50.log` |
| Fermare la sessione tmux | `tmux kill-session -t pred_yolo_saw_sample50` |

### Riprendere dopo disconnessione

```bash
tmux ls
tmux attach -t pred_yolo_saw_sample50
```

Se la sessione e gia terminata, controlla gli output:

```bash
ls -lh data/output/experiments/pred_yolo_saw_sample50
ls -lh data/output/experiments/pred_yolo_saw_sample50/overlays_Test | head
```

## Note operative

- Usa sempre la root principale del dataset con `--dataset-dir`.
- Usa `--max-test-items` per controlli rapidi e `0` per il benchmark completo.
- Usa sempre `--seed` quando vuoi un campione riproducibile.
- `--conf 0` e il default per non filtrare per confidence; `--max-detections-per-image 1` mantiene leggibile l'output su dataset a singola persona per frame.
- Per confronti qualitativi con VitPose++, mantieni separati `--output-dir` e `--overlays-dir` per ogni checkpoint.
