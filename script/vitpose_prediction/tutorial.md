# Tutorial: predict_vitpose_frame.sh

Questo tutorial descrive lo script parametrico per predire keypoint con VitPose++, calcolare metriche sul Test split e generare overlay frame/KP.

Script principale:

```bash
script/vitpose_prediction/predict_vitpose_frame.sh
```

## Cosa fa

Lo script:

1. riceve un checkpoint VitPose++;
2. riceve la root principale del dataset, per esempio `data/intermediate/SAW_frames`;
3. trova automaticamente `DATASET/_train_canonical` e `DATASET/_VitPosePP`;
4. crea una vista Test effettiva sotto `--output-dir`, con eventuale campionamento random e filtro bbox;
5. genera `effective_config.py` con i path Test effettivi;
6. forza crop/heatmap del Test in modo coerente con il checkpoint;
7. esegue il Test VitPose++ con `src/vitpose_base/tools/test.py`;
8. esporta `result_keypoints.json`, `kp_Test.json` e `metrics_Test.json`;
9. genera overlay frame/KP dal JSON delle predizioni.

## Struttura attesa del dataset

`--dataset-dir` deve puntare alla root principale del dataset, non a `_train_canonical` e non a `_VitPosePP`.

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

Esempi errati:

```bash
--dataset-dir data/intermediate/SAW_frames/_train_canonical
--dataset-dir data/intermediate/SAW_frames/_VitPosePP
```

Lo script rifiuta questi path per evitare mismatch tra dataset, config e split di Test.

## Comando minimo

```bash
bash script/vitpose_prediction/predict_vitpose_frame.sh \
  --checkpoint runs/vitpose_SAW_frames_20260615/checkpoint/best_AP_epoch_18.pth \
  --dataset-dir data/intermediate/SAW_frames
```

Con questo comando lo script usa tutto il Test split e scrive gli output in:

```text
data/output/experiments/vitpose_prediction_best_AP_epoch_18/
```

## Parametri

| Parametro | Default | Descrizione |
|---|---|---|
| `--checkpoint PATH` | nessuno | Checkpoint VitPose++ `.pth` da usare per predizione e metriche. |
| `--dataset-dir PATH` | nessuno | Root principale del dataset. Deve contenere `_train_canonical/` e `_VitPosePP/`. |
| `--base-config PATH` | `DATASET/_VitPosePP/generated_configs/swimxyz_vitposepp_huge.py` | Config base VitPose++ usata per generare `effective_config.py`. |
| `--max-test-items N` | `0` | Numero di elementi Test random da valutare. `0` significa tutto il Test split. |
| `--seed N` | opzionale; `0` quando si campiona | Seed per campionare gli elementi Test. |
| `--conf FLOAT` | `0` | Threshold bbox per costruire la vista Test. Se `0`, lo script tiene solo la prima bbox per immagine. |
| `--output-dir PATH` | `data/output/experiments/vitpose_prediction_<checkpoint-name>` | Directory per `result_keypoints.json`, `kp_Test.json`, `metrics_Test.json`, `effective_config.py` e `predict.log`. |
| `--overlays-dir PATH` | `OUTPUT_DIR/overlays_Test` | Directory dove salvare gli overlay frame/KP. |
| `--device VALUE` | `auto` | `auto`, `cpu`, `cuda:0`, ecc. |
| `--crop-size WxH` | `384x128` | Crop Test da usare per VitPose++. Deve combaciare con il checkpoint; il checkpoint `runs/vitpose_SAW_frames_20260615/checkpoint/best_AP_epoch_18.pth` e stato addestrato con `384x128`. |
| `--batch-size N` | `1` | Batch size del Test dataloader. |
| `--num-workers N` | `1` | Numero di worker del Test dataloader. |
| `--split NAME` | `test` | Split da valutare. Normalmente lasciare `test`. |
| `--help` | off | Mostra l'help dello script. |

## Esempi d'uso

### Test completo

```bash
bash script/vitpose_prediction/predict_vitpose_frame.sh \
  --checkpoint runs/vitpose_SAW_frames_20260615/checkpoint/best_AP_epoch_18.pth \
  --dataset-dir data/intermediate/SAW_frames \
  --output-dir data/output/experiments/pred_vitpose_saw_full \
  --overlays-dir data/output/experiments/pred_vitpose_saw_full/overlays_Test
```

Questo valuta tutto il Test split e genera overlay per tutti i frame.

### Campione random riproducibile

```bash
bash script/vitpose_prediction/predict_vitpose_frame.sh \
  --checkpoint runs/vitpose_SAW_frames_20260615/checkpoint/best_AP_epoch_18.pth \
  --dataset-dir data/intermediate/SAW_frames \
  --max-test-items 50 \
  --seed 42 \
  --output-dir data/output/experiments/pred_vitpose_saw_sample50 \
  --overlays-dir data/output/experiments/pred_vitpose_saw_sample50/overlays_Test
```

Quando usi `--max-test-items`, lo script crea una vista dataset sotto `--output-dir`:

```text
dataset_view_test_n50_seed42_conf0/
  annotations/
    person_keypoints_test.json
  test2017/
  sample_report.json
```

Metriche, `kp_Test.json` e overlay usano tutti lo stesso sottoinsieme.

### Threshold bbox esplicito

```bash
bash script/vitpose_prediction/predict_vitpose_frame.sh \
  --checkpoint runs/vitpose_SAW_frames_20260615/checkpoint/best_AP_epoch_18.pth \
  --dataset-dir data/intermediate/SAW_frames \
  --conf 0.25 \
  --output-dir data/output/experiments/pred_vitpose_saw_conf025
```

Usa questa forma quando vuoi scartare bbox sotto una certa soglia prima del Test top-down.

### Crop size esplicito

```bash
bash script/vitpose_prediction/predict_vitpose_frame.sh \
  --checkpoint runs/vitpose_SAW_frames_20260615/checkpoint/best_AP_epoch_18.pth \
  --dataset-dir data/intermediate/SAW_frames_EntireSwim \
  --crop-size 384x128 \
  --output-dir data/output/experiments/pred_vitpose_saw_crop384x128
```

Usa `--crop-size` quando il checkpoint e stato addestrato con una risoluzione diversa dal config base del dataset. Lo script deriva automaticamente `heatmap_size` dividendo il crop per 4.

### Esecuzione su CPU

```bash
bash script/vitpose_prediction/predict_vitpose_frame.sh \
  --checkpoint runs/vitpose_SAW_frames_20260615/checkpoint/best_AP_epoch_18.pth \
  --dataset-dir data/intermediate/SAW_frames \
  --max-test-items 10 \
  --device cpu \
  --output-dir data/output/experiments/pred_vitpose_saw_cpu_smoke
```

Utile per smoke test piccoli quando la GPU non e disponibile.

## Output prodotti

Con `--output-dir data/output/experiments/pred_vitpose_saw_sample50`, lo script produce:

```text
data/output/experiments/pred_vitpose_saw_sample50/
  effective_config.py
  result_keypoints.json
  kp_Test.json
  metrics_Test.json
  predict.log
  dataset_view_test_n50_seed42_conf0/
  overlays_Test/
```

`metrics_Test.json` contiene le metriche COCO keypoint di MMPose/VitPose++, per esempio:

```text
AP
AP .5
AP .75
AP (M)
AP (L)
AR
AR .5
AR .75
AR (M)
AR (L)
```

## Lancio con tmux

Lo script non crea una sessione tmux automaticamente. Per run lunghe su SSH, lancialo dentro una sessione tmux manuale.

### Avvio sessione

```bash
tmux new-session -d -s pred_vitpose_saw_sample50 \
  "cd /home/albertosco/HPE && bash script/vitpose_prediction/predict_vitpose_frame.sh \
    --checkpoint runs/vitpose_SAW_frames_20260615/checkpoint/best_AP_epoch_18.pth \
    --dataset-dir data/intermediate/SAW_frames \
    --max-test-items 50 \
    --seed 42 \
    --conf 0 \
    --output-dir data/output/experiments/pred_vitpose_saw_sample50 \
    --overlays-dir data/output/experiments/pred_vitpose_saw_sample50/overlays_Test \
    2>&1 | tee logs/pred_vitpose_saw_sample50.log"
```

### Gestione sessione

| Azione | Comando |
|---|---|
| Elencare sessioni | `tmux ls` |
| Entrare nella sessione | `tmux attach -t pred_vitpose_saw_sample50` |
| Uscire lasciando il job attivo | `Ctrl-b`, poi `d` |
| Vedere il log senza entrare | `tail -n 80 logs/pred_vitpose_saw_sample50.log` |
| Fermare la sessione tmux | `tmux kill-session -t pred_vitpose_saw_sample50` |

### Riprendere dopo disconnessione

```bash
tmux ls
tmux attach -t pred_vitpose_saw_sample50
```

Se la sessione e gia terminata, controlla gli output:

```bash
ls -lh data/output/experiments/pred_vitpose_saw_sample50
ls -lh data/output/experiments/pred_vitpose_saw_sample50/overlays_Test | head
```

## Note operative

- Usa sempre la root principale del dataset con `--dataset-dir`.
- Usa `--max-test-items` per controlli rapidi e `0` per il benchmark completo.
- Usa sempre `--seed` quando vuoi un campione riproducibile.
- Con `--conf 0`, lo script costruisce una vista Test top-1 bbox per immagine, coerente con il comportamento richiesto.
- `effective_config.py` e `sample_report.json` rendono il Test riproducibile e leggibile a posteriori.
- Per confronti qualitativi con YOLO26x-Pose, mantieni separati `--output-dir` e `--overlays-dir` per ogni checkpoint.
