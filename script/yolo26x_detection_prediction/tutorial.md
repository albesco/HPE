# Tutorial: YOLO26x-Detection prediction

Questo tutorial descrive gli script operativi di predizione per YOLO26x-Detection e il ponte YOLO bbox -> VitPose++.

Script principali:

```bash
script/yolo26x_detection_prediction/evaluate_yolo_vitpose_map.py
script/yolo26x_detection_prediction/render_yolo_detection_overlays.py
script/yolo26x_detection_prediction/export_yolo_detection_top1_labels.py
script/yolo26x_detection_prediction/preview_yolo_bbox_predictions.py
script/yolo26x_detection_prediction/yolo_detection_utils.py
```

## 1. Cosa fa la directory

La directory contiene quattro usi diversi:

1. `evaluate_yolo_vitpose_map.py` esegue la pipeline completa YOLO -> VitPose++ e valuta COCO mAP keypoint.
2. `render_yolo_detection_overlays.py` disegna overlay con una sola bbox per immagine.
3. `export_yolo_detection_top1_labels.py` esporta le predizioni top-1 in formato label YOLO.
4. `preview_yolo_bbox_predictions.py` produce anteprime rapide su immagini campione.

Tutti gli script usano la stessa regola di selezione bbox:

- confidenza piu alta vince;
- in caso di pari confidenza, vince la bbox con area maggiore.

Questa regola e centralizzata in `yolo_detection_utils.py`.

## 2. Allestimento ambiente

L'ambiente atteso e sempre `vitpose`:

```bash
conda activate vitpose
```

Controlli rapidi utili:

```bash
python -c "import ultralytics, cv2, numpy"
python -c "import xtcocotools, mmpose"
yolo --help
```

### Prerequisiti dati e modelli

Per la pipeline completa servono:

- la root del dataset canonico, per esempio `data/intermediate/Side_above_water`;
- il YAML VitPose++ del dataset;
- il checkpoint YOLO26x detector da usare come bbox provider;
- il checkpoint VitPose++ da valutare;
- la config VitPose++ base del dataset.

Se lavori solo sulle overlay o sulle label YOLO, serve soltanto il modello YOLO e il dataset detection.

## 3. Pipeline completa: `evaluate_yolo_vitpose_map.py`

Questo e il bridge operativo per la valutazione YOLO bbox -> VitPose++.

### Comando minimo

```bash
python script/yolo26x_detection_prediction/evaluate_yolo_vitpose_map.py
```

Con i default, la pipeline usa:

- dataset root: `data/intermediate/Side_above_water/_train_canonical`;
- split: `test`;
- YOLO model: `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt`;
- VitPose++ config: `data/intermediate/Side_above_water/_train_canonical/generated_configs/swimxyz_vitposepp_huge.py`;
- VitPose++ checkpoint: `runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth`;
- output root: `data/output/experiments/YoloVitPose_mAP`.

### Parametri

| Parametro | Default | Significato |
|---|---|---|
| `--dataset-root` | `data/intermediate/Side_above_water/_train_canonical` | root canonica con `annotations/` e `train2017/val2017/test2017` |
| `--split` | `test` | split da valutare; valori ammessi `train`, `val`, `test` |
| `--yolo-model` | `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt` | checkpoint YOLO da usare come bbox provider |
| `--vitpose-config` | `data/intermediate/Side_above_water/_train_canonical/generated_configs/swimxyz_vitposepp_huge.py` | config base VitPose++ usata da `init_pose_model` |
| `--vitpose-checkpoint` | `runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth` | checkpoint VitPose++ |
| `--output-root` | `data/output/experiments/YoloVitPose_mAP` | root di output dell'esperimento |
| `--imgsz` | `1280` | risoluzione inferita dal detector |
| `--conf` | `0.25` | soglia confidenza YOLO |
| `--overlay-count` | `20` | numero di overlay da salvare su campione casuale |
| `--seed` | `20260516` | seed per il campione overlay |
| `--keypoint-score-threshold` | `0.3` | soglia score keypoint usata nel punteggio finale |
| `--device` | `cuda:0` | device per YOLO e MMPose |

### Output prodotti

```text
<output-root>/<split>_<timestamp>/
  yolo_vitpose_keypoints_results.json
  failures.json
  summary.json
  overlays/
```

`summary.json` include metadati, modelli, conteggi e metriche COCO keypoint.

### Esempio con output separato

```bash
python script/yolo26x_detection_prediction/evaluate_yolo_vitpose_map.py   --dataset-root data/intermediate/Side_above_water/_train_canonical   --split test   --yolo-model runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt   --vitpose-checkpoint runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth   --output-root data/output/experiments/YoloVitPose_mAP
```

## 4. Overlay detector: `render_yolo_detection_overlays.py`

Questo script crea immagini con una sola bbox YOLO per frame.

### Parametri

| Parametro | Default | Significato |
|---|---|---|
| `--model` | obbligatorio | checkpoint YOLO da usare |
| `--source-dir` | `data/intermediate/Side_above_water/_Yolo26x_detection/images/test` | directory di immagini da annotare |
| `--output-dir` | `data/output/experiments/yolo26x_detection_overlays/top1` | directory di output |
| `--count` | `20` | numero di immagini da campionare |
| `--seed` | `20260523` | seed per il campione casuale |
| `--imgsz` | `768` | dimensione inferenza YOLO |
| `--conf` | `0.25` | soglia confidenza YOLO |
| `--bbox-thickness` | `3` | spessore rettangolo bbox |

### Esempio

```bash
conda run -n vitpose python script/yolo26x_detection_prediction/render_yolo_detection_overlays.py   --model runs/yolo26x_bbox_side_above_water/yolo26x-detection_from_cfg03_ep5_20260523_1923/weights/best.pt   --source-dir data/intermediate/Side_above_water/_Yolo26x_detection/images/test   --output-dir data/output/experiments/yolo26x_detection_overlays/top1_test20
```

### Criticita

- se non trova una bbox, salva comunque l'immagine senza rettangolo;
- il risultato e solo qualitativo, non e una valutazione;
- `count` deve essere minore o uguale al numero di immagini disponibili.

## 5. Export label top-1: `export_yolo_detection_top1_labels.py`

Questo script esporta le predizioni detector in file `.txt` YOLO, una bbox per immagine.

### Parametri

| Parametro | Default | Significato |
|---|---|---|
| `--model` | `runs/yolo26x_bbox_side_above_water/yolo26x-detection_from_cfg03_ep5_20260523_1923/weights/best.pt` | checkpoint YOLO |
| `--dataset-root` | `data/intermediate/Side_above_water/_Yolo26x_detection` | root del dataset detection |
| `--output-root` | `data/output/experiments/yolo26x_detection_predictions/top1_from_cfg03_ep5_20260523_1923` | root di output |
| `--splits` | `val test` | split da esportare |
| `--imgsz` | `768` | dimensione inferenza |
| `--conf` | `0.25` | soglia confidenza |
| `--overwrite` | off | cancella l'output esistente se presente |

### Esempio

```bash
conda run -n vitpose python script/yolo26x_detection_prediction/export_yolo_detection_top1_labels.py   --model runs/yolo26x_bbox_side_above_water/yolo26x-detection_from_cfg03_ep5_20260523_1923/weights/best.pt   --dataset-root data/intermediate/Side_above_water/_Yolo26x_detection   --splits val test   --output-root data/output/experiments/yolo26x_detection_predictions/top1_demo   --overwrite
```

### Criticita

- le label vuote indicano nessuna detection sopra soglia;
- il formato esportato include anche la confidence come sesto valore;
- senza `--overwrite`, l'export si ferma se l'output esiste gia.

## 6. Preview rapida: `preview_yolo_bbox_predictions.py`

Questo script serve solo per una verifica visiva veloce su poche immagini.

### Parametri

| Parametro | Default | Significato |
|---|---|---|
| `--model` | `runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox_padded10_20ep/frozen_checkpoints/best_epoch10_user_20260514_070621.pt` | modello YOLO da usare |
| `--source-dir` | `data/intermediate/Side_above_water/_train_canonical/val2017` | directory immagini sorgente |
| `--output-dir` | `data/intermediate/epoch_10_yolo_bbox` | directory di output |
| `--count` | `20` | numero di immagini campionate |
| `--seed` | `20260514` | seed del campione |
| `--imgsz` | `1280` | inferenza YOLO |
| `--conf` | `0.25` | soglia confidenza |

### Esempio

```bash
conda run -n vitpose python script/yolo26x_detection_prediction/preview_yolo_bbox_predictions.py   --model runs/yolo26x_bbox_side_above_water/yolo26x-detection_from_cfg03_ep5_20260523_1923/weights/best.pt   --source-dir data/intermediate/Side_above_water/_train_canonical/val2017   --output-dir data/output/experiments/yolo26x_detection_preview/demo20
```

### Criticita

- il default `--model` e storico; quasi sempre conviene sovrascriverlo;
- lo script non salva metriche, solo immagini annotate;
- `count` troppo alto rispetto alla cartella sorgente causa errore.

## 7. Lancio con tmux

Per run lunghe, usa `tmux` manualmente.

### Esempio per la pipeline completa

```bash
tmux new-session -d -s yolo26x_detection_eval_20260702   "cd /home/albertosco/HPE && conda run -n vitpose python script/yolo26x_detection_prediction/evaluate_yolo_vitpose_map.py     --dataset-root data/intermediate/Side_above_water/_train_canonical     --split test     --output-root data/output/experiments/YoloVitPose_mAP 2>&1 | tee logs/yolo26x_detection_eval_20260702.log"
```

### Gestione sessione

```bash
tmux attach -t yolo26x_detection_eval_20260702
tmux ls
tmux kill-session -t yolo26x_detection_eval_20260702
```

## 8. Criticita operative comuni

- `evaluate_yolo_vitpose_map.py` richiede sia YOLO sia VitPose++ e quindi e piu pesante degli altri script.
- `--device cuda:0` e il default: su CPU devi impostarlo esplicitamente.
- `--conf 0.25` puo filtrare troppe bbox se il detector e debole; abbassalo solo se sai perche lo stai facendo.
- Il confronto qualitativo corretto per YOLO+VitPose usa la stessa bbox rule di `yolo_detection_utils.py`.
- Se cambi il checkpoint YOLO o VitPose++, registra sempre il nuovo output root per non confondere i risultati con run precedenti.
