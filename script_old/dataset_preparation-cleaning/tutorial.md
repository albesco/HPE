# Tutorial Preparazione Dataset Frame SwimXYZ

## Panoramica
Questo flusso prepara `data/input/subset_xyz/Side_above_water_frames` nel layout standard usato dal progetto.

## Input
- Immagini frame: `.png`, `.jpg` oppure `.jpeg`
- Label ground-truth: `__COCO__2D_cam.txt`
- Una coppia immagine+label per frame

## Output
- Dataset canonico: `data/intermediate/Side_above_water_frames/_train_canonical`
- Export modello: `_VitPosePP`, `_Yolo26x_detection`, `_Yolo26x_pose`

## Build
```bash
python script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py \
  --input-root data/input/subset_xyz/Side_above_water_frames \
  --output-root data/intermediate/Side_above_water_frames \
  --copy-mode symlink
```

## Struttura Canonica
Il builder crea:
- `train2017/`, `val2017/`, `test2017/`
- `annotations/person_keypoints_train.json`
- `annotations/person_keypoints_val.json`
- `annotations/person_keypoints_test.json`

## Lancio con tmux
Esempio di comando `tmux` che lancia lo script con i suoi parametri:
```bash
tmux new-session -d -s prepare_swimxyz_frames_dataset \
"cd /home/albertosco/HPE && \
conda run -n vitpose python script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py \
  --input-root data/input/subset_xyz/Side_above_water_frames \
  --output-root data/intermediate/Side_above_water_frames \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --copy-mode symlink \
  --vitpose-work-dir runs/vitposepp_side_above_water_frames \
  --yolo-pose-link-mode symlink \
  --overwrite \
  2>&1 | tee logs/prepare_swimxyz_frames_dataset.log"
```

### Gestione sessione con bash
```bash
# elenca le sessioni tmux attive
bash -lc 'tmux ls'

# entra nella sessione del dataset
bash -lc 'tmux attach -t prepare_swimxyz_frames_dataset'

# stacca dalla sessione senza fermarla
# dentro tmux: premi Ctrl-b, poi d

# chiudi la sessione se serve
bash -lc 'tmux kill-session -t prepare_swimxyz_frames_dataset'
```

## Note
- La materializzazione immagini di default è `symlink`.
- Il builder filtra i frame usando le stesse regole canoniche COCO/VitPose/YOLO del resto del repository.
