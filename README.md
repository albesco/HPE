# HPE VitPose++ Dataset Pipeline

Pipeline consolidata per preparare dataset di addestramento SwimXYZ -> VitPose++
single-head.

## Struttura

```text
HPE/
├─ configs/
│  └─ pipeline/
├─ data/
│  ├─ dataset/
│  ├─ input/
│  ├─ intermediate/
│  └─ output/
├─ models/
│  ├─ detection/
│  └─ pose/
├─ script/
├─ src/
│  └─ vitpose_base/
└─ logs/
```

`data/`, `models/`, `logs/` e gli output runtime non vanno versionati.

## Ambiente

Ambiente confermato durante il setup:

```text
Python 3.9
torch 1.9.0+cu111
mmcv-full 1.3.17
numpy 1.26.4
opencv-python 4.9.0
```

Attivazione:

```bash
conda activate vitpose
cd ~/HPE
```

Dipendenze runtime installabili da file:

```bash
pip install -r requirements_runtime.txt
```

Installazioni pip specifiche validate:

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
  -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.3.17 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html \
  --only-binary=:all:
```

Il file `environment.example.yml` documenta la base Conda/Pip, ma PyTorch CUDA e MMCV vanno installati con i comandi sopra.

## Script attivi

La root `script/` contiene solo gli script della pipeline consolidata:

```text
CSV-2-JSON_Keypoint-Conversion.py
kp_check_swimxyz_video_frames.py
prepare_swimxyz_vitposepp_utils.py
prepare_swimxyz_vitposepp_single_head.py
prepare_swimxyz_vitposepp_train.py
```

Gli script storici o sperimentali sono stati spostati in `script/old_trials/`.

## Verifica ambiente

```bash
python script/old_trials/smoke_test.py
```

## Conversione in `intermediate`

```bash
conda run -n vitpose python script/CSV-2-JSON_Keypoint-Conversion.py \
  --source-root data/input/subset_xyz \
  --output-root data/intermediate
```

Questo step:

- copia video e label nella struttura `data/intermediate/<dataset>/_converted`
- genera il `manifest.json` usato dagli step successivi
- genera i sidecar `*_coco.json` per le label `COCO/2D`

## Preparazione dataset VitPose++

```bash
conda run -n vitpose python script/prepare_swimxyz_vitposepp_train.py \
  --dataset-root data/intermediate
```

Per un singolo video si può usare:

```bash
conda run -n vitpose python script/prepare_swimxyz_vitposepp_train.py \
  --dataset-entry "path/video.webm::path/label__COCO__2D_cam.txt" \
  --output-root data/intermediate/<dataset>/<video>_train_vitposepp_swap_ears
```

L'output contiene:

- `train2017`, `val2017`, `test2017`
- i corrispondenti overlay `*_with-KP.jpg`
- `annotations/person_keypoints_{train,val,test}.json`
- `annotations/dataset_exceptions.log`
- `reports/preparation_report.json`
- `generated_configs/...py`

## Logica consolidata

- inversione dell'asse `Y` da SwimXYZ a coordinate immagine
- scambio `LEye/REye` e `LEar/REar` prima della ricodifica COCO17
- `z` non interpretata come confidence
- visibility ricostruita con:
  `v = 0` se `x == 0` oppure `x == max_x`
  `v = 2` se `0 < x < max_x`
- logging dei salti anomali della bounding box
- overlay finali con soli skeleton e punti, senza etichette

Il dettaglio operativo aggiornato è documentato in `note.md`.

## Verifica visiva

Per controllare frame singoli o l'intero video:

```bash
conda run -n vitpose python script/kp_check_swimxyz_video_frames.py \
  --video path/video.webm \
  --labels path/label__COCO__2D_cam.txt \
  --frames 10 20 30 \
  --output-dir data/intermediate/kp_check_example \
  --flip-y
```

Oppure:

```bash
conda run -n vitpose python script/kp_check_swimxyz_video_frames.py \
  --video path/video.webm \
  --labels path/label__COCO__2D_cam.txt \
  --all-frames \
  --output-dir data/intermediate/kp_check_full_video \
  --flip-y
```

## Note Git

Versionare:

```text
README.md
.gitignore
configs/
script/
src/hpe_project/
```

Non versionare:

```text
data/
models/
logs/
outputs/
*.pt
*.pth
*.ckpt
```
