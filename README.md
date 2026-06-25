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
kp_check_swimxyz_video_frames.py
prepare_swimxyz_vitposepp_utils.py
prepare_swimxyz_vitposepp_single_head.py
prepare_swimxyz_vitposepp_train.py
```

Gli script storici o sperimentali sono stati spostati in `script_old/old_trials/`.

## Verifica ambiente

```bash
python script_old/old_trials/smoke_test.py
```

## Preparazione dataset consolidata

```bash
conda run -n vitpose python script_old/prepare_swimxyz_vitposepp_train.py \
  --source-root data/input/subset_xyz \
  --intermediate-root data/intermediate
```

Questo step:

- normalizza i video e le label SwimXYZ nella struttura `data/intermediate/<dataset>/_converted`
- genera il `manifest.json` usato dagli step successivi
- pulisce e converte i keypoint nella rappresentazione COCO17 usata da VitPose++
- genera il dataset finale pronto per il training/inferenza top-down:
  - `train2017`, `val2017`, `test2017`
  - `annotations/person_keypoints_{train,val,test}.json`
  - overlay `*_with-KP.jpg`
  - `reports/preparation_report.json`
  - `generated_configs/...py`

Se i dati sono gia stati normalizzati in `data/intermediate/.../_converted`, si puo partire da li:

```bash
conda run -n vitpose python script_old/prepare_swimxyz_vitposepp_train.py \
  --dataset-root data/intermediate
```

Per un singolo video si può usare:

```bash
conda run -n vitpose python script_old/prepare_swimxyz_vitposepp_train.py \
  --dataset-entry "path/video.webm::path/label__COCO__2D_cam.txt" \
  --output-root data/intermediate/<dataset>/<video>_train_canonical
```

## Preparazione dataset model-specific

Una volta ottenuto il dataset canonico `_train_canonical`, gli exporter
model-specific operano solo sulle label/config e non preparano immagini:

```bash
conda run -n vitpose python script/dataset_preparation-cleaning/prepare_vitposepp_dataset.py --overwrite
conda run -n vitpose python script/dataset_preparation-cleaning/prepare_yolo_detection_dataset.py --overwrite
conda run -n vitpose python script/dataset_preparation-cleaning/prepare_yolo_pose_dataset.py --overwrite
```

In questo modo la pipeline dati consolidata diventa:

1. `data/input/subset_xyz/...`
2. `script_old/prepare_swimxyz_vitposepp_train.py --source-root ...`
3. `data/intermediate/<dataset>/_converted`
4. `data/intermediate/<dataset>/_train_canonical`
5. `data/intermediate/<dataset>/_VitPosePP`
6. `data/intermediate/<dataset>/_Yolo26x_detection`
7. `data/intermediate/<dataset>/_Yolo26x_pose`

## Logica consolidata

- inversione dell'asse `Y` da SwimXYZ a coordinate immagine
- scambio `LEye/REye` e `LEar/REar` prima della ricodifica COCO17
- `z` non interpretata come confidence
- visibility ricostruita con:
  `v = 0` se `x == 0` oppure `x == max_x`
  `v = 2` se `0 < x < max_x`
- logging dei salti anomali della bounding box
- overlay finali con soli skeleton e punti, senza etichette

Il consolidamento storico del data cleaning è documentato in
`docs/ai/data-cleaning-consolidation.md`; lo stato operativo aggiornato è in
`docs/ai/context.md` e `docs/ai/handoff.md`.

## Verifica visiva

Per controllare frame singoli o l'intero video:

```bash
conda run -n vitpose python script_old/kp_check_swimxyz_video_frames.py \
  --video path/video.webm \
  --labels path/label__COCO__2D_cam.txt \
  --frames 10 20 30 \
  --output-dir data/intermediate/kp_check_example \
  --flip-y
```

Oppure:

```bash
conda run -n vitpose python script_old/kp_check_swimxyz_video_frames.py \
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
