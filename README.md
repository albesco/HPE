# HPE ViTPose Pipeline

Pipeline per Human Pose Estimation con YOLO per la detection delle persone e ViTPose-H/ViTPose++ per la stima dei keypoint.

Il checkpoint YOLO predefinito del workspace e' `models/detection/yolo26x.pt`.

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

## Smoke Test

```bash
python script/smoke_test.py
```

## Pipeline Immagine

```bash
python script/run_pipeline.py \
  --input data/input/bus.jpg \
  --output-dir data/output/pipeline_bus
```

Output:

```text
data/output/pipeline_bus/bus.jpg
data/output/pipeline_bus/bus.csv
```

## Pipeline Video

```bash
python script/run_pipeline.py \
  --input data/input/video.mp4 \
  --output-dir data/output/pipeline_video
```

Output:

```text
data/output/pipeline_video/video_pose.mp4
data/output/pipeline_video/video_pose.csv
```

Per video `.webm`, lo script accetta il formato se OpenCV sul server riesce a leggerlo.

## CSV Keypoint

I CSV usano separatore `;` e virgola decimale. Le coordinate sono riferite all'immagine o frame originale, con origine nel punto in alto a sinistra.

Il modello COCO produce 17 keypoint. Lo script esporta il formato BODY_25 richiesto, calcolando `Neck` e `MidHip` come medie, mentre i keypoint di dita/piedi non disponibili nel modello COCO restano vuoti.

## Intermedi

I file intermedi sono mantenuti di default in:

```text
data/intermediate/pipeline/
```

Per eliminarli a fine elaborazione:

```bash
python script/run_pipeline.py \
  --input data/input/bus.jpg \
  --output-dir data/output/pipeline_bus \
  --clean-intermediate
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
