# YOLO Training Datasets

Scripts for Side_above_water YOLO26x detection and pose labels derived from the canonical SwimXYZ COCO keypoint dataset.

## Prepare YOLO26x Detection Labels

```bash
python script/yolo_training/prepare_yolo_detection_dataset.py --overwrite
```

Output:

- Dataset root: `data/intermediate/Side_above_water/_Yolo26x_detection/`
- Data YAML: `data/intermediate/Side_above_water/_Yolo26x_detection/swimxyz_side_above_water_yolo26x_detection.yaml`
- Labels: `labels/{train,val,test}`
- Class: `swimmer`

The exporter reads GT bboxes from `_train_canonical` and writes labels only. It does not copy, link, resize, or modify images. Keep `--bbox-padding-ratio 0.0` unless a new experiment deliberately changes the bbox convention.

## Prepare YOLO26x Pose Labels

```bash
python script/yolo_training/prepare_yolo_pose_dataset.py --overwrite
```

Output:

- Dataset root: `data/intermediate/Side_above_water/_Yolo26x_pose/`
- Data YAML: `data/intermediate/Side_above_water/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml`
- Labels: `labels/{train,val,test}`
- Keypoints: COCO17, `kpt_shape: [17, 3]`

This exporter reads `_train_canonical`, writes YOLO pose labels, and exposes the canonical images under `images/{train,val,test}` with symlinks by default. Frame inclusion is therefore coordinated with VitPose++ and YOLO detection through the canonical dataset.

Sanity check one generated label should contain `56` fields: class, bbox `xywh`, and `17 * 3` keypoint values.

## Train YOLO Detection

```bash
script/yolo_training/train_yolo_side_above_water.sh
```

Defaults:

- Base model: `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt`
- Data: `data/intermediate/Side_above_water/_Yolo26x_detection/swimxyz_side_above_water_yolo26x_detection.yaml`
- Epochs: `100`
- Image size: `768`
- Batch: `2`
- Learning rate: `0.00067`
- Patience: `2`
- Save period: every epoch (`SAVE_PERIOD=1`)
- Periodic retention: keep latest `3` epoch checkpoints (`KEEP_EPOCH_CKPTS=3`)
- Output: `runs/yolo26x_bbox_side_above_water/yolo26x-detection_<tag>/`
- Status: `runs/yolo26x_bbox_side_above_water/yolo26x-detection_<tag>/training_status.txt`
- Log: `logs/yolo26x_detection_side_above_water_<timestamp>.log`

The launcher auto-resumes from the target run `weights/last.pt` when present; otherwise it starts from the selected cfg_03 `last.pt` above. Stop the incremental phase when training loss keeps decreasing while validation mAP50-95 plateaus or declines, using the default patience of `2` epochs.

Override defaults with environment variables:

```bash
TAG=plateau_probe_01 EPOCHS=20 BATCH=2 script/yolo_training/train_yolo_side_above_water.sh
```

`yolo26x.pt` with `imgsz=1280` and `batch=8` exceeds the 32 GB V100 GPU memory.


## Render YOLO26x Detection Overlays

Use the dedicated detector renderer when qualitative bbox outputs must contain exactly one prediction per frame. It selects the box with highest confidence; when confidences tie, it keeps the larger-area box.

Example:

    conda run -n vitpose python script/yolo_training/render_yolo_detection_overlays.py \
      --model runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt \
      --source-dir data/intermediate/Side_above_water/_Yolo26x_detection/images/test \
      --output-dir data/output/experiments/yolo26x_detection_overlays/top1_test20 \
      --count 20

## Train YOLO26x Pose

```bash
script/yolo_training/train_yolo_pose_side_above_water.sh
```

Defaults:

- Model: `models/pose/yolo26x-pose.pt`
- Data: `data/intermediate/Side_above_water/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml`
- Epochs: `100`
- Image size: `1280`
- Batch: `1`
- Save period: every epoch (`SAVE_PERIOD=1`)
- Periodic retention: keep latest `3` epoch checkpoints (`KEEP_EPOCH_CKPTS=3`)
- Output: `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17/`
- Weights: `weights/best.pt`, `weights/last.pt`, plus retained `weights/epoch*.pt`
- Status: `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17/training_status.txt`
- Log: `logs/yolo26x_pose_side_above_water_<timestamp>.log`

Override defaults with environment variables:

```bash
EPOCHS=50 BATCH=1 IMGSZ=1024 script/yolo_training/train_yolo_pose_side_above_water.sh
```

The launcher requires local weights by default. To let Ultralytics resolve/download the pretrained weights explicitly, run with `MODEL=yolo26x-pose.pt`.

Resume from a previous run by setting `MODEL` to the run `last.pt` and `RESUME=True`:

```bash
MODEL=runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17/weights/last.pt RESUME=True   script/yolo_training/train_yolo_pose_side_above_water.sh
```

## Evaluate YOLO -> VitPose++

Consolidated end-to-end pipeline:

```bash
conda run -n vitpose python script/yolo_training/evaluate_yolo_vitpose_map.py \
  --split test \
  --yolo-model runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt \
  --vitpose-checkpoint runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth \
  --overlay-count 20 \
  --output-root data/output/experiments/YoloVitPose_mAP
```

Visualization convention: MMPose `vis_pose_result` predicted keypoints/skeleton plus a red YOLO bbox.


## Render YOLO26x Pose Overlays

Use the dedicated renderer instead of raw yolo pose predict when qualitative YOLO26x-pose outputs need the same visual format as VitPose++ overlays.

Example:

    conda run -n vitpose python script/yolo_training/render_yolo_pose_overlays.py \
      --model runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/weights/best.pt \
      --source data/intermediate/Side_above_water/_Yolo26x_pose/images/test \
      --output-dir data/output/experiments/yolo26x_pose_side_above_water/overlays_mmpose_style \
      --max-images 20

The renderer uses COCO/MMPose keypoint and skeleton colors, radius 3, skeleton thickness 2, and a red bbox with thickness 3.
