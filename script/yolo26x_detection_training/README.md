# YOLO26x-Detection training

Operational scripts for the one-class YOLO26x swimmer detector used as bbox provider for the YOLO -> VitPose++ pipeline.

## Entry points

- `train_yolo26x_detection_frame.sh`: parametric detector training for frame datasets; resolves `_Yolo26x_detection`, enforces `min_delta` early stopping via monitor, exports validation reports, and evaluates Test with bbox JSON/overlays.
- `prune_yolo_epoch_checkpoints.py`: helper used by the training launcher to retain recent periodic checkpoints.

## Defaults

- Dataset YAML: `data/intermediate/Side_above_water/_Yolo26x_detection/swimxyz_side_above_water_yolo26x_detection.yaml`.
- Selected grid checkpoint: `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt`.
- Useful trained checkpoint: `runs/yolo26x-detection_SUW_frames_20260701/weights/best.pt`.
- Training root: `runs/yolo26x-detection_SUW_frames_20260701/`.

## Example

```bash
bash script/yolo26x_detection_training/train_yolo26x_detection_frame.sh \
  --dataset-dir data/intermediate/SUW_frames \
  --pretrained-checkpoint models/detection/yolo26x.pt \
  --run-name yolo26x-detection_SUW_frames_20260701_example \
  --use-tmux yes
```


## Parametric frame training

```bash
bash script/yolo26x_detection_training/train_yolo26x_detection_frame.sh \
  --dataset-dir data/intermediate/SUW_frames \
  --pretrained-checkpoint models/detection/yolo26x.pt \
  --run-name yolo26x-detection_SUW_frames_20260701 \
  --early-stop-patience 3 \
  --early-stop-min-delta 0.007 \
  --keep-last-n-checkpoints 10 \
  --use-tmux yes
```

The launcher refuses to overwrite existing run/Test outputs unless `--overwrite` is passed.
