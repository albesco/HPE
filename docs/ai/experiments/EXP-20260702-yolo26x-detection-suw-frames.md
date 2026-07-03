# EXP-20260702 YOLO26x-Detection SUW Frames

## Purpose

Train a YOLO26x one-class swimmer detector on `data/intermediate/SUW_frames` using the consolidated detector launcher.

## Launch

- Date: 2026-07-02 UTC
- tmux session: `train_yolo26x_detection_SUW_frames_20260701`
- Launcher: `script/yolo26x_detection_training/train_yolo_side_above_water.sh`
- Dataset YAML: `data/intermediate/SUW_frames/_Yolo26x_detection/swimxyz_side_above_water_yolo26x_detection.yaml`
- Initial checkpoint: `models/detection/yolo26x.pt`
- Run name: `yolo26x-detection_SUW_frames_20260701`
- Run directory: `runs/yolo26x-detection_SUW_frames_20260701`
- Log: `logs/yolo26x_detection_side_above_water_20260702_080859.log`

## Settings

- `epochs=100`
- `imgsz=768`
- `batch=2`
- `lr0=0.00067`
- `patience=3`
- `save_period=1`
- `keep_epoch_ckpts=10`
- `resume=False`
- `run_test=False`
- External `min_delta` monitor: not active for this run.

## Notes

- Optional monitor created for future runs: `script/yolo26x_detection_training/monitor_yolo_detection_patience.py`.
- The monitor can apply `min_delta=0.007` to `metrics/mAP50-95(B)`, but this launch intentionally uses native Ultralytics early stopping only.
- Metrics pending.

## Relaunch After Dataset Layout Fix

- The first launch wrote zero precision/recall/mAP and zero box losses because Ultralytics resolved directory symlinks in `_Yolo26x_detection/images/{train,val,test}` to `_train_canonical/*2017` and cached images with no labels.
- Fix applied on 2026-07-02: replaced split directory symlinks with real directories containing per-file symlinks, removed stale `_train_canonical/train2017.cache`, `_train_canonical/val2017.cache`, and `_train_canonical/test2017.cache`, removed the bad run directory, and relaunched the same run name from `models/detection/yolo26x.pt`.
- Verification after relaunch: `_Yolo26x_detection/labels/train.cache` contains `6044` boxes and `_Yolo26x_detection/labels/val.cache` contains `1727` boxes.
