# 2026-07-02 Train YOLO26x-Detection SUW Frames

## Summary

- Added `script/yolo26x_detection_training/monitor_yolo_detection_patience.py`, an optional external monitor for YOLO detection `min_delta` early stopping on `metrics/mAP50-95(B)`.
- Validated the monitor with `conda run -n vitpose python -m py_compile`.
- Validated the detector launcher with `bash -n script_old/yolo26x_Detection-Training/train_yolo_side_above_water.sh`.
- Launched `yolo26x-detection_SUW_frames_20260701` in tmux without the external `min_delta` monitor, per user instruction.

## Run

- tmux session: `train_yolo26x_detection_SUW_frames_20260701`
- Run directory: `runs/yolo26x-detection_SUW_frames_20260701`
- Status file: `runs/yolo26x-detection_SUW_frames_20260701/training_status.txt`
- Log: `logs/yolo26x_detection_side_above_water_20260702_080859.log`
- Dataset YAML: `data/intermediate/SUW_frames/_Yolo26x_detection/swimxyz_side_above_water_yolo26x_detection.yaml`
- Initial checkpoint: `models/detection/yolo26x.pt`
- Settings: `epochs=100`, `imgsz=768`, `batch=2`, `lr0=0.00067`, `patience=3`, `keep_epoch_ckpts=10`.

## Pending

- Monitor training progress and report final metrics/checkpoints after completion.

## Dataset Layout Fix

- Observed `results.csv` with precision/recall/mAP all `0` and box losses `0`.
- Root cause: `_Yolo26x_detection/images/{train,val,test}` were directory symlinks, so Ultralytics resolved image paths to `_train_canonical/*2017` and created `_train_canonical/*.cache` files with zero labels.
- Replaced those directory symlinks with real split directories containing per-file symlinks to canonical images.
- Removed stale `_train_canonical/*.cache` files and the bad run directory.
- Relaunched tmux session `train_yolo26x_detection_SUW_frames_20260701` with the same run name and settings.
- Verification: new `_Yolo26x_detection/labels/train.cache` has `6044` boxes and `_Yolo26x_detection/labels/val.cache` has `1727` boxes.

## Parametric Detector Launcher

- Added `script/yolo26x_detection_training/train_yolo26x_detection_frame.sh` as a generalized CLI launcher based on `train_yolo_side_above_water.sh`.
- Added `script/yolo26x_detection_training/export_yolo_detection_training_report.py` for Val metrics CSV and validation loss/mAP PNGs.
- Added `script/yolo26x_detection_training/evaluate_yolo_detection_split.py` for Test bbox JSON, metrics JSON, and bbox overlays.
- Static validation passed: `bash -n`, Python `py_compile`, and launcher `--help`.

## Tutorial Refresh

- Rewrote `script/yolo26x_detection_training/tutorial.md` around the parametric launcher `train_yolo26x_detection_frame.sh`.
- Added installation notes, parameter descriptions, defaults, output layout, tmux workflow, examples, and troubleshooting.
- Documented the exact semantics of `START_EPOCH`, `MAX_EPOCHS`, and the external `min_delta` monitor.

## Training Directory Cleanup

- Moved historical `train_yolo_side_above_water.sh` and `yolo26x_detector_grid2x2.py` from `script/yolo26x_detection_training/` to `script_old/yolo26x_Detection-Training/`.
- Kept active runtime files in `script/yolo26x_detection_training/`: `train_yolo26x_detection_frame.sh`, monitor, pruner, report exporter, Test evaluator, README, and tutorial.
- Removed the obsolete `BASE_SCRIPT` existence check from `train_yolo26x_detection_frame.sh` so the archived historical launcher is no longer required at runtime.
- Validation passed: `bash -n`, helper `py_compile`, and launcher `--help`.
