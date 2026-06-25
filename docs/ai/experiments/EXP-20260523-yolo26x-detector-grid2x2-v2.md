# EXP-20260523: YOLO26x detector 2x2 grid (v2) selection

Date: 2026-05-23 UTC  
Last updated: 2026-05-23 16:29 UTC

## Purpose

Select the YOLO26x one-class `swimmer` detector checkpoint to use as the bbox provider in the YOLO->VitPose++ pipeline (`YoloVitPose_mAP`).

## Selection constraint

Hyperparameter selection is **validation-only** (no test split).

## Run root

- `runs/hparam_search/yolo26x_detector_v2/`

## Script

- `script_old/yolo_training/yolo26x_detector_grid2x2.py`

## Data

- Train/val YAML: `runs/hparam_search/yolo26x_detector_v2/dataset_view/swimxyz_side_above_water_yolo26x_detection_train_val.yaml`

## Grid

- lr0 ∈ {0.00067, 0.00100}
- imgsz ∈ {640, 768}
- epochs=5, split=val, one class (`swimmer`)

## Selection rule

Priority: recall -> (AP75/IoU when available) -> mAP50-95 -> prefer imgsz=640.

Note: AP75/IoU are not produced by Ultralytics `results.csv` here, so they remain null.

## Results (val, epoch 5)

Source: `runs/hparam_search/yolo26x_detector_v2/summary.csv`

- `cfg_01_lr0_0.00067_imgsz_640`: precision 0.98865, recall 0.99095, mAP50 0.99330, mAP50-95 0.87363
- `cfg_02_lr0_0.00100_imgsz_640`: precision 0.98865, recall 0.99095, mAP50 0.99330, mAP50-95 0.87363
- `cfg_03_lr0_0.00067_imgsz_768`: precision 0.99440, recall 0.99519, mAP50 0.99267, mAP50-95 0.86614
- `cfg_04_lr0_0.00100_imgsz_768`: precision 0.99440, recall 0.99519, mAP50 0.99267, mAP50-95 0.86614

## Selected checkpoint

- Consolidated config: `cfg_03_lr0_0.00067_imgsz_768` (ties `cfg_04` on reported metrics; chosen as canonical per `best_config.json`).
- Use this checkpoint in the YOLO->VitPose++ pipeline:
  - `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/best.pt`

## Repro pointers

- Summary: `runs/hparam_search/yolo26x_detector_v2/summary.csv`
- Report: `runs/hparam_search/yolo26x_detector_v2/report.md`
- Best-config JSON: `runs/hparam_search/yolo26x_detector_v2/best_config.json`
