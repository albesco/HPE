# Session Note: hyper-choose Yolo26x-Detection

Date: 2026-05-23 UTC  
Last updated: 2026-05-23 16:28 UTC

## Goal

Select and consolidate the best YOLO26x one-class `swimmer` detector checkpoint from the v2 2x2 grid search for use as the bbox provider in the YOLO->VitPose++ pipeline.

## Constraint

Model selection uses **validation only** (no test split).

## Search

- Root: `runs/hparam_search/yolo26x_detector_v2/`
- Script: `script_old/yolo_training/yolo26x_detector_grid2x2.py`
- Dataset YAML (train/val only): `runs/hparam_search/yolo26x_detector_v2/dataset_view/swimxyz_side_above_water_yolo26x_detection_train_val.yaml`

## Selection rule

Priority: recall -> (AP75/IoU when available) -> mAP50-95 -> prefer imgsz=640.

## Outcome

- Consolidated best config: `cfg_03_lr0_0.00067_imgsz_768`
- Best checkpoint to use: `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/best.pt`
- Note: `cfg_03` and `cfg_04` tie on reported metrics; `best_config.json` selects `cfg_03` as canonical.

## Next

- Rerun full end-to-end `YoloVitPose_mAP` evaluation using the consolidated detector checkpoint above.
