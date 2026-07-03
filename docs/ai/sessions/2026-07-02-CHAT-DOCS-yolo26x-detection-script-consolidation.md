# Session Note: CHAT-DOCS YOLO26x-Detection script consolidation

## Purpose

Promote the consolidated YOLO26x-Detection training and prediction scripts from `script_old/yolo_training/` into the current operational `script/` tree.

## Changes

- Created `script/yolo26x_detection_training/` for detector training and detector grid-search scripts.
- Created `script/yolo26x_detection_prediction/` for top-1 bbox prediction, detector overlays, label export, and YOLO bbox -> VitPose++ evaluation.
- Kept `script_old/yolo_training/` as historical source only.

## VitPose++ pipeline note

YOLO26x-Detection provides the bbox handoff for the VitPose++ pipeline. The operational YOLO -> VitPose++ bridge is `script/yolo26x_detection_prediction/evaluate_yolo_vitpose_map.py`.

## Validation

`bash -n` passed for `script/yolo26x_detection_training/train_yolo_side_above_water.sh`. `python3 -m py_compile` passed for the Python helpers in `script/yolo26x_detection_training/` and `script/yolo26x_detection_prediction/`.
