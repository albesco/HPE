# 2026-05-21 - CHAT-DATASET-2

Logical title: Dataset conversion (data cleaning & preparation)
Role: dataset-conversion
Predecessor: `Elenca file e cartelle`

## Scope

Verified and implemented coordinated model-specific dataset exporters from the canonical SwimXYZ COCO keypoint dataset.

## What was done

- Confirmed `_train_canonical` is the coordinated source of accepted frames.
- Added `script/dataset_preparation/prepare_vitposepp_dataset.py` to export VitPose++ annotations/config without image operations.
- Updated `script/dataset_preparation/prepare_yolo_detection_dataset.py` to export YOLO26x detection labels only into `_Yolo26x_detection`.
- Added `script/dataset_preparation/prepare_yolo_pose_dataset.py` to export YOLO26x pose labels into `_Yolo26x_pose`.
- Updated `script_old/yolo_training/train_yolo_side_above_water.sh` to use the new detection YAML.

## Outputs

- `data/intermediate/Side_above_water/_VitPosePP/`
- `data/intermediate/Side_above_water/_Yolo26x_detection/`
- `data/intermediate/Side_above_water/_Yolo26x_pose/`

All exporters report `image_operations: none`.

## Validation

- `python -m py_compile script/dataset_preparation/prepare_vitposepp_dataset.py script/dataset_preparation/prepare_yolo_detection_dataset.py script/dataset_preparation/prepare_yolo_pose_dataset.py` passed.
- Generated all three outputs with `--overwrite`.
- Verified train/val/test counts: `18181 / 5195 / 2597`.
- Detection labels have exactly `5` fields per row.
- Pose labels have exactly `56` fields per row.
- Detection and pose label stems exactly match canonical image stems; missing/extra counts are `0 / 0` for every split.

## Notes

The canonical preparation script still materializes images and decides frame inclusion. The downstream VitPose++/YOLO26x detection/YOLO26x pose exporters now operate only on labels/config derived from that canonical frame set.
