# Dataset Preparation

Consolidated RAW/frame+label to train-ready dataset preparation flow.

## Main entrypoint

- `prepare_swimxyz_frames_dataset.py`: builds a canonical COCO-style dataset from SwimXYZ frame+label inputs and regenerates VitPose++, YOLO26x detection, and YOLO26x pose exports.

## Supporting exporters/libraries

- `prepare_swimxyz_vitposepp_utils.py`
- `prepare_vitposepp_dataset.py`
- `prepare_yolo_detection_dataset.py`
- `prepare_yolo_pose_dataset.py`

## Docs

- `tutorial.md`
