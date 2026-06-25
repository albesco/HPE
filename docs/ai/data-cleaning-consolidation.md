# Data Cleaning Consolidation

This note preserves the useful outcome of the initial SwimXYZ data-cleaning work.
It replaces the older root-level `note.md`, which mixed durable data-cleaning
decisions with operational tests that are no longer part of the current workflow.

## Consolidated Value

The initial data-cleaning work remains important because it established the
canonical SwimXYZ -> COCO17 preparation assumptions used by the current training
pipeline:

- convert SwimXYZ image coordinates to image-space coordinates by flipping the
  vertical axis;
- swap `LEye/REye` and `LEar/REar` before COCO17 recoding;
- treat SwimXYZ `z` as a spatial coordinate, not keypoint confidence;
- reconstruct visibility from horizontal image position;
- derive bounding boxes from valid keypoints and apply the current padding
  convention in canonical preparation;
- log anomalous bbox jumps during dataset preparation;
- keep model-specific exporters downstream of `_train_canonical`, rather than
  letting each model independently choose/drop frames.

## Current Canonical Pipeline

Current source-of-truth files:

- `docs/ai/context.md`
- `docs/ai/decision-log.md`
- `docs/ai/task-board.md`
- `docs/ai/tests-and-results.md`
- `script_old/prepare_swimxyz_vitposepp_train.py`
- `script_old/prepare_swimxyz_vitposepp.py`
- `script/dataset_preparation-cleaning/prepare_swimxyz_vitposepp_utils.py`
- `script/dataset_preparation-cleaning/prepare_vitposepp_dataset.py`
- `script/dataset_preparation-cleaning/prepare_yolo_detection_dataset.py`
- `script/dataset_preparation-cleaning/prepare_yolo_pose_dataset.py`

The active dataset preparation flow is:

1. normalize SwimXYZ inputs into `_converted`;
2. create the canonical accepted frame/keypoint dataset in `_train_canonical`;
3. export model-specific label/config outputs into `_VitPosePP`,
   `_Yolo26x_detection`, and `_Yolo26x_pose`.

## Deprecated Parts Of The Old Note

Operational smoke tests and early verification commands from the old root
`note.md` are intentionally not preserved here as active instructions.

VitPose++ training now uses official/pretrained checkpoints documented in the
current training memory and run notes, so old local setup tests are historical
only. Future sessions should use the current `docs/ai/` memory and experiment
notes instead of treating the old note as an operational runbook.
