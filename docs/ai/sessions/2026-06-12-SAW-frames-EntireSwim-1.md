# 2026-06-12 - SAW-frames-EntireSwim-1

Logical title: SAW frames EntireSwim preparation
Role: data preparation / frames
Status: completed

## Scope

Extend the standalone frame-preparation pipeline so it can ingest the `SAW_frames_EntireSwim` layout and regenerate the standard canonical and model-specific outputs.

## What was done

- Updated `script_old/cleaning_frames/prepare_swimxyz_frames/prepare_swimxyz_frames_dataset.py` to accept `.png`, `.jpg`, and `.jpeg` frame images.
- Added support for both `__frame_` and `__frm_` frame index tokens in standalone frame filenames.
- Ran the frame-based pipeline on `data/input/subset_xyz/SAW_frames_EntireSwim/` with image symlinks and standard exporters enabled.

## Outputs

- `data/intermediate/SAW_frames_EntireSwim/_train_canonical/`
- `data/intermediate/SAW_frames_EntireSwim/_VitPosePP/`
- `data/intermediate/SAW_frames_EntireSwim/_Yolo26x_detection/`
- `data/intermediate/SAW_frames_EntireSwim/_Yolo26x_pose/`

## Validation

- `python3 -m py_compile script_old/cleaning_frames/prepare_swimxyz_frames/prepare_swimxyz_frames_dataset.py script/dataset_preparation/prepare_vitposepp_dataset.py script/dataset_preparation/prepare_yolo_detection_dataset.py script/dataset_preparation/prepare_yolo_pose_dataset.py` passed.
- Canonical preparation report: `data/intermediate/SAW_frames_EntireSwim/_train_canonical/reports/swimxyz_frames_preparation_report.json`.
- Source pairs total: `14984`; accepted samples total: `14984`; rejected pairs: none.
- Canonical split counts: train `10489`, val `2997`, test `1498`.
- Exporter reports were produced for `_VitPosePP`, `_Yolo26x_detection`, and `_Yolo26x_pose`.
