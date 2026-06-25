# 2026-06-10 - cleaning_frames prepare_swimxyz_frames

Logical title: prepare_swimxyz_frames
Role: data preparation
Status: completed

## Goal
- Prepare a new dataset builder for standalone frame+label pairs under `data/input/subset_xyz/Side_above_water_frames/`.
- Recreate the standard dataset structure used by `Side_above_water`, including `_train_canonical`, `_VitPosePP`, `_Yolo26x_detection`, and `_Yolo26x_pose`.

## What changed
- Added `script_old/cleaning_frames/prepare_swimxyz_frames/prepare_swimxyz_frames_dataset.py`.
- Added `script_old/cleaning_frames/prepare_swimxyz_frames/run_prepare_swimxyz_frames_dataset_tmux.sh`.
- The builder expects `COCO__2D_cam` `.png + .txt` pairs, reuses the standard bbox/keypoint conversion rules, and writes canonical image symlinks before calling the existing exporters.

## Validation
- `python -m py_compile script_old/cleaning_frames/prepare_swimxyz_frames/prepare_swimxyz_frames_dataset.py`
- `bash -n script_old/cleaning_frames/prepare_swimxyz_frames/run_prepare_swimxyz_frames_dataset_tmux.sh`
- `python script_old/cleaning_frames/prepare_swimxyz_frames/prepare_swimxyz_frames_dataset.py --help`
