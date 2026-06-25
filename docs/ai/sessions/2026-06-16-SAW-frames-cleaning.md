# 2026-06-16 - SAW frames cleaning

Logical title: SwimXYZ frames preparation
Role: data preparation
Status: completed

## Goal
- Move the standalone frame-preparation flow into `script/dataset_preparation-cleaning/`.
- Translate the tutorial into Italian and add tmux launch/management instructions.

## What changed
- Copied the frame dataset builder and tmux launcher into `script/dataset_preparation-cleaning/`.
- Added `script/dataset_preparation-cleaning/tutorial.md` in Italian.
- Updated `docs/ai/tutorials/tutorial_train_yolo26x_pose_frame.md` and the YOLO README to reference the new directory.

## Validation
- `python -m py_compile script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py`
- `bash -n script/dataset_preparation-cleaning/run_prepare_swimxyz_frames_dataset_tmux.sh`
- Visual inspection of the translated tutorial and updated references
