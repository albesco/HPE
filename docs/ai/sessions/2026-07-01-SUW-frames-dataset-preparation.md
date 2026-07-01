# 2026-07-01 - SUW_frames Dataset Preparation

## Scope

Prepare raw `data/input/subset_xyz/SUW_frames` for training with VitPose++, YOLO26x detection, and YOLO26x pose.

## Commands

Launched in tmux via:

```bash
SESSION_NAME=prepare_SUW_frames_dataset \
INPUT_ROOT_REL=data/input/subset_xyz/SUW_frames \
OUTPUT_ROOT_REL=data/intermediate/SUW_frames \
VITPOSE_WORK_DIR_REL=runs/vitposepp_SUW_frames \
LOG_REL=logs/prepare_SUW_frames_dataset_20260701_181926.log \
bash script/dataset_preparation-cleaning/run_prepare_swimxyz_frames_dataset_tmux.sh
```

## Preflight

- Raw images: `15000`.
- Raw `__COCO__2D_cam.txt` labels: `15000`.
- Output root did not exist before launch.
- Static checks passed:
  - `bash -n script/dataset_preparation-cleaning/run_prepare_swimxyz_frames_dataset_tmux.sh`
  - `python -m py_compile script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py`

## Current Status

- tmux session: `prepare_SUW_frames_dataset`.
- Log: `logs/prepare_SUW_frames_dataset_20260701_181926.log`.
- Output root: `data/intermediate/SUW_frames`.
- Status: completed.
- Accepted samples: `8634/15000`.
- Rejected pairs: `6366` with `no_valid_keypoints`.
- Split counts: train `6044`, val `1727`, test `863`.
- YOLO26x detection labels: all rows have `5` fields.
- YOLO26x pose labels: all rows have `56` fields.

## Expected Outputs

- `data/intermediate/SUW_frames/_train_canonical`
- `data/intermediate/SUW_frames/_VitPosePP`
- `data/intermediate/SUW_frames/_Yolo26x_detection`
- `data/intermediate/SUW_frames/_Yolo26x_pose`
