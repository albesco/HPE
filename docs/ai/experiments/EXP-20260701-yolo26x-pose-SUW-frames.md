# EXP-20260701-yolo26x-pose-SUW-frames

## Purpose

Train YOLO26x-Pose on the prepared `SUW_frames` dataset.

## Inputs

- Dataset: `data/intermediate/SUW_frames`
- Data YAML: `data/intermediate/SUW_frames/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml`
- Launcher: `script/yolo26x_pose_training/train_yolo26x_pose_frame.sh`
- Pretrained checkpoint: `models/pose/yolo26x-pose.pt`

## Run

- Run name: `yolo26x-pose_SUW_frames_20260701`
- tmux session: `train_yolo26x_pose_yolo26x-pose_SUW_frames_20260701`
- Run dir: `runs/yolo26x-pose_SUW_frames_20260701/`
- Output dir: `data/output/experiments/yolo26x-pose_SUW_frames_20260701/`
- Hyperparameters: `epochs=100`, `imgsz=768`, `lr0=0.001`, `batch=1`, `optimizer=AdamW`, `device=0`, `workers=2`
- Early stop monitor: Pose mAP50-95 with `patience=3`, `min_delta=0.007`
- Checkpoint retention: best plus last `10` periodic checkpoints mirrored to `runs/yolo26x-pose_SUW_frames_20260701/checkpoint/`

## Status

- Launched on 2026-07-01 UTC.
- Metrics pending.
