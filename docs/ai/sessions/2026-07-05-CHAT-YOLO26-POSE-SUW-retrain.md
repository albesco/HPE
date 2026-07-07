# CHAT-YOLO26-POSE-SUW-retrain

- Date: 2026-07-05 UTC
- Role: training / YOLO pose
- Status: active
- Dataset: `data/intermediate/SUW_frames`
- Launcher: `script/yolo26x_pose_training/train_yolo26x_pose_frame.sh`
- tmux session: `train_yolo26x_pose_yolo26x-pose_SUW_frames_20260705`
- Run name: `yolo26x-pose_SUW_frames_20260705`
- Run dir: `runs/yolo26x-pose_SUW_frames_20260705/`
- Output dir: `data/output/experiments/yolo26x-pose_SUW_frames_20260705/`

## Launch

```bash
bash script/yolo26x_pose_training/train_yolo26x_pose_frame.sh --dataset-dir data/intermediate/SUW_frames --run-name SUW_frames_20260705
```

## Effective setup

- Pretrained checkpoint: `models/pose/yolo26x-pose.pt`
- Data YAML: `data/intermediate/SUW_frames/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml`
- `epochs=100`, `imgsz=768`, `lr0=0.001`, `batch=1`, `device=0`, `workers=2`, `optimizer=AdamW`
- External monitor: Pose mAP50-95, `patience=3`, `min_delta=0.007`, keep last `10`

## Status

- Training launched in tmux; metrics pending.
