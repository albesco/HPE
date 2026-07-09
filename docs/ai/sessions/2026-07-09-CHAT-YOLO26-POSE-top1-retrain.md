# 2026-07-09 - CHAT-YOLO26-POSE top-1 retrain

- Updated YOLO26x-Pose train/eval/prediction scripts so validation, exported keypoints, and overlays use only the highest-confidence detection per frame (`max_det=1`).
- Launched tmux session `train_yolo26x_pose_yolo26x-pose_SUW_frames_20260709_top1` on `data/intermediate/SUW_frames`.
- Run directory: `runs/yolo26x-pose_SUW_frames_20260709_top1/`.
- Output directory: `data/output/experiments/yolo26x-pose_SUW_frames_20260709_top1/`.
- Settings: pretrained `models/pose/yolo26x-pose.pt`, `imgsz=768`, `lr0=0.001`, `min_epochs=20`, `patience=3`, `min_delta=0.007`, `keep_last=10`.
