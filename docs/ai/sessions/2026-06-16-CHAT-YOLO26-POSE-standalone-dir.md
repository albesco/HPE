# 2026-06-16 - CHAT-YOLO26-POSE standalone dir

## Scope
- Create a standalone copy of the frame-based YOLO26x-Pose training pipeline under `script/yolo26x_pose_training/`.
- Keep the launcher, monitor, report export, evaluator, and tutorial self-contained in that directory.

## Added directory
- `script/yolo26x_pose_training/`

## Files copied
- `train_yolo26x_pose_frame.sh`
- `monitor_yolo_pose_patience.py`
- `export_yolo_pose_training_report.py`
- `evaluate_yolo_pose_split.py`
- `tutorial.md`

## Outcome
- The copied launcher now resolves its helper scripts from the same directory.
- The copied tutorial now documents direct launch and tmux launch/management from `script/yolo26x_pose_training/`.
