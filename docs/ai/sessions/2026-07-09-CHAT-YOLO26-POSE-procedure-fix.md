# CHAT-YOLO26-POSE procedure fix

- Date: 2026-07-09 UTC
- Role: training / YOLO pose
- Scope: fix YOLO26x-Pose train procedure after an early-stop retry evaluated Ultralytics `best.pt` instead of the external Pose mAP50-95 best.

## Changes

- `script/yolo26x_pose_training/monitor_yolo_pose_patience.py` now processes all new `results.csv` rows, supports `--min-epochs`, and copies the metric-best checkpoint to `weights/best_map50_95_pose.pt`.
- `script/yolo26x_pose_training/train_yolo26x_pose_frame.sh` defaults to `patience=8`, `min_epochs=20`, keeps `min_delta=0.007`, mirrors `best_map50_95_pose.pt` into `checkpoint/`, and evaluates Test on the metric-best checkpoint before falling back to Ultralytics `best.pt`.

## Validation

- `bash -n script/yolo26x_pose_training/train_yolo26x_pose_frame.sh` passed.
- `python3 -m py_compile script/yolo26x_pose_training/monitor_yolo_pose_patience.py` passed.
- `--help` checks passed for both updated entrypoints.
- Non-GPU monitor smoke test selected the expected metric-best checkpoint from a synthetic `results.csv`.
