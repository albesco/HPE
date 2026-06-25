# 2026-06-15 - CHAT-YOLO26-POSE SAW rename

## Scope
- Align the SAW frame-based YOLO26x-Pose launcher, tutorial, and saved run config with the renamed `yolo26x-pose_SAW_frames_20260614` run/output convention.

## Updated files
- `script/yolo26x_pose_training/train_yolo26x_pose_frame.sh`
- `script_old/yolo_training/tutorial.md`
- `runs/yolo26x-pose_SAW_frames_20260614/args.yaml`

## Outcome
- The frame-based launcher now prefixes bare `--run-name` values with `yolo26x-pose_`.
- The saved run config now reports `name: yolo26x-pose_SAW_frames_20260614` and the matching `save_dir`.
- Tutorial examples now reference `runs/yolo26x-pose_SAW_frames_20260614/`.
