# 2026-06-13 - CHAT-YOLO26-POSE SAW frames 20260612

## Scope
- Adapt the last useful YOLO26x-Pose training launcher for `data/intermediate/SAW_frames_EntireSwim`.
- Keep the same pretrained-start / cfg04-style setup, but add the SAW_frames Val report and Test export outputs.

## Prepared files
- `script_old/yolo_training/train_yolo26x_pose_SAW_frames_EntireSwim_20260612.sh`
- `script_old/yolo_training/run_yolo26x_pose_SAW_frames_EntireSwim_20260612_tmux.sh`
- `script/yolo26x_pose_training/export_yolo_pose_training_report.py`
- Updated `script/yolo26x_pose_training/evaluate_yolo_pose_split.py` with optional metrics JSON export.

## Runtime contract
- Training uses pretrained `models/pose/yolo26x-pose.pt`.
- Training uses `imgsz=768`, `lr0=0.00100`, early stopping on Pose mAP50-95 with `patience=3` and `min_delta=0.007`.
- Checkpoint retention keeps `best.pt` plus the last 10 periodic epoch checkpoints.
- Val outputs go to `runs/yolo26x-pose_SAW_frames_EntireSwim_20260612/reports/`.
- Test outputs go to `data/output/experiments/yolo26x-pose_SAW_frames_EntireSwim_20260612/`.

## Validation
- `bash -n script_old/yolo_training/train_yolo26x_pose_SAW_frames_EntireSwim_20260612.sh`
- `bash -n script_old/yolo_training/run_yolo26x_pose_SAW_frames_EntireSwim_20260612_tmux.sh`
- `python -m py_compile script/yolo26x_pose_training/export_yolo_pose_training_report.py script/yolo26x_pose_training/evaluate_yolo_pose_split.py`
- Report helper tested on `runs/yolo26x-pose_A_20260605/results.csv` and wrote a CSV plus loss/mAP PNGs under `/tmp/yolo26x_pose_report_test/`.
