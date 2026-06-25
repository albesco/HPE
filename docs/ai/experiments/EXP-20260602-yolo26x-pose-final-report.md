# EXP-20260602: YOLO26x-Pose final training report

## Inputs
- Grid run: `runs/hparam_search/yolo26x_pose/cfg_04_lr0_0.00100_imgsz_768/`
- Incremental run: `runs/pose/runs/yolo26x_pose_side_above_water/yolo26x-pose-incremental-from-cfg04/`
- Dataset YAML: `data/intermediate/Side_above_water/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml`
- Final checkpoint: `runs/pose/runs/yolo26x_pose_side_above_water/yolo26x-pose-incremental-from-cfg04/weights/best.pt`

## Script
- `script_old/yolo_training/report_yolo26x_pose_final_training.py`

## Outputs
- Report dir: `runs/pose/runs/yolo26x_pose_side_above_water/yolo26x-pose-incremental-from-cfg04/reports/final_training/`
- Base CSV: `loss_pose_map50_95_by_epoch.csv`
- Loss plot: `loss_pose_by_epoch.png`
- Validation mAP plot: `val_map50_95_by_epoch.png`
- Test mAP plot target: `test_map50_95_by_epoch.png`
- Test overlay directory target: `test_overlays_best/`

## Status
- Objective changed to validation metrics only; heavy tmux test job `yolo26x_pose_final_report_20260602` was stopped on 2026-06-02.
- Final Val table combines grid cfg_04 epochs 1-5 and incremental epochs 1-29.
- Best Val Pose mAP50-95 is `0.95705` at total epoch `26` / incremental epoch `21`.
