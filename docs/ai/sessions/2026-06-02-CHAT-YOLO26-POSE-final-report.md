# 2026-06-02 — CHAT-YOLO26-POSE final report

## Scope
- User requested final YOLO26x-Pose training parameters, model/script links, per-epoch Loss-Pose and test mAP50-95 table/plots, and test-set keypoint overlays.

## Confirmed source
- Final YOLO26x-Pose training starts from `cfg_04_lr0_0.00100_imgsz_768`, not cfg_02.
- Final run: `runs/pose/runs/yolo26x_pose_side_above_water/yolo26x-pose-incremental-from-cfg04/`.
- Best observed validation Pose mAP50-95 remains incremental epoch `21`.

## Work done
- Added `script_old/yolo_training/report_yolo26x_pose_final_training.py`.
- Generated immediate report artifacts under `runs/pose/runs/yolo26x_pose_side_above_water/yolo26x-pose-incremental-from-cfg04/reports/final_training/`:
  - `loss_pose_map50_95_by_epoch.csv`
  - `loss_pose_by_epoch.png`
  - `val_map50_95_by_epoch.png`
- Launched tmux session `yolo26x_pose_final_report_20260602` to evaluate test mAP50-95 per available checkpoint and render test overlays.

## Important limitation
- The grid run retained only `best.pt` and `last.pt`; therefore per-epoch test metrics for grid epochs 1-4 cannot be produced from existing artifacts. Test metrics can be generated from grid epoch 5 and the incremental epoch checkpoints.

## Objective update
- User clarified that Loss-Pose and mAP over epochs should be validation metrics, not test metrics.
- Stopped tmux session `yolo26x_pose_final_report_20260602`; no full test-per-checkpoint report is required.
- Val artifacts are complete under `runs/pose/runs/yolo26x_pose_side_above_water/yolo26x-pose-incremental-from-cfg04/reports/final_training/`.

## Test report script update
- Updated `script/yolo26x_pose_training/evaluate_yolo_pose_split.py` to optionally export predicted bbox/keypoint JSON and VitPose++-style overlays, in addition to mAP CSV.
- Validation: `conda run -n vitpose python -m py_compile script/yolo26x_pose_training/evaluate_yolo_pose_split.py` passed.
- Full Test run is prepared but not launched in this update.
