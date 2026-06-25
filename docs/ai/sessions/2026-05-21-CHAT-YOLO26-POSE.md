# 2026-05-21 - CHAT-YOLO26-POSE

Role: training / YOLO pose.

## Summary

Implemented and validated the initial Ultralytics YOLO26x-pose training setup, separate from YOLO detection and VitPose++.

## Changes

- Updated `script/dataset_preparation/prepare_yolo_pose_dataset.py` to generate YOLO pose labels and expose canonical images with symlinks under `_Yolo26x_pose/images/{train,val,test}`.
- Added `script_old/yolo_training/train_yolo_pose_side_above_water.sh` for `yolo pose train`.
- Updated `script_old/yolo_training/README.md` with YOLO26x-pose preparation and training commands.
- Regenerated `data/intermediate/Side_above_water/_Yolo26x_pose/` from `_train_canonical`.

## Validation

Commands run:

```bash
conda run -n vitpose python -m py_compile script/dataset_preparation/prepare_yolo_pose_dataset.py
bash -n script_old/yolo_training/train_yolo_pose_side_above_water.sh
conda run -n vitpose python script/dataset_preparation/prepare_yolo_pose_dataset.py --overwrite
```

Results:

- Train: `18181` image symlinks, `18181` label files, label field count `{56: 18181}`.
- Val: `5195` image symlinks, `5195` label files, label field count `{56: 5195}`.
- Test: `2597` image symlinks, `2597` label files, label field count `{56: 2597}`.

## Pending

- `yolo26x-pose.pt` was resolved explicitly via `MODEL=yolo26x-pose.pt` for the 1-epoch smoke run; default launcher path remains `models/pose/yolo26x-pose.pt` for local-weight runs.
- Recheck GPU/driver status before a long training run; earlier `nvidia-smi` reported `Driver/library version mismatch`, while `torch.cuda.is_available()` was true.
- Launch training with conservative defaults first (`batch=1`, `imgsz=1280`) or lower `imgsz` if memory is tight.

## Checkpoint and logging policy update

Aligned YOLO26x-pose training with the VitPose++ criteria where Ultralytics supports equivalent behavior:

- `save=True` and `save_period=1` for periodic epoch checkpoints.
- Preserve Ultralytics `weights/best.pt` and `weights/last.pt`.
- Prune periodic `weights/epoch*.pt` to the latest `3` after a successful training run using `script_old/yolo_training/prune_yolo_epoch_checkpoints.py`.
- Write timestamped stdout/stderr logs under `logs/`.
- Write a run `training_status.txt` with phase, config, log path, and exit code.

Additional validation:

```bash
conda run -n vitpose python -m py_compile script_old/yolo_training/prune_yolo_epoch_checkpoints.py
bash -n script_old/yolo_training/train_yolo_pose_side_above_water.sh
```

A `/tmp` pruning test with `epoch0.pt` through `epoch4.pt` retained only `epoch2.pt`, `epoch3.pt`, `epoch4.pt`, plus untouched `best.pt` and `last.pt`.

## User instruction

For future heavy scripts/training runs, use `tmux` by default.

## 1-epoch training and test evaluation

Completed requested YOLO26x-pose smoke training and test evaluation.

Outputs:

- Run: `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap`
- Train log: `logs/yolo26x_pose_side_above_water_20260521_133719.log`
- Test eval log: `logs/yolo26x_pose_test_eval_20260521_1ep.log`
- Checkpoints: `best.pt`, `last.pt`, `epoch0.pt` under `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/weights/`
- Loss plot: `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/reports/loss_epoch1_train_val.png`
- Test mAP plot: `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/reports/map_epoch1_test.png`
- Test metrics CSV: `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/reports/test_metrics_epoch1.csv`

Metrics:

- Validation after epoch 1: Pose mAP50 `0.90037`, Pose mAP50-95 `0.49212`.
- Test split: Pose precision `0.861`, recall `0.902`, mAP50 `0.928`, mAP50-95 `0.537`; Box mAP50 `0.965`, Box mAP50-95 `0.656`.

## Metric semantics note

- VitPose++ `AP` is COCO keypoint OKS AP averaged over `0.50:0.05:0.95`, so it corresponds to the stricter AP/mAP50-95-style metric rather than AP50.
- YOLO26x-pose reports Ultralytics Pose mAP50 and Pose mAP50-95. To compare against VitPose++ rigorously, export YOLO keypoint predictions to COCO result JSON and evaluate with the same COCO/OKS evaluator and test GT.
