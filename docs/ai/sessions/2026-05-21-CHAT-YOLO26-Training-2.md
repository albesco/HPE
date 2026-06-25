# 2026-05-21 - CHAT-YOLO26-Training-2

Role: training.

## Summary

Launched incremental YOLO26x-pose training in tmux, resuming from the 1-epoch run `yolo26x_pose_coco17_1ep_testmap` and continuing to a total of 4 epochs.

## Command

- tmux session: `yolo26x_pose_resume_to_4ep`
- run dir: `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/`

Environment variables used:

- `NAME=yolo26x_pose_coco17_1ep_testmap`
- `PROJECT=runs/yolo26x_pose_side_above_water`
- `EPOCHS=4`
- `RESUME=True`
- `MODEL=runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/weights/last.pt`
- `DATA=data/intermediate/Side_above_water/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml`
- `RUN_TEST=True`, `TEST_SPLIT=test`, `TEST_WEIGHTS=best`

## Outputs

- training log (timestamped): `logs/yolo26x_pose_side_above_water_20260521_160620.log`
- run status file: `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/training_status.txt`
- test metrics CSV (after training completes): `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/reports/test_metrics.csv`

## Notes

- The training launcher runs an end-of-run test evaluation by default and records both Pose/Box `mAP50` and `mAP50-95`.

## Plots and overlays

- YOLO26x-pose training curves: runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/results.png and runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/results.csv
- YOLO26x-pose test metrics CSV: runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/reports/test_metrics.csv
- Sampled test images (20): data/output/experiments/yolo26x_pose_side_above_water/test_overlays_20260521_215252/images/
- Predicted keypoint overlays (20): runs/pose/data/output/experiments/yolo26x_pose_side_above_water/test_overlays_20260521_215252/pred/pred/

## Incremental resume to 10 epochs

Launched another incremental resume in tmux to continue from 4 total epochs to 10 total epochs.

- tmux session: yolo26x_pose_resume_to_10ep
- run dir: runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/
- log: logs/yolo26x_pose_side_above_water_20260521_220901.log
- status: runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/training_status.txt

## YOLO26x-pose overlay style update

Added script_old/yolo_training/render_yolo_pose_overlays.py to render YOLO26x-pose qualitative overlays with the same visual convention used by VitPose++ overlays: COCO/MMPose keypoint and skeleton colors, keypoint radius 3, skeleton line thickness 2, and red bbox thickness 3.

Also fixed the final print statements in script/yolo26x_pose_training/evaluate_yolo_pose_split.py so end-of-run test evaluation can print Pose/Box mAP50 and mAP50-95 without a runtime NameError.

Validation:
- conda run -n vitpose --no-capture-output python -m py_compile script_old/yolo_training/render_yolo_pose_overlays.py script/yolo26x_pose_training/evaluate_yolo_pose_split.py
- conda run -n vitpose --no-capture-output python script_old/yolo_training/render_yolo_pose_overlays.py --help

Generated MMPose-style YOLO26x-pose overlays during validation:
- data/output/experiments/yolo26x_pose_side_above_water/overlays_mmpose_style/

## Regenerated YOLO26x-pose overlays from current best checkpoint

Regenerated the 20 MMPose-style YOLO26x-pose overlays using the current best checkpoint from the resumed training run:
- checkpoint: runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/weights/best.pt
- output dir: data/output/experiments/yolo26x_pose_side_above_water/overlays_mmpose_style/
- command: conda run -n vitpose --no-capture-output python script_old/yolo_training/render_yolo_pose_overlays.py --model runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/weights/best.pt --source data/intermediate/Side_above_water/_Yolo26x_pose/images/test --output-dir data/output/experiments/yolo26x_pose_side_above_water/overlays_mmpose_style --max-images 20 --seed 0 --device 0
- result: 20 images, 20 detections

Note: the resume-to-10-epochs tmux training process was still running at generation time, so these overlays use the best checkpoint available at that moment.
