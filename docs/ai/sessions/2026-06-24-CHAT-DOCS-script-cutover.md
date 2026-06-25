# 2026-06-24 - CHAT-DOCS script cutover

## Purpose

Finalize the safe script cutover: promote the validated consolidated tree to `script/` and move the previous broad tree to `script_old/`.

## Scope

The current operational root is `script/`. The previous broad script tree is now `script_old/` and should be used only for past versions, variants, or tests.

## Consolidated entrypoints

- Dataset preparation RAW/frame+label -> train-ready exports: `script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py`
- YOLO26x-Pose train on frame+label datasets: `script/yolo26x_pose_training/train_yolo26x_pose_frame.sh`
- VitPose++ train on frame+label datasets: `script/vitpose_training/train_vitpose_frame.sh`
- YOLO26x-Pose prediction/KP/overlay: `script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh`
- VitPose++ prediction/KP/overlay: `script/vitpose_prediction/predict_vitpose_frame.sh`
- Overlay comparison/GT overlays: `script/overlays/GT_KP_overlays.py`, `script/overlays/overlay_sample_comparison.py`
- Direct/cross metric report tables: `script/hpe_report/build_hpe_report_tables.py`
- Metric/loss/mAP plotting: `script/plot-metrics/plot_vitpose_metrics_with_checkpoints.py`

## Validation

- `python3 -m py_compile` passed for all Python files under `script/`.
- `bash -n` passed for all shell files under `script/`.
- `--help` smoke checks passed for the main entrypoints listed above.

## Notes

- Internal operational references were updated to the final `script/...` paths where needed.
- Historical `script/` references in old docs/session notes are not source-of-truth for new work.
- Final cutover completed after user approval: use `script/` for current workflows and `script_old/` only for legacy reference.
