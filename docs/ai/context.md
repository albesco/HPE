# Shared AI Context

## Current project state
- Goal: train/evaluate VitPose++ on a SwimXYZ subset.
- Environment: VS Code over SSH to Linux server.
- Repository: GitHub.
- Main workstreams:
  - Dataset conversion: SwimXYZ -> VitPose++/MMPose-compatible format.
  - Training: configs, launch scripts, checkpoints, metrics.
  - Workspace Q&A/debugging.
  - Documentation.

## Dataset notes
- Source dataset: SwimXYZ subset.
- Inputs: videos and labels.
- Target format: VitPose++ training format.

## Current conventions
- Python modules should be small and explicit.
- Scripts should expose CLI arguments.
- Paths must not be hardcoded.
- Generated artifacts should go outside Git or under ignored directories.

## Open questions
- Exact SwimXYZ label schema.
- Exact target annotation schema required by the chosen VitPose++ implementation.
- Train/val/test split policy.
- Evaluation metrics.

## Codex sessions
- 2026-05-11 | CHAT-DATASET | SwimXYZ 2 VitPose++ | Role: dataset-conversion (see `docs/ai/chat-index.md`).
- 2026-05-12 | CHAT-TRAINING | VitPose++ Training | Role: training (see `docs/ai/chat-index.md`).

## Training readiness (Side_above_water)
- Prepared dataset root: `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/`
- Generated config: `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/generated_configs/swimxyz_vitposepp_huge_single_head_swap_ears.py`
- Work dir: `runs/vitposepp_single_head_subset_xyz_swap_ears/`
- Helper launcher (train 10 epochs + checkpoint every 5 + plots): `script/run_train_side_above_water_10ep.sh`
- VitPose++ training supports a lightweight current-status file via `--status-file` and `--status-interval`; launchers write `runs/vitposepp_single_head_subset_xyz_swap_ears/training_status.txt`.
- Active VitPose++ anisotropic GT training: tmux session `vitpose_side_above_water_aniso_30ep`, work dir `runs/vitposepp_side_above_water_aniso_20x25_min15/`, status file `runs/vitposepp_side_above_water_aniso_20x25_min15/training_status.txt`, checkpoints every epoch, total epochs `30`.
- Current active training status as of 2026-05-14 17:03 UTC: epoch `3/30`, checkpoints available through `epoch_2.pth`.

## YOLO detector preparation (Side_above_water)
- Goal: train a one-class `swimmer` detector from GT bboxes, then use YOLO bboxes before VitPose++ inference.
- Scripts: `script/yolo_training/prepare_yolo_detection_dataset.py`, `script/yolo_training/train_yolo_side_above_water.sh`
- Generated YOLO dataset: `data/intermediate/Side_above_water/_yolo_detection/`
- Generated data YAML: `data/intermediate/Side_above_water/_yolo_detection/swimxyz_side_above_water_yolo.yaml`
- Current VitPose++ GT annotations use bbox padding ratio `0.20` before train/val/test split.
- Current VitPose++ GT annotations use anisotropic bbox padding: horizontal `0.20`, vertical `0.25`, minimum `15 px` per side.
- YOLO conversion reads those already padded GT bboxes with `bbox_padding_ratio=0.0`; do not add padding again downstream unless explicitly retraining/evaluating a new convention.
- YOLO diagnostic: `yolo26x.pt` with `imgsz=1280` and `batch=8` fails on the 32 GB V100 with CUDA OOM; launcher defaults were reduced to `batch=2`, `workers=2`, with persistent logs under `logs/`.
- YOLO training `yolo26x_swimmer_gt_bbox_padded10_20ep` was stopped after epoch 11 was recorded in `results.csv`.
- Frozen checkpoints are in `runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox_padded10_20ep/frozen_checkpoints/`: `best_epoch10_user_20260514_070621.pt`, `last_epoch11_user_20260514_070621.pt`, and `results_stopped_20260514_070621.csv`.
- Epoch-10/best validation metrics: precision `0.985`, recall `0.986`, mAP50 `0.992`, mAP50-95 `0.872`; validation output directory `runs/detect/runs/yolo_side_above_water/val_best_epoch10_user_20260514_070621/`.
- Random validation bbox previews from the frozen epoch-10/best YOLO checkpoint were generated, then removed during intermediate workspace cleanup.
- Padding-20 GT dataset was generated with train `18181`, val `5195`, test `2597`; YOLO dataset was regenerated from it with no extra padding.
- YOLO padding-20 smoke training `yolo26x_swimmer_gt_bbox_padded20_5ep` was stopped before completion to change the bbox padding convention.
- Anisotropic-padding GT dataset was generated with train `18181`, val `5195`, test `2597`; YOLO dataset was regenerated from it with no extra padding.
- YOLO anisotropic-padding smoke training completed for 5 epochs: run `runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox_aniso_20x25y_min15_5ep/`, weights `weights/best.pt` and `weights/last.pt`.
- YOLO anisotropic epoch-5 validation metrics: precision `0.99227`, recall `0.99249`, mAP50 `0.99266`, mAP50-95 `0.87601`.

## VitPose++ qualitative preview note
- Previous custom preview inference passed `xyxy` coordinates into MMPose `_box2cs`, which expects `xywh`; this caused visually bad predicted keypoint overlays despite GT bboxes.
- Fixed custom preview inference and centralized overlay helpers in `script/pose_overlay_utils.py`; all active keypoint preview scripts should use this module so bbox format and skeleton drawing stay consistent.
- Updated `script/compare_test_overlays.py`, `script/preview_test_predictions.py`, and `script/visualize_gt_vs_pred_keypoints.py` to use the shared utility.
- Corrected GT-vs-pred overlays are in `data/output/preview/gt_vs_pred_keypoints_fixed_bbox_format/`; consolidation smoke-test outputs are in `data/output/preview/consistency_check/`.

## GT bbox visual checks
- Added bbox-only visualization script: `script/visualize_gt_bboxes.py`.
- Generated GT bbox overlays for visual checks, then removed intermediate preview directories during cleanup.

## Intermediate workspace state
- `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/`: active VitPose++ dataset, about `15G` after removing non-annotation `*_with-KP.jpg` previews.
- `data/intermediate/Side_above_water/_yolo_detection/`: current YOLO dataset derived from anisotropic GT bboxes, about `224M`.
- `data/intermediate/Side_above_water/_converted/`: conversion intermediate retained for possible dataset regeneration, about `3.4G`.
- Removed obsolete visual preview directories from `data/intermediate/`: `bbox_val`, `bbox_val_padded20_sample50`, `bbox_val_anisotropic_sample50`, `epoch_10_yolo_bbox`, `yolo_aniso_ep5_test_bbox_sample20`.
