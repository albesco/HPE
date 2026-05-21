# Shared AI Context

## Current project state
- Goal: train/evaluate VitPose++ on a SwimXYZ Side_above_water subset and evaluate realistic YOLO-bbox -> VitPose++ inference.
- Environment: VS Code over SSH to Linux server.
- Repository: GitHub.
- Main workstreams:
  - Dataset conversion: SwimXYZ -> VitPose++/MMPose-compatible format.
  - Training: configs, launch scripts, checkpoints, metrics.
  - Detection + pose evaluation: YOLO bbox -> VitPose++ keypoints -> COCO mAP.
  - Documentation and reproducibility memory under `docs/ai/`.

## Dataset notes
- Source dataset: SwimXYZ subset.
- Prepared dataset root: `data/intermediate/Side_above_water/_train_vitposepp/`.
- Target format: COCO-style keypoint annotations for MMPose/VitPose++.
- Current GT bbox convention: anisotropic padding, horizontal `0.20`, vertical `0.25`, minimum `15 px` per side, clipped to image boundaries.
- YOLO conversion reads the already padded GT bboxes with no extra padding; do not add YOLO-side/downstream padding unless a new experiment explicitly changes convention.

## Current conventions
- Python modules should be small and explicit.
- Scripts should expose CLI arguments.
- Paths must not be hardcoded.
- Generated artifacts should go outside Git or under ignored directories.
- For current VitPose++ training, retain best validation checkpoint plus latest three periodic checkpoints (`max_keep_ckpts=3`).

## Codex sessions
- 2026-05-11 | CHAT-DATASET | SwimXYZ 2 VitPose++ | Role: dataset-conversion.
- 2026-05-12 | CHAT-TRAINING | VitPose++ Training | Role: training.
- 2026-05-15/19 | CHAT-TRAINING-2 | VitPose++ Training 2 | Role: training.

## VitPose++ training state
- Active work dir: `runs/vitposepp_side_above_water_aniso_20x25_min15/`.
- Generated config: `data/intermediate/Side_above_water/_train_vitposepp/generated_configs/swimxyz_vitposepp_huge_single_head.py`.
- Final completed training status: phase `finished`, target `40` epochs, status timestamp `2026-05-16 17:19:43 UTC`.
- Final log: `runs/vitposepp_side_above_water_aniso_20x25_min15/20260516_100449.log`.
- Latest periodic checkpoint: `runs/vitposepp_side_above_water_aniso_20x25_min15/epoch_40.pth`.
- Latest symlink: `runs/vitposepp_side_above_water_aniso_20x25_min15/latest.pth -> epoch_40.pth`.
- Best validation checkpoint: `runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth`.
- Retained periodic checkpoints visible: `epoch_38.pth`, `epoch_39.pth`, `epoch_40.pth`.
- Older best retained: `best_AP_epoch_30.pth`.
- Mild plateau / early overfitting signal after epoch 35: AP `0.9821` at epoch 35, AP `0.9812` at epoch 40 while loss continued decreasing.
- Recommended VitPose++ checkpoint for downstream use: `best_AP_epoch_35.pth`.

## VitPose++ validation metrics
- Epoch 5 AP `0.8604`
- Epoch 10 AP `0.9424`
- Epoch 15 AP `0.9549`
- Epoch 20 AP `0.9727`
- Epoch 25 AP `0.9739`
- Epoch 30 AP `0.9776`
- Epoch 35 AP `0.9821`
- Epoch 40 AP `0.9812`

Latest plots:
- `data/intermediate/Side_above_water/_train_vitposepp/reports/training_plots/loss_epoch_avg__20260519_230159_completed_epochs.png`
- `data/intermediate/Side_above_water/_train_vitposepp/reports/training_plots/mAP_validation__20260519_230159_completed_epochs.png`
- `data/intermediate/Side_above_water/_train_vitposepp/reports/training_plots/loss_map_summary__20260519_230159_completed_epochs.csv`

## YOLO detector state
- Current YOLO detector run: `runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox_aniso_20x25y_min15_5ep/`.
- Weights: `weights/best.pt`, `weights/last.pt`.
- Epoch-5 validation metrics: precision `0.99227`, recall `0.99249`, mAP50 `0.99266`, mAP50-95 `0.87601`.
- YOLO OOM diagnostic: `logs/yolo_diagnostic_1280_b8_20260513_135040.log`; `yolo26x.pt`, `imgsz=1280`, `batch=8` exhausted 32 GB V100.

## End-to-end YOLO+VitPose++ pipeline
- Consolidated experiment name: `YoloVitPose_mAP`.
- Consolidated script: `script/yolo_training/evaluate_yolo_vitpose_map.py`.
- Pipeline: YOLO predicts absolute `xyxy` bbox on the full frame; code converts bbox to COCO `xywh`; VitPose++ receives full image + `xywh` bbox; MMPose performs the top-down crop/affine internally; COCO keypoint mAP evaluates predictions.
- Visualization convention: use MMPose `vis_pose_result` for predicted keypoints/skeleton and draw only the YOLO bbox in red; do not use custom fuchsia/GT-mixed skeleton renderers for YOLO+VitPose outputs.
- YOLO checkpoint: `runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox_aniso_20x25y_min15_5ep/weights/best.pt`.
- VitPose++ checkpoint: `runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth`.
- Current qualitative comparison set: `data/output/experiments/YoloVitPose_mAP/test_20260520_0852_best35_vispose_overlays/` (`20` GT-bbox references plus `20` YOLO->VitPose outputs; `2` YOLO no-detection markers at `conf=0.25`).
- Previous failed YOLO+VitPose outputs and saved result JSON were invalidated because saved keypoints were not reproducible with the current bbox/model path; those failed experiment artifacts were removed.

## Open questions / next work
- Rerun the full consolidated `YoloVitPose_mAP` test evaluation with `script/yolo_training/evaluate_yolo_vitpose_map.py` to regenerate metrics from the corrected pipeline.
- Use the consolidated visualization directory to inspect remaining YOLO no-detection cases and decide whether to tune YOLO confidence/fallback behavior.
