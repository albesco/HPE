# AI Handoff

Last updated: 2026-05-20 UTC  
Current session: `CHAT-TRAINING-2`  
Logical title: VitPose++ Training 2  
Role: training

## Current state

- VitPose++ Side_above_water training is complete at epoch `40`.
- Recommended VitPose++ checkpoint: `runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth`.
- Active prepared dataset root: `data/intermediate/Side_above_water/_train_vitposepp/`.
- Active config: `data/intermediate/Side_above_water/_train_vitposepp/generated_configs/swimxyz_vitposepp_huge_single_head.py`.
- Consolidated end-to-end experiment name: `YoloVitPose_mAP`.
- Consolidated pipeline script: `script/yolo_training/evaluate_yolo_vitpose_map.py`.

## What changed in this session

- Consolidated the YOLO -> VitPose++ pipeline.
- Renamed active dataset/config/pipeline references to the consolidated VitPose++ naming convention.
- Renamed the end-to-end experiment root to `data/output/experiments/YoloVitPose_mAP/`.
- Removed failed/ambiguous previous YOLO+VitPose artifacts and old preview folders.
- Removed obsolete YOLO+VitPose rendering/preview scripts and custom GT-vs-pred keypoint visualization scripts.
- Kept one visualization convention: MMPose `vis_pose_result` predicted skeleton plus red YOLO bbox only.

## Consolidated pipeline

Data flow:
1. YOLO predicts swimmer bbox as absolute `xyxy` on the full frame.
2. The pipeline converts YOLO `xyxy` to COCO `xywh`.
3. VitPose++ receives full image path plus `xywh` bbox.
4. MMPose performs top-down crop/affine internally.
5. Outputs are COCO keypoint results plus optional MMPose overlays.

Use:

```bash
conda run -n vitpose python script/yolo_training/evaluate_yolo_vitpose_map.py \
  --split test \
  --yolo-model runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox_aniso_20x25y_min15_5ep/weights/best.pt \
  --vitpose-checkpoint runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth \
  --overlay-count 20 \
  --seed 20260520 \
  --output-root data/output/experiments/YoloVitPose_mAP
```

## Current qualitative sample

Directory:

`data/output/experiments/YoloVitPose_mAP/test_20260520_0852_best35_vispose_overlays/`

Contents:
- `20` GT-bbox reference overlays.
- `18` current YOLO -> VitPose++ overlays.
- `2` YOLO no-detection marker images at `conf=0.25`.

## Validation and results

- No valid full-test mAP is currently recorded for the consolidated `YoloVitPose_mAP` pipeline.
- Previous failed end-to-end AP should not be used; its saved keypoints were not reproducible from identical YOLO bboxes and current VitPose++ checkpoint.
- The next required validation is a fresh full-test run with `script/yolo_training/evaluate_yolo_vitpose_map.py`.

## Files changed in consolidation

- `script/yolo_training/evaluate_yolo_vitpose_map.py`
- `script/pose_overlay_utils.py`
- `script/prepare_swimxyz_vitposepp_single_head.py`
- `script/run_resume_side_above_water_to_25ep_tmux.sh`
- `script/run_train_side_above_water_10ep.sh`
- `script/yolo_training/prepare_yolo_detection_dataset.py`
- `data/intermediate/Side_above_water/_train_vitposepp/generated_configs/swimxyz_vitposepp_huge_single_head.py`
- `docs/ai/context.md`
- `docs/ai/task-board.md`
- `docs/ai/chat-index.md`
- `docs/ai/tests-and-results.md`
- `docs/ai/experiments/EXP-20260520-YoloVitPose-consolidation.md`

## Removed obsolete files/artifacts

- Old custom/preview keypoint visualization scripts.
- Old result-JSON overlay renderer for the invalidated run.
- Previous failed YOLO+VitPose output root.
- Old generated preview folders under `data/output/preview/`.

## Next step

Run the full consolidated `YoloVitPose_mAP` test evaluation and update `docs/ai/tests-and-results.md` plus a new/updated experiment note with the generated metrics.
