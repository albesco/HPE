# EXP-20260520 - YoloVitPose pipeline consolidation

## Purpose

Consolidate the correct YOLO -> VitPose++ pipeline and remove failed/ambiguous visualization artifacts.

## Consolidated pipeline

- Script: `script/yolo_training/evaluate_yolo_vitpose_map.py`
- Experiment root: `data/output/experiments/YoloVitPose_mAP/`
- Dataset root: `data/intermediate/Side_above_water/_train_vitposepp/`
- Config: `data/intermediate/Side_above_water/_train_vitposepp/generated_configs/swimxyz_vitposepp_huge_single_head.py`
- YOLO checkpoint: `runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox_aniso_20x25y_min15_5ep/weights/best.pt`
- VitPose++ checkpoint: `runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth`

Data flow:
1. YOLO predicts swimmer bbox as absolute `xyxy` on the full frame.
2. Pipeline converts `xyxy` to COCO `xywh`.
3. VitPose++ receives the full image plus `xywh` bbox.
4. MMPose performs the top-down crop/affine internally.
5. Evaluation writes COCO keypoint result JSON and optional MMPose overlays.

## Consolidated visualization

Use only MMPose `vis_pose_result` for predicted keypoints/skeleton and draw only the YOLO bbox in red. Removed custom/mixed GT-vs-pred skeleton visualizations from the consolidated YOLO->VitPose workflow.

## Outputs

Qualitative sample:

`data/output/experiments/YoloVitPose_mAP/test_20260520_0852_best35_vispose_overlays/`

Contents:
- `20` GT-bbox reference overlays.
- `18` current YOLO->VitPose++ overlays.
- `2` current YOLO no-detection marker images at `conf=0.25`.

## Invalidated artifacts

Removed previous failed/ambiguous artifacts:
- the previous failed YOLO+VitPose output root
- old generated preview folders under `data/output/preview/`
- old result-rendering and custom visualization scripts.

Reason: the previous saved keypoints were not reproducible from identical YOLO bboxes and the current VitPose++ checkpoint, while recomputed YOLO-bbox and GT-bbox keypoints are nearly identical for checked samples.

## Next step

Run full-test metrics with the consolidated pipeline before reporting end-to-end AP:

```bash
conda run -n vitpose python script/yolo_training/evaluate_yolo_vitpose_map.py \
  --split test \
  --yolo-model runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox_aniso_20x25y_min15_5ep/weights/best.pt \
  --vitpose-checkpoint runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth \
  --overlay-count 20 \
  --seed 20260520 \
  --output-root data/output/experiments/YoloVitPose_mAP
```
