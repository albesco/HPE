# YOLO26x-Detection prediction

Operational scripts for detector inference, top-1 bbox export, detector overlays, and the YOLO bbox -> VitPose++ evaluation bridge.

## Entry points

- `render_yolo_detection_overlays.py`: renders one selected bbox per image.
- `export_yolo_detection_top1_labels.py`: exports top-1 detector labels with confidence.
- `evaluate_yolo_vitpose_map.py`: runs YOLO detector bboxes through VitPose++ and evaluates COCO keypoint mAP.
- `preview_yolo_bbox_predictions.py`: legacy-style random bbox preview helper.
- `yolo_detection_utils.py`: shared top-1 bbox selection rule.

## Pipeline convention

The detector keeps exactly one bbox per frame: highest confidence wins, with larger area as tie-breaker. This is the bbox handoff convention for the VitPose++ prediction pipeline.

## Example

```bash
conda run -n vitpose python script/yolo26x_detection_prediction/render_yolo_detection_overlays.py \
  --model runs/yolo26x_bbox_side_above_water/yolo26x-detection_from_cfg03_ep5_20260523_1923/weights/best.pt \
  --source-dir data/intermediate/Side_above_water/_Yolo26x_detection/images/test \
  --output-dir data/output/experiments/yolo26x_detection_overlays/top1_test20
```
