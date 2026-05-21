# EXP-20260514 - YOLO anisotropic bbox padding smoke training

## Status

Completed 5 epochs.

## Goal

Validate YOLO detector training using the current anisotropic GT bbox convention.

## Dataset

Current YOLO dataset:

`data/intermediate/Side_above_water/_yolo_detection/`

Derived from VitPose++ GT annotations with:
- horizontal padding: `0.20`
- vertical padding: `0.25`
- minimum: `15 px` per side
- YOLO-side padding: `0.0`

## Run

`runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox_aniso_20x25y_min15_5ep/`

## Weights

- `weights/best.pt`
- `weights/last.pt`

## Validation metrics

- precision: `0.99227`
- recall: `0.99249`
- mAP50: `0.99266`
- mAP50-95: `0.87601`

## Interpretation

The anisotropic bbox convention is viable for YOLO detector training.

## Follow-up

Use YOLO-predicted bboxes before VitPose++ inference/evaluation and compare against GT bbox pipeline.