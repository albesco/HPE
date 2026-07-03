# script

Consolidated operational scripts for the HPE pose-estimation workflow.

The previous broad script tree was moved to the legacy archive. This directory is now the current operational script root.

## Layout

- `dataset_preparation-cleaning/`: RAW/frame+label to canonical train-ready datasets plus VitPose++/YOLO exports.
- `yolo26x_detection_training/`: YOLO26x-Detection training and detector grid-search scripts.
- `yolo26x_pose_training/`: YOLO26x-Pose training on frame+label datasets.
- `vitpose_training/`: VitPose++ training on frame+label datasets.
- `yolo26x_detection_prediction/`: YOLO26x-Detection top-1 bbox prediction, overlays, and YOLO bbox -> VitPose++ evaluation bridge.
- `yolo26x_pose_prediction/`: YOLO26x-Pose Test prediction, KP JSON, metrics, overlays.
- `vitpose_prediction/`: VitPose++ Test prediction, KP JSON, metrics, overlays.
- `overlays/`: GT overlays and qualitative YOLO/VitPose overlay comparison.
- `hpe_report/`: direct/cross report table generation for the HPE report.
- `plot-metrics/`: metric and checkpoint plotting helpers.

## Legacy

Use the legacy archive only for past versions, variants, or tests that were intentionally not consolidated.
