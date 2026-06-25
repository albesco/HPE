# EXP-20260527: YOLO26x-pose incremental from cfg04

## Summary

Incremental YOLO26x-pose training from the selected grid winner `cfg_04_lr0_0.00100_imgsz_768`.

## Inputs

- Dataset YAML: `data/intermediate/Side_above_water/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml`
- Initial checkpoint: `runs/hparam_search/yolo26x_pose/cfg_04_lr0_0.00100_imgsz_768/weights/best.pt`
- Run directory: `runs/pose/runs/yolo26x_pose_side_above_water/yolo26x-pose-incremental-from-cfg04/`
- Results file: `runs/pose/runs/yolo26x_pose_side_above_water/yolo26x-pose-incremental-from-cfg04/results.csv`

## Configuration

- `epochs=100`
- `patience=10`
- `imgsz=768`
- `batch=1`
- `lr0=0.00100`
- `optimizer=AdamW`

## Status

- Training is not active as of 2026-05-27.
- Last recorded epoch in `results.csv`: `29`.
- No active `tmux` session or YOLO/VitPose process was observed during memory consolidation on 2026-05-27.

## Best Observed Result

Best by Pose mAP50-95 in `results.csv`:

| Epoch | Pose Precision | Pose Recall | Pose mAP50 | Pose mAP50-95 | Box mAP50 | Box mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|
| `21` | `0.98101` | `0.98191` | `0.99211` | `0.95705` | `0.99242` | `0.89588` |

Last recorded epoch:

| Epoch | Pose Precision | Pose Recall | Pose mAP50 | Pose mAP50-95 | Box mAP50 | Box mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|
| `29` | `0.98516` | `0.98633` | `0.99193` | `0.95271` | `0.99217` | `0.90251` |

## Artifacts

- Plot: `runs/pose/runs/yolo26x_pose_side_above_water/yolo26x-pose-incremental-from-cfg04/map_plot.png`
- Checkpoints: `weights/best.pt`, `weights/last.pt`, and periodic `epoch*.pt` files under the run directory.

## Notes

- This experiment should be treated as a stopped run, not an active training job.
- Direct comparison with VitPose++ still requires a common COCO/OKS evaluator on the same test split.
