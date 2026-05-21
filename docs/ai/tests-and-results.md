# Tests and Results

This file tracks validation commands, training/evaluation outcomes, and important metrics.

## Latest known results

| Date | Area | Run | Result | Notes |
|---|---|---|---|---|
| 2026-05-20 | YOLO+VitPose++ consolidation | `YoloVitPose_mAP` | consolidated 20-image qualitative sample | output `data/output/experiments/YoloVitPose_mAP/test_20260520_0852_best35_vispose_overlays/`; `20` GT-bbox references, `18` YOLO->VitPose overlays, `2` YOLO no-detection markers at `conf=0.25` |
| 2026-05-20 | YOLO+VitPose++ diagnostics | previous failed YOLO+VitPose run | invalidated and removed | old saved keypoints were not reproducible from identical YOLO bboxes and current VitPose++ checkpoint; do not use old AP `0.6269` result |
| 2026-05-19 | VitPose++ plots | `vitposepp_side_above_water_aniso_20x25_min15` | generated completed-epoch loss/mAP plots | outputs timestamped `20260519_230159_completed_epochs`; includes validation points through epoch 40 |
| 2026-05-16 | VitPose++ | `vitposepp_side_above_water_aniso_20x25_min15` | finished target epoch `40` | best checkpoint `best_AP_epoch_35.pth`; latest `epoch_40.pth`; mild plateau/early overfitting after epoch 35 |
| 2026-05-14 | YOLO detector | `yolo26x_swimmer_gt_bbox_aniso_20x25y_min15_5ep` | completed 5 epochs | best epoch-5 metrics: precision `0.99227`, recall `0.99249`, mAP50 `0.99266`, mAP50-95 `0.87601` |

Related experiment notes:
- `docs/ai/experiments/EXP-20260520-YoloVitPose-consolidation.md`
- `docs/ai/experiments/EXP-20260514-yolo-aniso-5ep.md`
- `docs/ai/experiments/EXP-20260514-vitpose-aniso-stopped-epoch4.md`

## YOLO metrics

| Run | Epoch | Precision | Recall | mAP50 | mAP50-95 | Train Box Loss | Val Box Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `yolo26x_swimmer_gt_bbox_aniso_20x25y_min15_5ep` | 5 | `0.99227` | `0.99249` | `0.99266` | `0.87601` | `0.72344` | `0.58282` |

## VitPose++ validation metrics

| Run | Epoch | AP | AP50 | AP75 | AP(M) | AP(L) | AR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vitposepp_side_above_water_aniso_20x25_min15` | 5 | `0.8604` | `0.9900` | `0.9457` | `0.4083` | `0.8696` | `0.8952` |
| `vitposepp_side_above_water_aniso_20x25_min15` | 10 | `0.9424` | `0.9901` | `0.9697` | `0.4999` | `0.9498` | `0.9544` |
| `vitposepp_side_above_water_aniso_20x25_min15` | 15 | `0.9549` | `0.9901` | `0.9799` | `0.5934` | `0.9612` | `0.9649` |
| `vitposepp_side_above_water_aniso_20x25_min15` | 20 | `0.9727` | `0.9901` | `0.9900` | `0.7892` | `0.9768` | `0.9799` |
| `vitposepp_side_above_water_aniso_20x25_min15` | 25 | `0.9739` | `0.9901` | `0.9900` | `0.8261` | `0.9762` | `0.9818` |
| `vitposepp_side_above_water_aniso_20x25_min15` | 30 | `0.9776` | `0.9901` | `0.9901` | `0.8500` | `0.9806` | `0.9842` |
| `vitposepp_side_above_water_aniso_20x25_min15` | 35 | `0.9821` | `0.9901` | `0.9900` | `0.8444` | `0.9842` | `0.9880` |
| `vitposepp_side_above_water_aniso_20x25_min15` | 40 | `0.9812` | `0.9901` | `0.9901` | `0.7993` | `0.9844` | `0.9871` |

## YOLO+VitPose++ end-to-end metrics

No valid full-test metric is currently recorded for the consolidated `YoloVitPose_mAP` pipeline. Rerun `script/yolo_training/evaluate_yolo_vitpose_map.py` before reporting end-to-end AP.

## Qualitative outputs

- Consolidated sample: `data/output/experiments/YoloVitPose_mAP/test_20260520_0852_best35_vispose_overlays/`
- Visualization convention: MMPose `vis_pose_result` predicted skeleton plus red YOLO bbox only.

## Generated plots

Latest completed-epoch plots:
- `data/intermediate/Side_above_water/_train_vitposepp/reports/training_plots/loss_epoch_avg__20260519_230159_completed_epochs.png`
- `data/intermediate/Side_above_water/_train_vitposepp/reports/training_plots/mAP_validation__20260519_230159_completed_epochs.png`
- `data/intermediate/Side_above_water/_train_vitposepp/reports/training_plots/loss_map_summary__20260519_230159_completed_epochs.csv`

## Validation commands

```bash
python -m py_compile script/yolo_training/evaluate_yolo_vitpose_map.py script/pose_overlay_utils.py
```

## Errors and diagnostics

- YOLO OOM diagnostic: `logs/yolo_diagnostic_1280_b8_20260513_135040.log`; `yolo26x.pt`, `imgsz=1280`, `batch=8` exhausted the 32 GB V100.
- VitPose++ resume diagnostic: failed resume attempts are in `runs/vitposepp_side_above_water_aniso_20x25_min15/20260514_184736.log` and `runs/vitposepp_side_above_water_aniso_20x25_min15/20260514_190748.log`; fixed by tolerating invalid checkpoint metadata config during resume.
- Previous failed YOLO+VitPose result was invalidated: saved keypoints diverged from recomputed YOLO->VitPose keypoints despite identical YOLO bboxes.
