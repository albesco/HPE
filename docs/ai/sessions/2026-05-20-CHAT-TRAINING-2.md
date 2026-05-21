# Session 2026-05-20 - CHAT-TRAINING-2

## Scope

Consolidated the YOLO -> VitPose++ inference/evaluation pipeline and removed failed/ambiguous visualization artifacts.

## What was done

- Renamed active prepared dataset root to `data/intermediate/Side_above_water/_train_vitposepp/`.
- Renamed generated config to `data/intermediate/Side_above_water/_train_vitposepp/generated_configs/swimxyz_vitposepp_huge_single_head.py`.
- Renamed consolidated end-to-end script to `script/yolo_training/evaluate_yolo_vitpose_map.py`.
- Renamed active experiment output root to `data/output/experiments/YoloVitPose_mAP/`.
- Removed old custom/mixed keypoint visualization scripts and old failed output artifacts.
- Kept one visualization convention: MMPose `vis_pose_result` predicted skeleton plus red YOLO bbox.
- Preserved the 20-sample qualitative comparison set in `data/output/experiments/YoloVitPose_mAP/test_20260520_0852_best35_vispose_overlays/`.

## Validation

Executed:

```bash
python -m py_compile script/yolo_training/evaluate_yolo_vitpose_map.py script/pose_overlay_utils.py script/yolo_training/prepare_yolo_detection_dataset.py script/prepare_swimxyz_vitposepp_single_head.py
bash -n script/run_resume_side_above_water_to_25ep_tmux.sh
bash -n script/run_train_side_above_water_10ep.sh
bash -n script/yolo_training/train_yolo_side_above_water.sh
```

Results: syntax checks passed.

## Current sample output

`data/output/experiments/YoloVitPose_mAP/test_20260520_0852_best35_vispose_overlays/`

Counts verified:
- `20` GT-bbox reference overlays.
- `20` YOLO-side outputs, including `2` no-detection marker images.

## Not verified

- Full-test mAP for the consolidated `YoloVitPose_mAP` pipeline has not been rerun yet.

## Next step

Run full-test evaluation with `script/yolo_training/evaluate_yolo_vitpose_map.py` and update `docs/ai/tests-and-results.md` plus `docs/ai/experiments/EXP-20260520-YoloVitPose-consolidation.md` with the metrics.
