# 2026-06-04 - Image-type-split-1

Logical title: Image type split 1
Role: data preparation
Status: completed

## Goal
- Rebuild `data/intermediate/Side_above_water_EntireSwim/` from the complementary datasets `data/intermediate/Side_above_water` and `data/intermediate/Side_above_water_VideoTest2`, keeping only samples with all 17 keypoints visible and inside the image.
- Regenerate the model-specific training exports for VitPose++, YOLO26x detection, and YOLO26x pose.

## What changed
- Replaced `script_old/prepare_entire_swim_dataset.py` with a full rebuild workflow that consumes both source datasets, rebuilds `_train_canonical`, writes root/canonical manifests, and calls the three downstream exporter scripts.
- Added `script_old/run_prepare_entire_swim_dataset_tmux.sh` as the standard tmux launcher for the heavy rebuild.
- Deleted the previous `data/intermediate/Side_above_water_EntireSwim/` tree and rebuilt it from scratch in tmux session `prepare_entire_swim_dataset_full`.

## Result
- Aggregate canonical root: `data/intermediate/Side_above_water_EntireSwim/_train_canonical/`
- Export roots: `_VitPosePP`, `_Yolo26x_detection`, `_Yolo26x_pose` under `data/intermediate/Side_above_water_EntireSwim/`
- Copy mode: `symlink`
- Source `Side_above_water` accepted: train `13674`, val `3933`, test `2041`
- Source `Side_above_water_VideoTest2` accepted: train `14058`, val `4016`, test `1960`
- Aggregate totals: train `27732`, val `7949`, test `4001`

## Files
- `script_old/prepare_entire_swim_dataset.py`
- `script_old/run_prepare_entire_swim_dataset_tmux.sh`
- `data/intermediate/Side_above_water_EntireSwim/manifest.json`
- `data/intermediate/Side_above_water_EntireSwim/_train_canonical/reports/entire_swim_preparation_report.json`
- `data/intermediate/Side_above_water_EntireSwim/_VitPosePP/`
- `data/intermediate/Side_above_water_EntireSwim/_Yolo26x_detection/`
- `data/intermediate/Side_above_water_EntireSwim/_Yolo26x_pose/`

## Follow-up
- Added `script_old/prepare_entire_swim_ab_split.py` to split `data/intermediate/Side_above_water_EntireSwim/` into complementary datasets `data/intermediate/Side_above_water_EntireSwim_A/` and `data/intermediate/Side_above_water_EntireSwim_B/`.
- The split is reproducible through `--seed`, uses default `--prob-a 0.3`, preserves Train/Val/Test separation, recreates canonical image symlinks, rewrites manifests/reports, and regenerates `_VitPosePP`, `_Yolo26x_detection`, `_Yolo26x_pose` for both outputs.
- Added `script_old/run_prepare_entire_swim_ab_split_tmux.sh` as the standard launcher for the heavy split run; it is ready but has not been launched in this session.
