# 2026-05-23 - hyper-choose VitPose++

Role: Hyperparameters Setup for VitPose++.

## Summary

Created a local VitPose++ huge 2x2 hyperparameter search script for the Side_above_water top-down YOLO26x-detector -> VitPose++ pipeline.

## Method

- Script: `script_old/hparam_search/vitposepp_huge_grid2x2.py`
- Output root: `runs/hparam_search/vitposepp_huge/`
- Baseline config: `data/intermediate/Side_above_water/_train_canonical/generated_configs/swimxyz_vitposepp_huge.py`
- Pretrained weights: `models/pose/wholebody.pth`
- Search grid:
  - `cfg_01_lr_0.00067_crop_384x128`
  - `cfg_02_lr_0.00100_crop_384x128`
  - `cfg_03_lr_0.00067_crop_512x128`
  - `cfg_04_lr_0.00100_crop_512x128`
- Each run trains for `5` epochs.
- Selection uses validation Keypoint AP@[OKS 0.50:0.95] only; test split is not used.
- Crop convention: requested crop is width x height; MMPose `image_size=[W,H]`; ViT backbone `img_size=(H,W)`.
- Bboxes: canonical padded GT bboxes via `use_gt_bbox=True`; detector bbox files are `not reconstructible from workspace files`.

## Outputs per config

Each config directory is designed to contain `effective_config.py`, `command.txt`, `training_args.json`, `stdout_stderr.log` after real training starts, `status.json`, `validation_metrics.json` after training, and retained best/latest checkpoints.

Search-level outputs include `summary.csv`, `summary.json`, `report.md`, `best_config.json`, `search_args.json`, and `static_validation.json`.

## Validation

Executed:

```bash
python -m py_compile script_old/hparam_search/vitposepp_huge_grid2x2.py
python script_old/hparam_search/vitposepp_huge_grid2x2.py --dry-run
python -m py_compile runs/hparam_search/vitposepp_huge/cfg_*/effective_config.py
conda run -n vitpose --no-capture-output python /tmp/validate_vitposepp_grid_configs.py
```

Results:

- Static validation passed.
- Dry-run wrote artifacts under `runs/hparam_search/vitposepp_huge/`.
- Verified generated configs use `optimizer.lr` plus crop-related fields only as varied hyperparameters.
- `mmcv.Config` validation confirmed optimizer is constant except lr and `train_pipeline` is inherited unchanged.
- No training was launched.

## Next step

Launch the full search in `tmux` when ready:

```bash
tmux new-session -d -s vitposepp_huge_grid2x2 'cd /home/albertosco/HPE && python script_old/hparam_search/vitposepp_huge_grid2x2.py'
```
