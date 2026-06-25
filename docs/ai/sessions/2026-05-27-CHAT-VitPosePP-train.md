# 2026-05-27 - CHAT-VitPose++ train

Role: training / VitPose++

## Summary

Prepared the next VitPose++ training launch without starting it, then smoke-tested the deferred launcher path.

## What was prepared

- Added `script_old/hparam_search/prepare_vitposepp_grid_best_resume.py` to generate a fresh VitPose++ config only after `runs/hparam_search/vitposepp_huge/best_config.json` contains a completed winner.
- Added `script_old/run_train_side_above_water_from_grid_best_tmux.sh` to launch the prepared config in tmux later, with status-file and final-test behavior aligned to the existing VitPose++ workflow.
- Chose `load_from=<winner checkpoint from runs/hparam_search/vitposepp_huge/...>` instead of `--resume-from` so the grid-selected learning rate and crop size apply cleanly in a new run/work dir while continuing from the selected grid winner itself.

## Validation

Planned static validation:

```bash
conda run -n vitpose python -m py_compile script_old/hparam_search/prepare_vitposepp_grid_best_resume.py
bash -n script_old/run_train_side_above_water_from_grid_best_tmux.sh
```

No training was launched.

## Next step

Wait for `runs/hparam_search/vitposepp_huge/best_config.json` to become `status=completed`, then launch:

```bash
script_old/run_train_side_above_water_from_grid_best_tmux.sh
```

## Smoke test


- Added `script_old/hparam_search/trigger_vitposepp_post_grid.py` so the post-grid preparation can be driven directly by the completed VitPose++ grid result.
- Added `script_old/hparam_search/monitor_vitpose_patience.py` so the generated tmux launcher can enforce AP-based `patience=5` despite the lack of a native MMCV/YOLO-style patience flag in the current VitPose++ workflow.
- Temporarily simulated a completed launcher path with stubbed `tmux` to verify end-to-end readiness without starting real training.
- Confirmed generated config/work dir and final train command are valid.
- Verified the actual `runs/hparam_search/vitposepp_huge/best_config.json` now records winner `cfg_02_lr_0.00100_crop_384x128`, so the launcher no longer needs to wait for grid completion.
