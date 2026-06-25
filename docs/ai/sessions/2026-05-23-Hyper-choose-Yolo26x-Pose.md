# 2026-05-23 - Hyper-choose Yolo26x-Pose

Role: Hyperparameters Setup for Yolo26x-Pose.

## Summary

Created a local YOLO26x-pose 2x2 hyperparameter search script for the Side_above_water swimmer pose task.

## Method

- Script: `script_old/yolo_training/yolo26x_pose_grid2x2.py`
- Output root: `runs/hparam_search/yolo26x_pose/`
- Dataset YAML: `data/intermediate/Side_above_water/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml`
- Model: `yolo26x-pose.pt`
- Search grid:
  - `cfg_01_lr0_0.00067_imgsz_640`
  - `cfg_02_lr0_0.00100_imgsz_640`
  - `cfg_03_lr0_0.00067_imgsz_768`
  - `cfg_04_lr0_0.00100_imgsz_768`
- Each run trains for `5` epochs.
- Selection uses validation only; test split is not used.
- Checkpoint resume is disabled for this turn because prior checkpoints are not compatible.
- Periodic epoch checkpoints are disabled with `save_period=-1`; each run should keep only `weights/best.pt` and `weights/last.pt` unless Ultralytics produces an additional mandatory artifact.

## Optimizer note

Prior YOLO logs show `optimizer=auto` ignores `lr0`. The search script sets `optimizer=AdamW`, matching the optimizer type selected automatically in prior YOLO26x-pose logs, so the grid's `lr0` values are actually applied.

## Outputs per config

Each config directory is designed to contain `command.txt`, `training_args.json`, copied `data.yaml`, `stdout_stderr.log`, `status.json`, `validation_metrics.json`, Ultralytics outputs, `weights/best.pt`, `weights/last.pt`, and no intermediate checkpoint unless produced mandatorily by Ultralytics.

Search-level outputs include `summary.csv`, `summary.json`, `report.md`, `best_config.json`, `search_args.json`, and `static_validation.json`.

## Validation

Executed:

```bash
python -m py_compile script_old/yolo_training/yolo26x_pose_grid2x2.py
python script_old/yolo_training/yolo26x_pose_grid2x2.py --dry-run
```

Results:

- Static validation passed.
- Dry-run wrote artifacts under `runs/hparam_search/yolo26x_pose/`.
- Verified only `lr0` and `imgsz` vary across the four generated training commands; `save_period=-1` is constant across all configs.
- No training was launched.

## Next step

Launch the full search in `tmux` when ready:

```bash
tmux new-session -d -s yolo26x_pose_grid2x2 'cd /home/albertosco/HPE && python script_old/yolo_training/yolo26x_pose_grid2x2.py'
```

## Launch state

Launched the full 2x2 search in a detached tmux session.

- tmux session: `yolo26x_pose_grid2x2`
- Started: `2026-05-23T20:29:48+00:00`
- Search root: `runs/hparam_search/yolo26x_pose/`
- First active config: `cfg_01_lr0_0.00067_imgsz_640`
- Status file: `runs/hparam_search/yolo26x_pose/cfg_01_lr0_0.00067_imgsz_640/status.json`
- Log file: `runs/hparam_search/yolo26x_pose/cfg_01_lr0_0.00067_imgsz_640/stdout_stderr.log`
- Command uses `split=val`, `save_period=-1`, `optimizer=AdamW`, `resume=False`.
- Concurrent process observed: `yolo26x_det_train_from_cfg03_ep5_20260523_1923` also appears active on `device=0`; monitor for GPU contention or OOM.

## Interruption state

The full search was interrupted by user request to avoid GPU contention with a concurrent YOLO detection training job on `device=0`.

- Interrupted config: `cfg_01_lr0_0.00067_imgsz_640`
- No `results.csv`, `weights/best.pt`, or `weights/last.pt` existed at interruption time.
- The interrupted config cannot resume from an epoch checkpoint; it should be rerun from scratch.
- Status file was updated to `interrupted`.
- Detection training was left running.

## Detection-triggered relaunch

A watcher was installed in `tmux` session `resume_yolo26x_pose_after_detection` to launch the pose grid after YOLO26x detection training completed.

- Watcher log: `logs/resume_yolo26x_pose_after_detection_20260523_2048.log`
- Detection run watched: `yolo26x-detection_from_cfg03_ep5_20260523_1923`
- Trigger fired at `2026-05-23T20:53:24Z`.
- Pose grid command: `python script_old/yolo_training/yolo26x_pose_grid2x2.py --rerun-running --rerun-failed`
- Active config after relaunch: `cfg_01_lr0_0.00067_imgsz_640`
- Active status: `runs/hparam_search/yolo26x_pose/cfg_01_lr0_0.00067_imgsz_640/status.json`
- The grid is running inside the watcher tmux session; attach with `tmux attach -t resume_yolo26x_pose_after_detection`.

## Post-consolidation notes

- Update (2026-05-24): the 2x2 grid evaluation winner is `cfg_04_lr0_0.00100_imgsz_768` (model 04).
