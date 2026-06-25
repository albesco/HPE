# 2026-05-23 - Ricostruisci contesto progetto handoff to Choose Yolo26x detection

Role: scelta degli iperparametri per Yolo26x detection.

## Summary

This session implemented and launched a local 2x2 hyperparameter search for the YOLO26x one-class `swimmer` detector used as the bbox component in the YOLO26x-detector -> VitPose++ huge pipeline.

## Work performed

- Implemented the hyperparameter grid-search script: `script_old/yolo_training/yolo26x_detector_grid2x2.py`.
- Created a dedicated search root and per-config run subdirectories under `runs/hparam_search/`.
- Launched the corrected search in `tmux`.

## Files read

- AI memory core files under `docs/ai/` (see repository memory protocol).
- YOLO training scripts under `script_old/yolo_training/`.

## Files modified or added

- Modified:
  - `docs/ai/handoff.md`
  - `docs/ai/chat-index.md`
  - `docs/ai/tests-and-results.md`
- Added:
  - `docs/ai/sessions/2026-05-23-Ricostruisci contesto progetto-handoff-to-Choose Yolo26x detection.md`

## Commands executed (reconstructible)

- Launch tmux search:

```bash
tmux attach -t yolo26x_detector_grid2x2_v2
```

Note: the exact `tmux new-session ...` invocation is not reconstructible from workspace files; use the script help and the handoff commands.

## Artifacts / checkpoints / logs

Search root (valid):

- `runs/hparam_search/yolo26x_detector_v2/`

Per-config artifacts (example):

- `runs/hparam_search/yolo26x_detector_v2/cfg_01_lr0_0.00067_imgsz_640/results.csv`
- `runs/hparam_search/yolo26x_detector_v2/cfg_01_lr0_0.00067_imgsz_640/weights/best.pt`
- `runs/hparam_search/yolo26x_detector_v2/cfg_01_lr0_0.00067_imgsz_640/weights/last.pt`

## Results / metrics

- Config `cfg_01_lr0_0.00067_imgsz_640` completed (epoch 5):
  - precision(B)=0.98865
  - recall(B)=0.99095
  - mAP50(B)=0.99330
  - mAP50-95(B)=0.87363

Other configs were running/pending at handoff time.

## Issues encountered

- The earlier search root `runs/hparam_search/yolo26x_detector/` produced all-zero metrics with `WARNING: no labels found in detect set`. Those runs must not be used.

## State left to the next chat

- A valid search is in progress under tmux session `yolo26x_detector_grid2x2_v2` and root `runs/hparam_search/yolo26x_detector_v2/`.
- Next chat should wait for all 4 configs to complete, then select best config based on validation recall and secondary criteria.

## Not verified

- Final best config across all 4 grid points (grid not complete yet).
- Downstream end-to-end metrics impact with the chosen detector.

## Not reconstructible from workspace files

- The exact interactive command history and any unpersisted console output.
