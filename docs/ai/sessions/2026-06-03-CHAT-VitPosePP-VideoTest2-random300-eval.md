# CHAT-VitPose++ VideoTest2 random300 evaluation

## Goal
- Sample a deterministic 300-frame subset across Train/Val/Test from `data/intermediate/Side_above_water_VideoTest2/_train_canonical/` and evaluate the current VitPose++ winner checkpoint on that subset.

## Prepared assets
- Subset generator: `script_old/prepare_vitposepp_video_test2_subset.py`
- tmux launcher: `script_old/run_vitposepp_video_test2_random50_eval_tmux.sh` with `NUM_FRAMES=300` overrides
- Subset root: `data/intermediate/Side_above_water_VideoTest2/_train_canonical/subsets/random300_seed20260603/`
- Eval work dir: `runs/vitposepp_video_test2_random300_eval_best_AP_epoch_24/`
- Checkpoint used: `runs/vitposepp_side_above_water_grid_winner_resume/best_AP_epoch_24.pth`

## Result
- Selected frames: `300` total (`224` train, `48` val, `28` test)
- AP (mAP50-95): `0.8108414950663122`
- AP50: `0.9778092342413344`
- AP75: `0.889591367785784`
- AR: `0.8480000000000001`

## Notes
- The tmux-launched eval completed normally and wrote `test_predictions.pkl`, `result_keypoints.json`, and `test_eval_stdout.log`.

## Overlay export
- Script: `script/vitpose_training/vitpose_generate_test_overlays_from_json.py`
- Output directory: `data/intermediate/Side_above_water_VideoTest2/_train_canonical/reports/test_overlays/vitposepp_video_test2_random300_best_AP_epoch_24/`
- Result: `300` overlay images generated with GT bboxes and predicted keypoints from `runs/vitposepp_video_test2_random300_eval_best_AP_epoch_24/result_keypoints.json`.
