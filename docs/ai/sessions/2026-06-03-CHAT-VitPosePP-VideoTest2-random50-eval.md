# CHAT-VitPose++ VideoTest2 random50 evaluation

## Goal
- Sample a deterministic 50-frame subset across Train/Val/Test from `data/intermediate/Side_above_water_VideoTest2/_train_canonical/` and evaluate the current VitPose++ winner checkpoint on that subset.

## Prepared assets
- Subset generator: `script_old/prepare_vitposepp_video_test2_subset.py`
- tmux launcher: `script_old/run_vitposepp_video_test2_random50_eval_tmux.sh`
- Subset root: `data/intermediate/Side_above_water_VideoTest2/_train_canonical/subsets/random50_seed20260603/`
- Eval work dir: `runs/vitposepp_video_test2_random50_eval_best_AP_epoch_24/`
- Checkpoint used: `runs/vitposepp_side_above_water_grid_winner_resume/best_AP_epoch_24.pth`

## Result
- Selected frames: `50` total (`39` train, `4` val, `7` test)
- AP (mAP50-95): `0.8354957059367897`
- AP50: `1.0`
- AP75: `0.9233146792940162`
- AR: `0.858`

## Notes
- The tmux-launched eval completed normally and wrote `test_predictions.pkl`, `result_keypoints.json`, and `test_eval_stdout.log`.
