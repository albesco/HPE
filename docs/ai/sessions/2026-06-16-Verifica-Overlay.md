# 2026-06-16 Verifica overlay V-Y

- Role: data analysis.
- Diagnosed that the previous VitPose++ `kp_Test.json` for `vitpose_SAW_frames_20260615` had been generated on `data/intermediate/SAW_frames_EntireSwim/_train_canonical`, while YOLO26x-Pose used `data/intermediate/SAW_frames/_train_canonical`.
- Updated `script_old/train_vitpose_SAW_frames_20260615.sh` so `--train-dataset-root` and `--test-dataset-root` are launch parameters, with test defaulting to train.
- Ran VitPose++ Test on `data/intermediate/SAW_frames/_train_canonical/test2017` using `runs/vitpose_SAW_frames_20260615/checkpoint/best_AP_epoch_18.pth`.
- Outputs regenerated under `data/output/experiments/vitpose_SAW_frames_20260615/`: `result_keypoints.json`, `kp_Test.json`, `metrics_Test.json`, and `overlays_Test/`.
- Result: `1359` Test images, AP `0.93288`, AP50 `0.97968`, AP75 `0.94886`, AR `0.93804`.

## VitPose launcher consolidation

- Added `script/vitpose_training/train_vitpose_frame.sh` as the consolidated parametric VitPose++ launcher.
- Added local helper scripts under `script/vitpose_training/`: `monitor_vitpose_patience.py`, `export_vitpose_val_metrics.py`, `plot_vitpose_training_log.py`, `vitpose_generate_test_overlays_from_json.py`, and `pose_overlay_utils.py`.
- Added tutorial `script/vitpose_training/tutorial_train_vitpose_frame.md` with description, parameters, defaults, and examples.
- Important convention: `--dataset-dir` points to the dataset root, e.g. `data/intermediate/SAW_frames`; the launcher finds `_train_canonical` and `_VitPosePP` internally.
- `--test-dataset-dir` defaults to `--dataset-dir`, preventing the previous hidden `SAW_frames` vs `SAW_frames_EntireSwim` Test mismatch.
- Static validation passed: `bash -n`, helper `py_compile`, `--help`, and rejection of `_train_canonical` as `--dataset-dir`.
- Updated `script/vitpose_training/tutorial_train_vitpose_frame.md` with normalized Markdown tables and a tmux launch/management section.
