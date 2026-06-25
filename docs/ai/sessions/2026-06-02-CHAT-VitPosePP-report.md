# 2026-06-02 — CHAT-VitPosePP-report

## Scope
- Summarize the current VitPose++ training setup and artifacts.
- Produce per-epoch reporting assets if missing.

## Work completed
- Added `script_old/vitpose_epoch_test_report.py` to combine the grid-winner phase and the winner-resume phase, with support for per-checkpoint test evaluation.
- Added `script_old/vitpose_generate_test_overlays.py` to export predicted keypoint overlays on the canonical test split using GT bounding boxes.
- Generated combined validation-only artifacts from existing logs:
  - `data/intermediate/Side_above_water/_train_canonical/reports/training_plots/loss_pose_validation_map_by_epoch__grid_winner_resume.csv`
  - `data/intermediate/Side_above_water/_train_canonical/reports/training_plots/loss_pose_by_epoch__grid_winner_resume.png`
  - `data/intermediate/Side_above_water/_train_canonical/reports/training_plots/validation_map50_95_by_epoch__grid_winner_resume.png`

## Blocker
- Per-checkpoint test-set mAP generation and test-overlay export were attempted but not completed because GPU 0 was effectively full during model load (`RuntimeError: CUDA out of memory`; PyTorch reported ~12 MiB free). No repo-local VitPose training/test process was visible in `ps`, so the contention appears external to this repo session.

## Suggested next step
- Re-run `conda run -n vitpose python script_old/vitpose_epoch_test_report.py` and `conda run -n vitpose python script_old/vitpose_generate_test_overlays.py` once GPU memory is available.
