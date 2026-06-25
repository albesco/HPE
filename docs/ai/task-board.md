# AI Task Board

## Backlog
- Create a script to compare model performance by calculating and plotting per-keypoint OKS.
- Compute common COCO/OKS keypoint AP for YOLO26x-pose and VitPose++ on the same test split before direct model comparison.
- Rerun full-test `YoloVitPose_mAP` metrics with the consolidated pipeline.
- Decide whether YOLO no-detection cases need confidence tuning or fallback behavior.

## In progress

## Done
- Consolidated AI memory: `handoff.md` is deprecated as a source of truth; current state lives in `context.md`, active work in `task-board.md`, decisions in `decision-log.md`, and results in `tests-and-results.md`; current scripts live in `script/`, while `script_old/` is legacy/archive.
- Added `script/overlays/overlay_sample_comparison.py` to compare Yolo26x-Pose and VitPose++ overlays on random Test frames, with original-frame copies and per-model overlay outputs.
- Prepared the SAW_frames YOLO26x-Pose launcher set: `script_old/yolo_training/train_yolo26x_pose_SAW_frames_EntireSwim_20260612.sh`, `script_old/yolo_training/run_yolo26x_pose_SAW_frames_EntireSwim_20260612_tmux.sh`, `script/yolo26x_pose_training/export_yolo_pose_training_report.py`, and the updated `script/yolo26x_pose_training/evaluate_yolo_pose_split.py`; the run will use pretrained `yolo26x-pose.pt`, `imgsz=768`, `lr0=0.00100`, early stopping `patience=3` / `min_delta=0.007`, keep best plus last 10 checkpoints, export `reports/val_metrics_by_epoch.csv`, `loss_by_epoch.png`, `map50_95_by_epoch.png`, and post-train `kp_Test.json`, `metrics_Test.json`, `overlays_Test/` for `data/intermediate/SAW_frames_EntireSwim`.
- Copied the standalone frame-preparation flow into `script/dataset_preparation-cleaning/`, translated the tutorial into Italian, and kept the tmux launcher aligned to `data/input/subset_xyz/Side_above_water_frames` -> `data/intermediate/Side_above_water_frames`.
- Added `script/overlays/GT_KP_overlays.py` for COCO-style GT keypoint overlays from image file/dir inputs and annotation file/dir inputs, with optional red GT bbox drawing.
- Rebuilt `data/intermediate/Side_above_water_EntireSwim/` from the complementary canonical datasets `Side_above_water` and `Side_above_water_VideoTest2`, keeping only samples with all keypoints visible/in-frame; regenerated `_train_canonical`, `_VitPosePP`, `_Yolo26x_detection`, `_Yolo26x_pose`, `manifest.json`, and aggregate totals train `27732`, val `7949`, test `4001`.
- Prepared `script_old/prepare_entire_swim_ab_split.py` and `script_old/run_prepare_entire_swim_ab_split_tmux.sh` to split `data/intermediate/Side_above_water_EntireSwim/` into complementary datasets `..._A` and `..._B` with default probability `A=30%`, symlinked canonical images, regenerated manifests, and rebuilt VitPose++ / YOLO26x detection / YOLO26x pose exports for both outputs.
- Evaluated VitPose++ on deterministic random 300-frame and 50-frame subsets of `Side_above_water_VideoTest2` sampled from Train/Val/Test; subset and config are reproducible from `script_old/prepare_vitposepp_video_test2_subset.py` and `script_old/run_vitposepp_video_test2_random50_eval_tmux.sh`.
- Prepared a deferred VitPose++ launcher/config generator that waits for `runs/hparam_search/vitposepp_huge/best_config.json` before starting a fresh run from `best_AP_epoch_35.pth`.
- Removed the temporary post-grid VitPose++ trigger machinery after grid completion and kept a direct tmux launcher aligned to winner `cfg_02_lr_0.00100_crop_384x128` and best checkpoint `best_AP_epoch_5.pth`.
- Stopped YOLO26x-pose incremental training from `cfg_04` after epoch `29`; best observed Pose mAP50-95 was `0.95705` at epoch `21`.
- Added YOLO26x detector top-1 bbox postprocessing for detector overlays and YOLO->VitPose++ evaluation: highest confidence, then larger area on ties.
- Prepared YOLO26x detection dataset image symlinks and updated the detector launcher for incremental cfg_03 training under `runs/yolo26x_bbox_side_above_water/`.
- Completed YOLO26x detector v2 2x2 grid search and consolidated best detector config `cfg_03_lr0_0.00067_imgsz_768` (val-only) for the VitPose++ pipeline.
- Created VitPose++ huge 2x2 hyperparameter search script with train/val-only selection, horizontal crop-size convention, dry-run validation, per-config artifacts, summaries, report, and best-config selection rules.
- Created YOLO26x-pose 2x2 hyperparameter search script with train/val-only selection, dry-run validation, per-config artifacts, summaries, report, and best-config selection rules.
- Completed YOLO26x-pose 1-epoch smoke training and test evaluation; generated loss and test mAP plots.
- Aligned YOLO26x-pose launcher with VitPose++ checkpoint/log criteria: periodic checkpoints, latest-three retention, `best.pt`/`last.pt`, timestamped log, and run status file.
- Implemented YOLO26x-pose dataset symlink export and a separate `yolo pose train` launcher; regenerated `_Yolo26x_pose` with train `18181`, val `5195`, test `2597` and 56-field pose labels.
- Added label-only model-specific exporters for `_VitPosePP`, `_Yolo26x_detection`, and `_Yolo26x_pose` from `_train_canonical`; verified coordinated split counts and label formats.
- Prepared `data/intermediate/SAW_frames_EntireSwim/` from `data/input/subset_xyz/SAW_frames_EntireSwim/` with the frame-based pipeline, using image symlinks and regenerated `_VitPosePP`, `_Yolo26x_detection`, `_Yolo26x_pose`.
- Prepared Side_above_water YOLO detection dataset from GT bboxes.
- Added dedicated YOLO training scripts under `script_old/yolo_training/`.
- Completed YOLO anisotropic-padding smoke training for 5 epochs; best epoch-5 metrics: precision `0.99227`, recall `0.99249`, mAP50 `0.99266`, mAP50-95 `0.87601`.
- Regenerated Side_above_water VitPose++ and YOLO datasets with anisotropic GT bbox padding: `0.20` horizontal, `0.25` vertical, min `15 px`.
- Added lightweight VitPose++ training status monitor file support.
- Patched VitPose++ resume to tolerate invalid checkpoint metadata config and successfully resumed from `epoch_4.pth`.
- Updated VitPose++ checkpoint retention to keep best validation plus latest three periodic checkpoints.
- Completed VitPose++ incremental training through epoch `40`; best validation checkpoint is `best_AP_epoch_35.pth`, latest is `epoch_40.pth`.
- Generated completed-epoch loss/mAP plots for the full VitPose++ run.
- Invalidated and removed failed previous YOLO+VitPose artifacts whose saved keypoints were not reproducible.
- Consolidated `YoloVitPose_mAP` pipeline and MMPose-only visualization style.
- Generated 20 paired GT-bbox vs current YOLO->VitPose qualitative samples.
- Documented CHAT-TRAINING and CHAT-TRAINING-2 handoff/session state in `docs/ai/`.
- Completed CHAT-TRAINING-2 handoff / context preservation for training and end-to-end evaluation state.
