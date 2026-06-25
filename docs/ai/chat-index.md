# Chat Index (Codex Sessions)

This file tracks Codex chat sessions for this repository.

## Active / Recent
### CHAT-DOCS-ai-memory-consolidation
- Logical title: AI memory consolidation
- Role: documentation / AI memory maintenance
- Status: completed
- Purpose: simplify `docs/ai` memory, deprecate long operational handoff notes, and clarify source-of-truth files
- Outcome: `handoff.md` is now a short compatibility pointer; current state, tasks, decisions, and results are consolidated in their dedicated core files; `script/` remains operational and `script_old/` legacy/archive
- Key files:
  - `docs/ai/start-here.md`
  - `docs/ai/context.md`
  - `docs/ai/task-board.md`
  - `docs/ai/tests-and-results.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/handoff.md`
  - `docs/ai/sessions/2026-06-24-CHAT-DOCS-ai-memory-consolidation.md`
- Started: 2026-06-24 (UTC)
- Completed: 2026-06-24 (UTC)

### CHAT-DOCS-script-cutover
- Logical title: Script cutover
- Role: documentation / repository organization
- Status: completed
- Purpose: finalize the consolidated operational `script/` tree and move the previous broad script tree to `script_old/`
- Outcome: final cutover complete; `script/` contains organized dataset preparation, YOLO26x-Pose training/prediction, VitPose++ training/prediction, overlays, HPE report, and plot-metrics entrypoints; `script_old/` keeps the previous broad tree; static validation and `--help` smoke checks passed
- Key files:
  - `script/README.md`
  - `script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py`
  - `script/yolo26x_pose_training/train_yolo26x_pose_frame.sh`
  - `script/vitpose_training/train_vitpose_frame.sh`
  - `script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh`
  - `script/vitpose_prediction/predict_vitpose_frame.sh`
  - `script/overlays/GT_KP_overlays.py`
  - `script/overlays/overlay_sample_comparison.py`
  - `script/hpe_report/build_hpe_report_tables.py`
  - `script/plot-metrics/plot_vitpose_metrics_with_checkpoints.py`
  - `docs/ai/sessions/2026-06-24-CHAT-DOCS-script-cutover.md`
- Started: 2026-06-24 (UTC)
- Completed: 2026-06-24 (UTC)

### CHAT-DOCS-HPE-report-metrics
- Logical title: HPE report direct/cross metrics memory
- Role: documentation / AI memory maintenance
- Status: completed
- Purpose: consolidate the report-table workflow for direct and cross train metrics from the local MD and PPTX into project memory
- Outcome: added an AI runbook for `script/hpe_report/build_hpe_report_tables.py`, recorded the report inputs/outputs/default thresholds, and captured the key direct/cross AP results from the PPTX
- Key files:
  - `docs/ai/runbooks/hpe-report-direct-cross-metrics.md`
  - `docs/ai/context.md`
  - `docs/ai/tests-and-results.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/handoff.md`
  - `docs/ai/sessions/2026-06-24-CHAT-DOCS-HPE-report-metrics.md`
  - `script/hpe_report/build_hpe_report_tables.py`
  - `script/hpe_report/7) Diagrammi sulle metriche dirette e cross.md`
  - `script/hpe_report/20260617_Report_Fine-Tuning_Senza-Immagini.pptx`
- Started: 2026-06-24 (UTC)
- Completed: 2026-06-24 (UTC)

### Predictions_Yolo-Vitpose
- Logical title: Predizioni Y e V
- Role: Test / prediction
- Status: completed
- Purpose: add parametric YOLO26x-Pose and VitPose++ prediction launchers for Test split metrics, KP JSON export, and overlays
- Outcome: `script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh` now wraps the existing evaluator and can run full or sampled Test predictions from any selected checkpoint/dataset
- Key files:
  - `script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh`
  - `script/vitpose_prediction/predict_vitpose_frame.sh`
  - `script/yolo26x_pose_training/evaluate_yolo_pose_split.py`
  - `docs/ai/sessions/2026-06-18-Predictions-Yolo-Vitpose.md`
- Started: 2026-06-18 (UTC)
- Completed: 2026-06-18 (UTC)

### SAW-frames-cleaning
- Logical title: SwimXYZ frames preparation
- Role: data preparation
- Status: completed
- Purpose: move the standalone frame-preparation flow into `script/dataset_preparation-cleaning/` and translate the tutorial into Italian with tmux launch/management notes
- Outcome: `script/dataset_preparation-cleaning/` now contains the copied builder and launcher; `script/dataset_preparation-cleaning/tutorial.md` and `docs/ai/tutorials/tutorial_train_yolo26x_pose_frame.md` are in Italian and document tmux usage
- Key files:
  - `script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py`
  - `script/dataset_preparation-cleaning/run_prepare_swimxyz_frames_dataset_tmux.sh`
  - `script/dataset_preparation-cleaning/tutorial.md`
  - `docs/ai/tutorials/tutorial_train_yolo26x_pose_frame.md`
  - `docs/ai/sessions/2026-06-16-SAW-frames-cleaning.md`
- Started: 2026-06-16 (UTC)
- Completed: 2026-06-16 (UTC)

### CHAT-YOLO26-POSE-standalone-dir
- Logical title: YOLO26x-Pose standalone training directory
- Role: training / YOLO pose
- Status: completed
- Purpose: copy the frame-based YOLO26x-Pose launcher and its required helpers into a standalone `script/yolo26x_pose_training/` directory and document tmux usage there
- Outcome: `script/yolo26x_pose_training/` now contains a self-contained launcher, monitor, report exporter, evaluator, and updated tutorial
- Key files:
  - `script/yolo26x_pose_training/train_yolo26x_pose_frame.sh`
  - `script/yolo26x_pose_training/monitor_yolo_pose_patience.py`
  - `script/yolo26x_pose_training/export_yolo_pose_training_report.py`
  - `script/yolo26x_pose_training/evaluate_yolo_pose_split.py`
  - `script/yolo26x_pose_training/tutorial.md`
  - `docs/ai/sessions/2026-06-16-CHAT-YOLO26-POSE-standalone-dir.md`
- Started: 2026-06-16 (UTC)
- Completed: 2026-06-16 (UTC)


### overlay-sample-comparison-tutorial-1
- Logical title: Overlay sample comparison tutorial
- Role: documentation
- Status: completed
- Purpose: write a user-facing tutorial for `script/overlays/overlay_sample_comparison.py` with function summary, parameters, defaults, and example commands
- Outcome: `script/overlays/tutorial_overlay_sample_comparison.md` now documents the comparison workflow, parameter defaults, and common usage patterns
- Key files:
  - `script/overlays/tutorial_overlay_sample_comparison.md`
  - `docs/ai/sessions/2026-06-15-overlay-sample-comparison-tutorial-1.md`
- Started: 2026-06-15 (UTC)
- Completed: 2026-06-15 (UTC)

### overlay-sample-comparison-1
- Logical title: Overlay sample comparison
- Role: visualization
- Status: completed
- Purpose: compare Yolo26x-Pose and VitPose++ overlays on random Test frames using the saved `kp_Test.json` files from both experiments
- Outcome: `script/overlays/overlay_sample_comparison.py` now copies the sampled Test frames into the comparison directory and writes `_GT`, `_Yolo26x-Pose`, and `_VitPosePP` overlays plus `_manifest.json`; VitPose mapping now prefers the run manifest when available
- Key files:
  - `script/overlays/overlay_sample_comparison.py`
  - `docs/ai/sessions/2026-06-15-overlay-sample-comparison-1.md`
- Started: 2026-06-15 (UTC)
- Completed: 2026-06-15 (UTC)

### CHAT-YOLO26-POSE-SAW-rename
- Logical title: YOLO26x-Pose SAW rename alignment
- Role: training / YOLO pose
- Status: completed
- Purpose: align the frame-based YOLO26x-Pose launcher, tutorial, and saved run config with the renamed `yolo26x-pose_SAW_frames_20260614` run/output convention
- Outcome: `script/yolo26x_pose_training/train_yolo26x_pose_frame.sh` now normalizes bare `--run-name` values to `yolo26x-pose_<tag>`; `runs/yolo26x-pose_SAW_frames_20260614/args.yaml` and tutorial examples were updated to match
- Key files:
  - `script/yolo26x_pose_training/train_yolo26x_pose_frame.sh`
  - `script_old/yolo_training/tutorial.md`
  - `runs/yolo26x-pose_SAW_frames_20260614/args.yaml`
  - `docs/ai/sessions/2026-06-15-CHAT-YOLO26-POSE-SAW-rename.md`
- Started: 2026-06-15 (UTC)
- Completed: 2026-06-15 (UTC)

### CHAT-YOLO26-POSE-SAW-frames-20260612
- Logical title: YOLO26x-Pose SAW_frames launcher adaptation
- Role: training / YOLO pose
- Status: completed
- Purpose: adapt the last useful YOLO26x-Pose training script for `SAW_frames_EntireSwim`, preserve the same pretrained-start / cfg04 hyperparameters, and add post-train Val/Test reporting
- Outcome: new launcher set prepared under `script_old/yolo_training/train_yolo26x_pose_SAW_frames_EntireSwim_20260612.sh`, `script_old/yolo_training/run_yolo26x_pose_SAW_frames_EntireSwim_20260612_tmux.sh`, and `script/yolo26x_pose_training/export_yolo_pose_training_report.py`; Test outputs are wired to `data/output/experiments/yolo26x-pose_SAW_frames_EntireSwim_20260612/`
- Key files:
  - `script_old/yolo_training/train_yolo26x_pose_SAW_frames_EntireSwim_20260612.sh`
  - `script_old/yolo_training/run_yolo26x_pose_SAW_frames_EntireSwim_20260612_tmux.sh`
  - `script/yolo26x_pose_training/export_yolo_pose_training_report.py`
  - `script/yolo26x_pose_training/evaluate_yolo_pose_split.py`
  - `data/intermediate/SAW_frames_EntireSwim/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml`
- Started: 2026-06-13 (UTC)
- Completed: 2026-06-13 (UTC)


### SAW-frames-EntireSwim-1
- Logical title: SAW frames EntireSwim preparation
- Role: data preparation / frames
- Status: completed
- Purpose: extend the standalone frame-preparation pipeline to support the `SAW_frames_EntireSwim` JPG + `__frm_` layout and generate canonical plus model-specific exports
- Outcome: `data/intermediate/SAW_frames_EntireSwim/` now contains `_train_canonical`, `_VitPosePP`, `_Yolo26x_detection`, `_Yolo26x_pose`; canonical splits are train `10489`, val `2997`, test `1498`
- Key files:
  - `script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py`
  - `script/dataset_preparation-cleaning/run_prepare_swimxyz_frames_dataset_tmux.sh`
  - `data/intermediate/SAW_frames_EntireSwim/_train_canonical/reports/swimxyz_frames_preparation_report.json`
  - `docs/ai/sessions/2026-06-12-SAW-frames-EntireSwim-1.md`
- Started: 2026-06-12 (UTC)
- Completed: 2026-06-12 (UTC)

### GT-KP-overlays-1
- Logical title: GT keypoint overlays
- Role: visualization
- Status: completed
- Purpose: add a parametrized GT keypoint overlay renderer that accepts image file/dir inputs plus COCO-style annotation file/dir inputs and matches the YOLO pose visual style
- Outcome: `script/overlays/GT_KP_overlays.py` now renders GT keypoints from COCO annotations with optional GT bbox drawing and defaults to `data/intermediate/Side_above_water/_train_canonical/test2017`, `data/intermediate/Side_above_water/_train_canonical/annotations/person_keypoints_test.json`, and `data/intermediate/Side_above_water/_train_canonical/reports/test_overlays/GT`
- Key files:
  - `script/overlays/GT_KP_overlays.py`
  - `docs/ai/sessions/2026-06-11-GT-KP-overlays-1.md`
- Started: 2026-06-11 (UTC)
- Completed: 2026-06-11 (UTC)

### Image-type-split-1
- Logical title: Image type split 1
- Role: data preparation
- Status: completed
- Predecessor: `Aggiorna logica directory data`
- Purpose: rebuild `data/intermediate/Side_above_water_EntireSwim/` from the complementary canonical datasets `Side_above_water` and `Side_above_water_VideoTest2`, then prepare the complementary `EntireSwim_A` / `EntireSwim_B` split workflow and exporters
- Outcome: `data/intermediate/Side_above_water_EntireSwim/` now contains rebuilt `_train_canonical`, `_VitPosePP`, `_Yolo26x_detection`, `_Yolo26x_pose`, root/canonical `manifest.json`, aggregate totals train `27732`, val `7949`, test `4001`; the complementary split utilities `script_old/prepare_entire_swim_ab_split.py` and `script_old/run_prepare_entire_swim_ab_split_tmux.sh` are ready with default `A=30%`
- Key files:
  - `script_old/prepare_entire_swim_dataset.py`
  - `script_old/run_prepare_entire_swim_dataset_tmux.sh`
  - `script_old/prepare_entire_swim_ab_split.py`
  - `script_old/run_prepare_entire_swim_ab_split_tmux.sh`
  - `data/intermediate/Side_above_water_EntireSwim/manifest.json`
  - `data/intermediate/Side_above_water_EntireSwim/_train_canonical/reports/entire_swim_preparation_report.json`
  - `docs/ai/sessions/2026-06-04-Image-type-split-1.md`
- Started: 2026-06-04 (UTC)
- Completed: 2026-06-04 (UTC)

### CHAT-VitPose++ VideoTest2 random50 eval
- Logical title: VitPose++ VideoTest2 random50 evaluation
- Role: evaluation / VitPose++
- Status: completed
- Purpose: sample a deterministic 50-frame subset across Train/Val/Test from `Side_above_water_VideoTest2` and evaluate the current VitPose++ winner checkpoint on that diagnostic subset
- Outcome: subset root `data/intermediate/Side_above_water_VideoTest2/_train_canonical/subsets/random50_seed20260603/`; eval outputs `runs/vitposepp_video_test2_random50_eval_best_AP_epoch_24/`; AP `0.83550`, AP50 `1.00000`, AP75 `0.92331`, AR `0.85800`
- Key files:
  - `script_old/prepare_vitposepp_video_test2_subset.py`
  - `script_old/run_vitposepp_video_test2_random50_eval_tmux.sh`
  - `data/intermediate/Side_above_water_VideoTest2/_train_canonical/subsets/random50_seed20260603/`
  - `runs/vitposepp_video_test2_random50_eval_best_AP_epoch_24/`
  - `docs/ai/sessions/2026-06-03-CHAT-VitPosePP-VideoTest2-random50-eval.md`
- Started: 2026-06-03 (UTC)
- Completed: 2026-06-03 (UTC)

### CHAT-VitPose++ VideoTest2 random300 overlay export
- Logical title: VitPose++ VideoTest2 random300 overlay export
- Role: visualization / VitPose++
- Status: completed
- Purpose: render bbox+keypoint overlays for the 300-frame Test2 subset from the saved VitPose++ predictions JSON
- Outcome: overlays written to `data/intermediate/Side_above_water_VideoTest2/_train_canonical/reports/test_overlays/vitposepp_video_test2_random300_best_AP_epoch_24/` using `script/vitpose_training/vitpose_generate_test_overlays_from_json.py`
- Key files:
  - `script/vitpose_training/vitpose_generate_test_overlays_from_json.py`
  - `runs/vitposepp_video_test2_random300_eval_best_AP_epoch_24/result_keypoints.json`
  - `data/intermediate/Side_above_water_VideoTest2/_train_canonical/reports/test_overlays/vitposepp_video_test2_random300_best_AP_epoch_24/`
  - `docs/ai/sessions/2026-06-03-CHAT-VitPosePP-VideoTest2-random300-eval.md`
- Started: 2026-06-03 (UTC)
- Completed: 2026-06-03 (UTC)

### CHAT-VitPose++ VideoTest2 random300 eval
- Logical title: VitPose++ VideoTest2 random300 evaluation
- Role: evaluation / VitPose++
- Status: completed
- Purpose: repeat the deterministic subset diagnostic with 300 frames sampled across Train/Val/Test from `Side_above_water_VideoTest2`
- Outcome: subset root `data/intermediate/Side_above_water_VideoTest2/_train_canonical/subsets/random300_seed20260603/`; eval outputs `runs/vitposepp_video_test2_random300_eval_best_AP_epoch_24/`; AP `0.81084`, AP50 `0.97781`, AP75 `0.88959`, AR `0.84800`
- Key files:
  - `script_old/prepare_vitposepp_video_test2_subset.py`
  - `script_old/run_vitposepp_video_test2_random50_eval_tmux.sh`
  - `data/intermediate/Side_above_water_VideoTest2/_train_canonical/subsets/random300_seed20260603/`
  - `runs/vitposepp_video_test2_random300_eval_best_AP_epoch_24/`
  - `docs/ai/sessions/2026-06-03-CHAT-VitPosePP-VideoTest2-random300-eval.md`
- Started: 2026-06-03 (UTC)
- Completed: 2026-06-03 (UTC)

### CHAT-VitPose++ train
- Logical title: VitPose++ Training
- Role: training / VitPose++
- Status: active
- Purpose: prepare the next VitPose++ training launch so it starts from the completed best checkpoint while waiting for the still-running VitPose++ grid search to publish its best hyperparameters
- Key files:
  - `docs/ai/context.md`
  - `docs/ai/task-board.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/tests-and-results.md`
  - `docs/ai/handoff.md`
  - `docs/ai/sessions/2026-05-27-CHAT-VitPosePP-train.md`
  - `script_old/hparam_search/prepare_vitposepp_grid_best_resume.py`
  - `script_old/hparam_search/trigger_vitposepp_post_grid.py`
  - `script_old/hparam_search/monitor_vitpose_patience.py`
  - `script_old/run_train_side_above_water_from_grid_best_tmux.sh`
  - `runs/hparam_search/vitposepp_huge/best_config.json`
  - `runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth`
- Started: 2026-05-27 (UTC)

### hyper-choose Yolo26x-Detection
- Logical title: Choose YOLO26x Detection
- Role: selection / detector
- Status: completed
- Purpose: selected and consolidated the best YOLO26x one-class swimmer detector checkpoint for the YOLO->VitPose++ pipeline (val-only).
- Outcome: selected `cfg_03_lr0_0.00067_imgsz_768`
- Key files:
  - `runs/hparam_search/yolo26x_detector_v2/summary.csv`
  - `runs/hparam_search/yolo26x_detector_v2/report.md`
  - `runs/hparam_search/yolo26x_detector_v2/best_config.json`
  - `docs/ai/tests-and-results.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/handoff.md`
- Started: 2026-05-23 (UTC)
- Completed: 2026-05-23 (UTC)

### CHAT-YOLO26-Training-2
- Logical title: VitPose++ Training 2
- Role: training
- Status: active
- Predecessor: `Audit-Training-yolo26-pose`
- Purpose: audit and continue YOLO26x-pose training runs (launcher, checkpoints, metrics) while keeping YOLO detection and VitPose++ artifacts separate
- Key files:
  - `docs/ai/context.md`
  - `docs/ai/task-board.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/sessions/2026-05-21-CHAT-YOLO26-POSE.md`
  - `script_old/yolo_training/train_yolo_pose_side_above_water.sh`
  - `script/dataset_preparation-cleaning/prepare_yolo_pose_dataset.py`
  - `script_old/yolo_training/prune_yolo_epoch_checkpoints.py`
  - `script_old/yolo_training/report_yolo26x_pose_final_training.py`
  - `runs/pose/runs/yolo26x_pose_side_above_water/yolo26x-pose-incremental-from-cfg04/reports/final_training/`
  - `data/intermediate/Side_above_water/_Yolo26x_pose/`
  - `runs/yolo26x_pose_side_above_water/`
- Started: 2026-05-21 (UTC)

### CHAT-YOLO26-POSE
- Logical title: YOLO26x Pose Training
- Role: training / YOLO pose
- Status: active
- Purpose: prepare and launch Ultralytics YOLO26x-pose training while keeping YOLO detection, YOLO pose, and VitPose++ outputs separate
- Key files:
  - `script/dataset_preparation-cleaning/prepare_yolo_pose_dataset.py`
  - `script_old/yolo_training/train_yolo_pose_side_above_water.sh`
  - `script_old/yolo_training/prune_yolo_epoch_checkpoints.py`
  - `script_old/yolo_training/README.md`
  - `data/intermediate/Side_above_water/_Yolo26x_pose/`
  - `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/`
  - `docs/ai/sessions/2026-05-21-CHAT-YOLO26-POSE.md`
- Started: 2026-05-21 (UTC)

### CHAT-DATASET-2 (formerly: Data-Cleaning 2)
- Logical title: Dataset conversion (data cleaning & preparation)
- Session alias: `Dataset-Test-2`
- Role: dataset-conversion
- Status: active
- Predecessor: `Elenca file e cartelle`
- Purpose: inspect and prepare the current dataset/workspace state for dataset conversion work, using repository files and `docs/ai/` as source of truth
- Key files:
  - `AGENTS.md`
  - `docs/ai/context.md`
  - `docs/ai/task-board.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/chat-index.md`
  - `docs/ai/chat-roles.md`
  - `docs/ai/handoff.md`
  - `docs/ai/tests-and-results.md`
  - `docs/ai/sessions/`
  - `docs/ai/experiments/`
- Started: 2026-05-20 (UTC)

### CHAT-DOCS
- Logical title: AI memory maintenance
- Role: documentation / AI memory maintenance
- Purpose: manutenzione di `docs/ai/` per handoff tra sessioni Codex con finestra di contesto limitata
- Key files:
  - `AGENTS.md`
  - `docs/ai/start-here.md`
  - `docs/ai/context.md`
  - `docs/ai/task-board.md`
  - `docs/ai/handoff.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/tests-and-results.md`
  - `docs/ai/chat-index.md`
  - `docs/ai/chat-roles.md`
  - `docs/ai/sessions/`
  - `docs/ai/experiments/`
  - `docs/ai/runbooks/`
- Started: 2026-05-15 (UTC)

### CHAT-DATASET
- Logical title: SwimXYZ 2 VitPose++
- Role: dataset-conversion
- Purpose: conversione del subset SwimXYZ nel formato richiesto da VitPose++/MMPose
- Key files:
  - `AGENTS.md`
  - `docs/ai/context.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/chat-index.md`
  - `scripts/convert_swimxyz_to_vitpose.py`
  - `scripts/validate_vitpose_dataset.py`
  - `docs/dataset-format.md`
- Started: 2026-05-11 (UTC)

### CHAT-TRAINING
- Logical title: VitPose++ Training
- Role: training
- Purpose: training/eval pipeline, configs, launch, checkpoints, metrics
- Status: closing; successor planned as `CHAT-TRAINING-2`
- Key files:
  - `AGENTS.md`
  - `docs/ai/context.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/chat-index.md`
  - `docs/ai/task-board.md`
  - `configs/`
  - `script/`
  - `script_old/yolo_training/`
  - `script_old/visualize_gt_bboxes.py`
  - `src/`
  - `docs/ai/sessions/2026-05-12-CHAT-TRAINING.md`
- Started: 2026-05-11 (UTC)

### CHAT-TRAINING-2
- Logical title: VitPose++ Training 2
- Role: training
- Status: active
- Predecessor: `CHAT-TRAINING`
- Purpose: continue VitPose++/YOLO training evaluation after CHAT-TRAINING handoff, using workspace files as source of truth
- Key files:
  - `AGENTS.md`
  - `docs/ai/start-here.md`
  - `docs/ai/context.md`
  - `docs/ai/task-board.md`
  - `docs/ai/handoff.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/tests-and-results.md`
  - `docs/ai/sessions/2026-05-15-CHAT-TRAINING-handoff-to-CHAT-TRAINING-2.md`
  - `docs/ai/sessions/2026-05-20-CHAT-TRAINING-2.md`
  - `runs/vitposepp_side_above_water_aniso_20x25_min15/training_status.txt`
  - `script_old/run_resume_side_above_water_to_25ep_tmux.sh`
  - `data/output/experiments/YoloVitPose_mAP/`
  - `script_old/yolo_training/evaluate_yolo_vitpose_map.py`
- Started: 2026-05-15 (UTC)
- Handoff updated: 2026-05-19 (UTC)


### Choose Yolo26x detection
- Logical title: Choose Yolo26x detection
- Role: scelta degli iperparametri per Yolo26x detection
- Status: completed
- Predecessor: `Ricostruisci contesto progetto`
- Purpose: select the best YOLO26x detection hyperparameters using validation-only evidence
- Outcome: selected `cfg_03_lr0_0.00067_imgsz_768`
- Handoff source: `docs/ai/handoff.md`
- Key files:
  - `docs/ai/handoff.md`
  - `docs/ai/sessions/2026-05-23-hyper-choose-Yolo26x-Detection.md`
  - `docs/ai/experiments/EXP-20260523-yolo26x-detector-grid2x2-v2.md`
  - `runs/hparam_search/yolo26x_detector_v2/`
  - `script_old/yolo_training/yolo26x_detector_grid2x2.py`
  - `docs/ai/tests-and-results.md`
- Completed: 2026-05-23 (UTC)


### Train Yolo26x detection
- Logical title: Train Yolo26x detection
- Role: Train the weights of Yolo26x Model on the best hyperparameters found with `Choose Yolo26x detection`
- Status: completed
- Predecessor: `Choose Yolo26x detection`
- Purpose: incrementally train YOLO26x one-class `swimmer` detection weights from the selected cfg_03 checkpoint until validation mAP50-95 plateaus/overfits
- Outcome: useful checkpoint `runs/yolo26x_bbox_side_above_water/yolo26x-detection_from_cfg03_ep5_20260523_1923/weights/best.pt`; best observed mAP50-95 `0.86827` at epoch `1`
- Training policy: first continuation starts from `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt`; later continuations auto-resume from the run `weights/last.pt`; use `patience=2`
- Key files:
  - `docs/ai/context.md`
  - `docs/ai/task-board.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/handoff.md`
  - `docs/ai/tests-and-results.md`
  - `docs/ai/sessions/2026-05-23-Train-Yolo26x-detection.md`
  - `script_old/yolo_training/train_yolo_side_above_water.sh`
  - `data/intermediate/Side_above_water/_Yolo26x_detection/swimxyz_side_above_water_yolo26x_detection.yaml`
  - `data/intermediate/Side_above_water/_Yolo26x_detection/images/`
  - `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt`
  - `runs/yolo26x_bbox_side_above_water/`
- Started: 2026-05-23 (UTC)
- Active smoke run: `runs/yolo26x_bbox_side_above_water/yolo26x-detection_smoke_1ep_20260523_1740`


### hyper-choose VitPose++
- Logical title: hyper-choose VitPose++
- Role: Hyperparameters Setup for VitPose++
- Status: planned / pending activation
- Predecessor: `Ricostruisci questo progetto`
- Purpose: choose/validate next VitPose++ hyperparameters against the current canonical dataset and existing baselines
- Key files:
  - `docs/ai/context.md`
  - `docs/ai/task-board.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/tests-and-results.md`
  - `docs/ai/handoff.md`
  - `data/intermediate/Side_above_water/_train_canonical/generated_configs/swimxyz_vitposepp_huge.py`
  - `runs/vitposepp_side_above_water_aniso_20x25_min15/`
  - `script_old/prepare_swimxyz_vitposepp.py`
  - `script_old/yolo_training/evaluate_yolo_vitpose_map.py`
  - `script_old/hparam_search/vitposepp_huge_grid2x2.py`
  - `runs/hparam_search/vitposepp_huge/`
- Started: 2026-05-23 (UTC)

### Hyper-choose Yolo26x-Pose
- Logical title: Hyper-choose Yolo26x-Pose
- Role: Hyperparameters Setup for Yolo26x-Pose
- Status: completed
- Predecessor: `Ricostruisci questo progetto`
- Purpose: define and select YOLO26x-pose hyperparameters using train/val only (no test-driven selection)
- Outcome: selected `cfg_04_lr0_0.00100_imgsz_768` (model 04)
- Key files:
  - `docs/ai/context.md`
  - `docs/ai/task-board.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/tests-and-results.md`
  - `data/intermediate/Side_above_water/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml`
  - `script_old/yolo_training/train_yolo_pose_side_above_water.sh`
  - `runs/yolo26x_pose_side_above_water/`
- Started: 2026-05-23 (UTC)
