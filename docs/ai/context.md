# Shared AI Context

## Current project state
- Goal: train/evaluate VitPose++ on a SwimXYZ Side_above_water subset and evaluate realistic YOLO-bbox -> VitPose++ inference.
- Environment: VS Code over SSH to Linux server.
- Repository: GitHub.
- Main workstreams:
  - Dataset conversion: SwimXYZ -> VitPose++/MMPose-compatible format.
  - Training: configs, launch scripts, checkpoints, metrics.
  - Detection + pose evaluation: YOLO bbox -> VitPose++ keypoints -> COCO mAP.
  - Documentation and reproducibility memory under `docs/ai/`.

## AI memory source of truth
- Use `docs/ai/context.md` for consolidated current project state.
- Use `docs/ai/task-board.md` for active work and backlog.
- Use `docs/ai/decision-log.md` for durable technical decisions.
- Use `docs/ai/tests-and-results.md` for validation, metrics, and run outcomes.
- `docs/ai/handoff.md` is deprecated as an operational handoff and kept only as a short compatibility pointer.
- `docs/ai/sessions/` is historical archive; paths and plans there may be stale.

## Dataset notes
- Source dataset: SwimXYZ subset.
- Prepared dataset root: `data/intermediate/Side_above_water/_train_canonical/`.
- Target format: COCO-style keypoint annotations for MMPose/VitPose++.
- Current GT bbox convention: anisotropic padding, horizontal `0.20`, vertical `0.25`, minimum `15 px` per side, clipped to image boundaries.
- YOLO conversion reads the already padded GT bboxes with no extra padding; do not add YOLO-side/downstream padding unless a new experiment explicitly changes convention.


## Dataset preparation architecture
- Historical data-cleaning consolidation note: `docs/ai/data-cleaning-consolidation.md`; it preserves durable assumptions from the initial data-cleaning phase and drops obsolete smoke-test instructions.
- Canonical prepared dataset root: `data/intermediate/Side_above_water/_train_canonical/`. This is the only stage that materializes the accepted frame set; downstream model-specific exporters must not make independent frame-drop decisions.
- Model-specific label/config outputs generated from the canonical dataset:
  - VitPose++: `data/intermediate/Side_above_water/_VitPosePP/` via `script/dataset_preparation-cleaning/prepare_vitposepp_dataset.py`
  - YOLO26x detection: `data/intermediate/Side_above_water/_Yolo26x_detection/` via `script/dataset_preparation-cleaning/prepare_yolo_detection_dataset.py`
  - YOLO26x pose: `data/intermediate/Side_above_water/_Yolo26x_pose/` via `script/dataset_preparation-cleaning/prepare_yolo_pose_dataset.py`
- VitPose++ exporter is label/config-only. YOLO26x detection and YOLO26x pose expose canonical images under their `images/{train,val,test}` roots with symlinks so Ultralytics can pair `images/` and `labels/` directly.
- Verified split alignment after YOLO26x pose export: train `18181`, val `5195`, test `2597`; pose labels have `56` fields (`class + bbox + 17*3`), and generated image/label counts match for every split.
- Aggregate filtered dataset root: `data/intermediate/Side_above_water_EntireSwim/_train_canonical/` via `script_old/prepare_entire_swim_dataset.py`; it is rebuilt from the complementary canonical sources `data/intermediate/Side_above_water/_train_canonical` and `data/intermediate/Side_above_water_VideoTest2/_train_canonical`, uses image symlinks, writes `manifest.json`, and totals train `27732`, val `7949`, test `4001`.
- EntireSwim model exports are rebuilt from that aggregate canonical dataset: `_VitPosePP`, `_Yolo26x_detection`, and `_Yolo26x_pose` live under `data/intermediate/Side_above_water_EntireSwim/`; detection images are exposed with split symlink dirs, and pose images are exposed with per-file symlinks.
- Prepared complementary split utility: `script_old/prepare_entire_swim_ab_split.py` can split `data/intermediate/Side_above_water_EntireSwim/` into `data/intermediate/Side_above_water_EntireSwim_A/` and `_B/` with deterministic seed control, default `prob_a=0.3`, canonical image symlinks, regenerated manifests, and refreshed `_VitPosePP`, `_Yolo26x_detection`, `_Yolo26x_pose` exports for both outputs.
- Prepared standalone-frame builder: `script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py` reads `data/input/subset_xyz/Side_above_water_frames/` as `COCO__2D_cam` frame+label pairs, rebuilds `data/intermediate/Side_above_water_frames/_train_canonical/` with image symlinks, and regenerates `_VitPosePP`, `_Yolo26x_detection`, `_Yolo26x_pose` via the standard exporters; the local tutorial lives at `script/dataset_preparation-cleaning/tutorial.md` and the tmux launcher is `script/dataset_preparation-cleaning/run_prepare_swimxyz_frames_dataset_tmux.sh`.
- The standalone-frame builder now also supports JPG/JPEG inputs and both `__frame_` and `__frm_` frame tokens; `data/input/subset_xyz/SAW_frames_EntireSwim/` was prepared into `data/intermediate/SAW_frames_EntireSwim/` with train `10489`, val `2997`, test `1498`, using image symlinks and regenerated `_VitPosePP`, `_Yolo26x_detection`, `_Yolo26x_pose`.

## Operational script root
- Current consolidated script root for new work: `script/`.
- Legacy script root: `script_old/`; use it only for past versions, variants, or tests that were intentionally not consolidated.
- Cutover status: final directory rename completed; the validated staging tree is now `script/`, and the previous broad tree is `script_old/`.
- Historical sections may mention `script_old/` launchers that produced past results; do not treat them as current entrypoints for new work unless explicitly restoring an old workflow.
- Consolidated entrypoints:
  - Dataset preparation RAW/frame+label -> train-ready exports: `script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py`.
  - YOLO26x-Pose training: `script/yolo26x_pose_training/train_yolo26x_pose_frame.sh`.
  - VitPose++ training: `script/vitpose_training/train_vitpose_frame.sh`.
  - YOLO26x-Pose prediction/KP/overlays: `script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh`.
  - VitPose++ prediction/KP/overlays: `script/vitpose_prediction/predict_vitpose_frame.sh`.
  - Overlay GT/comparison: `script/overlays/GT_KP_overlays.py` and `script/overlays/overlay_sample_comparison.py`.
  - HPE direct/cross metric report tables: `script/hpe_report/build_hpe_report_tables.py`.
  - Metric/loss/mAP plotting: `script/plot-metrics/plot_vitpose_metrics_with_checkpoints.py`.

## Current conventions
- Python modules should be small and explicit.
- Scripts should expose CLI arguments.
- Overlay comparison utility: `script/overlays/overlay_sample_comparison.py` compares Yolo26x-Pose and VitPose++ Test overlays on random frames, saving the original frame plus `_GT`, `_Yolo26x-Pose`, and `_VitPosePP` renders and a `_manifest.json` in the comparison directory. VitPose frame association prefers `<KP-VITPOSE>/../overlays_Test/_manifest.json` when available.
- Tutorial doc added: `script/overlays/tutorial_overlay_sample_comparison.md` explains function, parameters, examples, defaults, and the `_GT` / `_Yolo26x-Pose` / `_VitPosePP` output layout for the overlay comparison workflow.
- Consolidated VitPose++ frame launcher: `script/vitpose_training/train_vitpose_frame.sh`; pass `--dataset-dir` as the dataset root such as `data/intermediate/SAW_frames`, not `_train_canonical`. The script resolves `_train_canonical` and `_VitPosePP` internally; `--test-dataset-dir` defaults to the same root as `--dataset-dir` to avoid hidden train/test mismatches.
- Paths must not be hardcoded.
- Generated artifacts should go outside Git or under ignored directories.
- For current VitPose++ training, retain best validation checkpoint plus latest three periodic checkpoints (`max_keep_ckpts=3`).

## Codex sessions
- 2026-05-11 | CHAT-DATASET | SwimXYZ 2 VitPose++ | Role: dataset-conversion.
- 2026-05-12 | CHAT-TRAINING | VitPose++ Training | Role: training.
- 2026-05-15/19 | CHAT-TRAINING-2 | VitPose++ Training 2 | Role: training.

## VitPose++ training state
- Active work dir: `runs/vitposepp_side_above_water_aniso_20x25_min15/`.
- Generated config: `data/intermediate/Side_above_water/_train_canonical/generated_configs/swimxyz_vitposepp_huge.py`.
- Final completed training status: phase `finished`, target `40` epochs, status timestamp `2026-05-16 17:19:43 UTC`.
- Final log: `runs/vitposepp_side_above_water_aniso_20x25_min15/20260516_100449.log`.
- Latest periodic checkpoint: `runs/vitposepp_side_above_water_aniso_20x25_min15/epoch_40.pth`.
- Latest symlink: `runs/vitposepp_side_above_water_aniso_20x25_min15/latest.pth -> epoch_40.pth`.
- Best validation checkpoint: `runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth`.
- Retained periodic checkpoints visible: `epoch_38.pth`, `epoch_39.pth`, `epoch_40.pth`.
- Older best retained: `best_AP_epoch_30.pth`.
- Mild plateau / early overfitting signal after epoch 35: AP `0.9821` at epoch 35, AP `0.9812` at epoch 40 while loss continued decreasing.
- Recommended VitPose++ checkpoint for downstream use: `best_AP_epoch_35.pth`.
- Deferred post-grid launcher prepared: `script_old/run_train_side_above_water_from_grid_best_tmux.sh`; it generates `data/intermediate/Side_above_water/_train_canonical/generated_configs/swimxyz_vitposepp_huge_grid_best_from_best35.py` only after `runs/hparam_search/vitposepp_huge/best_config.json` becomes `status=completed`, then starts a fresh run from `best_AP_epoch_35.pth` weights via `load_from`.
- Current VitPose++ grid winner recorded in `runs/hparam_search/vitposepp_huge/best_config.json`: `cfg_02_lr_0.00100_crop_384x128` with validation AP `0.91044`; the deferred launcher smoke test confirmed it can now generate the downstream config and training command immediately.
- VitPose++ grid winner is `cfg_02_lr_0.00100_crop_384x128`; best checkpoint `runs/hparam_search/vitposepp_huge/cfg_02_lr_0.00100_crop_384x128/best_AP_epoch_5.pth`, latest `runs/hparam_search/vitposepp_huge/cfg_02_lr_0.00100_crop_384x128/latest.pth`.
- Direct launcher retained: `script_old/run_train_side_above_water_from_grid_best_tmux.sh` uses `data/intermediate/Side_above_water/_train_canonical/generated_configs/swimxyz_vitposepp_huge_grid_winner_resume.py`, which now loads the winner best checkpoint `best_AP_epoch_5.pth`. Trigger-specific scripts were removed.

## VitPose++ validation metrics
- Epoch 5 AP `0.8604`
- Epoch 10 AP `0.9424`
- Epoch 15 AP `0.9549`
- Epoch 20 AP `0.9727`
- Epoch 25 AP `0.9739`
- Epoch 30 AP `0.9776`
- Epoch 35 AP `0.9821`
- Epoch 40 AP `0.9812`

Latest plots:
- `data/intermediate/Side_above_water/_train_canonical/reports/training_plots/loss_epoch_avg__20260519_230159_completed_epochs.png`
- `data/intermediate/Side_above_water/_train_canonical/reports/training_plots/mAP_validation__20260519_230159_completed_epochs.png`
- `data/intermediate/Side_above_water/_train_canonical/reports/training_plots/loss_map_summary__20260519_230159_completed_epochs.csv`

## YOLO detector state
- Historical 5-epoch detector run: `runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox_aniso_20x25y_min15_5ep/`; epoch-5 validation metrics were precision `0.99227`, recall `0.99249`, mAP50 `0.99266`, mAP50-95 `0.87601`.
- Completed val-only hyperparameter selection: `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/` selected as canonical cfg_03; selection metric priority was recall, then AP75/IoU if available, then mAP50-95.
- Selected cfg_03 validation metrics at epoch 5: precision `0.99440`, recall `0.99519`, mAP50 `0.99267`, mAP50-95 `0.86614`.
- Incremental training starts from `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt`; subsequent runs auto-resume from `runs/yolo26x_bbox_side_above_water/yolo26x-detection_<tag>/weights/last.pt` when present.
- Detector launcher: `script_old/yolo_training/train_yolo_side_above_water.sh`; defaults include `imgsz=768`, `batch=2`, `lr0=0.00067`, `patience=2`, `save_period=1`, keep latest `3` periodic checkpoints, run root `runs/yolo26x_bbox_side_above_water/`.
- Detector dataset YAML: `data/intermediate/Side_above_water/_Yolo26x_detection/swimxyz_side_above_water_yolo26x_detection.yaml`; `images/{train,val,test}` are symlinks to canonical `train2017`, `val2017`, and `test2017`.
- Stop criterion for the incremental phase: stop when training loss keeps decreasing while validation mAP50-95 plateaus or degrades, using patience of `2` epochs to consolidate the plateau.
- Smoke run `runs/yolo26x_bbox_side_above_water/yolo26x-detection_smoke_1ep_20260523_1740` completed successfully as a launcher/checkpoint smoke test (`best.pt`, `last.pt`, `epoch0.pt`), but its validation metrics are not useful because the log reports `no labels found in detect set` and `results.csv` contains zeros.
- Useful YOLO26x detection continuation from cfg_03 completed: `runs/yolo26x_bbox_side_above_water/yolo26x-detection_from_cfg03_ep5_20260523_1923/`; start checkpoint `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt`; useful checkpoint to carry forward `runs/yolo26x_bbox_side_above_water/yolo26x-detection_from_cfg03_ep5_20260523_1923/weights/best.pt`; best observed mAP50-95 `0.86827` at epoch `1`, last epoch `3`.
- YOLO OOM diagnostic: `logs/yolo_diagnostic_1280_b8_20260513_135040.log`; `yolo26x.pt`, `imgsz=1280`, `batch=8` exhausted 32 GB V100.

## YOLO26x pose state
- SUW_frames YOLO26x-Pose training launched on 2026-07-01 via tmux session `train_yolo26x_pose_yolo26x-pose_SUW_frames_20260701` using `script/yolo26x_pose_training/train_yolo26x_pose_frame.sh` and dataset `data/intermediate/SUW_frames`; normalized run name `yolo26x-pose_SUW_frames_20260701`, run dir `runs/yolo26x-pose_SUW_frames_20260701/`, Test outputs target `data/output/experiments/yolo26x-pose_SUW_frames_20260701/`; metrics pending.
- Dataset root: `data/intermediate/Side_above_water/_Yolo26x_pose/`.
- Data YAML: `data/intermediate/Side_above_water/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml`.
- Converter: `script/dataset_preparation-cleaning/prepare_yolo_pose_dataset.py`; default `--link-mode symlink`.
- Training launcher: `script_old/yolo_training/train_yolo_pose_side_above_water.sh`.
- Hyperparameter search launcher: `script_old/yolo_training/yolo26x_pose_grid2x2.py`; output root `runs/hparam_search/yolo26x_pose/`; grid varies only `lr0` and `imgsz` across four 5-epoch train/val runs, does not use test for selection, and sets `save_period=-1` to avoid intermediate checkpoints.
- Grid evaluation winner: `cfg_04_lr0_0.00100_imgsz_768` (model 04).
- Launcher defaults: local model `models/pose/yolo26x-pose.pt`, data YAML above, output `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17/`, `imgsz=1280`, `batch=1`, `save_period=1`, keep latest `3` periodic `epoch*.pt` checkpoints after training, plus Ultralytics `best.pt` and `last.pt`.
- YOLO26x-pose training logs go to `logs/yolo26x_pose_side_above_water_<timestamp>.log`; status goes to the run `training_status.txt`. A 1-epoch smoke run has completed using explicitly resolved pretrained `yolo26x-pose.pt`.
- SAW_frames launcher prepared: `script_old/yolo_training/train_yolo26x_pose_SAW_frames_EntireSwim_20260612.sh` and `script_old/yolo_training/run_yolo26x_pose_SAW_frames_EntireSwim_20260612_tmux.sh`; it targets `data/intermediate/SAW_frames_EntireSwim/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml`, keeps best plus last 10 checkpoints, exports `reports/val_metrics_by_epoch.csv` plus `loss_by_epoch.png` / `map50_95_by_epoch.png`, and writes Test outputs as `kp_Test.json`, `metrics_Test.json`, and `overlays_Test/`.
- Frame-based launcher naming now normalizes `--run-name` to the `yolo26x-pose_<tag>` convention; the 2026-06-14 SAW run lives under `runs/yolo26x-pose_SAW_frames_20260614/` and `data/output/experiments/yolo26x-pose_SAW_frames_20260614/`, with `args.yaml` aligned to that renamed run directory.

## YOLO26x pose 1-epoch result
- Run: `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap`.
- Status: completed 2026-05-21 UTC, `epochs=1`, `imgsz=1280`, `batch=1`, pretrained `yolo26x-pose.pt`.
- Checkpoints: `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/weights/best.pt`, `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/weights/last.pt`, `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/weights/epoch0.pt`.
- Validation after epoch 1: Pose mAP50 `0.90037`, Pose mAP50-95 `0.49212`.
- Test split eval: Pose mAP50 `0.928`, Pose mAP50-95 `0.537`; Box mAP50 `0.965`, Box mAP50-95 `0.656`.
- Plots: `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/reports/loss_epoch1_train_val.png`, `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/reports/map_epoch1_test.png`.

## YOLO26x pose incremental training
- Status: stopped/not active. Launched 2026-05-24 and stopped after epoch `29`; no active `tmux` session or YOLO/VitPose process was observed on 2026-05-27.
- Purpose: Incremental training of YOLO26x-pose, starting from the best model from the hyperparameter search.
- Run name: `yolo26x-pose-incremental-from-cfg04`.
- Run directory: `runs/pose/runs/yolo26x_pose_side_above_water/yolo26x-pose-incremental-from-cfg04/`.
- Results file: `runs/pose/runs/yolo26x_pose_side_above_water/yolo26x-pose-incremental-from-cfg04/results.csv`.
- Initial checkpoint: `runs/hparam_search/yolo26x_pose/cfg_04_lr0_0.00100_imgsz_768/weights/best.pt`.
- Checkpoints: `weights/best.pt`, `weights/last.pt`, and periodic `epoch*.pt` files exist in the run directory.
- Best observed Pose mAP50-95 from `results.csv`: epoch `21`, Pose precision `0.98101`, Pose recall `0.98191`, Pose mAP50 `0.99211`, Pose mAP50-95 `0.95705`; Box mAP50 `0.99242`, Box mAP50-95 `0.89588`.
- Last recorded epoch `29`: Pose mAP50 `0.99193`, Pose mAP50-95 `0.95271`; Box mAP50 `0.99217`, Box mAP50-95 `0.90251`.
- Hyperparameters: `epochs=100`, `patience=10`, `imgsz=768`, `batch=1`, `lr0=0.00100`, `optimizer=AdamW`.

## Metric semantics
- VitPose++ `AP` from MMPose/COCO is COCO keypoint OKS AP averaged across thresholds `0.50:0.05:0.95` and is the stricter AP/mAP50-95-style metric; `AP50` and `AP75` are threshold-specific.
- YOLO26x-pose reports Ultralytics Pose mAP50 and Pose mAP50-95. Direct comparison to VitPose++ should ideally use the same COCO/OKS evaluator on exported YOLO predictions and the same GT split.

- YOLO26x detector prediction convention: downstream scripts keep exactly one bbox per frame, selected by highest confidence and then by larger area on ties. This applies to detector overlays and YOLO->VitPose++ bbox handoff.

## HPE report direct/cross metrics
- Reproducible report-table script: `script/hpe_report/build_hpe_report_tables.py`; example config: `script/hpe_report/hpe_report_config.example.json`.
- User-facing documentation: `script/hpe_report/7) Diagrammi sulle metriche dirette e cross.md`; linked PPTX report: `script/hpe_report/20260617_Report_Fine-Tuning_Senza-Immagini.pptx`.
- AI runbook: `docs/ai/runbooks/hpe-report-direct-cross-metrics.md`.
- The script is headless and GPU-free; it reads existing COCO GT JSON, `kp_Test.json` predictions, and `val_metrics_by_epoch.csv` files, then writes `data/output/experiments/hpe_report/hpe_report_tables.xlsx`.
- Covered report slides: 8, 9, 10, 16, 17, 18, 19, and 20.
- Direct scenarios: Train/Test `SAW_frames` and Train/Test `SAW_frames_EntireSwim`; cross scenario: Train `SAW_frames` -> Test `SAW_frames_EntireSwim`.
- Default thresholds: YOLO26x-Pose `0.30`, VitPose++ `0.20`; Val best epoch is reconstructed with `patience=3` and `min_delta=0.007`.
- Per-KP difficulty in slides 19/20 uses combined P90 over both models in the thresholded cross scenario; groups are Easy `<=6`, Medium `<=9`, High `<=12`, Challenging `>12`.
- Note: the project MD currently shows examples under `script/report/...`, while the actual workspace path is `script/hpe_report/...`.

## End-to-end YOLO+VitPose++ pipeline
- Consolidated experiment name: `YoloVitPose_mAP`.
- Historical implementation: `script_old/yolo_training/evaluate_yolo_vitpose_map.py`; port or recreate this workflow under `script/` before using it for new operational work.
- Pipeline: YOLO predicts absolute `xyxy` bbox on the full frame; code converts bbox to COCO `xywh`; VitPose++ receives full image + `xywh` bbox; MMPose performs the top-down crop/affine internally; COCO keypoint mAP evaluates predictions.
- Visualization convention: use MMPose `vis_pose_result` for predicted keypoints/skeleton and draw only the YOLO bbox in red; do not use custom fuchsia/GT-mixed skeleton renderers for YOLO+VitPose outputs.
- YOLO checkpoint for consolidated YOLO->VitPose++ evaluation: `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt`.
- VitPose++ checkpoint: `runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth`.
- Current qualitative comparison set: `data/output/experiments/YoloVitPose_mAP/test_20260520_0852_best35_vispose_overlays/` (`20` GT-bbox references plus `20` YOLO->VitPose outputs; `2` YOLO no-detection markers at `conf=0.25`).
- Previous failed YOLO+VitPose outputs and saved result JSON were invalidated because saved keypoints were not reproducible with the current bbox/model path; those failed experiment artifacts were removed.

## Open questions / next work
- Full `YoloVitPose_mAP` test evaluation is pending; port or recreate the historical `script_old/yolo_training/evaluate_yolo_vitpose_map.py` workflow under `script/` before reporting new end-to-end AP.
- Use the consolidated visualization directory to inspect remaining YOLO no-detection cases and decide whether to tune YOLO confidence/fallback behavior.

## SUW_frames Prepared Dataset

- `data/intermediate/SUW_frames/` was prepared from raw `data/input/subset_xyz/SUW_frames/` on 2026-07-01 using `script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py` via tmux session `prepare_SUW_frames_dataset`.
- Raw preflight counts were `15000` images and `15000` `__COCO__2D_cam.txt` labels; final accepted samples were `8634`, with `6366` rejected as `no_valid_keypoints`.
- Final split counts are train `6044`, val `1727`, test `863` across `_train_canonical`, `_VitPosePP`, `_Yolo26x_detection`, and `_Yolo26x_pose`.
- YOLO26x detection labels have `5` fields per row; YOLO26x pose labels have `56` fields per row.

## Execution convention
- For heavy/long-running scripts, launch via `tmux` by default so sessions survive disconnects and can be monitored/attached later.

## Metric reporting (test)
- VitPose++: src/vitpose_base/tools/train.py supports --final-test and writes <work_dir>/test_metrics.json (includes AP and AP50).
- `nvidia-smi` is currently unusable on this host because NVML reports `Driver/library version mismatch` (`NVML library version: 580.159`); use PyTorch-level CUDA checks instead until the driver/NVML mismatch is fixed.
- YOLO26x-pose current Test evaluation/prediction should use `script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh` or `script/yolo26x_pose_training/evaluate_yolo_pose_split.py`; the old `script_old/yolo_training/train_yolo_pose_side_above_water.sh` behavior is historical.

## YOLO26x-pose visualization
- For new qualitative YOLO26x-pose overlays, use `script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh` outputs and `script/overlays/overlay_sample_comparison.py`; the old renderer under `script_old/yolo_training/` is historical.
- YOLO26x-Pose prediction launcher: `script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh` evaluates a selected checkpoint on a dataset Test split from the main dataset directory, resolves `_Yolo26x_pose` internally, optionally builds a reproducible random sampled Test view, writes `kp_Test.json` / `metrics_Test.json` / `metrics_Test.csv`, and renders frame/KP overlays.
- VitPose++ prediction launcher: `script/vitpose_prediction/predict_vitpose_frame.sh` evaluates a selected checkpoint on a dataset Test split from the main dataset directory, resolves `_train_canonical` and `_VitPosePP` internally, creates a reproducible filtered/sampled Test view, defaults to crop `384x128` for the `vitpose_SAW_frames_20260615` checkpoint family, writes `kp_Test.json` / `metrics_Test.json`, and renders frame/KP overlays.
- GT keypoint overlay renderer: `script/overlays/GT_KP_overlays.py` renders COCO-style GT keypoints from a single annotation file or annotation directory, using the same skeleton/colors/radius/thickness conventions as the YOLO pose renderer.

## VitPose++ hyperparameter search
- Script: `script_old/hparam_search/vitposepp_huge_grid2x2.py`.
- Output root: `runs/hparam_search/vitposepp_huge/`.
- Purpose: local 2x2 train/val-only search for VitPose++ huge in the YOLO26x-detector -> VitPose++ top-down pipeline.
- Grid: lr `0.00067` / `0.00100` x crop `384x128` / `512x128`, where crop is requested as width x height.
- Crop convention documented in generated configs: MMPose `data_cfg.image_size=[width, height]`; ViT backbone `img_size=(height, width)`.
- Each run trains for `5` epochs from `models/pose/wholebody.pth`; existing best/latest VitPose++ checkpoints are not used as intermediate initialization because the crop size changes.
- Selection uses validation Keypoint AP@[OKS 0.50:0.95] only; test split is not used.
- Checkpoint policy: MMPose checkpoint interval `1`, `max_keep_ckpts=1`, `create_symlink=True`, plus `evaluation.save_best='AP'`, so only best and latest/final are retained rather than all intermediate epoch checkpoints.
- Detector-produced bbox files for validation are `not reconstructible from workspace files`; the search uses the canonical padded GT bboxes with `use_gt_bbox=True` and does not add bbox expansion.

## VideoTest2 random50 subset evaluation

- Prepared a deterministic 50-frame subset from `data/intermediate/Side_above_water_VideoTest2/_train_canonical/` sampling uniformly from train/val/test (`39/4/7`) and saved it under `data/intermediate/Side_above_water_VideoTest2/_train_canonical/subsets/random50_seed20260603/`.
- Launched VitPose++ subset evaluation from `runs/vitposepp_side_above_water_grid_winner_resume/best_AP_epoch_24.pth` using `script_old/run_vitposepp_video_test2_random50_eval_tmux.sh`; the run completed and saved `test_predictions.pkl`, `result_keypoints.json`, and `test_eval_stdout.log` in `runs/vitposepp_video_test2_random50_eval_best_AP_epoch_24/`.
- Subset mAP result: AP `0.83550`, AP50 `1.00000`, AP75 `0.92331`, AR `0.85800`.

## VideoTest2 random300 subset evaluation

- Prepared a deterministic 300-frame subset from `data/intermediate/Side_above_water_VideoTest2/_train_canonical/` sampling uniformly from train/val/test (`224/48/28`) and saved it under `data/intermediate/Side_above_water_VideoTest2/_train_canonical/subsets/random300_seed20260603/`.
- Launched VitPose++ subset evaluation from `runs/vitposepp_side_above_water_grid_winner_resume/best_AP_epoch_24.pth` using `script_old/run_vitposepp_video_test2_random50_eval_tmux.sh` with `NUM_FRAMES=300`; the run completed and stored outputs in `runs/vitposepp_video_test2_random300_eval_best_AP_epoch_24/`.
- Subset mAP result: AP `0.81084`, AP50 `0.97781`, AP75 `0.88959`, AR `0.84800`.

## VideoTest2 random300 overlay export

- Added `script/vitpose_training/vitpose_generate_test_overlays_from_json.py` to render overlays from the saved `runs/vitposepp_video_test2_random300_eval_best_AP_epoch_24/result_keypoints.json` rather than recomputing predictions.
- Overlay images were generated under `data/intermediate/Side_above_water_VideoTest2/_train_canonical/reports/test_overlays/vitposepp_video_test2_random300_best_AP_epoch_24/` with GT bboxes and predicted keypoints drawn for all `300` frames.
