# Tests and Results

This file tracks validation commands, training/evaluation outcomes, and important metrics.

## Latest known results

| Date | Area | Run | Result | Notes |
|---|---|---|---|---|
| 2026-06-24 | AI memory consolidation | `docs/ai` source-of-truth cleanup | completed | `handoff.md` deprecated as operational source; current state belongs in `context.md`, active work in `task-board.md`, decisions in `decision-log.md`, results in `tests-and-results.md`; `script/` is operational and `script_old/` is legacy/archive. |
| 2026-06-24 | `script/` safe migration | consolidated operational script tree | static validation passed | `python3 -m py_compile` passed for all Python files; `bash -n` passed for all shell files; `--help` smoke checks passed for dataset prep, YOLO/VitPose training, YOLO/VitPose prediction, overlays, HPE report tables, and plot-metrics entrypoints. Final cutover completed: current root is `script/`; previous broad tree is `script_old/`. |
| 2026-06-17 | HPE report direct/cross metrics | `script/hpe_report/build_hpe_report_tables.py` + `20260617_Report_Fine-Tuning_Senza-Immagini.pptx` | report tables documented | Direct Test AP from slide 8: `SAW_frames_EntireSwim` YOLO `0.93190`, VitPose++ `0.97822`; `SAW_frames` YOLO `0.87474`, VitPose++ `0.93288`. Cross Train `SAW_frames` -> Test `SAW_frames_EntireSwim` thresholded AP from slide 16: YOLO `0.9774`, VitPose++ `0.9914`; without threshold: YOLO `0.9854`, VitPose++ `0.9914`. |
| 2026-06-18 | VitPose++ prediction smoke test | `script/vitpose_prediction/predict_vitpose_frame.sh --max-test-items 1 --device cpu` | completed | fixed heredoc config-generation failure; smoke output under `/tmp/vitpose_pred_smoke_1/` includes `effective_config.py`, `metrics_Test.json`, `kp_Test.json`, `result_keypoints.json`, and one overlay; AP `1.0` on the single sampled frame |
| 2026-06-18 | VitPose++ prediction launcher | `script/vitpose_prediction/predict_vitpose_frame.sh` | static validation passed | new parametric prediction launcher supports checkpoint, main dataset dir with internal `_train_canonical` / `_VitPosePP` resolution, random sample size/seed, bbox confidence filtering with `conf=0` top-1 behavior, KP/metrics output dir, and overlay dir; `bash -n`, helper `py_compile`, `--help`, and `_train_canonical` rejection passed |
| 2026-06-18 | YOLO26x-Pose prediction launcher | `script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh`, `script/yolo26x_pose_training/evaluate_yolo_pose_split.py` | static validation passed | new parametric prediction launcher supports checkpoint, main Test dataset dir with internal `_Yolo26x_pose` resolution, random sample size/seed, confidence threshold, KP/metrics output dir, and overlay dir; `bash -n`, `py_compile`, and `--help` passed |
| 2026-06-16 | VitPose training launcher consolidation | `script/vitpose_training/train_vitpose_frame.sh` | static validation passed | new launcher requires dataset root containing `_train_canonical` and `_VitPosePP`; `bash -n` passed; helper `py_compile` passed; `--help` works; `_train_canonical` passed as `--dataset-dir` is rejected with a clear error |
| 2026-06-16 | VitPose++ SAW_frames Test | `data/output/experiments/vitpose_SAW_frames_20260615/kp_Test.json` | completed on `data/intermediate/SAW_frames/_train_canonical/test2017` | checkpoint `runs/vitpose_SAW_frames_20260615/checkpoint/best_AP_epoch_18.pth`; images `1359`; AP `0.93288`, AP50 `0.97968`, AP75 `0.94886`, AR `0.93804`; regenerated `result_keypoints.json`, `kp_Test.json`, `metrics_Test.json`, and `overlays_Test/` aligned to `SAW_frames` |
| 2026-06-16 | Overlay sample comparison | `script/overlays/overlay_sample_comparison.py` | updated VitPose mapping to prefer run manifest; static validation passed | VitPose frame association now prefers `<KP-VITPOSE>/../overlays_Test/_manifest.json` when available, because `kp_Test.json` image ids do not necessarily match the COCO annotations image ids |
| 2026-06-15 | Overlay sample comparison tutorial | `script/overlays/tutorial_overlay_sample_comparison.md` | completed | tutorial documents function, parameters, defaults, and example commands for the comparison workflow |
| 2026-06-16 | YOLO26x-Pose standalone directory | `script/yolo26x_pose_training/` | static validation passed | copied standalone launcher plus helper scripts and tutorial; `python -m py_compile` passed for the copied Python helpers and `bash -n` passed for `script/yolo26x_pose_training/train_yolo26x_pose_frame.sh`; tutorial now includes tmux launch/attach/detach/stop guidance |
| 2026-06-15 | SAW frame YOLO26x-Pose launcher naming | `script/yolo26x_pose_training/train_yolo26x_pose_frame.sh`, `runs/yolo26x-pose_SAW_frames_20260614/args.yaml` | corrected and validated | launcher now normalizes `--run-name` to `yolo26x-pose_<tag>`; the renamed run config now reports `name: yolo26x-pose_SAW_frames_20260614` and `save_dir: /home/albertosco/HPE/runs/yolo26x-pose_SAW_frames_20260614`; tutorial examples updated accordingly |
| 2026-06-13 | YOLO26x-Pose SAW_frames launcher | `script_old/yolo_training/train_yolo26x_pose_SAW_frames_EntireSwim_20260612.sh`, `script/yolo26x_pose_training/export_yolo_pose_training_report.py`, `script/yolo26x_pose_training/evaluate_yolo_pose_split.py` | static validation passed | `bash -n` and `python -m py_compile` passed; the report helper exported per-epoch CSV/plots from `runs/yolo26x-pose_A_20260605/results.csv` to `/tmp/yolo26x_pose_report_test/`; no training launched yet |
| 2026-06-12 | SAW_frames_EntireSwim builder | `script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py` | completed on example dataset | supports standalone `COCO__2D_cam` frame+label pairs with `.png`, `.jpg`, or `.jpeg` images and `__frame_` / `__frm_` naming; generated `data/intermediate/SAW_frames_EntireSwim/` with canonical splits train `10489`, val `2997`, test `1498`, plus `_VitPosePP`, `_Yolo26x_detection`, `_Yolo26x_pose` |
| 2026-06-04 | YOLO26x-Pose A training | `runs/yolo26x-pose_A_20260605/` | launched in tmux | dataset `data/intermediate/Side_above_water_EntireSwim_A/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml`; pretrained `models/pose/yolo26x-pose.pt`; `imgsz=768`, `lr0=0.00100`, `optimizer=AdamW`; external monitor uses Pose mAP50-95 with `patience=2`, `min_delta=0.001`, and prunes epoch checkpoints outside best/current patience window; final test outputs target `data/output/experiments/yolo26x-pose_A_20260605/` |
| 2026-06-04 | EntireSwim full rebuild | `script_old/run_prepare_entire_swim_dataset_tmux.sh` -> `script_old/prepare_entire_swim_dataset.py --source-dataset-roots data/intermediate/Side_above_water data/intermediate/Side_above_water_VideoTest2 --output-dataset-root data/intermediate/Side_above_water_EntireSwim --copy-mode symlink --overwrite` | completed in tmux; all exports regenerated | rebuilt aggregate `_train_canonical` plus `_VitPosePP`, `_Yolo26x_detection`, `_Yolo26x_pose`; root/canonical manifests written; aggregate totals are train `27732`, val `7949`, test `4001` |
| 2026-06-04 | EntireSwim A/B split script | `script_old/prepare_entire_swim_ab_split.py`, `script_old/run_prepare_entire_swim_ab_split_tmux.sh` | static validation passed; not launched | `python -m py_compile` passed, `bash -n` passed, and `--help` confirms defaults `prob_a=0.3`, `seed=20260604`, symlinked canonical images, and regeneration of `_VitPosePP`, `_Yolo26x_detection`, `_Yolo26x_pose` for both outputs |
| 2026-06-16 | Standalone frame prep scripts | `script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py`, `script/dataset_preparation-cleaning/run_prepare_swimxyz_frames_dataset_tmux.sh`, `script/dataset_preparation-cleaning/tutorial.md` | static validation passed; not launched | new home for the frame-preparation flow; tutorial translated into Italian and tmux launch/management documented |
| 2026-06-10 | Side_above_water_frames builder | `script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py`, `script/dataset_preparation-cleaning/run_prepare_swimxyz_frames_dataset_tmux.sh` | static validation passed; not launched | input expected as standalone `COCO__2D_cam` `.png + .txt` pairs in `data/input/subset_xyz/Side_above_water_frames/`; builder reuses canonical COCO bbox/keypoint rules, writes image symlinks, and regenerates `_VitPosePP`, `_Yolo26x_detection`, `_Yolo26x_pose` under `data/intermediate/Side_above_water_frames/` |
| 2026-06-02 | YOLO26x-Pose Test report script | `script/yolo26x_pose_training/evaluate_yolo_pose_split.py` | updated and py_compile passed | script now can write mAP CSV, predicted bbox/keypoint JSON, and VitPose++-style keypoint+bbox overlays for a chosen split; full Test run not launched in this update |
| 2026-06-02 | YOLO26x-Pose final Val report | `runs/pose/runs/yolo26x_pose_side_above_water/yolo26x-pose-incremental-from-cfg04/reports/final_training/` | Val objective satisfied; test-per-checkpoint job stopped | final table combines 5 cfg_04 grid epochs plus 29 incremental epochs; best Val Pose mAP50-95 `0.95705` at total epoch `26` / incremental epoch `21`; heavy tmux test job `yolo26x_pose_final_report_20260602` was killed after objective changed |
| 2026-05-29 | VitPose++ current training plots | `script/vitpose_training/plot_vitpose_training_log.py` on `runs/vitposepp_side_above_water_grid_winner_resume/20260528_061439.log` | PNGs generated | wrote `loss_epoch_avg__20260529_055330_grid_winner_current.png`, `mAP_validation__20260529_055330_grid_winner_current.png`, and `loss_map_summary__20260529_055330_grid_winner_current.csv` under `data/intermediate/Side_above_water/_train_canonical/reports/training_plots/` |
| 2026-05-28 | VitPose++ launcher cleanup | `script_old/run_train_side_above_water_from_grid_best_tmux.sh` | updated to direct launcher | trigger scripts removed; direct launcher now uses `swimxyz_vitposepp_huge_grid_winner_resume.py` with winner best checkpoint `best_AP_epoch_5.pth`; best grid config remains `cfg_02_lr_0.00100_crop_384x128` |
| 2026-05-27 | VitPose++ trigger arming | `tmux` session `vitpose_post_grid_trigger` -> `vitpose_side_above_water_grid_best` | armed and training started | trigger launched downstream training from winner checkpoint `runs/hparam_search/vitposepp_huge/cfg_02_lr_0.00100_crop_384x128/epoch_5.pth`; detector best `cfg_03_lr0_0.00067_imgsz_768`; monitor `patience=5`; work dir `runs/vitposepp_side_above_water_grid_winner_resume/`; initial run log `20260527_204942.log` present |
| 2026-05-25 | YOLO26x-pose incremental training | `runs/pose/runs/yolo26x_pose_side_above_water/yolo26x-pose-incremental-from-cfg04` | stopped at epoch `29`; best Pose mAP50-95 `0.95705` at epoch `21` | best epoch metrics from `results.csv`: Pose P `0.98101`, R `0.98191`, mAP50 `0.99211`, mAP50-95 `0.95705`; last epoch `29` Pose mAP50-95 `0.95271`; `map_plot.png` generated |
| 2026-05-23 | YOLO26x detector top-1 bbox rule | `render_yolo_detection_overlays.py` samples | validation passed | `py_compile` passed; generated top-1 overlay samples for val and test (`20` images each, `no_detection=0`); val/test GT labels already have one bbox per image |
| 2026-05-23 | YOLO26x detection useful training | `runs/yolo26x_bbox_side_above_water/yolo26x-detection_from_cfg03_ep5_20260523_1923` | completed from selected cfg_03; carry forward `weights/best.pt` | start checkpoint `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt`; useful checkpoint `runs/yolo26x_bbox_side_above_water/yolo26x-detection_from_cfg03_ep5_20260523_1923/weights/best.pt`; best mAP50-95 `0.86827` at epoch `1`; last epoch `3` |
| 2026-05-24 | YOLO26x-pose hparam search | `runs/hparam_search/yolo26x_pose/` | grid eval winner: `cfg_04_lr0_0.00100_imgsz_768` (model 04) | selection val-only (no test); update downstream launchers to start from cfg_04 `weights/last.pt` or `weights/best.pt` as appropriate |
| 2026-05-23 | YOLO detector hparam search | `runs/hparam_search/yolo26x_detector_v2/` | grid completed; best=`cfg_03_lr0_0.00067_imgsz_768` | selection val-only (no test) by recall->mAP50-95->imgsz: cfg_03 recall `0.99519`, mAP50-95 `0.86614`; best checkpoint `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/best.pt` (cfg_04 ties metrics) |
| 2026-05-21 | Dataset preparation exporters | `_VitPosePP`, `_Yolo26x_detection`, `_Yolo26x_pose` | generated and verified label-only outputs | all from `_train_canonical`; train `18181`, val `5195`, test `2597`; detection field count `{5: 26973}`, pose field count `{56: 26973}`; no missing/extra label stems |
| 2026-05-20 | YOLO+VitPose++ consolidation | `YoloVitPose_mAP` | consolidated 20-image qualitative sample | output `data/output/experiments/YoloVitPose_mAP/test_20260520_0852_best35_vispose_overlays/`; `20` GT-bbox references, `18` YOLO->VitPose overlays, `2` YOLO no-detection markers at `conf=0.25` |
| 2026-05-20 | YOLO+VitPose++ diagnostics | previous failed YOLO+VitPose run | invalidated and removed | old saved keypoints were not reproducible from identical YOLO bboxes and current VitPose++ checkpoint; do not use old AP `0.6269` result |
| 2026-05-19 | VitPose++ plots | `vitposepp_side_above_water_aniso_20x25_min15` | generated completed-epoch loss/mAP plots | outputs timestamped `20260519_230159_completed_epochs`; includes validation points through epoch 40 |
| 2026-05-16 | VitPose++ | `vitposepp_side_above_water_aniso_20x25_min15` | finished target epoch `40` | best checkpoint `best_AP_epoch_35.pth`; latest `epoch_40.pth`; mild plateau/early overfitting after epoch 35 |

Related experiment notes:
- `docs/ai/experiments/EXP-20260523-yolo26x-detector-grid2x2-v2.md`
- `docs/ai/experiments/EXP-20260527-yolo26x-pose-incremental-cfg04.md`

## Reusable Tools

### `tools/plot_map.py`

This script generates a plot from a CSV file. It is useful for visualizing metrics from training runs.

**Usage:**
```bash
python3 tools/plot_map.py <percorso_del_file_csv> --x_column <nome_colonna_x> --y_column <nome_colonna_y>
```

**Example:**
```bash
python3 tools/plot_map.py runs/pose/runs/yolo26x_pose_side_above_water/yolo26x-pose-incremental-from-cfg04/results.csv --x_column epoch --y_column 'metrics/mAP50-95(P)'
```

## Dataset preparation exporter validation

Commands/results from 2026-05-21:

```bash
python -m py_compile script/dataset_preparation-cleaning/prepare_vitposepp_dataset.py script/dataset_preparation-cleaning/prepare_yolo_detection_dataset.py script/dataset_preparation-cleaning/prepare_yolo_pose_dataset.py
python script/dataset_preparation-cleaning/prepare_vitposepp_dataset.py --overwrite
python script/dataset_preparation-cleaning/prepare_yolo_detection_dataset.py --overwrite
python script/dataset_preparation-cleaning/prepare_yolo_pose_dataset.py --overwrite
```

Verified outputs:
- `_VitPosePP`: train/val/test images+annotations `18181 / 5195 / 2597`; `image_operations: none`.
- `_Yolo26x_detection`: train/val/test label files `18181 / 5195 / 2597`; field count `{5: 26973}`; `image_operations: none`.
- `_Yolo26x_pose`: train/val/test image symlinks and label files `18181 / 5195 / 2597`; field count `{56: 26973}`; `kpt_shape: [17, 3]`; `image_operations: symlink`.
- Detection and pose label stems exactly match canonical image stems for every split; missing/extra counts are `0 / 0`.

YOLO26x-pose setup validation from `CHAT-YOLO26-POSE`:

```bash
conda run -n vitpose python -m py_compile script/dataset_preparation-cleaning/prepare_yolo_pose_dataset.py
bash -n script_old/yolo_training/train_yolo_pose_side_above_water.sh
conda run -n vitpose python script/dataset_preparation-cleaning/prepare_yolo_pose_dataset.py --overwrite
```

Verified `_Yolo26x_pose` after regeneration: train `18181`, val `5195`, test `2597`; each label row has `56` fields. Training was not launched in this validation step.

YOLO26x-pose checkpoint/log policy validation:

```bash
conda run -n vitpose python -m py_compile script/dataset_preparation-cleaning/prepare_yolo_pose_dataset.py script_old/yolo_training/prune_yolo_epoch_checkpoints.py
bash -n script_old/yolo_training/train_yolo_pose_side_above_water.sh
conda run -n vitpose python script_old/yolo_training/prune_yolo_epoch_checkpoints.py --weights-dir /tmp/yolo_prune_test_codex --keep 3
```

The pruner removed older `epoch*.pt` files and retained the latest three periodic checkpoints while leaving `best.pt` and `last.pt` intact.

## YOLO26x-pose 1-epoch smoke run

Date: 2026-05-21 UTC.

Training command used `script_old/yolo_training/train_yolo_pose_side_above_water.sh` with `MODEL=yolo26x-pose.pt`, `EPOCHS=1`, `NAME=yolo26x_pose_coco17_1ep_testmap`, `BATCH=1`, `IMGSZ=1280`, `WORKERS=2`. This run was not launched in tmux; future heavy runs should use tmux by default per user instruction.

Outputs:
- Run dir: `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap`
- Train log: `logs/yolo26x_pose_side_above_water_20260521_133719.log`
- Checkpoints: `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/weights/best.pt`, `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/weights/last.pt`, `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/weights/epoch0.pt`
- Loss plot: `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/reports/loss_epoch1_train_val.png`
- Test mAP plot: `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/reports/map_epoch1_test.png`
- Test metrics CSV: `runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/reports/test_metrics_epoch1.csv`

Metrics:
- Train validation split after epoch 1: Pose mAP50 `0.90037`, Pose mAP50-95 `0.49212`; Box mAP50 `0.94901`, Box mAP50-95 `0.64381`.
- Test split (`2597` images/instances): Pose precision `0.861`, recall `0.902`, mAP50 `0.928`, mAP50-95 `0.537`; Box precision `0.884`, recall `0.933`, mAP50 `0.965`, mAP50-95 `0.656`.
- Test eval log: `logs/yolo26x_pose_test_eval_20260521_1ep.log`

## Metric semantics

- VitPose++ `AP` is COCO keypoint OKS AP averaged over thresholds `0.50:0.05:0.95`; it is the stricter AP/mAP50-95-style metric, not AP50.
- YOLO26x-pose Ultralytics reports Pose mAP50 and Pose mAP50-95. A rigorous direct comparison should export YOLO predictions to COCO keypoint result JSON and evaluate both models with the same COCO/OKS evaluator and GT test split.

## YOLO metrics

| Run | Epoch | Precision | Recall | mAP50 | mAP50-95 | Train Box Loss | Val Box Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `yolo26x_swimmer_gt_bbox_aniso_20x25y_min15_5ep` | 5 | `0.99227` | `0.99249` | `0.99266` | `0.87601` | `0.72344` | `0.58282` |

## VitPose++ validation metrics

| Run | Epoch | AP | AP50 | AP75 | AP(M) | AP(L) | AR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vitposepp_side_above_water_aniso_20x25_min15` | 5 | `0.8604` | `0.9900` | `0.9457` | `0.4083` | `0.8696` | `0.8952` |
| `vitposepp_side_above_water_aniso_20x25_min15` | 10 | `0.9424` | `0.9901` | `0.9697` | `0.4999` | `0.9498` | `0.9544` |
| `vitposepp_side_above_water_aniso_20x25_min15` | 15 | `0.9549` | `0.9901` | `0.9799` | `0.5934` | `0.9612` | `0.9649` |
| `vitposepp_side_above_water_aniso_20x25_min15` | 20 | `0.9727` | `0.9901` | `0.9900` | `0.7892` | `0.9768` | `0.9799` |
| `vitposepp_side_above_water_aniso_20x25_min15` | 25 | `0.9739` | `0.9901` | `0.9900` | `0.8261` | `0.9762` | `0.9818` |
| `vitposepp_side_above_water_aniso_20x25_min15` | 30 | `0.9776` | `0.9901` | `0.9901` | `0.8500` | `0.9806` | `0.9842` |
| `vitposepp_side_above_water_aniso_20x25_min15` | 35 | `0.9821` | `0.9901` | `0.9900` | `0.8444` | `0.9842` | `0.9880` |
| `vitposepp_side_above_water_aniso_20x25_min15` | 40 | `0.9812` | `0.9901` | `0.9901` | `0.7993` | `0.9844` | `0.9871` |

## YOLO+VitPose++ end-to-end metrics

No valid full-test metric is currently recorded for the consolidated `YoloVitPose_mAP` pipeline. Before reporting new end-to-end AP, port or recreate the historical `script_old/yolo_training/evaluate_yolo_vitpose_map.py` workflow under `script/`.

## Qualitative outputs

- Consolidated sample: `data/output/experiments/YoloVitPose_mAP/test_20260520_0852_best35_vispose_overlays/`
- Visualization convention: MMPose `vis_pose_result` predicted skeleton plus red YOLO bbox only.

## Generated plots

Latest completed-epoch plots:
- `data/intermediate/Side_above_water/_train_canonical/reports/training_plots/loss_epoch_avg__20260519_230159_completed_epochs.png`
- `data/intermediate/Side_above_water/_train_canonical/reports/training_plots/mAP_validation__20260519_230159_completed_epochs.png`
- `data/intermediate/Side_above_water/_train_canonical/reports/training_plots/loss_map_summary__20260519_230159_completed_epochs.csv`

## Validation commands

```bash
python -m py_compile script_old/yolo_training/evaluate_yolo_vitpose_map.py script/vitpose_training/pose_overlay_utils.py
```

## Errors and diagnostics

- YOLO OOM diagnostic: `logs/yolo_diagnostic_1280_b8_20260513_135040.log`; `yolo26x.pt`, `imgsz=1280`, `batch=8` exhausted the 32 GB V100.
- VitPose++ resume diagnostic: failed resume attempts are in `runs/vitposepp_side_above_water_aniso_20x25_min15/20260514_184736.log` and `runs/vitposepp_side_above_water_aniso_20x25_min15/20260514_190748.log`; fixed by tolerating invalid checkpoint metadata config during resume.
- Previous failed YOLO+VitPose result was invalidated: saved keypoints diverged from recomputed YOLO->VitPose keypoints despite identical YOLO bboxes.
## YOLO26x detection incremental setup validation

Commands/results from 2026-05-23:

```bash
bash -n script_old/yolo_training/train_yolo_side_above_water.sh
find -L data/intermediate/Side_above_water/_Yolo26x_detection/images/<split> -maxdepth 1 -type f | wc -l
find data/intermediate/Side_above_water/_Yolo26x_detection/labels/<split> -maxdepth 1 -type f -name "*.txt" | wc -l
```

Verified outputs:
- `bash -n` passed for `script_old/yolo_training/train_yolo_side_above_water.sh`.
- `_Yolo26x_detection/images/train -> ../../_train_canonical/train2017`, `val -> ../../_train_canonical/val2017`, `test -> ../../_train_canonical/test2017`.
- Image/label counts match: train `18181 / 18181`, val `5195 / 5195`, test `2597 / 2597`.
- Obsolete detector grid-search caches removed from `runs/hparam_search/yolo26x_detector_v2/dataset_view/labels/`; no detector training was launched.
## YOLO26x detection 1-epoch smoke run

Date: 2026-05-23 UTC.

Launched command via tmux session `yolo26x_det_smoke_1ep_20260523_1740`:

```bash
TAG=smoke_1ep_20260523_1740 EPOCHS=1 RUN_TEST=False script_old/yolo_training/train_yolo_side_above_water.sh
```

Current status as of 2026-05-23 18:16 UTC:
- Run dir: `runs/yolo26x_bbox_side_above_water/yolo26x-detection_smoke_1ep_20260523_1740`
- Log: `logs/yolo26x_detection_side_above_water_20260523_175456.log`
- Status file: `runs/yolo26x_bbox_side_above_water/yolo26x-detection_smoke_1ep_20260523_1740/training_status.txt`
- Startup artifacts written: `args.yaml`, `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg`
- PyTorch CUDA check: `cuda_available=True`, device `Tesla V100S-PCIE-32GB`
- `results.csv`, `best.pt`, and `last.pt` are still pending; no metric should be reported until the epoch completes.
## YOLO26x detector top-1 bbox convention

Date: 2026-05-23 UTC.

Validation:
- `conda run -n vitpose python -m py_compile script_old/yolo_training/yolo_detection_utils.py script_old/yolo_training/render_yolo_detection_overlays.py script_old/yolo_training/preview_yolo_bbox_predictions.py script_old/yolo_training/evaluate_yolo_vitpose_map.py` passed.
- `_Yolo26x_detection/labels/val` has `5195` files and `5195` bbox rows; `_Yolo26x_detection/labels/test` has `2597` files and `2597` bbox rows, so GT val/test labels already contain one bbox per frame.
- Top-1 overlay samples generated with highest-confidence/larger-area tie-break: `data/output/experiments/yolo26x_detection_overlays/top1_val_sample20/` and `data/output/experiments/yolo26x_detection_overlays/top1_test_sample20/`, each with `20` rendered images and `no_detection=0`.
## VitPose++ VideoTest2 random50 subset evaluation

Date: 2026-06-03 UTC.

Generated subset and launcher:
- Subset prep script: `script_old/prepare_vitposepp_video_test2_subset.py`
- tmux launcher: `script_old/run_vitposepp_video_test2_random50_eval_tmux.sh`
- Subset root: `data/intermediate/Side_above_water_VideoTest2/_train_canonical/subsets/random50_seed20260603/`
- Sampled frames: `50` total (`39` from train, `4` from val, `7` from test)

Evaluation outputs:
- Eval work dir: `runs/vitposepp_video_test2_random50_eval_best_AP_epoch_24/`
- Predictions dump: `runs/vitposepp_video_test2_random50_eval_best_AP_epoch_24/test_predictions.pkl`
- COCO-style keypoints JSON: `runs/vitposepp_video_test2_random50_eval_best_AP_epoch_24/result_keypoints.json`
- Eval log: `runs/vitposepp_video_test2_random50_eval_best_AP_epoch_24/test_eval_stdout.log`

Metrics on the subset:
- AP (mAP50-95): `0.8354957059367897`
- AP50: `1.0`
- AP75: `0.9233146792940162`
- AR: `0.858`

Checkpoint used:
- `runs/vitposepp_side_above_water_grid_winner_resume/best_AP_epoch_24.pth`

Notes:
- The job completed in tmux and then exited normally; no overlay export was requested for this subset run.
## VitPose++ VideoTest2 random300 subset evaluation

Date: 2026-06-03 UTC.

Generated subset and launcher:
- Subset prep script: `script_old/prepare_vitposepp_video_test2_subset.py`
- tmux launcher: `script_old/run_vitposepp_video_test2_random50_eval_tmux.sh` with `NUM_FRAMES=300` overrides
- Subset root: `data/intermediate/Side_above_water_VideoTest2/_train_canonical/subsets/random300_seed20260603/`
- Sampled frames: `300` total (`224` from train, `48` from val, `28` from test)

Evaluation outputs:
- Eval work dir: `runs/vitposepp_video_test2_random300_eval_best_AP_epoch_24/`
- Predictions dump: `runs/vitposepp_video_test2_random300_eval_best_AP_epoch_24/test_predictions.pkl`
- COCO-style keypoints JSON: `runs/vitposepp_video_test2_random300_eval_best_AP_epoch_24/result_keypoints.json`
- Eval log: `runs/vitposepp_video_test2_random300_eval_best_AP_epoch_24/test_eval_stdout.log`

Metrics on the subset:
- AP (mAP50-95): `0.8108414950663122`
- AP50: `0.9778092342413344`
- AP75: `0.889591367785784`
- AR: `0.8480000000000001`

Checkpoint used:
- `runs/vitposepp_side_above_water_grid_winner_resume/best_AP_epoch_24.pth`

Notes:
- The job completed in tmux and then exited normally; no overlay export was requested for this subset run.
## VitPose++ VideoTest2 random300 overlay export

Date: 2026-06-03 UTC.

Input:
- Predictions JSON: `runs/vitposepp_video_test2_random300_eval_best_AP_epoch_24/result_keypoints.json`
- Subset root: `data/intermediate/Side_above_water_VideoTest2/_train_canonical/subsets/random300_seed20260603/`

Script:
- `script/vitpose_training/vitpose_generate_test_overlays_from_json.py`

Output directory:
- `data/intermediate/Side_above_water_VideoTest2/_train_canonical/reports/test_overlays/vitposepp_video_test2_random300_best_AP_epoch_24/`

Result:
- `300` overlay images generated with keypoints and GT bboxes drawn from the saved Test2 random300 predictions + annotations.
## 2026-06-16: overlay_sample_comparison VitPose mapping fix

- Updated `script/overlays/overlay_sample_comparison.py` to read VitPose++ image associations from the run manifest when available, instead of relying only on the dataset COCO annotations.
- Verified that `data/output/experiments/vitpose_SAW_frames_20260615/overlays_Test/_manifest.json` maps `image_id 49` to `FSAW_Skin_0_75_Muscle_8__Water_Q_0_75_Hght_0_6__Light_rx_110_roty_360__Spd_3__pos_3_75_000080.jpg`.
- Smoke test: `python3 script/overlays/overlay_sample_comparison.py --n-frame 3 --d-test data/intermediate/SAW_frames/_train_canonical/test2017 --d-compare data/output/experiments/overlay_sample_comparison/SAW_frames__20260614-15 --kp-yolo data/output/experiments/yolo26x-pose_SAW_frames_20260614/kp_Test.json --kp-vitpose data/output/experiments/vitpose_SAW_frames_20260615/kp_Test.json` rendered 3 frames successfully.
## 2026-06-18: VitPose prediction crop-size smoke test

- Script: `script/vitpose_prediction/predict_vitpose_frame.sh`.
- Checkpoint: `runs/vitpose_SAW_frames_20260615/checkpoint/best_AP_epoch_18.pth`.
- Dataset: `data/intermediate/SAW_frames_EntireSwim`, sampled Test view with `--max-test-items 1 --seed 1 --conf 0`.
- Fix verified: effective config uses crop `384x128`, heatmap `96x32`, and backbone `img_size=(128,384)`.
- Result: CPU smoke run completed under `/tmp/vitpose_pred_smoke_crop384/`, wrote `metrics_Test.json`, `kp_Test.json`, `result_keypoints.json`, and one overlay; single-frame AP was `1.0`.
