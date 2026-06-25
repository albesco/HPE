# Session 2026-05-19 - CHAT-TRAINING-2

Logical title: VitPose++ Training 2  
Role: training  
Predecessor: `CHAT-TRAINING`

## Scope

This session continued the Side_above_water VitPose++ training/evaluation workflow and added an isolated YOLO+VitPose++ end-to-end mAP experiment on the test split.

## What was done

- Confirmed `CHAT-TRAINING-2` as the active logical training role and updated AI memory around that role.
- Reviewed training plots and logs to decide whether VitPose++ was still improving or showing overfitting risk.
- Modified checkpoint retention so future VitPose++ training keeps the best checkpoint plus latest periodic checkpoints (`last`, `last-1`, `last-2` via `max_keep_ckpts=3`).
- Modified plotting scripts to generate separate loss and mAP plots and to mark best/latest checkpoint epochs.
- Continued VitPose++ training from epoch 30 to epoch 40 after stopping an intermediate target-37 attempt before checkpointing.
- Generated completed-epoch loss and mAP plots through epoch 40.
- Implemented and ran `EspYoloVitPose_mAP`, where YOLO predicts bbox and VitPose++ predicts keypoints inside that bbox, then COCO mAP is computed on the test split.
- Diagnosed and corrected misleading YOLO+VitPose overlay visualization by switching to MMPose `vis_pose_result` for the predicted skeleton and drawing only the YOLO bbox in red.

## Files modified or added

- `script_old/run_resume_side_above_water_to_25ep_tmux.sh`
- `script_old/run_train_side_above_water_10ep.sh`
- `script_old/prepare_swimxyz_vitposepp.py`
- `script/dataset_preparation/prepare_swimxyz_vitposepp_utils.py`
- `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/generated_configs/swimxyz_vitposepp_huge_vitposepp_swap_ears.py`
- `script/vitpose_training/plot_vitpose_training_log.py`
- `script_old/plot_loss_map_summary.py`
- `script/plot-metrics/plot_vitpose_metrics_with_checkpoints.py`
- `script_old/yolo_training/evaluate_yolo_plus_vitpose_map.py`
- `script_old/yolo_training/render_yolo_vitpose_overlays_from_results.py`
- `docs/ai/context.md`
- `docs/ai/task-board.md`
- `docs/ai/chat-index.md`
- `docs/ai/tests-and-results.md`
- `docs/ai/decision-log.md`
- `docs/ai/handoff.md`
- `docs/ai/experiments/EXP-20260516-EspYoloVitPose-mAP.md`
- `docs/ai/sessions/2026-05-19-CHAT-TRAINING-2.md`

## Commands executed

Training resume to epoch 37, then stopped before checkpointing:

```bash
TOTAL_EPOCHS=37 RESUME_FROM=/home/albertosco/HPE/runs/vitposepp_side_above_water_aniso_20x25_min15/latest.pth SESSION_NAME=vitpose_side_above_water_aniso_ep37 BASE_LOG=/home/albertosco/HPE/runs/vitposepp_side_above_water_aniso_20x25_min15/20260514_191100.log bash script_old/run_resume_side_above_water_to_25ep_tmux.sh
```

Final training resume to epoch 40:

```bash
TOTAL_EPOCHS=40 RESUME_FROM=/home/albertosco/HPE/runs/vitposepp_side_above_water_aniso_20x25_min15/epoch_30.pth SESSION_NAME=vitpose_side_above_water_aniso_ep40 BASE_LOG=/home/albertosco/HPE/runs/vitposepp_side_above_water_aniso_20x25_min15/20260514_191100.log bash script_old/run_resume_side_above_water_to_25ep_tmux.sh
```

Generate completed-epoch plots:

```bash
conda run -n vitpose python script/vitpose_training/plot_vitpose_training_log.py \
  --log-file runs/vitposepp_side_above_water_aniso_20x25_min15/20260514_191100.log \
  --log-file runs/vitposepp_side_above_water_aniso_20x25_min15/20260516_100449.log \
  --output-dir data/intermediate/Side_above_water/_train_vitposepp_swap_ears/reports/training_plots \
  --timestamp 20260519_230159_completed_epochs
```

Run YOLO+VitPose++ mAP experiment:

```bash
conda run -n vitpose python script_old/yolo_training/evaluate_yolo_plus_vitpose_map.py \
  --split test \
  --yolo-model runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox_aniso_20x25y_min15_5ep/weights/best.pt \
  --vitpose-checkpoint runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth \
  --overlay-count 20 \
  --seed 20260516 \
  --output-root data/output/experiments/EspYoloVitPose_mAP
```

Render corrected full-test overlays:

```bash
conda run -n vitpose python script_old/yolo_training/render_yolo_vitpose_overlays_from_results.py \
  --split test \
  --checkpoint runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth \
  --results-json data/output/experiments/EspYoloVitPose_mAP/test_20260516_202955/yolo_vitpose_keypoints_results.json \
  --output-dir data/output/experiments/EspYoloVitPose_mAP/test_20260519_230159_best35_vispose_overlays
```

Validation commands:

```bash
python -m py_compile script/vitpose_training/plot_vitpose_training_log.py
python -m py_compile script_old/yolo_training/evaluate_yolo_plus_vitpose_map.py script_old/yolo_training/render_yolo_vitpose_overlays_from_results.py
bash -n script_old/run_resume_side_above_water_to_25ep_tmux.sh
bash -n script_old/run_train_side_above_water_10ep.sh
```

## Results

VitPose++ validation AP by epoch:

| Epoch | AP | AP50 | AP75 | AP(M) | AP(L) | AR |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | `0.8604` | `0.9900` | `0.9457` | `0.4083` | `0.8696` | `0.8952` |
| 10 | `0.9424` | `0.9901` | `0.9697` | `0.4999` | `0.9498` | `0.9544` |
| 15 | `0.9549` | `0.9901` | `0.9799` | `0.5934` | `0.9612` | `0.9649` |
| 20 | `0.9727` | `0.9901` | `0.9900` | `0.7892` | `0.9768` | `0.9799` |
| 25 | `0.9739` | `0.9901` | `0.9900` | `0.8261` | `0.9762` | `0.9818` |
| 30 | `0.9776` | `0.9901` | `0.9901` | `0.8500` | `0.9806` | `0.9842` |
| 35 | `0.9821` | `0.9901` | `0.9900` | `0.8444` | `0.9842` | `0.9880` |
| 40 | `0.9812` | `0.9901` | `0.9901` | `0.7993` | `0.9844` | `0.9871` |

YOLO+VitPose++ test mAP:

- Test images: `2597`
- Predictions: `2531`
- Failures: `66`
- AP: `0.6268862298064438`
- AP50: `0.9580692184411759`
- AP75: `0.8184334166882117`
- AP(M): `0.1461124872991501`
- AP(L): `0.6371632082144546`
- AR: `0.6681940700808625`

## Checkpoints and artifacts

- Best VitPose++ checkpoint: `runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth`
- Latest VitPose++ checkpoint: `runs/vitposepp_side_above_water_aniso_20x25_min15/epoch_40.pth`
- Latest symlink: `runs/vitposepp_side_above_water_aniso_20x25_min15/latest.pth -> epoch_40.pth`
- Retained periodic checkpoints: `epoch_38.pth`, `epoch_39.pth`, `epoch_40.pth`
- Completed-epoch plots: `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/reports/training_plots/loss_epoch_avg__20260519_230159_completed_epochs.png` and `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/reports/training_plots/mAP_validation__20260519_230159_completed_epochs.png`
- YOLO+VitPose++ run dir: `data/output/experiments/EspYoloVitPose_mAP/test_20260516_202955/`
- Corrected YOLO+VitPose++ overlays: `data/output/experiments/EspYoloVitPose_mAP/test_20260519_230159_best35_vispose_overlays/`

## Errors or blockers

- `apply_patch` failed repeatedly because the sandbox helper could not start `bwrap` (`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`); edits were applied with approved shell/Python commands.
- First target-40 tmux attempt exited early without a useful Python traceback in the log; relaunch via wrapper succeeded.
- Original YOLO+VitPose experiment overlays were visually misleading because they mixed GT/custom skeletons; corrected overlays now use MMPose visualization.
- `conda run` buffered stdout during long experiment execution.

## Not verified

- Corrected YOLO+VitPose++ overlays have not received recorded human visual QA.
- The `66` YOLO failures have not been categorized.
- No end-to-end mAP comparison using `epoch_40.pth` instead of `best_AP_epoch_35.pth` is recorded.
- No new YOLO training or VitPose++ fine-tuning with `evaluation.interval=1` was run.

## Recommended next step

Review corrected overlays in `data/output/experiments/EspYoloVitPose_mAP/test_20260519_230159_best35_vispose_overlays/` and inspect `data/output/experiments/EspYoloVitPose_mAP/test_20260516_202955/failures.json` to determine whether the AP drop is driven mainly by YOLO misses, bbox/crop quality, or medium/small swimmer cases.
