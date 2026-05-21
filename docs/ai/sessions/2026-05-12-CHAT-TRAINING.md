# 2026-05-12 - CHAT-TRAINING

Logical role: `CHAT-TRAINING`  
Last updated: 2026-05-15 12:14 UTC

## 1. Session objective

Train and evaluate a robust `Side_above_water` SwimXYZ pipeline with:
- a one-class YOLO detector for swimmer bounding boxes;
- VitPose++ for top-down keypoint prediction;
- reproducible launchers, checkpoint/resume support, progress monitoring, and qualitative overlays.

The priority decision for this session was accuracy of keypoint prediction, not training or inference speed.

## 2. VitPose++ configurations created or modified

- Active prepared dataset: `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/`
- Active generated config: `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/generated_configs/swimxyz_vitposepp_huge_single_head_swap_ears.py`
- Preparation scripts now generate GT bboxes with anisotropic padding:
  - horizontal padding ratio: `0.20`;
  - vertical padding ratio: `0.25`;
  - minimum padding: `15 px` per side;
  - clipping to image boundaries.
- `script/prepare_swimxyz_vitposepp_single_head.py` and `script/prepare_swimxyz_vitposepp_utils.py` were extended for anisotropic padding and checkpoint retention options.
- VitPose++ training code was extended with a current-status file:
  - `src/vitpose_base/tools/train.py` accepts `--status-file` and `--status-interval`;
  - `src/vitpose_base/mmpose/apis/train.py` writes current epoch, iteration, progress, ETA, and work dir.
- VitPose++ resume was patched to tolerate invalid checkpoint metadata config during `runner.resume`, preserving checkpoint state loading while ignoring unparsable `meta.config`.
- Checkpoint convention for active/future launches:
  - keep the latest periodic checkpoint with `checkpoint_config.max_keep_ckpts=1`;
  - preserve validation best checkpoint with `evaluation.save_best='AP'`;
  - keep `latest.pth` as symlink to the current latest periodic checkpoint.

## 3. Launchers created or modified

- `script/run_train_side_above_water_10ep.sh`
  - launcher for the original 10-epoch VitPose++ run with periodic checkpoints and plots.
- `script/run_resume_side_above_water_to_25ep_tmux.sh`
  - tmux launcher for the active anisotropic VitPose++ resume run;
  - now targets `max_epochs=30` from the retained checkpoint;
  - writes `runs/vitposepp_side_above_water_aniso_20x25_min15/training_status.txt`.
- `script/yolo_training/train_yolo_side_above_water.sh`
  - dedicated YOLO launcher under the separate YOLO training directory.
- Supporting scripts added/used during the session:
  - YOLO dataset preparation and evaluation/preview scripts under `script/yolo_training/`;
  - keypoint/bbox overlay helpers in `script/pose_overlay_utils.py`;
  - plot generation through `script/plot_vitpose_training_log.py`;
  - GT bbox visualization through `script/visualize_gt_bboxes.py`.

## 4. Checkpoints produced or removed

### VitPose++

- Original `runs/vitposepp_single_head_subset_xyz_swap_ears/` training artifacts were treated as obsolete after the bbox convention and dataset changed.
- Active work dir: `runs/vitposepp_side_above_water_aniso_20x25_min15/`
- First anisotropic GT run was stopped after `epoch_4.pth`.
- `epoch_1.pth`, `epoch_2.pth`, and `epoch_3.pth` were removed on request.
- Resume run continued from `epoch_4.pth`.
- Current checkpoint state at 2026-05-15 12:14 UTC:
  - `best_AP_epoch_25.pth` is the current best validation checkpoint;
  - `epoch_26.pth` is the latest periodic checkpoint;
  - `latest.pth -> epoch_26.pth`;
  - active training is in epoch `27/30`.
- Each VitPose++ checkpoint is about `11G` on disk.

### YOLO

- Current retained YOLO run:
  - `runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox_aniso_20x25y_min15_5ep/`
  - retained weights: `weights/best.pt`, `weights/last.pt`.
- Earlier YOLO padded-10 and padded-20 experiments were superseded by the anisotropic padding convention; relevant metrics/notes were retained in documentation.

## 5. YOLO experiments

- A one-class `swimmer` YOLO dataset was generated from the prepared VitPose++ annotations.
- YOLO conversion reads already padded GT bboxes and applies no extra padding (`bbox_padding_ratio=0.0`).
- Diagnostic attempt with `yolo26x.pt`, `imgsz=1280`, `batch=8` failed on the 32 GB V100 with CUDA OOM.
- Launcher defaults were reduced to `batch=2` and `workers=2`.
- Padding experiments:
  - uniform `0.10`: training stopped after epoch 11; epoch-10/best metrics were good but convention was superseded;
  - uniform `0.20`: smoke training was stopped before completion to switch padding convention;
  - anisotropic `0.20x / 0.25y / min 15 px`: 5-epoch smoke training completed successfully.

## 6. VitPose++ experiments

- Original Side_above_water 10-epoch run completed quickly enough to disprove the initial rough duration estimate.
- Qualitative preview issue was found: custom inference passed `xyxy` bboxes into MMPose `_box2cs`, which expects `xywh`; overlay scripts were consolidated to avoid inconsistent visualization.
- GT bboxes were visually judged too tight, leading to padded GT dataset regeneration.
- Active anisotropic VitPose++ experiment:
  - started from scratch, stopped at `epoch_4.pth`;
  - older checkpoints removed;
  - resumed via tmux from `epoch_4.pth`;
  - current target: `30` total epochs;
  - status is readable from `training_status.txt`.

## 7. Metrics

### YOLO anisotropic 5-epoch run

| Metric | Value |
|---|---:|
| Precision | `0.99227` |
| Recall | `0.99249` |
| mAP50 | `0.99266` |
| mAP50-95 | `0.87601` |
| Train box loss | `0.72344` |
| Val box loss | `0.58282` |

### VitPose++ validation AP during active run

| Epoch | AP | AP50 | AP75 | AP(M) | AP(L) | AR |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | `0.8604` | `0.9900` | `0.9457` | `0.4083` | `0.8696` | `0.8952` |
| 10 | `0.9424` | `0.9901` | `0.9697` | `0.4999` | `0.9498` | `0.9544` |
| 15 | `0.9549` | `0.9901` | `0.9799` | `0.5934` | `0.9612` | `0.9649` |
| 20 | `0.9727` | `0.9901` | `0.9900` | `0.7892` | `0.9768` | `0.9799` |
| 25 | `0.9739` | `0.9901` | `0.9900` | `0.8261` | `0.9762` | `0.9818` |

Generated plot files from the active log include:
- `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/reports/training_plots/loss_epoch_avg__20260515_090421.png`
- `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/reports/training_plots/mAP_validation__20260515_090421.png`

Those plot files were produced before epoch-25 validation completed, so the documented table is newer than the current plot images.

## 8. Errors encountered

- YOLO OOM:
  - `yolo26x.pt`, `imgsz=1280`, `batch=8` exhausted the V100 32 GB GPU;
  - diagnostic log: `logs/yolo_diagnostic_1280_b8_20260513_135040.log`;
  - mitigation: `batch=2`, `workers=2`.
- VitPose++ plot parsing:
  - an early plot command failed with `RuntimeError: No training epoch lines found in log file`;
  - the issue was caused by parsing a log format/file that did not contain the expected training lines.
- Keypoint preview visualization:
  - incorrect bbox format (`xyxy` vs `xywh`) produced misleading overlays;
  - fixed by centralizing bbox/keypoint drawing and inference helpers.
- VitPose++ resume:
  - resume initially failed because checkpoint metadata contained an unparsable config string;
  - patched training API to ignore invalid metadata config while preserving resume state.
- tmux/process handling:
  - at least one previous launch was not active in tmux when checked; active long runs should be launched and monitored through named tmux sessions.

## 9. Decisions on bbox padding

- Do not train YOLO or VitPose++ with the original tight GT bboxes.
- Store padding directly in the prepared GT annotations used by both models.
- Do not apply an additional padding step during YOLO conversion or after YOLO prediction unless a new experiment explicitly changes the convention.
- Current durable convention:
  - horizontal padding `20%`;
  - vertical padding `25%`;
  - minimum `15 px` per side;
  - clipped at image boundaries.

## 10. Current resume state

At 2026-05-15 12:14 UTC:

```text
tmux session: vitpose_side_above_water_aniso_resume
work dir: runs/vitposepp_side_above_water_aniso_20x25_min15/
epoch: 27 / 30
iteration in epoch: 4260 / 4546
global progress: 89.79%
ETA from status file: 6978 seconds
latest periodic checkpoint: epoch_26.pth
latest symlink: latest.pth -> epoch_26.pth
best checkpoint: best_AP_epoch_25.pth
```

Status file:

`runs/vitposepp_side_above_water_aniso_20x25_min15/training_status.txt`

## 11. Next steps

1. Let VitPose++ finish epoch 30.
2. Regenerate loss and mAP plots after the final validation.
3. Evaluate the YOLO + VitPose++ pipeline end-to-end on test images.
4. Generate qualitative overlays with YOLO bbox and VitPose++ keypoints.
5. Compare results against GT-bbox VitPose++ evaluation.
6. Decide whether to continue incremental training from the best or latest checkpoint.

## 12. Elements not verified

- Final epoch-30 VitPose++ metrics are not yet available.
- End-to-end YOLO + VitPose++ quantitative evaluation is not yet documented.
- Best-vs-latest checkpoint choice after epoch 30 is not yet decided.
- Final plots including epoch 25+ metrics need to be regenerated.
- Whether YOLO trained for only 5 epochs is sufficient for final inference quality remains to be validated on qualitative and quantitative downstream pose results.
