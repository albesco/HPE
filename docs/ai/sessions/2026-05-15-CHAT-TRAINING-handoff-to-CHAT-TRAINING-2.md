# 2026-05-15 - CHAT-TRAINING handoff to CHAT-TRAINING-2

## Purpose

Close `CHAT-TRAINING` and prepare the handoff to:

- Logical session: `CHAT-TRAINING-2`
- Logical title: VitPose++ Training 2
- Role: training
- Predecessor: `CHAT-TRAINING`

This note is based on workspace files only. Compacted chat memory is not a technical source of truth.

## Files read

- `AGENTS.md`
- `docs/ai/context.md`
- `docs/ai/task-board.md`
- `docs/ai/decision-log.md`
- `docs/ai/chat-index.md`
- `docs/ai/chat-roles.md`
- `docs/ai/start-here.md`
- `docs/ai/handoff.md`
- `docs/ai/tests-and-results.md`
- `docs/ai/experiments/EXP-20260514-vitpose-aniso-resume-30ep.md`
- `docs/ai/experiments/EXP-20260514-vitpose-aniso-stopped-epoch4.md`
- `docs/ai/experiments/EXP-20260514-yolo-aniso-5ep.md`
- `docs/ai/experiments/EXP-20260514-yolo-padded10-epoch10.md`
- `docs/ai/sessions/2026-05-11-CHAT-DATASET.md`
- `docs/ai/sessions/2026-05-12-CHAT-TRAINING.md`
- `runs/vitposepp_side_above_water_aniso_20x25_min15/training_status.txt`

## Current training state

File-based state at `2026-05-15 13:56:22 UTC` from `training_status.txt`:

```text
phase=train
epoch=29
max_epochs=30
iter_in_epoch=3160
iters_per_epoch=4546
global_iter=130448
total_iters=136380
progress_pct=95.65
eta_seconds=3069
work_dir=runs/vitposepp_side_above_water_aniso_20x25_min15
```

The generated status file contains a local absolute work dir; this session note records only the relative path.

## Checkpoint state

Visible files in `runs/vitposepp_side_above_water_aniso_20x25_min15/` at handoff:

- `best_AP_epoch_25.pth`
- `epoch_28.pth`
- `latest.pth -> epoch_28.pth`
- `training_status.txt`
- log files: `20260514_150859.log`, `20260514_184736.log`, `20260514_190748.log`, `20260514_191100.log` and related JSON logs

Checkpoint to use for generic resume at handoff:

`runs/vitposepp_side_above_water_aniso_20x25_min15/latest.pth`

Checkpoint to use for best-known validation model at handoff:

`runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_25.pth`

Which checkpoint should be used for the next training/evaluation decision is not yet decided.

## Work dir

`runs/vitposepp_side_above_water_aniso_20x25_min15/`

## Config and launchers

- VitPose++ config:
  - `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/generated_configs/swimxyz_vitposepp_huge_single_head_swap_ears.py`
- VitPose++ tmux/resume launcher:
  - `script/run_resume_side_above_water_to_25ep_tmux.sh`
- Original 10-epoch launcher:
  - `script/run_train_side_above_water_10ep.sh`
- YOLO training launcher:
  - `script/yolo_training/train_yolo_side_above_water.sh`
- YOLO dataset preparation:
  - `script/yolo_training/prepare_yolo_detection_dataset.py`

Launcher values reconstructed from `script/run_resume_side_above_water_to_25ep_tmux.sh`:

- default work dir: `runs/vitposepp_side_above_water_aniso_20x25_min15`
- default resume source: `runs/vitposepp_side_above_water_aniso_20x25_min15/epoch_4.pth`
- default total epochs: `30`
- default tmux session: `vitpose_side_above_water_aniso_resume`
- default status file: `runs/vitposepp_side_above_water_aniso_20x25_min15/training_status.txt`

## Current bbox convention

Durable bbox convention from docs and scripts:

- GT bboxes are padded during VitPose++ dataset preparation.
- Horizontal padding ratio: `0.20`.
- Vertical padding ratio: `0.25`.
- Minimum padding: `15 px` per side.
- Padding is clipped to image boundaries.
- YOLO conversion default extra padding is `0.0`.
- Do not add extra YOLO-side or downstream padding unless a new experiment explicitly changes the convention.

## Current checkpoint convention

- Periodic VitPose++ checkpoints keep only latest checkpoint: `checkpoint_config.max_keep_ckpts=1`.
- The generated config uses `checkpoint_config = dict(interval=1, max_keep_ckpts=1, create_symlink=True)`.
- The generated config uses `evaluation = dict(interval=5, metric='mAP', save_best='AP')`.
- Best validation checkpoint is preserved separately.
- Each VitPose++ checkpoint is documented as about `11G`.

## Documented YOLO results

Current YOLO run:

`runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox_aniso_20x25y_min15_5ep/`

Visible files:

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`

Documented metrics:

- precision: `0.99227`
- recall: `0.99249`
- mAP50: `0.99266`
- mAP50-95: `0.87601`
- train box loss: `0.72344`
- val box loss: `0.58282`

Earlier YOLO padded-10 result is documented but superseded by the anisotropic convention:

- precision: `0.985`
- recall: `0.986`
- mAP50: `0.992`
- mAP50-95: `0.872`

## Documented VitPose++ results

Validation AP documented through epoch 25:

| Epoch | AP | AP50 | AP75 | AP(M) | AP(L) | AR |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | `0.8604` | `0.9900` | `0.9457` | `0.4083` | `0.8696` | `0.8952` |
| 10 | `0.9424` | `0.9901` | `0.9697` | `0.4999` | `0.9498` | `0.9544` |
| 15 | `0.9549` | `0.9901` | `0.9799` | `0.5934` | `0.9612` | `0.9649` |
| 20 | `0.9727` | `0.9901` | `0.9900` | `0.7892` | `0.9768` | `0.9799` |
| 25 | `0.9739` | `0.9901` | `0.9900` | `0.8261` | `0.9762` | `0.9818` |

Generated plot files documented:

- `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/reports/training_plots/loss_epoch_avg__20260515_090421.png`
- `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/reports/training_plots/mAP_validation__20260515_090421.png`

The plot files are documented as older than the epoch-25 metric table.

## Known errors

- YOLO OOM with `yolo26x.pt`, `imgsz=1280`, `batch=8` on 32 GB V100; diagnostic log: `logs/yolo_diagnostic_1280_b8_20260513_135040.log`.
- VitPose++ resume failed initially due to unparsable checkpoint metadata config; code now tolerates invalid metadata config during resume.
- Plot parsing previously failed with `RuntimeError: No training epoch lines found in log file`.
- Preview overlays were previously misleading because `xyxy` bboxes were passed to MMPose code expecting `xywh`.

## Recommended next actions for CHAT-TRAINING-2

1. Read `AGENTS.md`, `docs/ai/start-here.md`, and the training-related memory files.
2. Check `runs/vitposepp_side_above_water_aniso_20x25_min15/training_status.txt`.
3. Verify whether epoch 30 completed and whether `latest.pth` advanced beyond `epoch_28.pth`.
4. Regenerate loss/mAP plots after final validation.
5. Update `docs/ai/tests-and-results.md` with final epoch-30 metrics.
6. Run YOLO + VitPose++ qualitative previews on test images.
7. Compare GT-bbox and YOLO-bbox VitPose++ evaluation results.
8. Decide whether future incremental training should resume from the best checkpoint or latest checkpoint.

## Not verified

- Final epoch-30 VitPose++ metrics.
- Final epoch-30 checkpoint state.
- Whether the tmux process is still active.
- End-to-end YOLO + VitPose++ quantitative evaluation.
- Qualitative YOLO + VitPose++ preview after the active VitPose++ run finishes.
- Whether the 5-epoch YOLO detector is sufficient for final downstream pose accuracy.

## Not reconstructible from workspace files

- Exact interactive command history for all launches.
- Details present only in compacted chat memory and not copied into `docs/ai/`.
- Whether the tmux process is active without running a process command.
- Human visual judgments that were not written into a persisted document.
