# EXP-20260514 - VitPose++ anisotropic GT resume to 30 epochs

## Status

In progress at 2026-05-15 12:14 UTC.

## Goal

Continue VitPose++ training on `Side_above_water` after the controlled stop at epoch 4, using the anisotropic GT bbox convention and retaining only latest periodic plus best validation checkpoints.

## Dataset

`data/intermediate/Side_above_water/_train_vitposepp_swap_ears/`

GT bbox convention:
- horizontal padding: `0.20`
- vertical padding: `0.25`
- minimum: `15 px` per side
- clipped to image boundaries

## Run

`runs/vitposepp_side_above_water_aniso_20x25_min15/`

Resume source:

`runs/vitposepp_side_above_water_aniso_20x25_min15/epoch_4.pth`

Active tmux session:

`vitpose_side_above_water_aniso_resume`

## Current state

At 2026-05-15 12:14 UTC:
- epoch: `27/30`
- iteration in epoch: `4260/4546`
- progress: `89.79%`
- latest periodic checkpoint: `epoch_26.pth`
- latest symlink: `latest.pth -> epoch_26.pth`
- best checkpoint: `best_AP_epoch_25.pth`

File-based update at 2026-05-15 13:56 UTC:
- epoch: `29/30`
- iteration in epoch: `3160/4546`
- progress: `95.65%`
- latest periodic checkpoint visible on disk: `epoch_28.pth`
- latest symlink: `latest.pth -> epoch_28.pth`
- best checkpoint visible on disk: `best_AP_epoch_25.pth`

## Validation metrics

| Epoch | AP | AP50 | AP75 | AP(M) | AP(L) | AR |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | `0.8604` | `0.9900` | `0.9457` | `0.4083` | `0.8696` | `0.8952` |
| 10 | `0.9424` | `0.9901` | `0.9697` | `0.4999` | `0.9498` | `0.9544` |
| 15 | `0.9549` | `0.9901` | `0.9799` | `0.5934` | `0.9612` | `0.9649` |
| 20 | `0.9727` | `0.9901` | `0.9900` | `0.7892` | `0.9768` | `0.9799` |
| 25 | `0.9739` | `0.9901` | `0.9900` | `0.8261` | `0.9762` | `0.9818` |

## Notes

- The run uses a `training_status.txt` file for quick progress checks.
- Resume required a patch to ignore invalid checkpoint metadata config during `runner.resume`.
- Plot files should be regenerated after epoch 30 so they include all validation points.
