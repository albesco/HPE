# EXP-20260514 - VitPose++ anisotropic GT training stopped at epoch 4

## Status

Superseded by resumed training.

This experiment captured the controlled stop after `epoch_4.pth`; the same work dir was later resumed toward epoch `30`.

## Work dir

`runs/vitposepp_side_above_water_aniso_20x25_min15/`

## Resume checkpoint

`runs/vitposepp_side_above_water_aniso_20x25_min15/epoch_4.pth`

At the time of the stop, `latest.pth` pointed to this checkpoint. After resume, `latest.pth` advances with the latest retained periodic checkpoint.

## Current convention

Future VitPose++ launches should:
- keep only the latest periodic checkpoint with `checkpoint_config.max_keep_ckpts=1`;
- preserve best validation checkpoint with `evaluation.save_best='AP'`.

## Follow-up

Follow-up experiment:

`docs/ai/experiments/EXP-20260514-vitpose-aniso-resume-30ep.md`
