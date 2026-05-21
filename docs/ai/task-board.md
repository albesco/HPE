# AI Task Board

## Backlog
- Define exact SwimXYZ input schema.
- Define target VitPose++ annotation format.
- Implement dataset conversion script.
- Add validation script for converted annotations.
- Document reproducible workflow.
- Rerun full-test `YoloVitPose_mAP` metrics with the consolidated pipeline.
- Decide whether YOLO no-detection cases need confidence tuning or fallback behavior.

## In progress
- None recorded.

## Done
- Prepared Side_above_water YOLO detection dataset from GT bboxes.
- Added dedicated YOLO training scripts under `script/yolo_training/`.
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
