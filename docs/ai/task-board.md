# AI Task Board

## Backlog
- Define exact SwimXYZ input schema.
- Define target VitPose++ annotation format.
- Implement dataset conversion script.
- Add validation script for converted annotations.
- Create training config.
- Create training launcher.
- Document reproducible workflow.

## In progress
- Side_above_water: train VitPose++ on anisotropic GT bboxes in tmux for 30 epochs.

## Done
- Prepared Side_above_water YOLO detection dataset from GT bboxes.
- Added dedicated YOLO training scripts under `script/yolo_training/`.
- Added lightweight VitPose++ training status monitor file support.
- Generated Side_above_water validation GT bbox-only overlays in `data/intermediate/bbox_val/`.
- Regenerated Side_above_water VitPose++ train/val/test annotations with GT bbox padding ratio `0.10`.
- Regenerated Side_above_water YOLO dataset from the padded GT annotations with no extra YOLO-side padding.
- Stopped YOLO training after epoch 11, froze best/last checkpoints, evaluated epoch-10/best checkpoint, and generated 20 random validation bbox previews.
- Regenerated Side_above_water VitPose++ and YOLO datasets with GT bbox padding ratio `0.20`.
- Stopped YOLO padding-20 smoke training before completion to switch to anisotropic GT bbox padding.
- Regenerated Side_above_water VitPose++ and YOLO datasets with anisotropic GT bbox padding: `0.20` horizontal, `0.25` vertical, min `15 px`.
- Completed YOLO anisotropic-padding smoke training for 5 epochs; best epoch-5 metrics: precision `0.99227`, recall `0.99249`, mAP50 `0.99266`, mAP50-95 `0.87601`.
- Cleaned obsolete `data/intermediate` visual preview directories and removed non-annotation `*_with-KP.jpg` previews from the active VitPose++ dataset.
