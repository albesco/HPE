# GT-KP-overlays-1

## Goal
- Add a parametrized GT keypoint overlay script under `script/overlays/` that renders COCO-style GT keypoints from either a single image or a directory of images.

## Prepared assets
- Script: `script/overlays/GT_KP_overlays.py`
- Defaults: `--source data/intermediate/Side_above_water/_train_canonical/test2017`, `--annotations data/intermediate/Side_above_water/_train_canonical/annotations/person_keypoints_test.json`, `--output-dir data/intermediate/Side_above_water/_train_canonical/reports/test_overlays/GT`
- Visual style: same skeleton/colors/radius/thickness conventions as `script_old/yolo_training/render_yolo_pose_overlays.py`

## Result
- Script added and syntax-checked with `python3 -m py_compile`.
- Supports image file or directory input, COCO-style annotation file or annotation directory, and optional GT bbox drawing.

## Notes
- Uses COCO dataset-info colors from `src/vitpose_base/configs/_base_/datasets/coco.py`.
