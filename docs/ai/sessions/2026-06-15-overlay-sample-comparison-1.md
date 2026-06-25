# overlay-sample-comparison-1

## Goal
- Add a parametrized script under `script/overlays/` that compares Yolo26x-Pose and VitPose++ overlays on random Test frames.

## Prepared assets
- Script: `script/overlays/overlay_sample_comparison.py`
- Default test frames: `data/intermediate/SAW_frames_EntireSwim/_train_canonical/test2017`
- Default KP inputs: `data/output/experiments/yolo26x-pose_SAW_frames_EntireSwim_20260612/kp_Test.json` and `data/output/experiments/vitpose_SAW_frames_EntireSwim_20260612/kp_Test.json`
- Default comparison output: `data/output/experiments/overlay_sample_comparison`

## Result
- Script added and syntax-checked with `python3 -m py_compile`.
- Samples random shared Test frames, copies the original frame into the comparison directory, and renders paired `_Yolo26x-Pose` and `_VitPosePP` overlays.

## Notes
- VitPose++ KP input is mapped back to file names through the inferred Test COCO annotations file at `<D-TEST>/../annotations/person_keypoints_test.json`, unless `--dataset-annotations` is provided.
