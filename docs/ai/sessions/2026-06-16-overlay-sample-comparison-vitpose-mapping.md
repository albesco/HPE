# 2026-06-16 overlay sample comparison VitPose mapping

- Fixed `script/overlays/overlay_sample_comparison.py` so VitPose++ keypoints are associated with image files through the run manifest at `<KP-VITPOSE>/../overlays_Test/_manifest.json` when available.
- Kept the GT overlay output as `_GT` in the comparison directory, alongside `_Yolo26x-Pose` and `_VitPosePP`.
- Confirmed the current VitPose++ run manifest maps `image_id 49` to `FSAW_Skin_0_75_Muscle_8__Water_Q_0_75_Hght_0_6__Light_rx_110_roty_360__Spd_3__pos_3_75_000080.jpg`, which explains the earlier mismatch against the COCO Test annotations.
- Validation: `python3 -m py_compile script/overlays/overlay_sample_comparison.py script/overlays/GT_KP_overlays.py` and a smoke test using the user command both passed.
