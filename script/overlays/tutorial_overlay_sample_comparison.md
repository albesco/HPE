# Tutorial: overlay_sample_comparison.py

This guide explains how to use `script/overlays/overlay_sample_comparison.py` to compare the qualitative overlays produced by Yolo26x-Pose and VitPose++ on randomly sampled Test frames.

## 1. What this script does

The script:

1. samples `N-FRAME` random frames from `D-TEST`,
2. loads the Yolo26x-Pose keypoints from `KP-YOLO`,
3. loads the VitPose++ keypoints from `KP-VITPOSE`,
4. renders two overlays for each selected frame using the same drawing style as `script/overlays/GT_KP_overlays.py`,
5. copies the original frame into the comparison directory, and
6. saves the results in `D-COMPARE` with a per-frame manifest.

For each selected image, the script writes:

- the original frame with the same file name,
- the GT overlay with suffix `_GT`,
- the Yolo26x-Pose overlay with suffix `_Yolo26x-Pose`,
- the VitPose++ overlay with suffix `_VitPosePP`,
- a `_manifest.json` file in the comparison directory.

---

## 2. Prerequisites

Before running the script, make sure:

- the Test frames exist in `D-TEST`,
- `KP-YOLO` points to a valid Yolo26x-Pose `kp_Test.json`,
- `KP-VITPOSE` points to a valid VitPose++ `kp_Test.json`,
- the matching COCO Test annotations file exists at `<D-TEST>/../annotations/person_keypoints_test.json`, or you pass it explicitly with `--dataset-annotations`,
- if available, the VitPose run manifest exists at `<KP-VITPOSE>/../overlays_Test/_manifest.json` so the script can map `image_id` values back to the original file names used by the VitPose run,
- the expected Python environment is available.

The renderer uses the COCO dataset-info colors from:

```text
src/vitpose_base/configs/_base_/datasets/coco.py
```

---

## 3. Minimum working example

From the repository root:

```bash
python3 script/overlays/overlay_sample_comparison.py \
  --n-frame 20 \
  --d-test data/intermediate/SAW_frames_EntireSwim/_train_canonical/test2017 \
  --d-compare data/output/experiments/overlay_sample_comparison \
  --kp-yolo data/output/experiments/yolo26x-pose_SAW_frames_EntireSwim_20260612/kp_Test.json \
  --kp-vitpose data/output/experiments/vitpose_SAW_frames_EntireSwim_20260612/kp_Test.json
```

This command:

- samples `20` random Test frames,
- writes the original frame plus both overlays,
- saves everything in `data/output/experiments/overlay_sample_comparison`.

---

## 4. Parameter reference

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--n-frame` | `20` | Number of random Test frames to compare. Use `0` or a value larger than the available shared frames to render all matching frames. |
| `--d-test` | `data/intermediate/SAW_frames_EntireSwim/_train_canonical/test2017` | Directory containing the original Test images. |
| `--d-compare` | `data/output/experiments/overlay_sample_comparison` | Output directory for the copied originals, overlay images, and `_manifest.json`. |
| `--kp-yolo` | `data/output/experiments/yolo26x-pose_SAW_frames_EntireSwim_20260612/kp_Test.json` | Yolo26x-Pose keypoint results for Test. |
| `--kp-vitpose` | `data/output/experiments/vitpose_SAW_frames_EntireSwim_20260612/kp_Test.json` | VitPose++ keypoint results for Test. |
| `--dataset-annotations` | inferred as `<D-TEST>/../annotations/person_keypoints_test.json` | COCO Test annotations used to map VitPose `image_id` values back to image file names. Pass this explicitly if your layout is different. |
| `--dataset-info` | `src/vitpose_base/configs/_base_/datasets/coco.py` | COCO skeleton and color palette used to draw the overlays. |
| `--vitpose-manifest` | inferred as `<KP-VITPOSE>/../overlays_Test/_manifest.json` when present | VitPose run manifest used to map `image_id` values to file names. This is the preferred source for VitPose frame association. |
| `--seed` | `0` | Random seed used for sampling the frames. |
| `--kpt-score-thr` | `0.3` | Minimum keypoint confidence required before drawing a joint or link. |
| `--radius` | `3` | Keypoint circle radius. |
| `--thickness` | `2` | Skeleton line thickness. |

### Notes on the most important options

- `--d-test` must point to the directory that contains the original Test frames.
- `--kp-yolo` and `--kp-vitpose` must both refer to JSON outputs computed on the same Test split.
- `--dataset-annotations` is still needed for GT overlays, because the GT frame labels come from the COCO annotations.
- `--vitpose-manifest` is the safest way to resolve VitPose overlays when the run produced an `overlays_Test/_manifest.json`.
- `--n-frame` is capped by the number of frames that are present in all three inputs.
- `--seed` makes the random sampling reproducible.

---

## 5. Example usages

### A. Compare 10 random Test frames

```bash
python3 script/overlays/overlay_sample_comparison.py \
  --n-frame 10
```

Use this when the defaults already match your dataset layout.

### B. Compare a custom Test directory

```bash
python3 script/overlays/overlay_sample_comparison.py \
  --n-frame 20 \
  --d-test data/intermediate/Side_above_water/_train_canonical/test2017 \
  --d-compare data/output/experiments/sample_compare_side_above_water
```

Use this when you want to store comparison outputs in a separate experiment folder.

### C. Reproduce the same sample set

```bash
python3 script/overlays/overlay_sample_comparison.py \
  --n-frame 20 \
  --seed 42
```

Use the same seed again to get the same sampled frames, provided the input files do not change.

### D. Use a non-standard annotations location

```bash
python3 script/overlays/overlay_sample_comparison.py \
  --n-frame 20 \
  --dataset-annotations data/intermediate/Side_above_water_VideoTest2/_train_canonical/annotations/person_keypoints_test.json
```

Use this when the Test annotations file is stored somewhere else.

---

## 6. Output layout

A typical comparison directory looks like this:

```text
D-COMPARE/
  sample_001.jpg
  sample_001_GT.jpg
  sample_001_Yolo26x-Pose.jpg
  sample_001_VitPosePP.jpg
  sample_002.jpg
  sample_002_Yolo26x-Pose.jpg
  sample_002_VitPosePP.jpg
  _manifest.json
```

The manifest records:

- the requested and rendered frame counts,
- the resolved input paths,
- one entry per selected frame with the original and overlay paths.

---

## 7. Recommended workflow

1. Start with the defaults if your Test split is the standard SAW layout.
2. Run the script with a small `--n-frame` first, such as `10` or `20`.
3. Inspect the paired overlays and the originals side by side.
4. Increase `--n-frame` if you want a broader qualitative sample.
5. Use a fixed `--seed` when you want a reproducible comparison set.

This keeps qualitative comparisons simple and makes Yolo26x-Pose and VitPose++ easier to review on the same frames.
