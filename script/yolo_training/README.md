# YOLO Detection Training

Scripts for the Side_above_water swimmer detector used before VitPose++.

## Prepare Dataset

```bash
python script/yolo_training/prepare_yolo_detection_dataset.py --overwrite --link-mode symlink
```

Output:

- Dataset root: `data/intermediate/Side_above_water/_yolo_detection/`
- Data YAML: `data/intermediate/Side_above_water/_yolo_detection/swimxyz_side_above_water_yolo.yaml`
- Class: `swimmer`

The converter reads GT bboxes from the prepared COCO keypoint annotations and writes Ultralytics labels.
The current Side_above_water GT annotations already contain the agreed bbox padding, so the YOLO converter must keep `--bbox-padding-ratio 0.0` to avoid applying padding twice.

When YOLO predictions are later passed to VitPose++, do not add another bbox padding step unless the detector has been retrained/evaluated with that same convention. The intended pipeline is: padded GT bbox for YOLO training, YOLO-predicted bbox as-is for VitPose++ inference.

## Train YOLO

```bash
script/yolo_training/train_yolo_side_above_water.sh
```

Defaults:

- Model: `models/detection/yolo26x.pt`
- Epochs: `100`
- Image size: `1280`
- Batch: `2`
- Output: `runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox/`
- Log: `logs/yolo_side_above_water_<timestamp>.log`

Override defaults with environment variables:

```bash
EPOCHS=200 BATCH=1 IMGSZ=1536 script/yolo_training/train_yolo_side_above_water.sh
```

`yolo26x.pt` with `imgsz=1280` and `batch=8` exceeds the 32 GB V100 GPU memory.
