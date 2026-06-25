# YOLOv8 Pose Training Pipeline Tutorial

## Overview
This tutorial explains the training pipeline for the `yolo26x-pose_SAW_frames_EntireSwim` model using the script `train_yolo26x_pose_SAW_frames_EntireSwim_20260612.sh`.

## Prerequisites
1. Dataset prepared with `prepare_entire_swim_dataset.py`
2. Configuration files in `configs/pipeline/pose/`
3. Base model weights in `models/pose/`

## Pipeline Steps
### 1. Data Preparation
```bash
# Example data preparation command
python script/prepare_entire_swim_dataset.py \
  --input_dir=data/input/entire_swim \
  --output_dir=data/intermediate/pose_dataset
```

### 2. Training Execution
```bash
# Training command from the script
CUDA_VISIBLE_DEVICES=0 \
  python -m torch.distributed.run --nproc_per_node=1 \
  src/train.py \
  --config configs/pipeline/pose/yolo26x_saw_entire_swim.yaml \
  --weights models/pose/yolo26x_pose_side_above_water.pt \
  --output runs/yolo26x-pose_SAW_frames_EntireSwim_20260612
```

### 3. Monitoring & Evaluation
- Training logs: `runs/yolo26x-pose_SAW_frames_EntireSwim_20260612/train.log`
- Metrics: `runs/yolo26x-pose_SAW_frames_EntireSwim_20260612/results.csv`
- Model checkpoints: `runs/yolo26x-pose_SAW_frames_EntireSwim_20260612/weights/`

## Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | 100 | Maximum training epochs |
| `batch_size` | 16 | Per-GPU batch size |
| `imgsz` | 640 | Input image resolution |
| `project` | `runs/yolo26x-pose_SAW_frames_EntireSwim_20260612` | Output directory |

## Troubleshooting
- GPU memory issues: Reduce `batch_size` in the config file
- Overfitting: Add regularization in `configs/pipeline/pose/yolo26x_saw_entire_swim.yaml`

## Next Steps
1. Evaluate results with `script/plot_yolo_pose_metrics.py`
2. Export model with `script/export_vitpose_val_metrics.py`
3. Test on new data with `script/kp_check_swimxyz_video_frames.py`

> Note: Modify `configs/pipeline/pose/yolo26x_saw_entire_swim.yaml` to customize training behavior.