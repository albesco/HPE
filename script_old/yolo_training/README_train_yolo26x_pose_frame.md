# Train YOLO26x Pose on Frame+Label Datasets

Parametric training pipeline for YOLO pose training on canonical frame+label datasets.

## Quick Start

Assuming your frame+label dataset has been prepared with `script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py`:

```bash
# Basic usage (with defaults)
bash script/yolo_training/train_yolo26x_pose_frame.sh \
  --dataset-dir data/intermediate/Side_above_water_frames

# With custom run name and hyperparameters
bash script/yolo_training/train_yolo26x_pose_frame.sh \
  --run-name my_experiment_20260613_1430 \
  --dataset-dir data/intermediate/Side_above_water_frames \
  --imgsz 640 \
  --lr 0.0005 \
  --patience 5 \
  --min-delta 0.01 \
  --epochs 150 \
  --keep-last 5

# With custom output directories
bash script/yolo_training/train_yolo26x_pose_frame.sh \
  --dataset-dir data/intermediate/Side_above_water_frames \
  --run-name frame_experiment_001 \
  --checkpoint-dir /custom/checkpoints \
  --reports-dir /custom/reports \
  --overlays-dir /custom/overlays \
  --test-kp-json /custom/keypoints.json \
  --test-metrics-json /custom/metrics.json

# Without tmux (synchronous execution)
bash script/yolo_training/train_yolo26x_pose_frame.sh \
  --dataset-dir data/intermediate/Side_above_water_frames \
  --use-tmux no

# With custom checkpoint (resume from previous run)
bash script/yolo_training/train_yolo26x_pose_frame.sh \
  --dataset-dir data/intermediate/Side_above_water_frames \
  --checkpoint runs/previous_run/weights/best.pt
```

## Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--run-name` | string | `YYYYMMDD_HHMM` | Symbolic experiment name |
| `--dataset-dir` | path | (required) | Directory with train2017/val2017/test2017 |
| `--use-tmux` | yes\|no | `yes` | Launch in tmux session |
| `--checkpoint` | path | `models/pose/yolo26x-pose.pt` | Base model checkpoint |
| `--imgsz` | int | `768` | Input image size |
| `--lr` | float | `0.001` | Initial learning rate |
| `--patience` | int | `3` | Early stopping patience (epochs) |
| `--min-delta` | float | `0.007` | Minimum improvement for patience counter |
| `--keep-last` | int | `10` | Number of recent checkpoints to keep |
| `--epochs` | int | `100` | Total training epochs |
| `--batch` | int | `1` | Batch size |
| `--device` | int\|str | `0` | GPU device ID or 'cpu' |
| `--workers` | int | `2` | Data loading workers |
| `--checkpoint-dir` | path | `runs/<run-name>/checkpoints/` | Checkpoint save location |
| `--reports-dir` | path | `runs/<run-name>/reports/` | Metrics & plots directory |
| `--test-kp-json` | path | `data/output/experiments/yolo26x_pose_<run-name>/kp_Test.json` | Test keypoints output |
| `--test-metrics-json` | path | `data/output/experiments/yolo26x_pose_<run-name>/metrics_Test.json` | Test metrics JSON |
| `--test-metrics-csv` | path | `data/output/experiments/yolo26x_pose_<run-name>/metrics_Test.csv` | Test metrics CSV |
| `--overlays-dir` | path | `data/output/experiments/yolo26x_pose_<run-name>/overlays_Test/` | Test visualization overlays |

## Dataset Requirements

The script expects a canonical YOLO dataset structure:

```
<dataset-dir>/
├── train2017/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── val2017/
│   ├── image1.jpg
│   └── ...
└── test2017/
    ├── image1.jpg
    └── ...
```

Annotations must be embedded in a `person_keypoints_*.json` file in an `annotations/` subdirectory (created by `script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py`).

## Output Structure

```
runs/<run-name>/
├── checkpoints/              # Saved model checkpoints
├── reports/                  # Training metrics & plots
│   ├── val_metrics_by_epoch.csv
│   ├── results.csv
│   └── *.png                 # Training loss & mAP plots
├── dataset.yaml             # Generated YOLO dataset config
└── weights/                 # YOLO training output
    ├── best.pt
    ├── last.pt
    └── ...

data/output/experiments/yolo26x_pose_<run-name>/
├── kp_Test.json             # Predicted keypoints
├── metrics_Test.json        # Performance metrics (mAP, AP50, etc.)
├── metrics_Test.csv         # Tabular metrics
└── overlays_Test/           # Annotated test images (if generated)
```

## Running in tmux Session

If using `--use-tmux yes` (default):

```bash
# Attach to running session
tmux attach -t train_yolo26x_pose_<run-name>

# List active sessions
tmux ls

# Kill session when done (or let it auto-exit)
tmux kill-session -t train_yolo26x_pose_<run-name>
```

## Integration with Previous Pipeline

If you prepared your dataset using `script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py`:

```bash
# 1. Prepare dataset from frames
python script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py \
  --input-root data/input/subset_xyz/Side_above_water_frames \
  --output-root data/intermediate/Side_above_water_frames

# 2. Train YOLO directly (no separate aggregation step needed)
bash script/yolo_training/train_yolo26x_pose_frame.sh \
  --dataset-dir data/intermediate/Side_above_water_frames/_train_canonical
```

## Notes

- The script automatically generates a YOLO dataset YAML at `runs/<run-name>/dataset.yaml`.
- Early stopping is monitored via `monitor_yolo_pose_patience.py` using the mAP50-95(P) metric.
- Test evaluation runs on the best checkpoint (weights/best.pt) after training completes.
- All logs are written to `logs/<run-name>_*.log`.
- The script uses conda environment `vitpose` by default.

## Environment Setup

Ensure your conda environment has YOLO and required packages:

```bash
conda activate vitpose
pip install ultralytics opencv-python
```
