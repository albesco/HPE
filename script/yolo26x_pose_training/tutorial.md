# Tutorial: train_yolo26x_pose_frame.sh

This guide explains how to run the YOLO26x pose training launcher at
`script/yolo26x_pose_training/train_yolo26x_pose_frame.sh` and how to tune its options.

## 1. What this script does

The script:

1. validates the input dataset directory,
2. resolves the YOLO pose dataset YAML automatically,
3. starts YOLO pose training with `conda run -n vitpose`,
4. monitors training with the patience/early-stopping helper,
5. exports reports and evaluates the best checkpoint on the test split.

It is designed for frame+label datasets prepared in the same workflow used by
the SwimXYZ / SAW frame exports.

---

## 2. Prerequisites

Before running the training:

- activate the expected environment:

  ```bash
  conda activate vitpose
  ```

- make sure the input dataset exists. For the current frame-dataset layout, the
  recommended path is the main dataset root:

  ```bash
  data/intermediate/SAW_frames
  ```

- make sure the YOLO pose YAML exists in the sibling pose export folder:

  ```bash
  data/intermediate/SAW_frames/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml
  ```

The launcher resolves the YAML for you when you pass the main dataset root,
the `_Yolo26x_pose` export folder, or the `_train_canonical` folder.

---

## 3. Minimum working example

From the repository root:

```bash
bash script/yolo26x_pose_training/train_yolo26x_pose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames \
  --run-name yolo26x-pose_SAW_frames_20260614 \
  --use-tmux no
```

This is the simplest end-to-end command:

- `--dataset-dir` is required.
- `--run-name` gives a friendly experiment name.
- `--use-tmux no` runs in the current shell so you can follow the logs directly.

---

## 4. Parameter reference

The script options are defined directly in `script/yolo26x_pose_training/train_yolo26x_pose_frame.sh`.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--run-name` | `$(date -u +%Y%m%d_%H%M)` | Experiment name used for `runs/<run-name>/...` and test output folders. |
| `--use-tmux` | `yes` | Start training in a detached tmux session. Use `no` for foreground runs. |
| `--checkpoint` | `models/pose/yolo26x-pose.pt` | Base YOLO pose checkpoint to start from. |
| `--dataset-dir` | required | Main dataset root, `_Yolo26x_pose` folder, or `_train_canonical` folder. The script auto-resolves the pose YAML. |
| `--imgsz` | `768` | Input image size passed to YOLO training. |
| `--lr` | `0.001` | Initial learning rate. |
| `--patience` | `3` | Early-stopping patience used by the monitoring helper. |
| `--min-delta` | `0.007` | Minimum improvement required to count as progress in early stopping. |
| `--keep-last` | `10` | Number of recent checkpoints to retain when the monitor cleans old weights. |
| `--checkpoint-dir` | `runs/<run-name>/checkpoint` | Directory where best/last checkpoints are copied for later reuse. |
| `--reports-dir` | `runs/<run-name>/reports` | Directory for exported metrics and plots. |
| `--test-kp-json` | `data/output/experiments/<run-name>/kp_Test.json` | Keypoint output on the test split. |
| `--test-metrics-json` | `data/output/experiments/<run-name>/metrics_Test.json` | JSON test metrics. |
| `--test-metrics-csv` | `data/output/experiments/<run-name>/metrics_Test.csv` | CSV test metrics. |
| `--overlays-dir` | `data/output/experiments/<run-name>/overlays_Test` | Visualization folder for test overlays. |
| `--epochs` | `100` | Total training epochs. |
| `--batch` | `1` | YOLO training batch size. |
| `--device` | `0` | GPU device index or `cpu`. |
| `--workers` | `2` | Data loading worker count. |

### Notes on the most important options

- `--dataset-dir` should normally point to the main dataset root, for example
  `data/intermediate/SAW_frames`. The script finds `_Yolo26x_pose` internally.
  Passing `_Yolo26x_pose` or `_train_canonical` is still supported.
- `--imgsz` has a big impact on memory usage. Start with `768` for a balance of
  speed and quality. Larger values such as `1024` or `1280` may need more GPU memory.
- `--batch` should be set according to your GPU memory. If training becomes OOM,
  lower it first.
- `--device` can be `0`, `1`, `2`, ... or `cpu`.
- `--use-tmux no` is recommended the first time you run the script so you can
  inspect the logs immediately.

---

## 5. Example usages

### A. Quick baseline run on the current SAW dataset

```bash
bash script/yolo26x_pose_training/train_yolo26x_pose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames \
  --run-name saw_pose_baseline \
  --use-tmux no \
  --epochs 50 \
  --batch 1 \
  --imgsz 768
```

Use this when you want a fast first experiment before longer training.

### B. Resume from an existing checkpoint

```bash
bash script/yolo26x_pose_training/train_yolo26x_pose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames \
  --run-name saw_pose_resume \
  --checkpoint runs/yolo26x-pose_SAW_frames_20260614/checkpoint/best.pt \
  --use-tmux no \
  --epochs 100
```

This is useful when you want to continue a prior run from the best saved weight.

### C. Launch in tmux and keep outputs in custom folders

```bash
bash script/yolo26x_pose_training/train_yolo26x_pose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames \
  --run-name saw_pose_custom \
  --checkpoint-dir runs/saw_pose_custom/checkpoint \
  --reports-dir runs/saw_pose_custom/reports \
  --overlays-dir data/output/experiments/saw_pose_custom/overlays_Test \
  --epochs 100 \
  --batch 2 \
  --device 0
```

This is the best choice when you want to keep experiment artifacts isolated.

### D. Lower the learning rate for a more conservative run

```bash
bash script/yolo26x_pose_training/train_yolo26x_pose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames \
  --run-name saw_pose_lr0005 \
  --lr 0.0005 \
  --patience 5 \
  --min-delta 0.01 \
  --use-tmux no
```

This is useful if the first run seems unstable or overfits too quickly.

---

## 6. Where the outputs go

A typical run creates the following folders:

```text
runs/<run-name>/
  checkpoint/          # copied best/last epoch checkpoints
  reports/             # exported metrics and plots
  weights/             # YOLO training outputs written by Ultralytics
  logs/train.log       # main training log
  logs/test.log        # test evaluation log

data/output/experiments/<run-name>/
  kp_Test.json
  metrics_Test.json
  metrics_Test.csv
  overlays_Test/
```

If you use `--use-tmux no`, you can inspect the real-time logs in the terminal.
If you use the tmux default, attach with:

```bash
tmux attach -t train_yolo26x_pose_<run-name>
```

---

## 7. Recommended workflow

1. Start with a short run using `--use-tmux no`.
2. Check the validation behavior and reports.
3. Increase `--epochs` or adjust `--lr` only after reviewing the first results.
4. Reuse the best checkpoint with `--checkpoint` for follow-up experiments.

This keeps the training loop simple and lets you compare runs by `--run-name`.


## 8. Launch and manage with tmux

The launcher already supports `tmux` through `--use-tmux yes`, which is the default.

### Start in tmux

```bash
bash script/yolo26x_pose_training/train_yolo26x_pose_frame.sh \
  --dataset-dir data/intermediate/SAW_frames \
  --run-name yolo26x-pose_SAW_frames_tmux_demo
```

This creates a tmux session named:

```bash
train_yolo26x_pose_yolo26x-pose_SAW_frames_tmux_demo
```

### Attach to the session

```bash
tmux attach -t train_yolo26x_pose_yolo26x-pose_SAW_frames_tmux_demo
```

### Detach without stopping the training

Press:

```text
Ctrl+b then d
```

### List active tmux sessions

```bash
tmux ls
```

### Inspect logs without attaching

```bash
tail -f runs/yolo26x-pose_SAW_frames_tmux_demo/logs/train.log
```

### Stop the tmux session

```bash
tmux kill-session -t train_yolo26x_pose_yolo26x-pose_SAW_frames_tmux_demo
```

If you stop the tmux session, the training process launched inside it stops as well.

