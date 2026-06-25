# Session Note: Train Yolo26x detection

Date: 2026-05-23 UTC  
Last updated: 2026-05-23 18:16 UTC

## Identity

- ID: `Train Yolo26x detection`
- Logical title: `Train Yolo26x detection`
- Role: train YOLO26x detection weights using the hyperparameters selected by `Choose Yolo26x detection`
- Predecessor: `Choose Yolo26x detection`
- Status in `docs/ai/chat-index.md`: active

## Starting point

- Selected config: `cfg_03_lr0_0.00067_imgsz_768`
- First continuation checkpoint: `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt`
- Selected checkpoint for downstream inference remains available at `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/best.pt`

## Prepared in this setup pass

- Updated `script_old/yolo_training/train_yolo_side_above_water.sh` for incremental cfg_03 training.
- Created `data/intermediate/Side_above_water/_Yolo26x_detection/images/{train,val,test}` symlinks to the canonical split image directories.
- Removed obsolete detector grid-search label caches from `runs/hparam_search/yolo26x_detector_v2/dataset_view/labels/`.
- Updated AI memory docs and `script_old/yolo_training/README.md`.

## Training policy

- Defaults: `imgsz=768`, `batch=2`, `lr0=0.00067`, `patience=2`, `save_period=1`.
- Output root: `runs/yolo26x_bbox_side_above_water/`.
- Default run name: `yolo26x-detection_cfg03_lr0_0.00067_imgsz_768_incremental`.
- Resume: target run `weights/last.pt` if present, otherwise selected cfg_03 `last.pt`.
- Stop when train loss continues decreasing while validation mAP50-95 plateaus or drops, with patience `2`.

## Next step

Launch the incremental detector training in `tmux` after activation/confirmation.
## Smoke run launched

Launched at 2026-05-23 17:54 UTC:

```bash
TAG=smoke_1ep_20260523_1740 EPOCHS=1 RUN_TEST=False script_old/yolo_training/train_yolo_side_above_water.sh
```

- tmux session: `yolo26x_det_smoke_1ep_20260523_1740`
- Run dir: `runs/yolo26x_bbox_side_above_water/yolo26x-detection_smoke_1ep_20260523_1740`
- Log: `logs/yolo26x_detection_side_above_water_20260523_175456.log`
- Startup verification: `args.yaml` and `train_batch*.jpg` created; PyTorch sees CUDA device `Tesla V100S-PCIE-32GB`.
- Completed: 1 epoch finished; val metrics: P `0.99094`, R `0.98902`, mAP50 `0.99250`, mAP50-95 `0.82533`; checkpoints: `best.pt`, `last.pt`, `epoch0.pt`.
