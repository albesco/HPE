from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from mmcv.parallel import DataContainer


DEFAULT_DATASET_ROOT = "data/intermediate/Side_above_water/_train_canonical"
DEFAULT_CONFIG = (
    "data/intermediate/Side_above_water/_train_canonical/generated_configs/"
    "swimxyz_vitposepp_huge.py"
)
DEFAULT_CHECKPOINT = "runs/vitposepp_subset_xyz/best_AP_epoch_10.pth"

def resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def split_paths(dataset_root: Path, split: str) -> tuple[Path, Path]:
    return dataset_root / "annotations" / f"person_keypoints_{split}.json", dataset_root / f"{split}2017"


def bbox_xywh_to_xyxy(bbox_xywh: list[float]) -> list[float]:
    x, y, width, height = bbox_xywh
    return [x, y, x + width, y + height]


def draw_bbox_xywh(image: np.ndarray, bbox: list[float], color: tuple[int, int, int]) -> None:
    x, y, width, height = bbox
    cv2.rectangle(image, (round(x), round(y)), (round(x + width), round(y + height)), color, 3)


def draw_bbox_xyxy(image: np.ndarray, bbox: list[float], color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), color, 3)


def dataset_info_from_model(model):
    from mmpose.datasets.dataset_info import DatasetInfo

    dataset_info_cfg = model.cfg.data.get("test", {}).get("dataset_info", None)
    return DatasetInfo(dataset_info_cfg) if isinstance(dataset_info_cfg, dict) else dataset_info_cfg


def _unwrap_cpu_data(value):
    if isinstance(value, DataContainer):
        data = value.data
        if isinstance(data, list) and len(data) == 1:
            return _unwrap_cpu_data(data[0])
        return data
    if isinstance(value, dict):
        return {k: _unwrap_cpu_data(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_unwrap_cpu_data(v) for v in value)
    if isinstance(value, list):
        return [_unwrap_cpu_data(v) for v in value]
    return value


def run_pose_prediction(model, image_path: Path, bbox_xywh: list[float], dataset_info):
    from mmcv.parallel import collate, scatter
    from mmpose.apis import inference as mmpose_inference
    from mmpose.datasets.pipelines import Compose

    flip_pairs = []
    if dataset_info is not None:
        flip_pairs = getattr(dataset_info, "flip_pairs", None)
        if flip_pairs is None:
            try:
                flip_pairs = dataset_info.get("flip_pairs")  # type: ignore[attr-defined]
            except Exception:
                flip_pairs = None
        if flip_pairs is None:
            flip_pairs = []

    center, scale = mmpose_inference._box2cs(model.cfg, bbox_xywh)  # _box2cs expects COCO xywh.
    data = {
        "center": center,
        "scale": scale,
        "bbox_score": 1.0,
        "bbox_id": 0,
        "dataset": "TopDownCocoDataset",
        "dataset_idx": 0,
        "joints_3d": np.zeros((model.cfg.data_cfg.num_joints, 3), dtype=np.float32),
        "joints_3d_visible": np.zeros((model.cfg.data_cfg.num_joints, 3), dtype=np.float32),
        "rotation": 0,
        "ann_info": {
            "image_size": np.array(model.cfg.data_cfg["image_size"]),
            "num_joints": model.cfg.data_cfg["num_joints"],
            "flip_pairs": flip_pairs,
        },
        "image_file": str(image_path),
    }

    data = Compose(model.cfg.test_pipeline)(data)
    batch_data = collate([data], samples_per_gpu=1)
    device = next(model.parameters()).device
    if device.type == 'cpu':
        batch_data = {key: _unwrap_cpu_data(value) for key, value in batch_data.items()}
    else:
        batch_data = scatter(batch_data, [device])[0]
    with torch.no_grad():
        result = model(
            img=batch_data["img"],
            img_metas=batch_data["img_metas"],
            return_loss=False,
            return_heatmap=False,
        )
    return result["preds"][0]
