from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_DATASET_ROOT = "data/intermediate/Side_above_water/_train_vitposepp_swap_ears"
DEFAULT_CONFIG = (
    "data/intermediate/Side_above_water/_train_vitposepp_swap_ears/generated_configs/"
    "swimxyz_vitposepp_huge_single_head_swap_ears.py"
)
DEFAULT_CHECKPOINT = "runs/vitposepp_single_head_subset_xyz_swap_ears/best_AP_epoch_10.pth"

COCO17_SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


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


def points_from_coco_keypoints(keypoints: list[float]) -> list[tuple[float, float, int]]:
    points: list[tuple[float, float, int]] = []
    for idx in range(0, len(keypoints), 3):
        points.append((keypoints[idx], keypoints[idx + 1], int(keypoints[idx + 2])))
    return points


def draw_coco_keypoints(
    image: np.ndarray,
    keypoints: list[float],
    color: tuple[int, int, int],
    line_color: tuple[int, int, int] | None = None,
    draw_indices: bool = False,
    index_offset: tuple[int, int] = (-12, -8),
) -> None:
    points = points_from_coco_keypoints(keypoints)
    skeleton_color = line_color if line_color is not None else color
    for start, end in COCO17_SKELETON:
        if points[start][2] > 0 and points[end][2] > 0:
            cv2.line(
                image,
                (round(points[start][0]), round(points[start][1])),
                (round(points[end][0]), round(points[end][1])),
                skeleton_color,
                2,
            )
    for idx, (x, y, visible) in enumerate(points):
        if visible <= 0:
            continue
        center = (round(x), round(y))
        cv2.circle(image, center, 5, color, -1)
        if draw_indices:
            cv2.putText(
                image,
                str(idx),
                (center[0] + index_offset[0], center[1] + index_offset[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )


def draw_prediction_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    score_threshold: float,
    color: tuple[int, int, int] = (0, 128, 255),
    line_color: tuple[int, int, int] = (255, 0, 255),
    draw_indices: bool = False,
    index_offset: tuple[int, int] = (7, 13),
) -> list[float] | None:
    valid = keypoints[:, 2] >= score_threshold
    for start, end in COCO17_SKELETON:
        if valid[start] and valid[end]:
            cv2.line(
                image,
                (round(float(keypoints[start, 0])), round(float(keypoints[start, 1]))),
                (round(float(keypoints[end, 0])), round(float(keypoints[end, 1]))),
                line_color,
                2,
            )
    for idx, (x, y, score) in enumerate(keypoints):
        if score < score_threshold:
            continue
        center = (round(float(x)), round(float(y)))
        cv2.circle(image, center, 5, color, -1)
        if draw_indices:
            cv2.putText(
                image,
                str(idx),
                (center[0] + index_offset[0], center[1] + index_offset[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                line_color,
                1,
                cv2.LINE_AA,
            )

    if not np.any(valid):
        return None

    xs = keypoints[valid, 0]
    ys = keypoints[valid, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def dataset_info_from_model(model):
    from mmpose.datasets.dataset_info import DatasetInfo

    dataset_info_cfg = model.cfg.data.get("test", {}).get("dataset_info", None)
    return DatasetInfo(dataset_info_cfg) if isinstance(dataset_info_cfg, dict) else dataset_info_cfg


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
    batch_data = scatter(batch_data, [next(model.parameters()).device])[0]
    with torch.no_grad():
        result = model(
            img=batch_data["img"],
            img_metas=batch_data["img_metas"],
            return_loss=False,
            return_heatmap=False,
        )
    return result["preds"][0]
