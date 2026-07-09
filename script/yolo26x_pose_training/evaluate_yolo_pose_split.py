#!/usr/bin/env python3

import argparse
import csv
import json
import random
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_COCO_INFO = "src/vitpose_base/configs/_base_/datasets/coco.py"
BBOX_RED_BGR = (0, 0, 255)


@dataclass
class SplitCounts:
    images: int
    instances: int


def _resolve_split_dir(data_yaml: Path, split: str) -> Path:
    cfg = yaml.safe_load(data_yaml.read_text())

    base = cfg.get("path")
    if base is None:
        raise ValueError(f"Missing path in data yaml: {data_yaml}")

    base_path = (data_yaml.parent / str(base)).resolve() if not Path(base).is_absolute() else Path(base)
    rel = cfg.get(split)
    if rel is None:
        raise ValueError(f"Missing {split} entry in data yaml: {data_yaml}")

    img_dir = (base_path / str(rel)).resolve()
    if not img_dir.exists():
        raise FileNotFoundError(f"Split images dir not found: {img_dir}")
    return img_dir


def _count_images_and_instances(data_yaml: Path, split: str) -> SplitCounts:
    img_dir = _resolve_split_dir(data_yaml, split)
    images = sum(1 for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)

    # SwimXYZ Side_above_water exports one person per image; keep instances=images for reporting consistency.
    return SplitCounts(images=images, instances=images)


def _get(d: Dict[str, Any], key: str) -> float:
    if key not in d:
        raise KeyError(f"Missing metric key: {key}. Available keys: {sorted(d.keys())}")
    v = d[key]
    return float(v)


def _collect_images(source: Path, max_images: int, seed: int) -> list[Path]:
    images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if max_images > 0 and len(images) > max_images:
        rng = random.Random(seed)
        images = sorted(rng.sample(images, max_images))
    return images


def _load_coco_style(dataset_info_path: Path):
    from mmpose.datasets.dataset_info import DatasetInfo

    dataset_info = DatasetInfo(runpy.run_path(str(dataset_info_path))["dataset_info"])
    return dataset_info.skeleton, dataset_info.pose_kpt_color, dataset_info.pose_link_color


def _draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    skeleton: list[list[int]],
    pose_kpt_color: np.ndarray,
    pose_link_color: np.ndarray,
    kpt_score_thr: float,
    radius: int,
    thickness: int,
) -> None:
    height, width = image.shape[:2]

    for link_id, (start_idx, end_idx) in enumerate(skeleton):
        start = keypoints[start_idx]
        end = keypoints[end_idx]
        start_xy = (int(start[0]), int(start[1]))
        end_xy = (int(end[0]), int(end[1]))
        if (
            0 < start_xy[0] < width
            and 0 < start_xy[1] < height
            and 0 < end_xy[0] < width
            and 0 < end_xy[1] < height
            and start[2] > kpt_score_thr
            and end[2] > kpt_score_thr
        ):
            color = tuple(int(c) for c in pose_link_color[link_id])
            cv2.line(image, start_xy, end_xy, color, thickness=thickness)

    for keypoint_id, (x_coord, y_coord, score) in enumerate(keypoints):
        if score > kpt_score_thr:
            color = tuple(int(c) for c in pose_kpt_color[keypoint_id])
            cv2.circle(image, (int(x_coord), int(y_coord)), radius, color, -1)


def _draw_bbox(image: np.ndarray, bbox_xyxy: np.ndarray, thickness: int) -> None:
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(
        image,
        (round(float(x1)), round(float(y1))),
        (round(float(x2)), round(float(y2))),
        BBOX_RED_BGR,
        thickness,
    )


def _write_predictions_and_overlays(
    model,
    image_dir: Path,
    out_keypoints_json: Optional[Path],
    overlays_dir: Optional[Path],
    dataset_info: Path,
    imgsz: int,
    conf: float,
    device: str,
    max_images: int,
    seed: int,
    max_detections_per_image: int,
    kpt_score_thr: float,
    bbox_thickness: int,
    radius: int,
    thickness: int,
) -> None:
    skeleton, pose_kpt_color, pose_link_color = _load_coco_style(dataset_info)
    images = _collect_images(image_dir, max_images, seed)
    predictions: list[dict[str, Any]] = []

    for index, image_path in enumerate(images):
        image = cv2.imread(str(image_path)) if overlays_dir else None
        results = model.predict(
            source=image_path.as_posix(),
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
            max_det=max_detections_per_image if max_detections_per_image > 0 else 300,
        )
        if not results:
            continue

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        box_scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty((0,))
        keypoints = (
            result.keypoints.data.cpu().numpy()
            if result.keypoints is not None
            else np.empty((0, 17, 3))
        )
        if max_detections_per_image > 0 and len(box_scores) > max_detections_per_image:
            keep_indices = np.argsort(-box_scores)[:max_detections_per_image]
            boxes = boxes[keep_indices]
            box_scores = box_scores[keep_indices]
            keypoints = keypoints[keep_indices]

        image_predictions = []
        for bbox_xyxy, score, pose_keypoints in zip(boxes, box_scores, keypoints):
            image_predictions.append(
                {
                    "bbox_xyxy": [float(v) for v in bbox_xyxy],
                    "score": float(score),
                    "keypoints": pose_keypoints.astype(float).tolist(),
                }
            )
            if image is not None:
                _draw_keypoints(
                    image=image,
                    keypoints=pose_keypoints,
                    skeleton=skeleton,
                    pose_kpt_color=pose_kpt_color,
                    pose_link_color=pose_link_color,
                    kpt_score_thr=kpt_score_thr,
                    radius=radius,
                    thickness=thickness,
                )
                _draw_bbox(image, bbox_xyxy, bbox_thickness)

        predictions.append({"image": image_path.name, "predictions": image_predictions})

        if image is not None and overlays_dir is not None:
            overlays_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlays_dir / f"pred_{index:04d}__{image_path.name}"), image)

    if out_keypoints_json is not None:
        out_keypoints_json.parent.mkdir(parents=True, exist_ok=True)
        out_keypoints_json.write_text(json.dumps(predictions, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Ultralytics YOLO pose on a split and optionally export keypoints/overlays."
    )
    parser.add_argument("--model", required=True, help="Path to .pt weights (best.pt/last.pt)")
    parser.add_argument("--data", required=True, help="Ultralytics data YAML")
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"], help="Split to evaluate"
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out-csv", required=True, help="Output CSV path")
    parser.add_argument("--out-metrics-json", help="Optional output JSON with the aggregate metrics row")
    parser.add_argument("--out-keypoints-json", help="Optional output JSON with predicted bboxes/keypoints")
    parser.add_argument("--overlays-dir", help="Optional output directory for VitPose++-style overlays")
    parser.add_argument("--dataset-info", default=DEFAULT_COCO_INFO)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--overlay-max-images", type=int, default=0)
    parser.add_argument("--overlay-seed", type=int, default=0)
    parser.add_argument("--max-detections-per-image", type=int, default=1)
    parser.add_argument("--kpt-score-thr", type=float, default=0.3)
    parser.add_argument("--bbox-thickness", type=int, default=3)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--thickness", type=int, default=2)
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    out_csv = Path(args.out_csv)
    out_metrics_json = Path(args.out_metrics_json) if args.out_metrics_json else None
    out_keypoints_json = Path(args.out_keypoints_json) if args.out_keypoints_json else None
    overlays_dir = Path(args.overlays_dir) if args.overlays_dir else None
    dataset_info_path = Path(args.dataset_info)

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not dataset_info_path.exists():
        raise FileNotFoundError(dataset_info_path)

    from ultralytics import YOLO

    y = YOLO(model_path.as_posix())
    val_kwargs = {
        "data": data_path.as_posix(),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "verbose": True,
        "plots": True,
    }
    if args.max_detections_per_image > 0:
        val_kwargs["max_det"] = args.max_detections_per_image
    if args.conf is not None:
        val_kwargs["conf"] = args.conf
    metrics = y.val(**val_kwargs)

    results: Dict[str, Any] = metrics.results_dict  # type: ignore[attr-defined]

    counts = _count_images_and_instances(data_path, args.split)

    row = {
        "split": args.split,
        "images": counts.images,
        "instances": counts.instances,
        "box_precision": _get(results, "metrics/precision(B)"),
        "box_recall": _get(results, "metrics/recall(B)"),
        "box_map50": _get(results, "metrics/mAP50(B)"),
        "box_map50_95": _get(results, "metrics/mAP50-95(B)"),
        "pose_precision": _get(results, "metrics/precision(P)"),
        "pose_recall": _get(results, "metrics/recall(P)"),
        "pose_map50": _get(results, "metrics/mAP50(P)"),
        "pose_map50_95": _get(results, "metrics/mAP50-95(P)"),
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = list(row.keys())
    out_csv.write_text(",".join(header) + "\n" + ",".join(str(row[h]) for h in header) + "\n")

    if out_metrics_json is not None:
        out_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        out_metrics_json.write_text(json.dumps(row, indent=2))

    print(f"Wrote: {out_csv}")
    print(f"Pose mAP50={row['pose_map50']:.6f} Pose mAP50-95={row['pose_map50_95']:.6f}")
    print(f"Box  mAP50={row['box_map50']:.6f} Box  mAP50-95={row['box_map50_95']:.6f}")

    if out_keypoints_json is not None or overlays_dir is not None:
        image_dir = _resolve_split_dir(data_path, args.split)
        prediction_conf = args.conf if args.conf is not None else 0.25
        _write_predictions_and_overlays(
            model=y,
            image_dir=image_dir,
            out_keypoints_json=out_keypoints_json,
            overlays_dir=overlays_dir,
            dataset_info=dataset_info_path,
            imgsz=args.imgsz,
            conf=prediction_conf,
            device=args.device,
            max_images=args.overlay_max_images,
            seed=args.overlay_seed,
            max_detections_per_image=args.max_detections_per_image,
            kpt_score_thr=args.kpt_score_thr,
            bbox_thickness=args.bbox_thickness,
            radius=args.radius,
            thickness=args.thickness,
        )
        if out_keypoints_json is not None:
            print(f"Wrote: {out_keypoints_json}")
        if overlays_dir is not None:
            print(f"Wrote overlays: {overlays_dir}")


if __name__ == "__main__":
    main()
