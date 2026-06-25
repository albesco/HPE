#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import runpy
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MODEL = 'runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/weights/best.pt'
DEFAULT_SOURCE = 'data/intermediate/Side_above_water/_Yolo26x_pose/images/test'
DEFAULT_OUTPUT_DIR = 'data/output/experiments/yolo26x_pose_side_above_water/overlays_mmpose_style'
DEFAULT_COCO_INFO = 'src/vitpose_base/configs/_base_/datasets/coco.py'

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
BBOX_RED_BGR = (0, 0, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render YOLO26x-pose predictions with the same visual style used by VitPose++ overlays.'
    )
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--source', default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--dataset-info', default=DEFAULT_COCO_INFO)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--kpt-score-thr', type=float, default=0.3)
    parser.add_argument('--max-images', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='0')
    parser.add_argument('--bbox-thickness', type=int, default=3)
    parser.add_argument('--radius', type=int, default=3)
    parser.add_argument('--thickness', type=int, default=2)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def collect_images(source: Path, max_images: int, seed: int) -> list[Path]:
    if source.is_file():
        return [source]

    images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if max_images > 0 and len(images) > max_images:
        rng = random.Random(seed)
        images = sorted(rng.sample(images, max_images))
    return images


def load_coco_style(dataset_info_path: Path):
    from mmpose.datasets.dataset_info import DatasetInfo

    dataset_info = DatasetInfo(runpy.run_path(str(dataset_info_path))['dataset_info'])
    return dataset_info.skeleton, dataset_info.pose_kpt_color, dataset_info.pose_link_color


def draw_keypoints(
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


def draw_bbox(image: np.ndarray, bbox_xyxy: np.ndarray, thickness: int) -> None:
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(
        image,
        (round(float(x1)), round(float(y1))),
        (round(float(x2)), round(float(y2))),
        BBOX_RED_BGR,
        thickness,
    )


def render_one(
    model,
    image_path: Path,
    output_path: Path,
    skeleton: list[list[int]],
    pose_kpt_color: np.ndarray,
    pose_link_color: np.ndarray,
    args: argparse.Namespace,
) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        return 0

    results = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        verbose=False,
    )
    if not results:
        return 0

    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
    keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else np.empty((0, 17, 3))

    for bbox_xyxy, pose_keypoints in zip(boxes, keypoints):
        draw_keypoints(
            image=image,
            keypoints=pose_keypoints,
            skeleton=skeleton,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            kpt_score_thr=args.kpt_score_thr,
            radius=args.radius,
            thickness=args.thickness,
        )
        draw_bbox(image, bbox_xyxy, args.bbox_thickness)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return len(keypoints)


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
    source = resolve_path(args.source)
    output_dir = resolve_path(args.output_dir)
    dataset_info_path = resolve_path(args.dataset_info)

    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not source.exists():
        raise FileNotFoundError(source)
    if not dataset_info_path.is_file():
        raise FileNotFoundError(dataset_info_path)

    from ultralytics import YOLO

    skeleton, pose_kpt_color, pose_link_color = load_coco_style(dataset_info_path)
    model = YOLO(str(model_path))

    images = collect_images(source, args.max_images, args.seed)
    rendered = 0
    detections = 0
    for index, image_path in enumerate(images):
        output_path = output_dir / f'pred_{index:02d}__{image_path.name}'
        detections += render_one(model, image_path, output_path, skeleton, pose_kpt_color, pose_link_color, args)
        rendered += 1

    print(f'Rendered images: {rendered}')
    print(f'Rendered detections: {detections}')
    print(f'Output directory: {output_dir}')


if __name__ == '__main__':
    main()
