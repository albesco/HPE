#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import runpy
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SOURCE = 'data/intermediate/Side_above_water/_train_canonical/test2017'
DEFAULT_ANNOTATIONS = 'data/intermediate/Side_above_water/_train_canonical/annotations/person_keypoints_test.json'
DEFAULT_OUTPUT_DIR = 'data/intermediate/Side_above_water/_train_canonical/reports/test_overlays/GT'
DEFAULT_COCO_INFO = 'src/vitpose_base/configs/_base_/datasets/coco.py'

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
BBOX_RED_BGR = (0, 0, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render GT keypoint overlays with the same visual style used by the YOLO pose renderer.'
    )
    parser.add_argument('--source', default=DEFAULT_SOURCE, help='Image file or image directory to render.')
    parser.add_argument(
        '--annotations',
        default=DEFAULT_ANNOTATIONS,
        help='COCO annotation file or a directory containing COCO-style JSON annotations.',
    )
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--dataset-info', default=DEFAULT_COCO_INFO)
    parser.add_argument('--max-images', type=int, default=0, help='Maximum number of images to render. 0 means all.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed used when sampling a subset of images.')
    parser.add_argument('--kpt-score-thr', type=float, default=0.3)
    parser.add_argument('--radius', type=int, default=3)
    parser.add_argument('--thickness', type=int, default=2)
    parser.add_argument('--draw-bbox', action='store_true', help='Also draw the GT bbox in red.')
    parser.add_argument('--bbox-thickness', type=int, default=3)
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
    dataset_info = runpy.run_path(str(dataset_info_path))['dataset_info']
    keypoint_info = sorted(dataset_info['keypoint_info'].values(), key=lambda item: int(item['id']))
    keypoint_name_to_id = {item['name']: int(item['id']) for item in keypoint_info}
    skeleton_info = sorted(dataset_info['skeleton_info'].values(), key=lambda item: int(item['id']))

    skeleton = [
        [keypoint_name_to_id[start], keypoint_name_to_id[end]]
        for item in skeleton_info
        for start, end in [item['link']]
    ]
    pose_kpt_color = np.asarray([item.get('color', [0, 255, 0]) for item in keypoint_info], dtype=np.int32)
    pose_link_color = np.asarray([item.get('color', [0, 255, 0]) for item in skeleton_info], dtype=np.int32)
    return skeleton, pose_kpt_color, pose_link_color


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


def draw_bbox(image: np.ndarray, bbox_xywh: list[float], thickness: int) -> None:
    x, y, width, height = bbox_xywh
    cv2.rectangle(
        image,
        (round(float(x)), round(float(y))),
        (round(float(x + width)), round(float(y + height))),
        BBOX_RED_BGR,
        thickness,
    )


def annotation_keys(file_name: str) -> list[str]:
    path = Path(file_name)
    keys = [file_name]
    for candidate in (path.as_posix(), path.name, path.stem):
        if candidate and candidate not in keys:
            keys.append(candidate)
    return keys


def add_annotations(index: dict[str, list[dict]], file_name: str, annotations: list[dict]) -> None:
    for key in annotation_keys(file_name):
        index.setdefault(key, []).extend(annotations)


def load_annotations(annotation_source: Path) -> dict[str, list[dict]]:
    annotation_paths: list[Path]
    if annotation_source.is_file():
        annotation_paths = [annotation_source]
    else:
        annotation_paths = sorted(p for p in annotation_source.iterdir() if p.suffix.lower() == '.json')
        if not annotation_paths:
            raise FileNotFoundError(f'No JSON annotations found in directory: {annotation_source}')

    annotations_by_image: dict[str, list[dict]] = {}
    for annotation_path in annotation_paths:
        payload = json.loads(annotation_path.read_text(encoding='utf-8'))

        if 'images' in payload and 'annotations' in payload:
            image_file_names = {int(image['id']): str(image['file_name']) for image in payload.get('images', [])}
            grouped: dict[str, list[dict]] = {}
            for annotation in payload.get('annotations', []):
                image_id = int(annotation.get('image_id', -1))
                file_name = image_file_names.get(image_id)
                if file_name:
                    grouped.setdefault(file_name, []).append(annotation)
            for file_name, annotations in grouped.items():
                add_annotations(annotations_by_image, file_name, annotations)
            continue

        file_name = payload.get('file_name') or payload.get('image_file') or payload.get('image')
        if file_name and 'keypoints' in payload:
            add_annotations(annotations_by_image, str(file_name), [payload])
            continue

        raise ValueError(
            f'Unsupported annotation format in {annotation_path}. '
            'Expected a COCO payload with images/annotations or a single-image payload with keypoints.'
        )

    return annotations_by_image


def candidate_annotation_keys(image_path: Path, source: Path) -> list[str]:
    candidates = [image_path.name, image_path.stem]
    try:
        relative = image_path.relative_to(source)
    except ValueError:
        relative = None

    if relative is not None:
        for candidate in (relative.as_posix(), relative.name, relative.stem):
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def render_one(
    image_path: Path,
    output_path: Path,
    annotations: list[dict],
    skeleton: list[list[int]],
    pose_kpt_color: np.ndarray,
    pose_link_color: np.ndarray,
    args: argparse.Namespace,
) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        return 0

    rendered_people = 0
    for annotation in annotations:
        keypoints = annotation.get('keypoints')
        if not keypoints or len(keypoints) < 3:
            continue
        keypoints_array = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)
        draw_keypoints(
            image=image,
            keypoints=keypoints_array,
            skeleton=skeleton,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            kpt_score_thr=args.kpt_score_thr,
            radius=args.radius,
            thickness=args.thickness,
        )
        if args.draw_bbox:
            bbox = annotation.get('bbox')
            if bbox and len(bbox) == 4:
                draw_bbox(image, bbox, args.bbox_thickness)
        rendered_people += 1

    if rendered_people == 0:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f'Unable to write overlay image: {output_path}')
    return rendered_people


def main() -> None:
    args = parse_args()
    source = resolve_path(args.source)
    annotation_source = resolve_path(args.annotations)
    output_dir = resolve_path(args.output_dir)
    dataset_info_path = resolve_path(args.dataset_info)

    if not source.exists():
        raise FileNotFoundError(source)
    if not annotation_source.exists():
        raise FileNotFoundError(annotation_source)
    if not dataset_info_path.is_file():
        raise FileNotFoundError(dataset_info_path)

    skeleton, pose_kpt_color, pose_link_color = load_coco_style(dataset_info_path)
    annotations_by_image = load_annotations(annotation_source)
    images = collect_images(source, args.max_images, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    rendered_images = 0
    rendered_people = 0
    for image_path in images:
        annotations = None
        for key in candidate_annotation_keys(image_path, source):
            annotations = annotations_by_image.get(key)
            if annotations:
                break
        if not annotations:
            continue

        output_path = output_dir / image_path.name
        people = render_one(
            image_path=image_path,
            output_path=output_path,
            annotations=annotations,
            skeleton=skeleton,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            args=args,
        )
        if people > 0:
            rendered_images += 1
            rendered_people += people

    if rendered_images == 0:
        raise RuntimeError('No GT overlays were written. Check the source images and annotations paths.')

    print(f'Rendered images: {rendered_images}')
    print(f'Rendered people: {rendered_people}')
    print(f'Output directory: {output_dir}')


if __name__ == '__main__':
    main()
