#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from GT_KP_overlays import (  # noqa: E402
    collect_images,
    draw_keypoints,
    load_annotations,
    load_coco_style,
    resolve_path,
)


DEFAULT_N_FRAME = 20
DEFAULT_D_TEST = 'data/intermediate/SAW_frames_EntireSwim/_train_canonical/test2017'
DEFAULT_D_COMPARE = 'data/output/experiments/overlay_sample_comparison'
DEFAULT_KP_YOLO = 'data/output/experiments/yolo26x-pose_SAW_frames_EntireSwim_20260612/kp_Test.json'
DEFAULT_KP_VITPOSE = 'data/output/experiments/vitpose_SAW_frames_EntireSwim_20260612/kp_Test.json'
DEFAULT_COCO_INFO = 'src/vitpose_base/configs/_base_/datasets/coco.py'
DEFAULT_VITPOSE_MANIFEST = ''

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    output_rel_path: Path
    gt_annotations: list[dict]
    yolo_predictions: list[dict]
    vitpose_predictions: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare Yolo26x-Pose and VitPose++ overlays on random Test frames.'
    )
    parser.add_argument('--n-frame', type=int, default=DEFAULT_N_FRAME, help='Number of random test frames to render.')
    parser.add_argument('--d-test', default=DEFAULT_D_TEST, help='Directory containing the original Test frames.')
    parser.add_argument('--d-compare', default=DEFAULT_D_COMPARE, help='Directory where comparison outputs are saved.')
    parser.add_argument('--kp-yolo', default=DEFAULT_KP_YOLO, help='Yolo26x-Pose kp_Test.json file.')
    parser.add_argument('--kp-vitpose', default=DEFAULT_KP_VITPOSE, help='VitPose++ kp_Test.json file.')
    parser.add_argument(
        '--dataset-annotations',
        default='',
        help=(
            'Optional COCO Test annotations file used to map VitPose image_id values to file names. '
            'Defaults to <D-TEST>/../annotations/person_keypoints_test.json.'
        ),
    )
    parser.add_argument('--dataset-info', default=DEFAULT_COCO_INFO)
    parser.add_argument(
        '--vitpose-manifest',
        default=DEFAULT_VITPOSE_MANIFEST,
        help=(
            'Optional VitPose++ manifest file with image_id/file_name pairs. '
            'Defaults to <KP-VITPOSE>/../overlays_Test/_manifest.json when present.'
        ),
    )
    parser.add_argument('--seed', type=int, default=0, help='Random seed used for frame sampling.')
    parser.add_argument('--kpt-score-thr', type=float, default=0.3)
    parser.add_argument('--radius', type=int, default=3)
    parser.add_argument('--thickness', type=int, default=2)
    return parser.parse_args()


def normalize_keys(value: str) -> list[str]:
    path = Path(value)
    keys = [value]
    for candidate in (path.as_posix(), path.name, path.stem):
        if candidate and candidate not in keys:
            keys.append(candidate)
    return keys


def relative_output_path(image_path: Path, source_root: Path) -> Path:
    try:
        relative = image_path.relative_to(source_root)
    except ValueError:
        return Path(image_path.name)
    if relative == Path('.'):
        return Path(image_path.name)
    return relative


def overlay_output_path(base_path: Path, suffix: str) -> Path:
    return base_path.with_name(f'{base_path.stem}_{suffix}{base_path.suffix}')


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding='utf-8'))


def load_yolo_predictions(predictions_path: Path) -> dict[str, list[dict]]:
    payload = read_json(predictions_path)
    if not isinstance(payload, list):
        raise ValueError(f'Unexpected Yolo26x-Pose JSON format: {predictions_path}')

    grouped: dict[str, list[dict]] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        image_name = entry.get('image') or entry.get('file_name') or entry.get('image_file')
        predictions = entry.get('predictions', [])
        if not image_name or not isinstance(predictions, list):
            continue
        for key in normalize_keys(str(image_name)):
            grouped.setdefault(key, []).extend(predictions)
    return grouped


def load_vitpose_predictions(
    predictions_path: Path,
    annotation_path: Path,
    manifest_path: Optional[Path],
) -> dict[str, list[dict]]:
    preds = read_json(predictions_path)
    if not isinstance(preds, list):
        raise ValueError(f'Unexpected VitPose++ JSON format: {predictions_path}')

    coco = read_json(annotation_path)
    if not isinstance(coco, dict):
        raise ValueError(f'Unexpected COCO annotations format: {annotation_path}')

    images_by_id = {int(image['id']): str(image['file_name']) for image in coco.get('images', [])}
    if manifest_path is not None and manifest_path.is_file():
        manifest = read_json(manifest_path)
        if isinstance(manifest, list):
            for entry in manifest:
                if not isinstance(entry, dict):
                    continue
                image_id = entry.get('image_id')
                file_name = entry.get('file_name')
                if image_id is None or not file_name:
                    continue
                images_by_id[int(image_id)] = str(file_name)

    grouped: dict[str, list[dict]] = {}
    for entry in preds:
        if not isinstance(entry, dict):
            continue
        image_id = entry.get('image_id')
        if image_id is None:
            continue
        file_name = images_by_id.get(int(image_id))
        if not file_name:
            continue
        for key in normalize_keys(file_name):
            grouped.setdefault(key, []).append(entry)
    return grouped


def resolve_annotation_path(d_test: Path, override: str) -> Path:
    if override:
        return resolve_path(override)
    return (d_test.parent / 'annotations' / 'person_keypoints_test.json').resolve()


def resolve_vitpose_manifest_path(kp_vitpose: Path, override: str) -> Optional[Path]:
    if override:
        return resolve_path(override)
    candidate = kp_vitpose.parent / 'overlays_Test' / '_manifest.json'
    return candidate if candidate.is_file() else None


def candidate_samples(
    images: list[Path],
    source_root: Path,
    gt_by_key: dict[str, list[dict]],
    yolo_by_key: dict[str, list[dict]],
    vitpose_by_key: dict[str, list[dict]],
) -> list[Sample]:
    samples: list[Sample] = []
    for image_path in images:
        gt_annotations = None
        yolo_predictions = None
        vitpose_predictions = None
        candidate_keys = normalize_keys(str(image_path.relative_to(source_root)) if image_path.is_relative_to(source_root) else image_path.name)
        for key in candidate_keys:
            if gt_annotations is None:
                gt_annotations = gt_by_key.get(key)
            if yolo_predictions is None:
                yolo_predictions = yolo_by_key.get(key)
            if vitpose_predictions is None:
                vitpose_predictions = vitpose_by_key.get(key)
            if gt_annotations and yolo_predictions and vitpose_predictions:
                break
        if not gt_annotations or not yolo_predictions or not vitpose_predictions:
            continue

        samples.append(
            Sample(
                image_path=image_path,
                output_rel_path=relative_output_path(image_path, source_root),
                gt_annotations=gt_annotations,
                yolo_predictions=yolo_predictions,
                vitpose_predictions=vitpose_predictions,
            )
        )
    return samples


def render_overlay(
    image_path: Path,
    predictions: list[dict],
    output_path: Path,
    skeleton: list[list[int]],
    pose_kpt_color: np.ndarray,
    pose_link_color: np.ndarray,
    args: argparse.Namespace,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f'Unable to read image: {image_path}')

    rendered = 0
    for prediction in predictions:
        keypoints = prediction.get('keypoints')
        if not keypoints:
            continue
        keypoints_array = np.asarray(keypoints, dtype=np.float32)
        if keypoints_array.ndim == 1:
            if keypoints_array.size % 3 != 0:
                continue
            keypoints_array = keypoints_array.reshape(-1, 3)
        elif keypoints_array.ndim == 2 and keypoints_array.shape[1] == 3:
            pass
        else:
            continue
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
        rendered += 1

    if rendered == 0:
        raise RuntimeError(f'No renderable keypoints found in: {image_path}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f'Unable to write overlay image: {output_path}')


def main() -> None:
    args = parse_args()
    d_test = resolve_path(args.d_test)
    d_compare = resolve_path(args.d_compare)
    kp_yolo = resolve_path(args.kp_yolo)
    kp_vitpose = resolve_path(args.kp_vitpose)
    dataset_info_path = resolve_path(args.dataset_info)
    annotation_path = resolve_annotation_path(d_test, args.dataset_annotations)
    vitpose_manifest_path = resolve_vitpose_manifest_path(kp_vitpose, args.vitpose_manifest)

    if not d_test.exists():
        raise FileNotFoundError(d_test)
    if not kp_yolo.is_file():
        raise FileNotFoundError(kp_yolo)
    if not kp_vitpose.is_file():
        raise FileNotFoundError(kp_vitpose)
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    if not dataset_info_path.is_file():
        raise FileNotFoundError(dataset_info_path)

    skeleton, pose_kpt_color, pose_link_color = load_coco_style(dataset_info_path)
    gt_by_key = load_annotations(annotation_path)
    yolo_by_key = load_yolo_predictions(kp_yolo)
    vitpose_by_key = load_vitpose_predictions(kp_vitpose, annotation_path, vitpose_manifest_path)

    images = collect_images(d_test, 0, args.seed)
    samples = candidate_samples(images, d_test, gt_by_key, yolo_by_key, vitpose_by_key)
    if not samples:
        raise RuntimeError('No shared Test frames found between D-TEST, KP-YOLO, and KP-VITPOSE.')

    sample_count = len(samples) if args.n_frame <= 0 else min(args.n_frame, len(samples))
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(samples, sample_count), key=lambda item: item.output_rel_path.as_posix())

    d_compare.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    for sample in selected:
        original_out = d_compare / sample.output_rel_path
        original_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image_path, original_out)

        gt_out = overlay_output_path(original_out, 'GT')
        yolo_out = overlay_output_path(original_out, 'Yolo26x-Pose')
        vitpose_out = overlay_output_path(original_out, 'VitPosePP')

        render_overlay(
            image_path=sample.image_path,
            predictions=sample.gt_annotations,
            output_path=gt_out,
            skeleton=skeleton,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            args=args,
        )
        render_overlay(
            image_path=sample.image_path,
            predictions=sample.yolo_predictions,
            output_path=yolo_out,
            skeleton=skeleton,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            args=args,
        )
        render_overlay(
            image_path=sample.image_path,
            predictions=sample.vitpose_predictions,
            output_path=vitpose_out,
            skeleton=skeleton,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            args=args,
        )

        manifest.append(
            {
                'image': sample.output_rel_path.as_posix(),
                'original': original_out.as_posix(),
                'gt_overlay': gt_out.as_posix(),
                'yolo_overlay': yolo_out.as_posix(),
                'vitpose_overlay': vitpose_out.as_posix(),
            }
        )

    (d_compare / '_manifest.json').write_text(
        json.dumps(
            {
                'n_frame_requested': args.n_frame,
                'n_frame_rendered': len(selected),
                'd_test': str(d_test),
                'd_compare': str(d_compare),
                'kp_yolo': str(kp_yolo),
                'kp_vitpose': str(kp_vitpose),
                'dataset_annotations': str(annotation_path),
                'vitpose_manifest': str(vitpose_manifest_path) if vitpose_manifest_path is not None else '',
                'samples': manifest,
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    print(f'Rendered frames: {len(selected)}')
    print(f'Output directory: {d_compare}')


if __name__ == '__main__':
    main()
