#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


DEFAULT_MODEL = 'runs/yolo26x_pose_side_above_water/yolo26x_pose_coco17_1ep_testmap/weights/best.pt'
DEFAULT_SOURCE = 'data/intermediate/Side_above_water/_Yolo26x_pose/images/test'
DEFAULT_OUTPUT_DIR = 'data/output/experiments/yolo26x_pose_side_above_water/predictions'

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run YOLO26x-pose prediction and save keypoints to JSON files.'
    )
    parser.add_argument('--model', required=True, help="Path to .pt weights (best.pt/last.pt)")
    parser.add_argument('--source', required=True, help="Path to image or directory of images")
    parser.add_argument('--output-dir', required=True, help="Directory to save JSON output")
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='0')
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


def predict_one(
    model,
    image_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> int:
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
    keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else np.empty((0, 17, 3))

    # Create a serializable representation of the keypoints
    output_data = {
        'image_path': str(image_path),
        'predictions': []
    }
    for i in range(len(keypoints)):
        instance_kpts = keypoints[i]
        output_data['predictions'].append(instance_kpts.tolist())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    return len(keypoints)


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
    source = resolve_path(args.source)
    output_dir = resolve_path(args.output_dir)

    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not source.exists():
        raise FileNotFoundError(source)

    from ultralytics import YOLO

    model = YOLO(str(model_path))

    images = collect_images(source, args.max_images, args.seed)
    predicted_files = 0
    detections = 0
    for index, image_path in enumerate(images):
        output_filename = image_path.stem + '.json'
        output_path = output_dir / output_filename
        detections += predict_one(model, image_path, output_path, args)
        predicted_files += 1
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1}/{len(images)} images...")

    print(f'Saved predictions for: {predicted_files} images')
    print(f'Total detections: {detections}')
    print(f'Output directory: {output_dir}')


if __name__ == '__main__':
    main()
