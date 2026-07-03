#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import cv2
from ultralytics import YOLO

PREDICTION_DIR = Path(__file__).resolve().parents[1] / 'yolo26x_detection_prediction'
sys.path.insert(0, str(PREDICTION_DIR))
from yolo_detection_utils import select_top_detection_box  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate YOLO26x detection on a split and export bbox JSON/overlays.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--split', default='test', choices=('train', 'val', 'test'))
    parser.add_argument('--imgsz', type=int, default=768)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--bbox-json', required=True)
    parser.add_argument('--metrics-json', required=True)
    parser.add_argument('--overlay-dir', required=True)
    parser.add_argument('--overlay-max-images', type=int, default=0, help='0 means all images')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def image_paths(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png'})


def draw_bbox(image: Any, xyxy: list[float], confidence: float) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(image, f'swimmer {confidence:.3f}', (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def metrics_payload(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, 'box', None)
    return json_safe({
        'precision': getattr(box, 'mp', None),
        'recall': getattr(box, 'mr', None),
        'mAP50': getattr(box, 'map50', None),
        'mAP50-95': getattr(box, 'map', None),
        'fitness': getattr(metrics, 'fitness', None),
        'results_dict': getattr(metrics, 'results_dict', None),
    })


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    data_yaml = Path(args.data).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    bbox_json = Path(args.bbox_json).expanduser().resolve()
    metrics_json = Path(args.metrics_json).expanduser().resolve()
    overlay_dir = Path(args.overlay_dir).expanduser().resolve()
    images_dir = dataset_root / 'images' / args.split

    for required in (model_path, data_yaml):
        if not required.is_file():
            raise FileNotFoundError(f'Missing required file: {required}')
    if not images_dir.is_dir():
        raise FileNotFoundError(f'Missing images directory: {images_dir}')
    for output in (bbox_json, metrics_json):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f'Output exists, pass --overwrite: {output}')
    if overlay_dir.exists() and any(overlay_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f'Overlay dir is not empty, pass --overwrite: {overlay_dir}')

    bbox_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers, verbose=False)

    predictions = []
    no_detection = 0
    overlay_limit = None if args.overlay_max_images == 0 else args.overlay_max_images
    overlays_written = 0
    for image_path in image_paths(images_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        results = model.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
        selected = select_top_detection_box(results[0].boxes if results else None)
        record: dict[str, Any] = {'image': image_path.name, 'width': width, 'height': height, 'bbox_xyxy': None, 'confidence': None, 'class_id': 0}
        if selected is None:
            no_detection += 1
        else:
            record['bbox_xyxy'] = selected.xyxy
            record['confidence'] = selected.confidence
            draw_bbox(image, selected.xyxy, selected.confidence)
        predictions.append(record)
        if overlay_limit is None or overlays_written < overlay_limit:
            cv2.imwrite(str(overlay_dir / image_path.name), image)
            overlays_written += 1

    bbox_payload = {
        'model': str(model_path),
        'data': str(data_yaml),
        'dataset_root': str(dataset_root),
        'split': args.split,
        'imgsz': args.imgsz,
        'conf': args.conf,
        'selection_rule': 'highest confidence, then largest xyxy area on confidence ties',
        'images': len(predictions),
        'no_detection': no_detection,
        'predictions': predictions,
    }
    bbox_json.write_text(json.dumps(bbox_payload, indent=2), encoding='utf-8')

    payload = metrics_payload(metrics)
    payload.update({'model': str(model_path), 'data': str(data_yaml), 'split': args.split, 'bbox_json': str(bbox_json), 'overlay_dir': str(overlay_dir)})
    metrics_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
