from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from yolo_detection_utils import select_top_detection_box


DEFAULT_MODEL = (
    "runs/yolo26x_bbox_side_above_water/"
    "yolo26x-detection_from_cfg03_ep5_20260523_1923/weights/best.pt"
)
DEFAULT_DATASET_ROOT = "data/intermediate/Side_above_water/_Yolo26x_detection"
DEFAULT_OUTPUT_ROOT = "data/output/experiments/yolo26x_detection_predictions/top1_from_cfg03_ep5_20260523_1923"
SPLITS = ("val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO26x detector predicted labels with at most one bbox per image."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_paths(images_dir: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png"}
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in suffixes)


def normalized_yolo_line(xyxy: list[float], confidence: float, image_width: int, image_height: int) -> str:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(image_width), float(x1)))
    y1 = max(0.0, min(float(image_height), float(y1)))
    x2 = max(0.0, min(float(image_width), float(x2)))
    y2 = max(0.0, min(float(image_height), float(y2)))
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    x_center = (x1 + width / 2.0) / float(image_width)
    y_center = (y1 + height / 2.0) / float(image_height)
    return (
        f"0 {x_center:.8f} {y_center:.8f} "
        f"{width / float(image_width):.8f} {height / float(image_height):.8f} "
        f"{confidence:.8f}"
    )


def export_split(model: YOLO, dataset_root: Path, output_root: Path, split: str, imgsz: int, conf: float) -> dict[str, int | str]:
    images_dir = dataset_root / "images" / split
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")

    split_output = output_root / split
    labels_output = split_output / "labels"
    labels_output.mkdir(parents=True, exist_ok=True)

    paths = image_paths(images_dir)
    images = 0
    detections = 0
    no_detection = 0
    unreadable = 0

    for image_path in paths:
        image = cv2.imread(str(image_path))
        if image is None:
            unreadable += 1
            continue
        image_height, image_width = image.shape[:2]
        results = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)
        boxes = results[0].boxes if results else None
        selected_box = select_top_detection_box(boxes)

        label_path = labels_output / f"{image_path.stem}.txt"
        if selected_box is None:
            label_path.write_text("", encoding="utf-8")
            no_detection += 1
        else:
            label_path.write_text(
                normalized_yolo_line(selected_box.xyxy, selected_box.confidence, image_width, image_height) + "\n",
                encoding="utf-8",
            )
            detections += 1
        images += 1

    summary = {
        "split": split,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_output),
        "images": images,
        "detections": detections,
        "no_detection": no_detection,
        "unreadable": unreadable,
        "label_format": "class x_center y_center width height confidence",
        "selection_rule": "highest confidence, then largest xyxy area on confidence ties",
    }
    (split_output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists, pass --overwrite: {output_root}")
    if output_root.exists():
        import shutil

        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    summaries = [export_split(model, dataset_root, output_root, split, args.imgsz, args.conf) for split in args.splits]
    payload = {
        "model": str(model_path),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "splits": summaries,
    }
    (output_root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
