from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
from ultralytics import YOLO


DEFAULT_MODEL = (
    "runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox_padded10_20ep/"
    "frozen_checkpoints/best_epoch10_user_20260514_070621.pt"
)
DEFAULT_SOURCE_DIR = "data/intermediate/Side_above_water/_yolo_detection/images/val"
DEFAULT_OUTPUT_DIR = "data/intermediate/epoch_10_yolo_bbox"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw YOLO predicted bbox overlays on random validation images.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def image_paths(source_dir: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png"}
    return sorted(path for path in source_dir.iterdir() if path.suffix.lower() in suffixes)


def draw_prediction(image, xyxy, confidence: float) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        image,
        f"swimmer {confidence:.3f}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source dir: {source_dir}")

    candidates = image_paths(source_dir)
    if len(candidates) < args.count:
        raise RuntimeError(f"Requested {args.count} images, found {len(candidates)} in {source_dir}")

    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: args.count]

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    saved = 0
    for image_path in selected:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        results = model.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
        boxes = results[0].boxes if results else None
        if boxes is not None and len(boxes) > 0:
            best_index = int(boxes.conf.argmax().item())
            draw_prediction(
                image,
                boxes.xyxy[best_index].detach().cpu().tolist(),
                float(boxes.conf[best_index].detach().cpu().item()),
            )

        cv2.imwrite(str(output_dir / image_path.name), image)
        saved += 1

    print(f"Saved {saved} YOLO bbox previews to: {output_dir}")


if __name__ == "__main__":
    main()
