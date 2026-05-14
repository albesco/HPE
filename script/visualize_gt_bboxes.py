from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from pose_overlay_utils import DEFAULT_DATASET_ROOT, draw_bbox_xywh, resolve_path, split_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw GT bbox overlays for a prepared COCO split.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output-dir", default="data/intermediate/bbox_val")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of images to process. 0 means all.")
    return parser.parse_args()


def output_path(output_dir: Path, source_name: str) -> Path:
    source_path = Path(source_name)
    return output_dir / f"{source_path.stem}_bbox{source_path.suffix}"


def main() -> None:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    output_dir = resolve_path(args.output_dir)
    ann_path, images_dir = split_paths(dataset_root, args.split)

    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    annotations_by_image_id: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        annotations_by_image_id.setdefault(int(ann["image_id"]), []).append(ann)

    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for image_info in coco.get("images", []):
        if args.limit > 0 and saved >= args.limit:
            break

        image_id = int(image_info["id"])
        image_name = image_info["file_name"]
        image_path = images_dir / image_name
        image_annotations = annotations_by_image_id.get(image_id, [])
        if not image_path.is_file() or not image_annotations:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        for ann in image_annotations:
            bbox = ann.get("bbox")
            if bbox and len(bbox) == 4:
                draw_bbox_xywh(image, bbox, (0, 255, 0))

        cv2.imwrite(str(output_path(output_dir, image_name)), image)
        saved += 1

    print(f"Saved {saved} bbox overlays for {args.split} split to: {output_dir}")


if __name__ == "__main__":
    main()
