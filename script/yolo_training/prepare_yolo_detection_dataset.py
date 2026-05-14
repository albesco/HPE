from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


DEFAULT_DATASET_ROOT = "data/intermediate/Side_above_water/_train_vitposepp_swap_ears"
DEFAULT_OUTPUT_ROOT = "data/intermediate/Side_above_water/_yolo_detection"
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the prepared SwimXYZ COCO bbox annotations into an "
            "Ultralytics YOLO detection dataset with one class: swimmer."
        )
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to expose source images under the YOLO dataset root.",
    )
    parser.add_argument(
        "--bbox-padding-ratio",
        type=float,
        default=0.0,
        help=(
            "Optional extra bbox expansion before YOLO normalization. Keep 0.0 when "
            "the source COCO dataset already contains padded GT bboxes."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output dataset before writing the new one.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def split_paths(dataset_root: Path, split: str) -> tuple[Path, Path]:
    return dataset_root / "annotations" / f"person_keypoints_{split}.json", dataset_root / f"{split}2017"


def padded_bbox_xywh(
    bbox: list[float],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> list[float]:
    x, y, width, height = bbox
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    x1 = max(0.0, x - pad_x)
    y1 = max(0.0, y - pad_y)
    x2 = min(float(image_width), x + width + pad_x)
    y2 = min(float(image_height), y + height + pad_y)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def yolo_line_from_bbox(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> str:
    x, y, width, height = bbox
    x_center = (x + width / 2.0) / image_width
    y_center = (y + height / 2.0) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height
    return (
        f"0 {x_center:.8f} {y_center:.8f} "
        f"{normalized_width:.8f} {normalized_height:.8f}"
    )


def link_image(source: Path, destination: Path, link_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if link_mode == "symlink":
        destination.symlink_to(source)
    elif link_mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def convert_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
    link_mode: str,
    bbox_padding_ratio: float,
) -> dict[str, int]:
    ann_path, images_root = split_paths(dataset_root, split)
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_root}")

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    annotations_by_image_id: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        annotations_by_image_id.setdefault(int(ann["image_id"]), []).append(ann)

    image_out_root = output_root / "images" / split
    label_out_root = output_root / "labels" / split
    image_count = 0
    label_count = 0
    skipped_count = 0

    for image_id, image_info in sorted(images_by_id.items()):
        file_name = image_info["file_name"]
        source_image = images_root / file_name
        if not source_image.is_file():
            skipped_count += 1
            continue

        anns = annotations_by_image_id.get(image_id, [])
        if not anns:
            skipped_count += 1
            continue

        destination_image = image_out_root / file_name
        link_image(source_image, destination_image, link_mode)
        image_count += 1

        image_width = int(image_info["width"])
        image_height = int(image_info["height"])
        label_lines: list[str] = []
        for ann in anns:
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            padded_bbox = padded_bbox_xywh(bbox, image_width, image_height, bbox_padding_ratio)
            if padded_bbox[2] <= 0 or padded_bbox[3] <= 0:
                continue
            label_lines.append(yolo_line_from_bbox(padded_bbox, image_width, image_height))

        if not label_lines:
            skipped_count += 1
            continue

        label_path = label_out_root / f"{Path(file_name).stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        label_count += len(label_lines)

    return {
        "images": image_count,
        "labels": label_count,
        "skipped": skipped_count,
    }


def write_dataset_yaml(output_root: Path) -> None:
    yaml_text = "\n".join(
        [
            f"path: {output_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: swimmer",
            "",
        ]
    )
    (output_root / "swimxyz_side_above_water_yolo.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    output_root = resolve_path(args.output_root)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists, pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_dataset_root": dataset_root.as_posix(),
        "output_root": output_root.as_posix(),
        "class_name": "swimmer",
        "bbox_padding_ratio": args.bbox_padding_ratio,
        "link_mode": args.link_mode,
        "splits": {},
    }
    for split in SPLITS:
        summary["splits"][split] = convert_split(
            dataset_root=dataset_root,
            output_root=output_root,
            split=split,
            link_mode=args.link_mode,
            bbox_padding_ratio=args.bbox_padding_ratio,
        )

    write_dataset_yaml(output_root)
    (output_root / "preparation_report.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"YOLO data yaml: {output_root / 'swimxyz_side_above_water_yolo.yaml'}")


if __name__ == "__main__":
    main()
