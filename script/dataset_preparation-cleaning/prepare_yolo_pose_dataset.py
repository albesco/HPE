from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


DEFAULT_DATASET_ROOT = "data/intermediate/Side_above_water/_train_canonical"
DEFAULT_OUTPUT_ROOT = "data/intermediate/Side_above_water/_Yolo26x_pose"
SPLITS = ("train", "val", "test")
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
COCO_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export canonical SwimXYZ COCO keypoint labels into an Ultralytics "
            "YOLO26x-pose dataset. This exporter writes labels and exposes images "
            "with symlinks, hardlinks, or copies; it does not resize or modify images."
        )
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to expose source images under the YOLO pose dataset root.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def split_annotation_path(dataset_root: Path, split: str) -> Path:
    return dataset_root / "annotations" / f"person_keypoints_{split}.json"


def split_image_root(dataset_root: Path, split: str) -> Path:
    return dataset_root / f"{split}2017"


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


def normalize_bbox_xywh(bbox: list[float], image_width: int, image_height: int) -> list[float]:
    x, y, width, height = bbox
    return [
        (x + width / 2.0) / image_width,
        (y + height / 2.0) / image_height,
        width / image_width,
        height / image_height,
    ]


def normalize_keypoints(keypoints: list[float], image_width: int, image_height: int) -> list[float]:
    if len(keypoints) != 51:
        raise ValueError(f"Expected 51 keypoint values, got {len(keypoints)}")

    normalized: list[float] = []
    for index in range(0, len(keypoints), 3):
        x, y, visibility = keypoints[index : index + 3]
        if visibility <= 0:
            normalized.extend([0.0, 0.0, 0.0])
            continue
        normalized.extend([
            min(max(x / image_width, 0.0), 1.0),
            min(max(y / image_height, 0.0), 1.0),
            float(int(visibility)),
        ])
    return normalized


def yolo_pose_line_from_annotation(annotation: dict, image_width: int, image_height: int) -> str | None:
    bbox = annotation.get("bbox")
    keypoints = annotation.get("keypoints")
    if not bbox or len(bbox) != 4 or not keypoints:
        return None
    if bbox[2] <= 0 or bbox[3] <= 0:
        return None

    formatted = ["0"]
    formatted.extend(f"{value:.8f}" for value in normalize_bbox_xywh(bbox, image_width, image_height))
    for index, value in enumerate(normalize_keypoints(keypoints, image_width, image_height)):
        formatted.append(str(int(value)) if index % 3 == 2 else f"{value:.8f}")
    return " ".join(formatted)


def convert_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
    link_mode: str,
) -> dict[str, int]:
    ann_path = split_annotation_path(dataset_root, split)
    images_root = split_image_root(dataset_root, split)
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_root}")

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    annotations_by_image_id: dict[int, list[dict]] = {}
    for annotation in coco.get("annotations", []):
        annotations_by_image_id.setdefault(int(annotation["image_id"]), []).append(annotation)

    image_out_root = output_root / "images" / split
    label_out_root = output_root / "labels" / split
    label_out_root.mkdir(parents=True, exist_ok=True)
    image_count = 0
    label_count = 0
    skipped_count = 0

    for image_id, image_info in sorted(images_by_id.items()):
        file_name = image_info["file_name"]
        source_image = images_root / file_name
        if not source_image.is_file():
            skipped_count += 1
            continue

        annotations = annotations_by_image_id.get(image_id, [])
        if not annotations:
            skipped_count += 1
            continue

        image_width = int(image_info["width"])
        image_height = int(image_info["height"])
        label_lines = [
            line
            for annotation in annotations
            if (line := yolo_pose_line_from_annotation(annotation, image_width, image_height)) is not None
        ]
        if not label_lines:
            skipped_count += 1
            continue

        link_image(source_image, image_out_root / file_name, link_mode)
        label_path = label_out_root / f"{Path(file_name).stem}.txt"
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        image_count += 1
        label_count += len(label_lines)

    return {"images": image_count, "labels": label_count, "skipped": skipped_count}


def write_dataset_yaml(output_root: Path) -> Path:
    yaml_path = output_root / "swimxyz_side_above_water_yolo26x_pose.yaml"
    lines = [
        f"path: {output_root.as_posix()}",
        "# Images are exposed under images/<split>; labels are under labels/<split>.",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "kpt_shape: [17, 3]",
        "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]",
        "names:",
        "  0: swimmer",
        "kpt_names:",
        "  0:",
    ]
    lines.extend(f"    - {name}" for name in COCO_KEYPOINT_NAMES)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


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
        "kpt_shape": [17, 3],
        "flip_idx": COCO_FLIP_IDX,
        "keypoint_names": COCO_KEYPOINT_NAMES,
        "image_operations": args.link_mode,
        "splits": {},
    }
    for split in SPLITS:
        summary["splits"][split] = convert_split(dataset_root, output_root, split, args.link_mode)

    yaml_path = write_dataset_yaml(output_root)
    (output_root / "preparation_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"YOLO26x-pose data yaml: {yaml_path}")


if __name__ == "__main__":
    main()
