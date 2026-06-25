#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from prepare_yolo_pose_dataset import (
    COCO_FLIP_IDX,
    COCO_KEYPOINT_NAMES,
    link_image,
    yolo_pose_line_from_annotation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a COCO keypoint subset to an Ultralytics YOLO pose test dataset."
    )
    parser.add_argument("--annotation", required=True, help="COCO keypoint annotation JSON")
    parser.add_argument("--images-dir", required=True, help="Directory containing subset images")
    parser.add_argument("--output-root", required=True, help="Output YOLO pose dataset root")
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_dataset_yaml(output_root: Path) -> Path:
    yaml_path = output_root / "yolo_pose_subset.yaml"
    lines = [
        f"path: {output_root.as_posix()}",
        "train: images/test",
        "val: images/test",
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
    annotation = Path(args.annotation).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists, pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    coco = json.loads(annotation.read_text(encoding="utf-8"))
    images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    annotations_by_image_id: dict[int, list[dict]] = {}
    for annotation_row in coco.get("annotations", []):
        annotations_by_image_id.setdefault(int(annotation_row["image_id"]), []).append(annotation_row)

    image_out_root = output_root / "images" / "test"
    label_out_root = output_root / "labels" / "test"
    label_out_root.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0
    skipped_count = 0
    for image_id, image_info in sorted(images_by_id.items()):
        file_name = image_info["file_name"]
        source_image = images_dir / file_name
        if not source_image.is_file():
            skipped_count += 1
            continue

        lines = []
        for annotation_row in annotations_by_image_id.get(image_id, []):
            line = yolo_pose_line_from_annotation(
                annotation_row,
                int(image_info["width"]),
                int(image_info["height"]),
            )
            if line is not None:
                lines.append(line)
        if not lines:
            skipped_count += 1
            continue

        link_image(source_image, image_out_root / file_name, args.link_mode)
        (label_out_root / f"{Path(file_name).stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        image_count += 1
        label_count += len(lines)

    yaml_path = write_dataset_yaml(output_root)
    report = {
        "annotation": annotation.as_posix(),
        "images_dir": images_dir.as_posix(),
        "output_root": output_root.as_posix(),
        "image_operations": args.link_mode,
        "kpt_shape": [17, 3],
        "flip_idx": COCO_FLIP_IDX,
        "splits": {"test": {"images": image_count, "labels": label_count, "skipped": skipped_count}},
        "yaml": yaml_path.as_posix(),
    }
    (output_root / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
