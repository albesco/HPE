from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from prepare_swimxyz_vitpose_train import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PRETRAINED_CHECKPOINT,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_WORK_DIR,
    Sample,
    build_train_output_root,
    build_coco_json,
    build_keypoints_and_bbox,
    iter_converted_dataset_roots,
    parse_dataset_entries,
    read_label_rows,
    resolve_path,
    write_training_config,
)


SPLITS = {
    "train": "train2017",
    "val": "val2017",
    "test": "test2017",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate only VitPose COCO annotation JSON files for an existing "
            "SwimXYZ image dataset, without extracting videos or rewriting images."
        )
    )
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing converted subset_xyz videos and labels.",
    )
    parser.add_argument(
        "--dataset-entry",
        action="append",
        default=[],
        help="Pair formatted as video_path::labels_path. Repeatable.",
    )
    parser.add_argument("--label-format", default="COCO")
    parser.add_argument("--label-dimension", default="2D")
    parser.add_argument("--label-reference", default="cam")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--pretrained-checkpoint", default=DEFAULT_PRETRAINED_CHECKPOINT)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--bbox-padding-ratio", type=float, default=0.05)
    parser.add_argument("--min-visible-keypoints", type=int, default=4)
    parser.add_argument(
        "--flip-y",
        dest="flip_y",
        action="store_true",
        default=True,
        help="Convert SwimXYZ bottom-origin y coordinates to image top-origin coordinates.",
    )
    parser.add_argument(
        "--no-flip-y",
        dest="flip_y",
        action="store_false",
        help="Keep label y coordinates unchanged.",
    )
    parser.add_argument("--samples-per-gpu", type=int, default=8)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--total-epochs", type=int, default=30)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Do not update the generated VitPose training config.",
    )
    return parser.parse_args()


def image_stem_to_source_and_frame(image_path: Path) -> tuple[str, int]:
    stem = image_path.stem.removesuffix("_with-KP")
    source_name, frame_value = stem.rsplit("_", 1)
    return source_name, int(frame_value)


def build_label_index(args: argparse.Namespace, project_root: Path) -> dict[str, list[dict[str, str]]]:
    entries = parse_dataset_entries(args, project_root)
    label_index = {}
    for entry in entries:
        source_name = entry.video_path.stem.replace(",", "_")
        label_index[source_name] = read_label_rows(entry.labels_path)
    return label_index


def image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def collect_split_samples(
    split_dir: Path,
    label_index: dict[str, list[dict[str, str]]],
    bbox_padding_ratio: float,
    min_visible_keypoints: int,
    flip_y: bool,
) -> list[Sample]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split image directory: {split_dir}")

    samples = []
    for image_path in sorted(split_dir.glob("*.jpg")):
        if image_path.stem.endswith("_with-KP"):
            continue

        source_name, frame_index = image_stem_to_source_and_frame(image_path)
        if source_name not in label_index:
            raise KeyError(f"No label entry found for source image prefix: {source_name}")

        label_rows = label_index[source_name]
        if frame_index >= len(label_rows):
            raise IndexError(
                f"Frame {frame_index} is outside label rows for source {source_name}"
            )

        width, height = image_size(image_path)
        prepared = build_keypoints_and_bbox(
            row=label_rows[frame_index],
            width=width,
            height=height,
            bbox_padding_ratio=bbox_padding_ratio,
            min_visible_keypoints=min_visible_keypoints,
            flip_y=flip_y,
        )
        if prepared is None:
            continue

        keypoints, num_keypoints, bbox, area = prepared
        samples.append(
            Sample(
                source_name=source_name,
                frame_index=frame_index,
                image_path=image_path,
                width=width,
                height=height,
                bbox=bbox,
                keypoints=keypoints,
                num_keypoints=num_keypoints,
                area=area,
            )
        )

    return samples


def regenerate_labels(
    args: argparse.Namespace,
    project_root: Path,
    dataset_root: Path,
    output_root: Path,
) -> None:
    scoped_args = argparse.Namespace(**vars(args))
    scoped_args.dataset_root = str(dataset_root)
    annotations_dir = output_root / "annotations"
    generated_config = output_root / "generated_configs" / "swimxyz_vitpose_huge.py"

    if not output_root.is_dir():
        raise FileNotFoundError(f"Missing existing VitPose dataset root: {output_root}")

    label_index = build_label_index(scoped_args, project_root)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    split_counts = {}
    for split_name, split_dir_name in SPLITS.items():
        split_dir = output_root / split_dir_name
        samples = collect_split_samples(
            split_dir=split_dir,
            label_index=label_index,
            bbox_padding_ratio=args.bbox_padding_ratio,
            min_visible_keypoints=args.min_visible_keypoints,
            flip_y=args.flip_y,
        )
        annotation_path = annotations_dir / f"person_keypoints_{split_name}.json"
        annotation_path.write_text(
            json.dumps(build_coco_json(samples, split_dir_name), indent=2),
            encoding="utf-8",
        )
        split_counts[split_name] = len(samples)

    if not args.skip_config:
        base_config = resolve_path(project_root, args.base_config)
        pretrained_checkpoint = resolve_path(project_root, args.pretrained_checkpoint)
        work_dir = resolve_path(project_root, args.work_dir)
        if not base_config.is_file():
            raise FileNotFoundError(f"Missing base config: {base_config}")
        if not pretrained_checkpoint.is_file():
            raise FileNotFoundError(f"Missing pretrained checkpoint: {pretrained_checkpoint}")

        write_training_config(
            output_path=generated_config,
            dataset_root=output_root,
            base_config=base_config,
            pretrained_checkpoint=pretrained_checkpoint,
            work_dir=work_dir,
            total_epochs=args.total_epochs,
            val_interval=args.val_interval,
            samples_per_gpu=args.samples_per_gpu,
            workers_per_gpu=args.workers_per_gpu,
        )

    print("Regenerated SwimXYZ VitPose annotation labels only.")
    print(f"Converted dataset root: {dataset_root}")
    print(f"Dataset root: {output_root}")
    print(f"Train samples: {split_counts['train']}")
    print(f"Val samples: {split_counts['val']}")
    print(f"Test samples: {split_counts['test']}")
    print(f"Annotations root: {annotations_dir}")
    if not args.skip_config:
        print(f"Generated config: {generated_config}")


def main() -> None:
    args = parse_args()
    project_root = resolve_path(Path.cwd(), args.project_root)
    dataset_root = resolve_path(project_root, args.dataset_root)

    if args.dataset_entry:
        output_root = build_train_output_root(project_root, dataset_root, args.output_root)
        regenerate_labels(args, project_root, dataset_root, output_root)
        return

    dataset_roots = iter_converted_dataset_roots(dataset_root)
    if len(dataset_roots) > 1 and args.output_root:
        raise ValueError(
            "When processing multiple converted datasets, do not pass a shared "
            "--output-root. Let the script infer one '_train_vitpose' directory per dataset."
        )
    for current_dataset_root in dataset_roots:
        output_root = build_train_output_root(
            project_root,
            current_dataset_root,
            args.output_root,
        )
        regenerate_labels(args, project_root, current_dataset_root, output_root)


if __name__ == "__main__":
    main()
