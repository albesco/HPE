from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from prepare_swimxyz_vitposepp_utils import (
    Sample,
    build_coco_json,
    build_keypoints_and_bbox,
    read_label_rows,
    split_samples,
)


DEFAULT_INPUT_ROOT = "data/input/subset_xyz/Side_above_water_frames"
DEFAULT_OUTPUT_ROOT = "data/intermediate/Side_above_water_frames"
DEFAULT_COPY_MODE = "symlink"
DEFAULT_VITPOSE_WORK_DIR = "runs/vitposepp_side_above_water_frames"
SPLITS = ("train", "val", "test")
REPORT_NAME = "swimxyz_frames_preparation_report.json"
MANIFEST_NAME = "manifest.json"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
FRAME_TOKENS = ("__frame_", "__frm_")


@dataclass(frozen=True)
class FrameEntry:
    image_path: Path
    label_path: Path
    source_name: str
    frame_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a canonical SwimXYZ frame dataset from standalone COCO 2D cam "
            "frame+label pairs, then regenerate the VitPose++, YOLO26x detection, "
            "and YOLO26x pose training exports."
        )
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--bbox-padding-x-ratio", type=float, default=0.20)
    parser.add_argument("--bbox-padding-y-ratio", type=float, default=0.25)
    parser.add_argument("--bbox-min-padding-px", type=float, default=15.0)
    parser.add_argument("--min-visible-keypoints", type=int, default=4)
    parser.add_argument("--copy-mode", choices=("copy", "symlink"), default=DEFAULT_COPY_MODE)
    parser.add_argument("--vitpose-work-dir", default=DEFAULT_VITPOSE_WORK_DIR)
    parser.add_argument("--yolo-pose-link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--skip-model-exports", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def ensure_removed(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def split_image_dir(split: str) -> str:
    return f"{split}2017"


def parse_frame_entry(image_path: Path) -> FrameEntry:
    suffix = "__COCO__2D_cam"
    stem = image_path.stem
    prefix = ""
    frame_token = ""

    # Preferred legacy formats, for example:
    #   source__frame_000002.jpg
    #   source__frm_000002.jpg
    for token in FRAME_TOKENS:
        if token in stem:
            prefix, frame_token = stem.rsplit(token, 1)
            break

    # SwimXYZ SAW/FSAW exports may instead end directly with the numeric frame
    # index, for example:
    #   FSAW_...__pos_1_75_000002.jpg
    # In that case, use the final underscore-separated numeric suffix as the
    # frame index and keep everything before it as the source name.
    if not prefix:
        match = re.fullmatch(r"(.+)_([0-9]+)", stem)
        if match is None:
            raise ValueError(
                f"Image filename missing supported frame token {FRAME_TOKENS} "
                f"and does not end with a numeric frame index: {image_path.name}"
            )
        prefix = match.group(1)
        frame_token = match.group(2)

    if not re.fullmatch(r"\d+", frame_token):
        raise ValueError(f"Invalid frame index token in filename: {image_path.name}")
    label_path = image_path.with_name(f"{stem}{suffix}.txt")
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing COCO 2D cam label for frame: {image_path.name}")
    return FrameEntry(
        image_path=image_path,
        label_path=label_path,
        source_name=prefix.replace(",", "_"),
        frame_index=int(frame_token),
    )


def discover_frame_entries(input_root: Path) -> list[FrameEntry]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Missing input root: {input_root}")
    image_paths = sorted(
        path
        for extension in IMAGE_EXTENSIONS
        for path in input_root.glob(f"*{extension}")
    )
    entries = [parse_frame_entry(path) for path in image_paths]
    if not entries:
        raise RuntimeError(
            f"No frame image files found in {input_root} for extensions {IMAGE_EXTENSIONS}"
        )
    return entries


def copy_or_link_image(source: Path, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def extract_sample(
    entry: FrameEntry,
    bbox_padding_x_ratio: float,
    bbox_padding_y_ratio: float,
    bbox_min_padding_px: float,
    min_visible_keypoints: int,
) -> tuple[Sample | None, str | None]:
    rows = read_label_rows(entry.label_path)
    if len(rows) != 1:
        return None, "expected_single_row"
    row = rows[0]

    import cv2
    image = cv2.imread(str(entry.image_path))
    if image is None:
        return None, "unreadable_image"
    height, width = image.shape[:2]

    prepared = build_keypoints_and_bbox(
        row=row,
        width=width,
        height=height,
        bbox_padding_x_ratio=bbox_padding_x_ratio,
        bbox_padding_y_ratio=bbox_padding_y_ratio,
        bbox_min_padding_px=bbox_min_padding_px,
        min_visible_keypoints=min_visible_keypoints,
        flip_y=True,
    )
    if prepared is None:
        return None, "no_valid_keypoints"

    keypoints, num_keypoints, bbox, area = prepared
    sample = Sample(
        source_name=entry.source_name,
        frame_index=entry.frame_index,
        image_path=entry.image_path,
        width=width,
        height=height,
        bbox=bbox,
        keypoints=keypoints,
        num_keypoints=num_keypoints,
        area=area,
    )
    return sample, None


def materialize_split(samples: list[Sample], destination_dir: Path, copy_mode: str) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        normalized_name = f"{sample.source_name}_{sample.frame_index:06d}{sample.image_path.suffix.lower()}"
        target = destination_dir / normalized_name
        copy_or_link_image(sample.image_path, target, copy_mode)
        sample.image_path = target


def run_subprocess(command: list[str], project_root: Path) -> None:
    subprocess.run(command, cwd=project_root, check=True)


def ensure_detection_image_symlinks(canonical_root: Path, detection_root: Path) -> None:
    for split in SPLITS:
        target = detection_root / "images" / split
        target.parent.mkdir(parents=True, exist_ok=True)
        ensure_removed(target)
        target.symlink_to((canonical_root / split_image_dir(split)).resolve())


def load_report(report_path: Path) -> dict[str, Any] | None:
    if not report_path.is_file():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def export_model_datasets(
    project_root: Path,
    canonical_root: Path,
    output_root: Path,
    vitpose_work_dir: Path,
    yolo_pose_link_mode: str,
) -> dict[str, Any]:
    vitpose_root = output_root / "_VitPosePP"
    detection_root = output_root / "_Yolo26x_detection"
    pose_root = output_root / "_Yolo26x_pose"

    run_subprocess(
        [
            sys.executable,
            "script/prepare_vitposepp_dataset.py",
            "--canonical-root", canonical_root.as_posix(),
            "--output-root", vitpose_root.as_posix(),
            "--work-dir", vitpose_work_dir.as_posix(),
            "--overwrite",
        ],
        project_root,
    )
    run_subprocess(
        [
            sys.executable,
            "script/yolo_training/prepare_yolo_detection_dataset.py",
            "--dataset-root", canonical_root.as_posix(),
            "--output-root", detection_root.as_posix(),
            "--overwrite",
        ],
        project_root,
    )
    ensure_detection_image_symlinks(canonical_root, detection_root)
    run_subprocess(
        [
            sys.executable,
            "script/yolo_training/prepare_yolo_pose_dataset.py",
            "--dataset-root", canonical_root.as_posix(),
            "--output-root", pose_root.as_posix(),
            "--link-mode", yolo_pose_link_mode,
            "--overwrite",
        ],
        project_root,
    )
    return {
        "vitposepp": load_report(vitpose_root / "preparation_report.json"),
        "yolo26x_detection": load_report(detection_root / "preparation_report.json"),
        "yolo26x_pose": load_report(pose_root / "preparation_report.json"),
    }


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path.cwd().resolve()
    input_root = resolve_path(args.input_root)
    output_root = resolve_path(args.output_root)
    canonical_root = output_root / "_train_canonical"

    if output_root.exists() or output_root.is_symlink():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists, pass --overwrite: {output_root}")
        ensure_removed(output_root)
    canonical_root.mkdir(parents=True, exist_ok=True)
    (canonical_root / "annotations").mkdir(parents=True, exist_ok=True)

    entries = discover_frame_entries(input_root)
    samples: list[Sample] = []
    skip_counters: dict[str, int] = {}
    for entry in entries:
        sample, reason = extract_sample(
            entry,
            bbox_padding_x_ratio=args.bbox_padding_x_ratio,
            bbox_padding_y_ratio=args.bbox_padding_y_ratio,
            bbox_min_padding_px=args.bbox_min_padding_px,
            min_visible_keypoints=args.min_visible_keypoints,
        )
        if sample is None:
            skip_counters[reason or "unknown"] = skip_counters.get(reason or "unknown", 0) + 1
            continue
        samples.append(sample)

    if not samples:
        raise RuntimeError("No valid frame samples were extracted.")

    samples.sort(key=lambda sample: (sample.source_name, sample.frame_index))
    train_samples, val_samples, test_samples = split_samples(samples, args.val_ratio, args.test_ratio)

    materialize_split(train_samples, canonical_root / "train2017", args.copy_mode)
    materialize_split(val_samples, canonical_root / "val2017", args.copy_mode)
    materialize_split(test_samples, canonical_root / "test2017", args.copy_mode)

    (canonical_root / "annotations" / "person_keypoints_train.json").write_text(
        json.dumps(build_coco_json(train_samples, "train2017"), indent=2), encoding="utf-8"
    )
    (canonical_root / "annotations" / "person_keypoints_val.json").write_text(
        json.dumps(build_coco_json(val_samples, "val2017"), indent=2), encoding="utf-8"
    )
    (canonical_root / "annotations" / "person_keypoints_test.json").write_text(
        json.dumps(build_coco_json(test_samples, "test2017"), indent=2), encoding="utf-8"
    )

    exports = {} if args.skip_model_exports else export_model_datasets(
        project_root=project_root,
        canonical_root=canonical_root,
        output_root=output_root,
        vitpose_work_dir=resolve_path(args.vitpose_work_dir),
        yolo_pose_link_mode=args.yolo_pose_link_mode,
    )

    report = {
        "input_root": input_root.as_posix(),
        "dataset_root": output_root.as_posix(),
        "canonical_root": canonical_root.as_posix(),
        "label_type": "COCO__2D_cam",
        "copy_mode": args.copy_mode,
        "split_ratios": {"train": 1.0 - args.val_ratio - args.test_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "bbox_padding": {
            "x_ratio": args.bbox_padding_x_ratio,
            "y_ratio": args.bbox_padding_y_ratio,
            "min_padding_px": args.bbox_min_padding_px,
        },
        "min_visible_keypoints": args.min_visible_keypoints,
        "source_pairs_total": len(entries),
        "accepted_samples_total": len(samples),
        "rejected_pairs": skip_counters,
        "canonical_splits": {
            "train": {"images": len(train_samples), "annotations": len(train_samples)},
            "val": {"images": len(val_samples), "annotations": len(val_samples)},
            "test": {"images": len(test_samples), "annotations": len(test_samples)},
        },
        "exports": exports,
    }
    (canonical_root / "reports").mkdir(parents=True, exist_ok=True)
    (canonical_root / "reports" / REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    (canonical_root / MANIFEST_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_root / MANIFEST_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = build_dataset(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
