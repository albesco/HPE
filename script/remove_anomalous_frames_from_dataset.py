from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


ANOMALY_LINE_RE = re.compile(
    r"^(?P<video>.+?) \| frame_index=(?P<frame_index>\d+) \| anomalous_bbox_shift\b"
)


@dataclass(frozen=True)
class AnomalousFrame:
    video_name: str
    frame_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove samples whose labels were flagged as anomalous (abrupt bbox jump) "
            "from an already-built dataset (train/val/test + COCO JSON)."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root containing train2017/val2017/test2017 and annotations/.",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional path to dataset_exceptions.log. Defaults to <dataset-root>/annotations/dataset_exceptions.log.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute removals but do not delete files or rewrite JSON.",
    )
    return parser.parse_args()


def source_name_from_video_filename(video_filename: str) -> str:
    # Must match the naming used by the dataset builder.
    return Path(video_filename).stem.replace(",", "_")


def parse_anomalous_frames(log_path: Path) -> list[AnomalousFrame]:
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing anomaly log: {log_path}")

    anomalies: list[AnomalousFrame] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = ANOMALY_LINE_RE.match(line.strip())
        if not match:
            continue
        anomalies.append(
            AnomalousFrame(
                video_name=match.group("video"),
                frame_index=int(match.group("frame_index")),
            )
        )
    return anomalies


def remove_images_for_anomalies(
    dataset_root: Path,
    anomalies: list[AnomalousFrame],
    dry_run: bool,
) -> tuple[int, int]:
    removed = 0
    missing = 0
    split_dirs = [
        dataset_root / "train2017",
        dataset_root / "val2017",
        dataset_root / "test2017",
    ]
    for anomaly in anomalies:
        source_name = source_name_from_video_filename(anomaly.video_name)
        filename = f"{source_name}_{anomaly.frame_index:06d}.jpg"
        overlay_filename = f"{source_name}_{anomaly.frame_index:06d}_with-KP.jpg"
        for split_dir in split_dirs:
            image_path = split_dir / filename
            overlay_path = split_dir / overlay_filename
            any_found = False
            for path in (image_path, overlay_path):
                if path.exists():
                    any_found = True
                    if not dry_run:
                        path.unlink()
                    removed += 1
            if not any_found:
                # Don't count as missing yet; it might live in another split.
                pass

    # Compute "missing" as anomalies that did not exist in any split (image file).
    for anomaly in anomalies:
        source_name = source_name_from_video_filename(anomaly.video_name)
        filename = f"{source_name}_{anomaly.frame_index:06d}.jpg"
        if not any((split / filename).exists() for split in split_dirs):
            missing += 1
    return removed, missing


def filter_coco_json(
    payload: dict,
    filenames_to_drop: set[str],
) -> dict:
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])

    kept_images = []
    old_image_id_to_new: dict[int, int] = {}
    for image in images:
        file_name = image.get("file_name")
        image_id = image.get("id")
        if not isinstance(file_name, str) or not isinstance(image_id, int):
            continue
        if file_name in filenames_to_drop:
            continue
        new_id = len(kept_images) + 1
        old_image_id_to_new[image_id] = new_id
        kept_images.append(
            {
                **image,
                "id": new_id,
            }
        )

    kept_annotations = []
    for annotation in annotations:
        image_id = annotation.get("image_id")
        if not isinstance(image_id, int):
            continue
        new_image_id = old_image_id_to_new.get(image_id)
        if new_image_id is None:
            continue
        new_ann_id = len(kept_annotations) + 1
        kept_annotations.append(
            {
                **annotation,
                "id": new_ann_id,
                "image_id": new_image_id,
            }
        )

    return {
        **payload,
        "images": kept_images,
        "annotations": kept_annotations,
    }


def rewrite_annotations(
    dataset_root: Path,
    anomalies: list[AnomalousFrame],
    dry_run: bool,
) -> dict[str, int]:
    annotations_dir = dataset_root / "annotations"
    files = [
        annotations_dir / "person_keypoints_train.json",
        annotations_dir / "person_keypoints_val.json",
        annotations_dir / "person_keypoints_test.json",
    ]

    filenames_to_drop = set()
    for anomaly in anomalies:
        source_name = source_name_from_video_filename(anomaly.video_name)
        filenames_to_drop.add(f"{source_name}_{anomaly.frame_index:06d}.jpg")

    removed_by_split: dict[str, int] = {}
    for path in files:
        split_name = path.stem.replace("person_keypoints_", "")
        if not path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        before = len(payload.get("images", []))
        filtered = filter_coco_json(payload, filenames_to_drop)
        after = len(filtered.get("images", []))
        removed_by_split[split_name] = before - after
        if not dry_run:
            path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")

    return removed_by_split


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    log_path = (
        Path(args.log_file).resolve()
        if args.log_file
        else (dataset_root / "annotations" / "dataset_exceptions.log")
    )

    anomalies = parse_anomalous_frames(log_path)
    if not anomalies:
        print(f"No anomalous frames found in log: {log_path}")
        return

    removed_files, missing_frames = remove_images_for_anomalies(
        dataset_root=dataset_root,
        anomalies=anomalies,
        dry_run=args.dry_run,
    )
    removed_by_split = rewrite_annotations(
        dataset_root=dataset_root,
        anomalies=anomalies,
        dry_run=args.dry_run,
    )

    print(f"Dataset root: {dataset_root}")
    print(f"Anomaly log: {log_path}")
    print(f"Anomalous frames (log entries): {len(anomalies)}")
    print(f"Removed image/overlay files: {removed_files}")
    print(f"Anomalous frames missing from splits: {missing_frames}")
    for split_name in ("train", "val", "test"):
        print(f"Removed from {split_name} annotations: {removed_by_split.get(split_name, 0)}")
    if args.dry_run:
        print("Dry-run enabled: no files were deleted and no JSON was rewritten.")


if __name__ == "__main__":
    main()

