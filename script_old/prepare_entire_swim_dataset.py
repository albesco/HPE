from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DATASET_ROOTS = [
    "data/intermediate/Side_above_water",
    "data/intermediate/Side_above_water_VideoTest2",
]
DEFAULT_OUTPUT_DATASET_ROOT = "data/intermediate/Side_above_water_EntireSwim"
DEFAULT_VITPOSE_WORK_DIR = "runs/vitposepp_side_above_water_entireswim"
DEFAULT_COPY_MODE = "symlink"
SPLITS = ("train", "val", "test")
REPORT_NAME = "entire_swim_preparation_report.json"
MANIFEST_NAME = "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the EntireSwim aggregate dataset from multiple complementary "
            "canonical SwimXYZ datasets, keeping only images whose keypoint "
            "annotations have every keypoint visible and inside the image."
        )
    )
    parser.add_argument(
        "--source-dataset-roots",
        nargs="+",
        default=list(DEFAULT_SOURCE_DATASET_ROOTS),
        help=(
            "Dataset roots that contain a `_train_canonical` child, or the canonical "
            "roots themselves. Sources are aggregated in the provided order."
        ),
    )
    parser.add_argument("--output-dataset-root", default=DEFAULT_OUTPUT_DATASET_ROOT)
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default=DEFAULT_COPY_MODE,
        help="How to materialize accepted canonical images in the aggregate dataset.",
    )
    parser.add_argument("--vitpose-work-dir", default=DEFAULT_VITPOSE_WORK_DIR)
    parser.add_argument(
        "--yolo-pose-link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
    )
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


def normalize_source_dataset_root(path_value: str) -> dict[str, Path]:
    dataset_root = resolve_path(path_value)
    canonical_root = dataset_root if dataset_root.name == "_train_canonical" else dataset_root / "_train_canonical"
    if dataset_root.name == "_train_canonical":
        dataset_root = dataset_root.parent
    return {
        "dataset_root": dataset_root,
        "canonical_root": canonical_root,
    }


def keypoints_are_entirely_visible(
    annotation: dict[str, Any],
    image: dict[str, Any],
    expected_keypoints: int,
) -> tuple[bool, str]:
    keypoints = annotation.get("keypoints")
    if not isinstance(keypoints, list):
        return False, "missing_keypoints"
    if len(keypoints) != expected_keypoints * 3:
        return False, "wrong_keypoint_count"

    width = image.get("width")
    height = image.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        return False, "invalid_image_size"

    for index in range(0, len(keypoints), 3):
        x_coord = keypoints[index]
        y_coord = keypoints[index + 1]
        visibility = keypoints[index + 2]
        if visibility != 2:
            return False, "not_visible"
        if not isinstance(x_coord, (int, float)) or not isinstance(y_coord, (int, float)):
            return False, "invalid_coordinate"
        if x_coord < 0 or x_coord >= width or y_coord < 0 or y_coord >= height:
            return False, "outside_image"

    return True, "accepted"


def copy_or_link_image(source: Path, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def load_coco_payload(annotation_path: Path) -> dict[str, Any]:
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
    return json.loads(annotation_path.read_text(encoding="utf-8"))


def aggregate_split(
    split: str,
    source_specs: list[dict[str, Path]],
    canonical_output_root: Path,
    copy_mode: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    output_images: list[dict[str, Any]] = []
    output_annotations: list[dict[str, Any]] = []
    source_entries: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    info: dict[str, Any] = {}
    licenses: list[dict[str, Any]] = []
    output_image_root = canonical_output_root / split_image_dir(split)
    next_image_id = 1
    next_annotation_id = 1

    for spec in source_specs:
        canonical_root = spec["canonical_root"]
        annotation_path = canonical_root / "annotations" / f"person_keypoints_{split}.json"
        image_root = canonical_root / split_image_dir(split)
        if not image_root.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_root}")

        payload = load_coco_payload(annotation_path)
        if not categories:
            categories = payload.get("categories", [])
            info = payload.get("info", {})
            licenses = payload.get("licenses", [])
        expected_keypoints = 17
        if categories and isinstance(categories[0], dict):
            expected_keypoints = len(categories[0].get("keypoints", [])) or expected_keypoints

        annotations_by_image_id: dict[int, list[dict[str, Any]]] = {}
        for annotation in payload.get("annotations", []):
            annotations_by_image_id.setdefault(int(annotation["image_id"]), []).append(annotation)

        accepted_images = 0
        accepted_annotations = 0
        rejected_images = 0
        rejection_counts: dict[str, int] = {}

        for image in payload.get("images", []):
            image_id = int(image["id"])
            annotations = annotations_by_image_id.get(image_id, [])
            if not annotations:
                rejected_images += 1
                rejection_counts["missing_annotation"] = rejection_counts.get("missing_annotation", 0) + 1
                continue

            image_rejection_reasons: list[str] = []
            for annotation in annotations:
                is_valid, reason = keypoints_are_entirely_visible(annotation, image, expected_keypoints)
                if not is_valid:
                    image_rejection_reasons.append(reason)
            if image_rejection_reasons:
                rejected_images += 1
                for reason in sorted(set(image_rejection_reasons)):
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue

            source_image = image_root / image["file_name"]
            if not source_image.is_file():
                raise FileNotFoundError(f"Missing source image: {source_image}")
            copy_or_link_image(source_image, output_image_root / image["file_name"], copy_mode)

            output_images.append(
                {
                    "id": next_image_id,
                    "file_name": image["file_name"],
                    "width": image["width"],
                    "height": image["height"],
                }
            )
            for annotation in annotations:
                output_annotations.append(
                    {
                        **annotation,
                        "id": next_annotation_id,
                        "image_id": next_image_id,
                        "num_keypoints": expected_keypoints,
                    }
                )
                next_annotation_id += 1
                accepted_annotations += 1
            next_image_id += 1
            accepted_images += 1

        source_entries.append(
            {
                "dataset_root": spec["dataset_root"].as_posix(),
                "canonical_root": canonical_root.as_posix(),
                "source_images": len(payload.get("images", [])),
                "source_annotations": len(payload.get("annotations", [])),
                "accepted_images": accepted_images,
                "accepted_annotations": accepted_annotations,
                "rejected_images": rejected_images,
                "rejection_reasons": dict(sorted(rejection_counts.items())),
            }
        )

    coco_payload = {
        "info": info,
        "licenses": licenses,
        "images": output_images,
        "annotations": output_annotations,
        "categories": categories,
    }
    split_summary = {
        "output_images_total": len(output_images),
        "output_annotations_total": len(output_annotations),
    }
    return coco_payload, split_summary, source_entries


def build_canonical_dataset(
    source_specs: list[dict[str, Path]],
    canonical_output_root: Path,
    copy_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate_summaries: dict[str, Any] = {}
    source_summaries_by_root = {
        spec["dataset_root"].as_posix(): {
            "dataset_root": spec["dataset_root"].as_posix(),
            "canonical_root": spec["canonical_root"].as_posix(),
            "splits": {},
        }
        for spec in source_specs
    }

    annotations_root = canonical_output_root / "annotations"
    annotations_root.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        payload, split_summary, source_entries = aggregate_split(split, source_specs, canonical_output_root, copy_mode)
        (annotations_root / f"person_keypoints_{split}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        aggregate_summaries[split] = split_summary
        for entry in source_entries:
            source_summaries_by_root[entry["dataset_root"]]["splits"][split] = {
                key: value for key, value in entry.items() if key not in {"dataset_root", "canonical_root"}
            }

    ordered_source_summaries = [source_summaries_by_root[spec["dataset_root"].as_posix()] for spec in source_specs]
    return aggregate_summaries, ordered_source_summaries


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
    output_dataset_root: Path,
    vitpose_work_dir: Path,
    yolo_pose_link_mode: str,
) -> dict[str, Any]:
    vitpose_root = output_dataset_root / "_VitPosePP"
    yolo_detection_root = output_dataset_root / "_Yolo26x_detection"
    yolo_pose_root = output_dataset_root / "_Yolo26x_pose"

    run_subprocess(
        [
            sys.executable,
            "script/prepare_vitposepp_dataset.py",
            "--canonical-root",
            canonical_root.as_posix(),
            "--output-root",
            vitpose_root.as_posix(),
            "--work-dir",
            vitpose_work_dir.as_posix(),
            "--overwrite",
        ],
        project_root,
    )
    run_subprocess(
        [
            sys.executable,
            "script/yolo_training/prepare_yolo_detection_dataset.py",
            "--dataset-root",
            canonical_root.as_posix(),
            "--output-root",
            yolo_detection_root.as_posix(),
            "--overwrite",
        ],
        project_root,
    )
    ensure_detection_image_symlinks(canonical_root, yolo_detection_root)
    run_subprocess(
        [
            sys.executable,
            "script/yolo_training/prepare_yolo_pose_dataset.py",
            "--dataset-root",
            canonical_root.as_posix(),
            "--output-root",
            yolo_pose_root.as_posix(),
            "--link-mode",
            yolo_pose_link_mode,
            "--overwrite",
        ],
        project_root,
    )

    return {
        "vitposepp": load_report(vitpose_root / "preparation_report.json"),
        "yolo26x_detection": load_report(yolo_detection_root / "preparation_report.json"),
        "yolo26x_pose": load_report(yolo_pose_root / "preparation_report.json"),
    }


def build_manifest(
    output_dataset_root: Path,
    canonical_root: Path,
    copy_mode: str,
    source_summaries: list[dict[str, Any]],
    canonical_summaries: dict[str, Any],
    exports: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_root": output_dataset_root.as_posix(),
        "canonical_root": canonical_root.as_posix(),
        "copy_mode": copy_mode,
        "filter": {
            "required_keypoints": 17,
            "required_visibility": 2,
            "coordinate_rule": "0 <= x < image.width and 0 <= y < image.height",
        },
        "source_datasets": source_summaries,
        "canonical_splits": {
            split: {
                "images": canonical_summaries[split]["output_images_total"],
                "annotations": canonical_summaries[split]["output_annotations_total"],
            }
            for split in SPLITS
        },
        "exports": exports,
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dataset_root = resolve_path(args.output_dataset_root)
    canonical_output_root = output_dataset_root / "_train_canonical"
    vitpose_work_dir = resolve_path(args.vitpose_work_dir)

    source_specs = [normalize_source_dataset_root(value) for value in args.source_dataset_roots]
    for spec in source_specs:
        if not spec["canonical_root"].is_dir():
            raise FileNotFoundError(f"Missing canonical source dataset: {spec['canonical_root']}")

    if output_dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset already exists, pass --overwrite: {output_dataset_root}")
        shutil.rmtree(output_dataset_root)
    output_dataset_root.mkdir(parents=True, exist_ok=True)

    canonical_summaries, source_summaries = build_canonical_dataset(
        source_specs=source_specs,
        canonical_output_root=canonical_output_root,
        copy_mode=args.copy_mode,
    )

    exports = {}
    if not args.skip_model_exports:
        exports = export_model_datasets(
            project_root=project_root,
            canonical_root=canonical_output_root,
            output_dataset_root=output_dataset_root,
            vitpose_work_dir=vitpose_work_dir,
            yolo_pose_link_mode=args.yolo_pose_link_mode,
        )

    report = {
        "output_dataset_root": output_dataset_root.as_posix(),
        "canonical_root": canonical_output_root.as_posix(),
        "copy_mode": args.copy_mode,
        "filter": {
            "required_keypoints": 17,
            "required_visibility": 2,
            "coordinate_rule": "0 <= x < image.width and 0 <= y < image.height",
        },
        "source_datasets": source_summaries,
        "canonical_splits": canonical_summaries,
        "exports": exports,
    }

    reports_root = canonical_output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    report_path = reports_root / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    manifest = build_manifest(
        output_dataset_root=output_dataset_root,
        canonical_root=canonical_output_root,
        copy_mode=args.copy_mode,
        source_summaries=source_summaries,
        canonical_summaries=canonical_summaries,
        exports=exports,
    )
    (output_dataset_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (canonical_output_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")
    print(f"Manifest: {output_dataset_root / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
