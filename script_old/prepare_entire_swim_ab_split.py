from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DATASET_ROOT = "data/intermediate/Side_above_water_EntireSwim"
DEFAULT_OUTPUT_DATASET_ROOT_A = "data/intermediate/Side_above_water_EntireSwim_A"
DEFAULT_OUTPUT_DATASET_ROOT_B = "data/intermediate/Side_above_water_EntireSwim_B"
DEFAULT_PROB_A = 0.3
DEFAULT_SEED = 20260604
DEFAULT_VITPOSE_WORK_DIR_A = "runs/vitposepp_side_above_water_entireswim_a"
DEFAULT_VITPOSE_WORK_DIR_B = "runs/vitposepp_side_above_water_entireswim_b"
SPLITS = ("train", "val", "test")
REPORT_NAME = "ab_split_report.json"
MANIFEST_NAME = "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split an EntireSwim canonical dataset into two complementary datasets A and B. "
            "Each element is assigned to A with probability prob_a, otherwise to B. "
            "Both outputs recreate the canonical dataset plus VitPose++, YOLO26x detection, and YOLO26x pose exports."
        )
    )
    parser.add_argument("--source-dataset-root", default=DEFAULT_SOURCE_DATASET_ROOT)
    parser.add_argument("--output-dataset-root-a", default=DEFAULT_OUTPUT_DATASET_ROOT_A)
    parser.add_argument("--output-dataset-root-b", default=DEFAULT_OUTPUT_DATASET_ROOT_B)
    parser.add_argument("--prob-a", type=float, default=DEFAULT_PROB_A)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default="symlink",
        help="How to materialize canonical images in datasets A and B.",
    )
    parser.add_argument("--vitpose-work-dir-a", default=DEFAULT_VITPOSE_WORK_DIR_A)
    parser.add_argument("--vitpose-work-dir-b", default=DEFAULT_VITPOSE_WORK_DIR_B)
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


def normalize_dataset_root(path_value: str) -> dict[str, Path]:
    dataset_root = resolve_path(path_value)
    canonical_root = dataset_root if dataset_root.name == "_train_canonical" else dataset_root / "_train_canonical"
    if dataset_root.name == "_train_canonical":
        dataset_root = dataset_root.parent
    return {
        "dataset_root": dataset_root,
        "canonical_root": canonical_root,
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def copy_or_link_image(source: Path, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def annotations_by_image_id(payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for annotation in payload.get("annotations", []):
        grouped.setdefault(int(annotation["image_id"]), []).append(annotation)
    return grouped


def build_split_payload(
    split: str,
    source_canonical_root: Path,
    output_canonical_root: Path,
    dataset_key: str,
    prob_a: float,
    rng: random.Random,
    copy_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_annotation_path = source_canonical_root / "annotations" / f"person_keypoints_{split}.json"
    source_image_root = source_canonical_root / split_image_dir(split)
    if not source_image_root.is_dir():
        raise FileNotFoundError(f"Missing image directory: {source_image_root}")

    payload = load_json(source_annotation_path)
    grouped_annotations = annotations_by_image_id(payload)
    output_images: list[dict[str, Any]] = []
    output_annotations: list[dict[str, Any]] = []
    output_image_root = output_canonical_root / split_image_dir(split)
    next_image_id = 1
    next_annotation_id = 1
    assigned_to_a = 0
    assigned_to_b = 0

    for image in payload.get("images", []):
        image_id = int(image["id"])
        annotations = grouped_annotations.get(image_id, [])
        if not annotations:
            continue
        goes_to_a = rng.random() < prob_a
        destination_key = "A" if goes_to_a else "B"
        if destination_key != dataset_key:
            if goes_to_a:
                assigned_to_a += 1
            else:
                assigned_to_b += 1
            continue

        source_image = source_image_root / image["file_name"]
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
                }
            )
            next_annotation_id += 1
        next_image_id += 1

        if goes_to_a:
            assigned_to_a += 1
        else:
            assigned_to_b += 1

    split_payload = {
        "info": payload.get("info", {}),
        "licenses": payload.get("licenses", []),
        "images": output_images,
        "annotations": output_annotations,
        "categories": payload.get("categories", []),
    }
    split_summary = {
        "source_images": len(payload.get("images", [])),
        "source_annotations": len(payload.get("annotations", [])),
        "selected_images": len(output_images),
        "selected_annotations": len(output_annotations),
        "assigned_to_a": assigned_to_a,
        "assigned_to_b": assigned_to_b,
    }
    return split_payload, split_summary


def build_canonical_dataset(
    source_canonical_root: Path,
    output_dataset_root: Path,
    dataset_key: str,
    prob_a: float,
    seed: int,
    copy_mode: str,
) -> dict[str, Any]:
    canonical_output_root = output_dataset_root / "_train_canonical"
    annotations_root = canonical_output_root / "annotations"
    annotations_root.mkdir(parents=True, exist_ok=True)

    split_summaries: dict[str, Any] = {}
    rng = random.Random(seed)

    for split in SPLITS:
        split_payload, split_summary = build_split_payload(
            split=split,
            source_canonical_root=source_canonical_root,
            output_canonical_root=canonical_output_root,
            dataset_key=dataset_key,
            prob_a=prob_a,
            rng=rng,
            copy_mode=copy_mode,
        )
        (annotations_root / f"person_keypoints_{split}.json").write_text(
            json.dumps(split_payload, indent=2),
            encoding="utf-8",
        )
        split_summaries[split] = split_summary

    return {
        "dataset_root": output_dataset_root.as_posix(),
        "canonical_root": canonical_output_root.as_posix(),
        "dataset_key": dataset_key,
        "copy_mode": copy_mode,
        "prob_a": prob_a,
        "seed": seed,
        "canonical_splits": {
            split: {
                "images": split_summaries[split]["selected_images"],
                "annotations": split_summaries[split]["selected_annotations"],
            }
            for split in SPLITS
        },
        "source_split_assignments": split_summaries,
    }


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


def write_report(
    source_dataset_root: Path,
    source_manifest: dict[str, Any] | None,
    output_dataset_root: Path,
    summary: dict[str, Any],
    exports: dict[str, Any],
) -> None:
    canonical_root = output_dataset_root / "_train_canonical"
    reports_root = canonical_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "source_dataset_root": source_dataset_root.as_posix(),
        "source_manifest": source_manifest,
        **summary,
        "exports": exports,
    }
    (reports_root / REPORT_NAME).write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    (canonical_root / MANIFEST_NAME).write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    root_manifest = {
        "dataset_root": output_dataset_root.as_posix(),
        "canonical_root": canonical_root.as_posix(),
        "source_dataset_root": source_dataset_root.as_posix(),
        "source_manifest": source_manifest,
        "dataset_key": summary["dataset_key"],
        "prob_a": summary["prob_a"],
        "seed": summary["seed"],
        "copy_mode": summary["copy_mode"],
        "canonical_splits": summary["canonical_splits"],
        "exports": exports,
    }
    (output_dataset_root / MANIFEST_NAME).write_text(json.dumps(root_manifest, indent=2), encoding="utf-8")


def rebuild_dataset(
    project_root: Path,
    source_dataset_root: Path,
    source_manifest: dict[str, Any] | None,
    output_dataset_root: Path,
    dataset_key: str,
    prob_a: float,
    seed: int,
    copy_mode: str,
    vitpose_work_dir: Path,
    yolo_pose_link_mode: str,
    skip_model_exports: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if output_dataset_root.exists() or output_dataset_root.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Output already exists, pass --overwrite: {output_dataset_root}")
        ensure_removed(output_dataset_root)
    output_dataset_root.mkdir(parents=True, exist_ok=True)

    summary = build_canonical_dataset(
        source_canonical_root=source_dataset_root / "_train_canonical",
        output_dataset_root=output_dataset_root,
        dataset_key=dataset_key,
        prob_a=prob_a,
        seed=seed,
        copy_mode=copy_mode,
    )
    exports = {} if skip_model_exports else export_model_datasets(
        project_root=project_root,
        canonical_root=output_dataset_root / "_train_canonical",
        output_dataset_root=output_dataset_root,
        vitpose_work_dir=vitpose_work_dir,
        yolo_pose_link_mode=yolo_pose_link_mode,
    )
    write_report(source_dataset_root, source_manifest, output_dataset_root, summary, exports)
    return {
        "dataset_root": output_dataset_root.as_posix(),
        "canonical_root": (output_dataset_root / "_train_canonical").as_posix(),
        "dataset_key": dataset_key,
        "canonical_splits": summary["canonical_splits"],
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.prob_a <= 1.0:
        raise ValueError("--prob-a must be between 0.0 and 1.0")

    project_root = Path.cwd().resolve()
    source_spec = normalize_dataset_root(args.source_dataset_root)
    source_dataset_root = source_spec["dataset_root"]
    source_canonical_root = source_spec["canonical_root"]
    if not source_canonical_root.is_dir():
        raise FileNotFoundError(f"Missing canonical dataset root: {source_canonical_root}")

    source_manifest = None
    source_manifest_path = source_dataset_root / MANIFEST_NAME
    if source_manifest_path.is_file():
        source_manifest = load_json(source_manifest_path)

    summary_a = rebuild_dataset(
        project_root=project_root,
        source_dataset_root=source_dataset_root,
        source_manifest=source_manifest,
        output_dataset_root=resolve_path(args.output_dataset_root_a),
        dataset_key="A",
        prob_a=args.prob_a,
        seed=args.seed,
        copy_mode=args.copy_mode,
        vitpose_work_dir=resolve_path(args.vitpose_work_dir_a),
        yolo_pose_link_mode=args.yolo_pose_link_mode,
        skip_model_exports=args.skip_model_exports,
        overwrite=args.overwrite,
    )
    summary_b = rebuild_dataset(
        project_root=project_root,
        source_dataset_root=source_dataset_root,
        source_manifest=source_manifest,
        output_dataset_root=resolve_path(args.output_dataset_root_b),
        dataset_key="B",
        prob_a=args.prob_a,
        seed=args.seed,
        copy_mode=args.copy_mode,
        vitpose_work_dir=resolve_path(args.vitpose_work_dir_b),
        yolo_pose_link_mode=args.yolo_pose_link_mode,
        skip_model_exports=args.skip_model_exports,
        overwrite=args.overwrite,
    )

    result = {
        "source_dataset_root": source_dataset_root.as_posix(),
        "prob_a": args.prob_a,
        "seed": args.seed,
        "copy_mode": args.copy_mode,
        "dataset_a": summary_a,
        "dataset_b": summary_b,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
