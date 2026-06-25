from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path("/home/albertosco/HPE")
DEFAULT_SOURCE_DATASET_ROOT = (
    "data/intermediate/Side_above_water_VideoTest2/_train_canonical"
)
DEFAULT_OUTPUT_ROOT = (
    "data/intermediate/Side_above_water_VideoTest2/_train_canonical/"
    "subsets/random50_seed20260603"
)
DEFAULT_BASE_CONFIG = (
    "data/intermediate/Side_above_water/_train_canonical/generated_configs/"
    "swimxyz_vitposepp_huge.py"
)
DEFAULT_CHECKPOINT = "runs/vitposepp_side_above_water_grid_winner_resume/best_AP_epoch_24.pth"
DEFAULT_WORK_DIR = "runs/vitposepp_videotest2_random50_eval_best_AP_epoch_24"
DEFAULT_SEED = 20260603
DEFAULT_NUM_FRAMES = 50
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a deterministic 50-frame subset from the canonical "
            "Side_above_water_VideoTest2 Train/Val/Test splits and prepare a "
            "VitPose++ eval-ready dataset plus config."
        )
    )
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--source-dataset-root", default=DEFAULT_SOURCE_DATASET_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def load_samples(source_dataset_root: Path) -> list[dict]:
    samples: list[dict] = []
    for split in SPLITS:
        annotation_path = (
            source_dataset_root / "annotations" / f"person_keypoints_{split}.json"
        )
        image_root = source_dataset_root / f"{split}2017"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
        if not image_root.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_root}")

        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_by_id = {image["id"]: image for image in payload.get("images", [])}
        annotation_by_image_id = {
            annotation["image_id"]: annotation for annotation in payload.get("annotations", [])
        }
        for image_id, image in image_by_id.items():
            annotation = annotation_by_image_id.get(image_id)
            if annotation is None:
                raise RuntimeError(
                    f"Missing annotation for image_id={image_id} in {annotation_path}"
                )
            source_path = image_root / image["file_name"]
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing source image: {source_path}")
            samples.append(
                {
                    "source_split": split,
                    "image_id": image_id,
                    "annotation_id": annotation["id"],
                    "image": image,
                    "annotation": annotation,
                    "source_path": source_path,
                }
            )
    return samples


def copy_selected_images(selected_samples: list[dict], output_image_root: Path) -> None:
    output_image_root.mkdir(parents=True, exist_ok=True)
    for sample in selected_samples:
        destination = output_image_root / sample["image"]["file_name"]
        shutil.copy2(sample["source_path"], destination)
        sample["output_path"] = destination


def build_coco_test_json(selected_samples: list[dict]) -> dict:
    images = []
    annotations = []
    for new_id, sample in enumerate(selected_samples, start=1):
        image = sample["image"]
        annotation = sample["annotation"]
        images.append(
            {
                "id": new_id,
                "file_name": image["file_name"],
                "width": image["width"],
                "height": image["height"],
            }
        )
        annotations.append(
            {
                "id": new_id,
                "image_id": new_id,
                "category_id": annotation.get("category_id", 1),
                "bbox": annotation["bbox"],
                "area": annotation["area"],
                "iscrowd": annotation.get("iscrowd", 0),
                "num_keypoints": annotation["num_keypoints"],
                "keypoints": annotation["keypoints"],
            }
        )
    return {
        "info": {
            "description": "Random 50-frame subset from Side_above_water_VideoTest2",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": [
                    "nose",
                    "left_eye",
                    "right_eye",
                    "left_ear",
                    "right_ear",
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                    "left_wrist",
                    "right_wrist",
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                ],
                "skeleton": [
                    [16, 14],
                    [14, 12],
                    [17, 15],
                    [15, 13],
                    [12, 13],
                    [6, 12],
                    [7, 13],
                    [6, 7],
                    [6, 8],
                    [7, 9],
                    [8, 10],
                    [9, 11],
                    [2, 3],
                    [1, 2],
                    [1, 3],
                    [2, 4],
                    [3, 5],
                    [4, 6],
                    [5, 7],
                ],
            }
        ],
    }


def write_config(
    base_config_path: Path,
    output_config_path: Path,
    source_dataset_root: Path,
    output_root: Path,
    checkpoint_path: Path,
    work_dir: Path,
) -> None:
    text = base_config_path.read_text(encoding="utf-8")
    replacements = {
        r"^data_root = .*$": f"data_root = '{source_dataset_root.as_posix()}'",
        r"^load_from = .*$": f"load_from = '{checkpoint_path.as_posix()}'",
        r"^work_dir = .*$": f"work_dir = '{work_dir.as_posix()}'",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

    text += (
        "\nsubset_metadata = dict(\n"
        f"    source_dataset_root='{source_dataset_root.as_posix()}',\n"
        f"    output_root='{output_root.as_posix()}',\n"
        f"    checkpoint='{checkpoint_path.as_posix()}',\n"
        "    subset_name='Side_above_water_VideoTest2 random 50',\n"
        ")\n"
    )
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    source_dataset_root = resolve_path(project_root, args.source_dataset_root)
    output_root = resolve_path(project_root, args.output_root)
    base_config = resolve_path(project_root, args.base_config)
    checkpoint_path = resolve_path(project_root, args.checkpoint)
    work_dir = resolve_path(project_root, args.work_dir)

    if not source_dataset_root.is_dir():
        raise FileNotFoundError(f"Missing source dataset root: {source_dataset_root}")
    if not base_config.is_file():
        raise FileNotFoundError(f"Missing base config: {base_config}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists, pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    all_samples = load_samples(source_dataset_root)
    if args.num_frames > len(all_samples):
        raise ValueError(
            f"Requested {args.num_frames} frames, but only {len(all_samples)} are available."
        )

    selected_samples = rng.sample(all_samples, args.num_frames)
    selected_samples.sort(
        key=lambda sample: (
            sample["source_split"],
            sample["image"]["file_name"],
        )
    )

    test_image_root = output_root / "test2017"
    annotations_root = output_root / "annotations"
    config_root = output_root / "generated_configs"
    annotations_root.mkdir(parents=True, exist_ok=True)

    copy_selected_images(selected_samples, test_image_root)
    coco_test = build_coco_test_json(selected_samples)
    (annotations_root / "person_keypoints_test.json").write_text(
        json.dumps(coco_test, indent=2),
        encoding="utf-8",
    )
    write_config(
        base_config_path=base_config,
        output_config_path=config_root / "swimxyz_vitposepp_videotest2_random50_eval.py",
        source_dataset_root=output_root,
        output_root=output_root,
        checkpoint_path=checkpoint_path,
        work_dir=work_dir,
    )

    split_counts: dict[str, int] = {split: 0 for split in SPLITS}
    for sample in selected_samples:
        split_counts[sample["source_split"]] += 1

    report = {
        "source_dataset_root": source_dataset_root.as_posix(),
        "output_root": output_root.as_posix(),
        "seed": args.seed,
        "num_frames": args.num_frames,
        "selected_split_counts": split_counts,
        "selected_frames": [
            {
                "source_split": sample["source_split"],
                "source_file": sample["image"]["file_name"],
                "image_id": sample["image_id"],
                "annotation_id": sample["annotation_id"],
            }
            for sample in selected_samples
        ],
        "config": (config_root / "swimxyz_vitposepp_videotest2_random50_eval.py").as_posix(),
        "test_annotation": (annotations_root / "person_keypoints_test.json").as_posix(),
    }
    (output_root / "subset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
