from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

# This script is the normalization step that reorganizes original SwimXYZ
# folders into the intermediate layout expected by the training pipeline.
# It copies videos and label files into a predictable structure and writes a
# manifest so downstream dataset builders can discover entries without relying
# on ad hoc path conventions.


DEFAULT_PROJECT_ROOT = Path("/home/albertosco/HPE")
DEFAULT_SOURCE_ROOT = "data/input/subset_xyz"
DEFAULT_OUTPUT_ROOT = "data/intermediate"
DEFAULT_IMAGE_WIDTH = 1920
DEFAULT_IMAGE_HEIGHT = 1080

EXPECTED_TRAILING_MISSING_HEADERS = [
    "LEar.x",
    "LEar.y",
    "LEar.z",
    "LBigToe.x",
    "LBigToe.y",
    "LBigToe.z",
    "LSmallToe.x",
    "LSmallToe.y",
    "LSmallToe.z",
    "LHeel.x",
    "LHeel.y",
    "LHeel.z",
    "RBigToe.x",
    "RBigToe.y",
    "RBigToe.z",
    "RSmallToe.x",
    "RSmallToe.y",
    "RSmallToe.z",
    "RHeel.x",
    "RHeel.y",
    "RHeel.z",
]

# SwimXYZ "COCO" files keep the BODY25 header but store only these 18 values.
SWIMXYZ_COCO18_ORDER = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "REar",
    "LEye",
    "LEar",
]

BODY25_TO_COCO = [
    ("Nose", "nose"),
    ("LEye", "left_eye"),
    ("REye", "right_eye"),
    ("LEar", "left_ear"),
    ("REar", "right_ear"),
    ("LShoulder", "left_shoulder"),
    ("RShoulder", "right_shoulder"),
    ("LElbow", "left_elbow"),
    ("RElbow", "right_elbow"),
    ("LWrist", "left_wrist"),
    ("RWrist", "right_wrist"),
    ("LHip", "left_hip"),
    ("RHip", "right_hip"),
    ("LKnee", "left_knee"),
    ("RKnee", "right_knee"),
    ("LAnkle", "left_ankle"),
    ("RAnkle", "right_ankle"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reorganize one or more SwimXYZ datasets from data/input/subset_xyz into "
            "data/intermediate/<dataset>/_converted and optionally export COCO-style "
            "JSON sidecars for 2D COCO labels."
        )
    )
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
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
        help="Keep label y coordinates unchanged in generated JSON sidecars.",
    )
    parser.add_argument(
        "--skip-json",
        action="store_true",
        help="Copy/reorganize files only, without generating *_coco.json sidecars.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def iter_source_datasets(source_root: Path) -> list[Path]:
    manifest_path = source_root / "manifest.json"
    if manifest_path.is_file():
        return [source_root]

    datasets = sorted(
        child for child in source_root.iterdir() if (child / "manifest.json").is_file()
    )
    if datasets:
        return datasets

    raise FileNotFoundError(
        "No source dataset manifest found in "
        f"{source_root}. Expected either {manifest_path} or child dataset directories."
    )


def build_output_root(intermediate_root: Path, dataset_root: Path) -> Path:
    return intermediate_root / dataset_root.name / "_converted"


def parse_decimal(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value.replace(",", "."))


def to_image_y(y: float, image_height: int, flip_y: bool) -> float:
    return float(image_height) - float(y) if flip_y else float(y)


def keypoint_row_from_values(keypoint_order: list[str], values: list[str]) -> dict[str, str]:
    row = {}
    for index, keypoint_name in enumerate(keypoint_order):
        offset = index * 3
        row[f"{keypoint_name}.x"] = values[offset]
        row[f"{keypoint_name}.y"] = values[offset + 1]
        row[f"{keypoint_name}.z"] = values[offset + 2]
    return row


def read_label_rows(labels_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = labels_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Label file is empty: {labels_path}")

    header = [column.strip() for column in lines[0].split(";") if column.strip()]
    rows: list[dict[str, str]] = []
    for raw_line in lines[1:]:
        if not raw_line.strip():
            continue
        values = [value.strip() for value in raw_line.split(";")]
        if values and values[-1] == "":
            values = values[:-1]

        if len(values) == len(SWIMXYZ_COCO18_ORDER) * 3 and len(header) > len(values):
            rows.append(keypoint_row_from_values(SWIMXYZ_COCO18_ORDER, values))
            continue

        if len(values) < len(header):
            missing_headers = header[len(values) :]
            if missing_headers != EXPECTED_TRAILING_MISSING_HEADERS[: len(missing_headers)]:
                raise ValueError(
                    "Unexpected truncated row in "
                    f"{labels_path}: missing headers {missing_headers}"
                )
            values.extend([""] * (len(header) - len(values)))
        elif len(values) > len(header):
            raise ValueError(
                f"Invalid row length in {labels_path}: expected {len(header)}, got {len(values)}"
            )

        rows.append(dict(zip(header, values)))
    return header, rows


def build_label_output_path(output_root: Path, video_stem: str, label_filename: str) -> tuple[str, str, Path]:
    stem_without_suffix, representation, label_kind = label_filename.rsplit("__", 2)
    if stem_without_suffix != video_stem:
        raise ValueError(
            f"Label {label_filename} does not match video stem {video_stem}"
        )

    normalized_label_kind = label_kind.removesuffix(".txt")
    output_dir = output_root / video_stem / representation
    output_file = output_dir / f"{normalized_label_kind}_{representation}.txt"
    return representation, normalized_label_kind, output_file


def build_coco_json(
    labels_path: Path,
    image_width: int,
    image_height: int,
    flip_y: bool,
) -> dict:
    _, rows = read_label_rows(labels_path)
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
    }

    for index, row in enumerate(rows, start=1):
        image_name = f"{labels_path.stem}_{index - 1}.png"
        coco_data["images"].append(
            {
                "id": index,
                "file_name": image_name,
                "width": image_width,
                "height": image_height,
            }
        )

        keypoints: list[float] = []
        x_coords: list[float] = []
        y_coords: list[float] = []
        for body25_name, _ in BODY25_TO_COCO:
            x = parse_decimal(row.get(f"{body25_name}.x"))
            y = parse_decimal(row.get(f"{body25_name}.y"))
            score = parse_decimal(row.get(f"{body25_name}.z"))
            if (
                x is None
                or y is None
                or score is None
                or math.isnan(x)
                or math.isnan(y)
                or math.isnan(score)
                or score <= 0
            ):
                keypoints.extend([0.0, 0.0, 0.0])
                continue

            image_y = to_image_y(float(y), image_height, flip_y)
            image_y = min(max(image_y, 0.0), float(image_height - 1))
            keypoints.extend([round(x, 2), round(image_y, 2), 2.0])
            x_coords.append(float(x))
            y_coords.append(image_y)

        if x_coords and y_coords:
            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)
            width = max_x - min_x
            height = max_y - min_y
            bbox = [
                round(min_x - width * 0.05, 2),
                round(min_y - height * 0.05, 2),
                round(width * 1.1, 2),
                round(height * 1.1, 2),
            ]
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]

        coco_data["annotations"].append(
            {
                "id": index,
                "image_id": index,
                "category_id": 1,
                "keypoints": keypoints,
                "bbox": bbox,
                "area": round(bbox[2] * bbox[3], 2),
                "iscrowd": 0,
                "num_keypoints": sum(1 for visibility in keypoints[2::3] if visibility > 0),
            }
        )

    return coco_data


def convert_dataset(args: argparse.Namespace, source_root: Path, output_root: Path) -> tuple[int, int, int]:
    # The output manifest is the contract used later by the VitPose++ dataset
    # preparer: each video keeps explicit references to the label variants that
    # were copied under the intermediate tree.
    source_manifest = source_root / "manifest.json"
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not items:
        raise RuntimeError(f"No dataset items found in manifest: {source_manifest}")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    generated_manifest: dict[str, list[dict[str, object]]] = {"items": []}
    copied_labels = 0
    generated_json = 0

    for item in items:
        video_file = item["video_file"]
        video_stem = Path(video_file).stem
        source_video_path = source_root / video_file
        if not source_video_path.is_file():
            raise FileNotFoundError(f"Missing source video file: {source_video_path}")

        output_video_path = output_root / video_file
        shutil.copy2(source_video_path, output_video_path)

        manifest_item: dict[str, object] = {
            "video_file": video_file,
            "video_path": video_file,
            "labels": [],
        }

        for label_filename in item.get("label_files", []):
            source_label_path = source_root / label_filename
            if not source_label_path.is_file():
                raise FileNotFoundError(f"Missing source label file: {source_label_path}")

            representation, label_kind, output_label_path = build_label_output_path(
                output_root=output_root,
                video_stem=video_stem,
                label_filename=label_filename,
            )
            output_label_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_label_path, output_label_path)
            copied_labels += 1

            label_record: dict[str, str] = {
                "representation": representation,
                "label_kind": label_kind,
                "source_file": label_filename,
                "output_relative_path": str(output_label_path.relative_to(output_root)),
            }

            if not args.skip_json and representation == "COCO" and label_kind.startswith("2D_"):
                json_path = output_label_path.with_name(f"{output_label_path.stem}_coco.json")
                json_path.write_text(
                    json.dumps(
                        build_coco_json(
                            labels_path=output_label_path,
                            image_width=args.image_width,
                            image_height=args.image_height,
                            flip_y=args.flip_y,
                        ),
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                label_record["json_relative_path"] = str(json_path.relative_to(output_root))
                generated_json += 1

            manifest_item["labels"].append(label_record)

        generated_manifest["items"].append(manifest_item)

    (output_root / "manifest.json").write_text(
        json.dumps(generated_manifest, indent=2),
        encoding="utf-8",
    )

    return len(items), copied_labels, generated_json


def main() -> None:
    args = parse_args()
    project_root = resolve_path(Path.cwd(), args.project_root)
    source_root = resolve_path(project_root, args.source_root)
    intermediate_root = resolve_path(project_root, args.output_root)

    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing source dataset directory: {source_root}")

    source_datasets = iter_source_datasets(source_root)
    converted_roots = []

    for dataset_root in source_datasets:
        output_root = build_output_root(intermediate_root, dataset_root)
        videos_count, copied_labels, generated_json = convert_dataset(
            args=args,
            source_root=dataset_root,
            output_root=output_root,
        )
        converted_roots.append(output_root)
        print(f"Converted SwimXYZ dataset: {dataset_root.name}")
        print(f"Source root: {dataset_root}")
        print(f"Output root: {output_root}")
        print(f"Videos copied: {videos_count}")
        print(f"Label files copied: {copied_labels}")
        print(f"COCO JSON files generated: {generated_json}")

    if len(converted_roots) > 1:
        print(f"Datasets converted: {len(converted_roots)}")


if __name__ == "__main__":
    main()
