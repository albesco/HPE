from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat

import cv2

# The CLI for this script has been restored to the stable standard COCO17
# pipeline. The experimental multi-dataset implementation below is kept in the
# file history context, but is bypassed when this script is executed directly.
from prepare_swimxyz_vitposepp import main as _prepare_main


if __name__ == "__main__":
    _prepare_main()
    raise SystemExit


DEFAULT_PROJECT_ROOT = Path("/home/albertosco/HPE")
DEFAULT_DATASET_ROOT = "data/intermediate"
DEFAULT_OUTPUT_ROOT = ""
DEFAULT_BASE_CONFIG = (
    "src/vitpose_base/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/"
    "vitPose+_huge_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py"
)
DEFAULT_PRETRAINED_CHECKPOINT = "models/pose/wholebody.pth"
DEFAULT_WORK_DIR = "runs/vitposepp_subset_xyz"
DEFAULT_FORMATS = ("COCO", "body25", "base")
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
BODY25_KEYPOINT_ORDER = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
]
BASE_KEYPOINT_ORDER = [
    "Pelvis",
    "Head",
    "HeadNub",
    "L Calf",
    "L Clavicle",
    "L Finger0",
    "L Finger1",
    "L Finger2",
    "L Finger3",
    "L Finger4",
    "L Foot",
    "L Forearm",
    "L Hand",
    "L Thigh",
    "L Heel",
    "L Toe0",
    "L Toe1",
    "L Toe2",
    "L Toe3",
    "L Toe4",
    "L UpperArm",
    "Neck",
    "R Calf",
    "R Clavicle",
    "R Finger0",
    "R Finger1",
    "R Finger2",
    "R Finger3",
    "R Finger4",
    "R Foot",
    "R Forearm",
    "R Hand",
    "R Thigh",
    "R Heel",
    "R Toe0",
    "R Toe1",
    "R Toe2",
    "R Toe3",
    "R Toe4",
    "R UpperArm",
    "Spine1",
    "Spine2",
    "Spine",
    "Ear_L",
    "Ear_R",
    "Eye_L",
    "Eye_R",
    "Nose",
]
COCO17_SOURCE_TO_OUTPUT = [
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
BODY25_TO_BASE_SHARED = {
    "Nose": "Nose",
    "Neck": "Neck",
    "RShoulder": "R UpperArm",
    "RElbow": "R Forearm",
    "RWrist": "R Hand",
    "LShoulder": "L UpperArm",
    "LElbow": "L Forearm",
    "LWrist": "L Hand",
    "MidHip": "Pelvis",
    "RHip": "R Thigh",
    "RKnee": "R Calf",
    "RAnkle": "R Foot",
    "LHip": "L Thigh",
    "LKnee": "L Calf",
    "LAnkle": "L Foot",
    "REye": "Eye_R",
    "LEye": "Eye_L",
    "REar": "Ear_R",
    "LEar": "Ear_L",
}
BODY25_SKELETON = [
    ("Nose", "Neck"),
    ("Neck", "RShoulder"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("Neck", "LShoulder"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
    ("Neck", "MidHip"),
    ("MidHip", "RHip"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
    ("RAnkle", "RHeel"),
    ("RAnkle", "RBigToe"),
    ("RBigToe", "RSmallToe"),
    ("MidHip", "LHip"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    ("LAnkle", "LHeel"),
    ("LAnkle", "LBigToe"),
    ("LBigToe", "LSmallToe"),
    ("Nose", "REye"),
    ("REye", "REar"),
    ("Nose", "LEye"),
    ("LEye", "LEar"),
]
BASE_SKELETON = [
    ("Pelvis", "Spine"),
    ("Spine", "Spine1"),
    ("Spine1", "Spine2"),
    ("Spine2", "Neck"),
    ("Neck", "Head"),
    ("Head", "HeadNub"),
    ("Head", "Nose"),
    ("Nose", "Eye_L"),
    ("Nose", "Eye_R"),
    ("Eye_L", "Ear_L"),
    ("Eye_R", "Ear_R"),
    ("Neck", "L Clavicle"),
    ("L Clavicle", "L UpperArm"),
    ("L UpperArm", "L Forearm"),
    ("L Forearm", "L Hand"),
    ("L Hand", "L Finger0"),
    ("L Hand", "L Finger4"),
    ("Pelvis", "L Thigh"),
    ("L Thigh", "L Calf"),
    ("L Calf", "L Foot"),
    ("L Foot", "L Heel"),
    ("L Foot", "L Toe0"),
    ("L Toe0", "L Toe4"),
    ("Neck", "R Clavicle"),
    ("R Clavicle", "R UpperArm"),
    ("R UpperArm", "R Forearm"),
    ("R Forearm", "R Hand"),
    ("R Hand", "R Finger0"),
    ("R Hand", "R Finger4"),
    ("Pelvis", "R Thigh"),
    ("R Thigh", "R Calf"),
    ("R Calf", "R Foot"),
    ("R Foot", "R Heel"),
    ("R Foot", "R Toe0"),
    ("R Toe0", "R Toe4"),
]
COCO_SKELETON = [
    ("left_ankle", "left_knee"),
    ("left_knee", "left_hip"),
    ("right_ankle", "right_knee"),
    ("right_knee", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_eye", "right_eye"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),
]


@dataclass
class DatasetBundle:
    video_path: Path
    labels_by_format: dict[str, Path]


@dataclass
class FormatSpec:
    name: str
    label_format: str
    source_names: list[str]
    output_names: list[str]
    skeleton: list[tuple[str, str]]
    dataset_idx: int


@dataclass
class PoseSample:
    width: int
    height: int
    bbox: list[float]
    keypoints: list[float]
    num_keypoints: int
    area: float


@dataclass
class FrameRecord:
    source_name: str
    frame_index: int
    image_name: str
    cache_path: Path
    width: int
    height: int
    samples_by_format: dict[str, PoseSample] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a multi-dataset SwimXYZ dataset for VitPose++ training using "
            "COCO, body25 and base labels on shared frames."
        )
    )
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--label-dimension", default="2D")
    parser.add_argument("--label-reference", default="cam")
    parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help="Comma-separated label formats to include. Default: COCO,body25,base",
    )
    parser.add_argument("--main-format", default="COCO")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--pretrained-checkpoint", default=DEFAULT_PRETRAINED_CHECKPOINT)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=0)
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
    parser.add_argument("--coherence-tolerance", type=float, default=1e-4)
    parser.add_argument("--samples-per-gpu", type=int, default=32)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--total-epochs", type=int, default=60)
    parser.add_argument("--val-interval", type=int, default=5)
    return parser.parse_args()


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def dataset_name_from_converted_root(dataset_root: Path) -> str:
    if dataset_root.name == "_converted":
        return dataset_root.parent.name
    if dataset_root.name.endswith("_converted"):
        return dataset_root.name.removesuffix("_converted")
    return dataset_root.name


def iter_converted_dataset_roots(dataset_root: Path) -> list[Path]:
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.is_file():
        return [dataset_root]

    nested_roots = sorted(
        child / "_converted"
        for child in dataset_root.iterdir()
        if child.is_dir() and (child / "_converted" / "manifest.json").is_file()
    )
    if nested_roots:
        return nested_roots

    flat_roots = sorted(
        child for child in dataset_root.iterdir() if (child / "manifest.json").is_file()
    )
    if flat_roots:
        return flat_roots

    raise FileNotFoundError(
        "No converted dataset manifest found in "
        f"{dataset_root}. Expected either a manifest in place, child '_converted' "
        "directories, or legacy '*_converted' directories."
    )


def build_output_root(project_root: Path, dataset_root: Path, output_root_value: str) -> Path:
    if output_root_value:
        return resolve_path(project_root, output_root_value)
    dataset_name = dataset_name_from_converted_root(dataset_root)
    return (project_root / "data" / "intermediate" / dataset_name / "_train_vitposepp").resolve()


def parse_formats(raw_formats: str, main_format: str) -> list[str]:
    formats = [item.strip() for item in raw_formats.split(",") if item.strip()]
    if not formats:
        raise ValueError("At least one label format must be provided.")
    if main_format not in formats:
        raise ValueError(f"--main-format {main_format} is not present in --formats {formats}.")
    ordered = [main_format]
    ordered.extend(item for item in formats if item != main_format)
    return ordered


def build_label_filename(label_format: str, label_dimension: str, label_reference: str) -> str:
    return f"{label_dimension}_{label_reference}_{label_format}.txt"


def discover_dataset_bundles(
    dataset_root: Path,
    label_formats: list[str],
    label_dimension: str,
    label_reference: str,
    max_videos: int,
) -> list[DatasetBundle]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")

    label_filenames = {
        label_format: build_label_filename(label_format, label_dimension, label_reference)
        for label_format in label_formats
    }
    manifest_path = dataset_root / "manifest.json"
    bundles: list[DatasetBundle] = []

    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in payload.get("items", []):
            video_rel_path = item.get("video_path") or item.get("video_file")
            if not video_rel_path:
                continue

            labels_by_format: dict[str, Path] = {}
            for label_format in label_formats:
                matching_label = next(
                    (
                        label["output_relative_path"]
                        for label in item.get("labels", [])
                        if label.get("representation") == label_format
                        and label.get("label_kind") == f"{label_dimension}_{label_reference}"
                    ),
                    None,
                )
                if matching_label is None:
                    labels_by_format = {}
                    break
                labels_by_format[label_format] = (dataset_root / matching_label).resolve()

            if not labels_by_format:
                continue

            video_path = (dataset_root / video_rel_path).resolve()
            if video_path.is_file() and all(path.is_file() for path in labels_by_format.values()):
                bundles.append(DatasetBundle(video_path=video_path, labels_by_format=labels_by_format))
    else:
        for video_path in sorted(dataset_root.glob("*.webm")):
            labels_by_format = {}
            for label_format in label_formats:
                labels_path = dataset_root / video_path.stem / label_format / label_filenames[label_format]
                if not labels_path.is_file():
                    labels_by_format = {}
                    break
                labels_by_format[label_format] = labels_path.resolve()
            if labels_by_format:
                bundles.append(DatasetBundle(video_path=video_path.resolve(), labels_by_format=labels_by_format))

    if max_videos > 0:
        bundles = bundles[:max_videos]

    if not bundles:
        raise RuntimeError(
            "No dataset bundles discovered in "
            f"{dataset_root} for formats {label_formats}/{label_dimension}_{label_reference}."
        )
    return bundles


def keypoint_row_from_values(keypoint_order: list[str], values: list[str]) -> dict[str, str]:
    row = {}
    for index, keypoint_name in enumerate(keypoint_order):
        offset = index * 3
        row[f"{keypoint_name}.x"] = values[offset]
        row[f"{keypoint_name}.y"] = values[offset + 1]
        row[f"{keypoint_name}.z"] = values[offset + 2]
    return row


def read_label_rows(labels_path: Path) -> list[dict[str, str]]:
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
            missing_headers = header[len(values):]
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
    return rows


def parse_decimal(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value.replace(",", "."))


def to_image_y(y: float, height: int, flip_y: bool) -> float:
    return float(height) - float(y) if flip_y else float(y)


def normalize_keypoint_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def infer_joint_type(name: str) -> str:
    lower_name = normalize_keypoint_name(name)
    if any(token in lower_name for token in ("hip", "knee", "ankle", "foot", "toe", "heel", "pelvis", "calf", "thigh")):
        return "lower"
    if any(token in lower_name for token in ("shoulder", "elbow", "wrist", "hand", "finger", "clavicle", "neck", "head", "ear", "eye", "nose", "spine")):
        return "upper"
    return ""


def derive_swap(name: str, names: set[str]) -> str:
    pairs = [
        ("left_", "right_"),
        ("l_", "r_"),
        ("_l", "_r"),
        ("left", "right"),
        ("ear_l", "ear_r"),
        ("eye_l", "eye_r"),
    ]
    normalized_name = normalize_keypoint_name(name)
    for left_token, right_token in pairs:
        if left_token in normalized_name:
            candidate = normalized_name.replace(left_token, right_token, 1)
            if candidate in names:
                return candidate
        if right_token in normalized_name:
            candidate = normalized_name.replace(right_token, left_token, 1)
            if candidate in names:
                return candidate
    if name.startswith("L "):
        candidate = normalize_keypoint_name("R " + name[2:])
        if candidate in names:
            return candidate
    if name.startswith("R "):
        candidate = normalize_keypoint_name("L " + name[2:])
        if candidate in names:
            return candidate
    return ""


def joint_color(name: str) -> list[int]:
    normalized_name = normalize_keypoint_name(name)
    if "left" in normalized_name or normalized_name.startswith("l_") or normalized_name.endswith("_l"):
        return [0, 255, 0]
    if "right" in normalized_name or normalized_name.startswith("r_") or normalized_name.endswith("_r"):
        return [255, 128, 0]
    return [51, 153, 255]


def make_dataset_info(dataset_name: str, keypoint_names: list[str], skeleton: list[tuple[str, str]]) -> dict:
    normalized_names = [normalize_keypoint_name(name) for name in keypoint_names]
    normalized_name_set = set(normalized_names)
    keypoint_info = {}
    for index, (raw_name, normalized_name) in enumerate(zip(keypoint_names, normalized_names)):
        keypoint_info[index] = dict(
            name=normalized_name,
            id=index,
            color=joint_color(raw_name),
            type=infer_joint_type(raw_name),
            swap=derive_swap(raw_name, normalized_name_set),
        )

    skeleton_info = {}
    for index, (left_name, right_name) in enumerate(skeleton):
        skeleton_info[index] = dict(
            link=(normalize_keypoint_name(left_name), normalize_keypoint_name(right_name)),
            id=index,
            color=[51, 153, 255],
        )

    return dict(
        dataset_name=dataset_name,
        paper_info=dict(
            author="SwimXYZ synthetic dataset",
            title="SwimXYZ multi-label pose dataset",
            container="local workspace",
            year="2026",
            homepage="",
        ),
        keypoint_info=keypoint_info,
        skeleton_info=skeleton_info,
        joint_weights=[1.0] * len(keypoint_names),
        sigmas=[0.05] * len(keypoint_names),
    )


def build_format_specs(formats: list[str]) -> dict[str, FormatSpec]:
    specs: dict[str, FormatSpec] = {}
    for dataset_idx, label_format in enumerate(formats):
        if label_format == "COCO":
            specs[label_format] = FormatSpec(
                name="swimxyz_coco",
                label_format=label_format,
                source_names=[source for source, _ in COCO17_SOURCE_TO_OUTPUT],
                output_names=[target for _, target in COCO17_SOURCE_TO_OUTPUT],
                skeleton=COCO_SKELETON,
                dataset_idx=dataset_idx,
            )
        elif label_format == "body25":
            specs[label_format] = FormatSpec(
                name="swimxyz_body25",
                label_format=label_format,
                source_names=BODY25_KEYPOINT_ORDER[:],
                output_names=[normalize_keypoint_name(item) for item in BODY25_KEYPOINT_ORDER],
                skeleton=BODY25_SKELETON,
                dataset_idx=dataset_idx,
            )
        elif label_format == "base":
            specs[label_format] = FormatSpec(
                name="swimxyz_base",
                label_format=label_format,
                source_names=BASE_KEYPOINT_ORDER[:],
                output_names=[normalize_keypoint_name(item) for item in BASE_KEYPOINT_ORDER],
                skeleton=BASE_SKELETON,
                dataset_idx=dataset_idx,
            )
        else:
            raise ValueError(f"Unsupported SwimXYZ format for VitPose++: {label_format}")
    return specs


def build_pose_sample(
    row: dict[str, str],
    width: int,
    height: int,
    spec: FormatSpec,
    bbox_padding_ratio: float,
    min_visible_keypoints: int,
    flip_y: bool,
) -> PoseSample | None:
    keypoints: list[float] = []
    visible_points: list[tuple[float, float]] = []

    for source_name in spec.source_names:
        x = parse_decimal(row.get(f"{source_name}.x"))
        y = parse_decimal(row.get(f"{source_name}.y"))
        score = parse_decimal(row.get(f"{source_name}.z"))
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

        clipped_x = min(max(float(x), 0.0), float(width - 1))
        image_y = to_image_y(float(y), height, flip_y)
        clipped_y = min(max(image_y, 0.0), float(height - 1))
        keypoints.extend([clipped_x, clipped_y, 2.0])
        visible_points.append((clipped_x, clipped_y))

    if len(visible_points) < min_visible_keypoints:
        return None

    min_x = min(point[0] for point in visible_points)
    max_x = max(point[0] for point in visible_points)
    min_y = min(point[1] for point in visible_points)
    max_y = max(point[1] for point in visible_points)
    pad_x = max((max_x - min_x) * bbox_padding_ratio, 2.0)
    pad_y = max((max_y - min_y) * bbox_padding_ratio, 2.0)

    bbox_x = max(0.0, min_x - pad_x)
    bbox_y = max(0.0, min_y - pad_y)
    bbox_w = min(float(width) - bbox_x, (max_x - min_x) + 2 * pad_x)
    bbox_h = min(float(height) - bbox_y, (max_y - min_y) + 2 * pad_y)
    bbox = [round(bbox_x, 2), round(bbox_y, 2), round(bbox_w, 2), round(bbox_h, 2)]
    area = round(bbox_w * bbox_h, 2)

    return PoseSample(
        width=width,
        height=height,
        bbox=bbox,
        keypoints=keypoints,
        num_keypoints=len(visible_points),
        area=area,
    )


def split_frame_records(
    records: list[FrameRecord],
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[FrameRecord], list[FrameRecord], list[FrameRecord]]:
    if not records:
        return [], [], []
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("val_ratio and test_ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1.")
    if len(records) == 1:
        return records, [], []
    if len(records) == 2:
        return [records[0]], [records[1]], []

    test_count = int(round(len(records) * test_ratio))
    val_count = int(round(len(records) * val_ratio))
    if test_ratio > 0:
        test_count = max(1, test_count)
    if val_ratio > 0:
        val_count = max(1, val_count)
    if val_count + test_count >= len(records):
        overflow = (val_count + test_count) - (len(records) - 1)
        if test_count >= overflow and test_count > 0:
            test_count -= overflow
        else:
            val_count = max(0, val_count - overflow)
    train_count = len(records) - val_count - test_count
    if train_count <= 0:
        raise ValueError("Split ratios leave no training records.")

    train_records = records[:train_count]
    val_end = train_count + val_count
    val_records = records[train_count:val_end]
    test_records = records[val_end:]
    return train_records, val_records, test_records


def build_category(spec: FormatSpec) -> dict:
    index_map = {name: idx + 1 for idx, name in enumerate(spec.output_names)}
    skeleton = []
    for left_name, right_name in spec.skeleton:
        normalized_left = normalize_keypoint_name(left_name)
        normalized_right = normalize_keypoint_name(right_name)
        if normalized_left in index_map and normalized_right in index_map:
            skeleton.append([index_map[normalized_left], index_map[normalized_right]])
    return {
        "id": 1,
        "name": "person",
        "supercategory": "person",
        "keypoints": spec.output_names,
        "skeleton": skeleton,
    }


def build_coco_json(records: list[FrameRecord], spec: FormatSpec) -> dict:
    images = []
    annotations = []
    image_id = 1
    for record in records:
        sample = record.samples_by_format.get(spec.label_format)
        if sample is None:
            continue
        images.append(
            {
                "id": image_id,
                "file_name": record.image_name,
                "width": record.width,
                "height": record.height,
            }
        )
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": sample.bbox,
                "area": sample.area,
                "iscrowd": 0,
                "num_keypoints": sample.num_keypoints,
                "keypoints": sample.keypoints,
            }
        )
        image_id += 1

    return {
        "info": {"description": f"SwimXYZ {spec.label_format} dataset for VitPose++"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [build_category(spec)],
    }


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def collect_existing_frame_index(output_root: Path, dataset_name: str) -> dict[str, Path]:
    candidates = [
        output_root,
        output_root.parent / "_train_vitpose",
        output_root.parent / "_train_vitposepp",
        output_root.parent.parent / f"{dataset_name}_train_vitpose",
        output_root.parent.parent / f"{dataset_name}_train_vitposepp",
    ]
    index: dict[str, Path] = {}
    for candidate_root in candidates:
        if not candidate_root.is_dir():
            continue
        for split_name in ("train2017", "val2017", "test2017"):
            split_dir = candidate_root / split_name
            if not split_dir.is_dir():
                continue
            for image_path in split_dir.glob("*.jpg"):
                index.setdefault(image_path.name, image_path.resolve())
        cache_dir = candidate_root / "_frame_cache"
        if cache_dir.is_dir():
            for image_path in cache_dir.rglob("*.jpg"):
                index.setdefault(image_path.name, image_path.resolve())
    return index


def image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Unable to read cached frame: {image_path}")
    height, width = image.shape[:2]
    return width, height


def extract_missing_frame(video_path: Path, frame_index: int) -> tuple[object, int, int] | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    height, width = frame.shape[:2]
    return frame, width, height


def validate_y_range(
    row: dict[str, str],
    source_names: list[str],
    height: int,
    flip_y: bool,
    labels_path: Path,
    frame_index: int,
) -> tuple[int, int]:
    checked = 0
    flipped = 0
    for source_name in source_names:
        y = parse_decimal(row.get(f"{source_name}.y"))
        score = parse_decimal(row.get(f"{source_name}.z"))
        if y is None or score is None or score <= 0:
            continue
        checked += 1
        image_y = to_image_y(y, height, flip_y)
        if not (0.0 <= image_y <= float(height)):
            raise ValueError(
                f"Y coordinate out of bounds after flip for {labels_path} frame {frame_index}: "
                f"{source_name} -> {image_y} not in [0, {height}]"
            )
        if flip_y:
            flipped += 1
    return checked, flipped


def validate_shared_keypoints(
    rows_by_format: dict[str, dict[str, str]],
    tolerance: float,
    labels_by_format: dict[str, Path],
    frame_index: int,
) -> int:
    validated = 0
    if "COCO" in rows_by_format and "body25" in rows_by_format:
        for source_name, _ in COCO17_SOURCE_TO_OUTPUT:
            for suffix in ("x", "y", "z"):
                coco_value = parse_decimal(rows_by_format["COCO"].get(f"{source_name}.{suffix}"))
                body_value = parse_decimal(rows_by_format["body25"].get(f"{source_name}.{suffix}"))
                if coco_value is None and body_value is None:
                    continue
                if coco_value is None or body_value is None or abs(coco_value - body_value) > tolerance:
                    raise ValueError(
                        f"COCO/body25 mismatch at frame {frame_index} for {source_name}.{suffix}: "
                        f"{coco_value} vs {body_value} in {labels_by_format['COCO']} and {labels_by_format['body25']}"
                    )
                validated += 1

    if "body25" in rows_by_format and "base" in rows_by_format:
        for body25_name, base_name in BODY25_TO_BASE_SHARED.items():
            for suffix in ("x", "y", "z"):
                body_value = parse_decimal(rows_by_format["body25"].get(f"{body25_name}.{suffix}"))
                base_value = parse_decimal(rows_by_format["base"].get(f"{base_name}.{suffix}"))
                if body_value is None and base_value is None:
                    continue
                if body_value is None or base_value is None or abs(body_value - base_value) > tolerance:
                    raise ValueError(
                        f"body25/base mismatch at frame {frame_index} for {body25_name}->{base_name}.{suffix}: "
                        f"{body_value} vs {base_value} in {labels_by_format['body25']} and {labels_by_format['base']}"
                    )
                validated += 1
    return validated


def prepare_frame_records(
    bundle: DatasetBundle,
    specs: dict[str, FormatSpec],
    cache_root: Path,
    existing_index: dict[str, Path],
    frame_step: int,
    max_frames_per_video: int,
    bbox_padding_ratio: float,
    min_visible_keypoints: int,
    flip_y: bool,
    coherence_tolerance: float,
    validation_counters: dict[str, int],
    skip_counters: dict[str, int],
    exception_log_entries: list[str],
) -> list[FrameRecord]:
    label_rows_by_format = {
        label_format: read_label_rows(labels_path)
        for label_format, labels_path in bundle.labels_by_format.items()
    }
    frame_limit = min(len(rows) for rows in label_rows_by_format.values())
    source_name = bundle.video_path.stem.replace(",", "_")
    source_cache_dir = cache_root / source_name
    source_cache_dir.mkdir(parents=True, exist_ok=True)

    records: list[FrameRecord] = []
    valid_count = 0
    for frame_index in range(0, frame_limit, frame_step):
        if max_frames_per_video > 0 and valid_count >= max_frames_per_video:
            break

        image_name = f"{source_name}_{frame_index:06d}.jpg"
        cache_path = source_cache_dir / image_name
        if cache_path.is_file():
            width, height = image_size(cache_path)
        else:
            existing_path = existing_index.get(image_name)
            if existing_path is not None and existing_path.is_file():
                hardlink_or_copy(existing_path, cache_path)
                width, height = image_size(cache_path)
            else:
                extracted = extract_missing_frame(bundle.video_path, frame_index)
                if extracted is None:
                    skip_counters["missing_frame_for_label"] += 1
                    exception_log_entries.append(
                        f"{bundle.video_path.name} | frame_index={frame_index} | missing_frame_for_label"
                    )
                    continue
                frame, width, height = extracted
                cv2.imwrite(str(cache_path), frame)

        rows_for_frame = {
            label_format: rows[frame_index]
            for label_format, rows in label_rows_by_format.items()
        }
        validation_counters["shared_keypoints"] += validate_shared_keypoints(
            rows_by_format=rows_for_frame,
            tolerance=coherence_tolerance,
            labels_by_format=bundle.labels_by_format,
            frame_index=frame_index,
        )

        for label_format, row in rows_for_frame.items():
            checked, flipped = validate_y_range(
                row=row,
                source_names=specs[label_format].source_names,
                height=height,
                flip_y=flip_y,
                labels_path=bundle.labels_by_format[label_format],
                frame_index=frame_index,
            )
            validation_counters["y_checked"] += checked
            validation_counters["y_flipped"] += flipped

        samples_by_format: dict[str, PoseSample] = {}
        for label_format, row in rows_for_frame.items():
            sample = build_pose_sample(
                row=row,
                width=width,
                height=height,
                spec=specs[label_format],
                bbox_padding_ratio=bbox_padding_ratio,
                min_visible_keypoints=min_visible_keypoints,
                flip_y=flip_y,
            )
            if sample is not None:
                samples_by_format[label_format] = sample

        if not samples_by_format:
            skip_counters["no_valid_keypoints"] += 1
            continue

        records.append(
            FrameRecord(
                source_name=source_name,
                frame_index=frame_index,
                image_name=image_name,
                cache_path=cache_path,
                width=width,
                height=height,
                samples_by_format=samples_by_format,
            )
        )
        valid_count += 1
    return records


def write_vitposepp_config(
    output_path: Path,
    output_root: Path,
    base_config: Path,
    pretrained_checkpoint: Path,
    work_dir: Path,
    total_epochs: int,
    val_interval: int,
    samples_per_gpu: int,
    workers_per_gpu: int,
    specs: list[FormatSpec],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_num_joints = max(len(spec.output_names) for spec in specs)
    dataset_infos = {
        spec.label_format: make_dataset_info(spec.name, spec.output_names, spec.skeleton)
        for spec in specs
    }
    channel_cfg_texts = []
    data_cfg_texts = []
    associate_heads = []
    train_dataset_blocks = []

    for index, spec in enumerate(specs):
        channel_cfg_name = f"{spec.label_format.lower()}_channel_cfg"
        data_cfg_name = f"{spec.label_format.lower()}_data_cfg"
        dataset_info_name = f"{spec.label_format.lower()}_dataset_info"
        joint_count = len(spec.output_names)
        channel_cfg_texts.append(
            f"{channel_cfg_name} = dict(\n"
            f"    num_output_channels={joint_count},\n"
            f"    dataset_joints={joint_count},\n"
            f"    dataset_channel=[list(range({joint_count}))],\n"
            f"    inference_channel=list(range({joint_count})),\n"
            f")\n"
        )
        data_cfg_texts.append(
            f"{data_cfg_name} = dict(\n"
            f"    image_size=[192, 256],\n"
            f"    heatmap_size=[48, 64],\n"
            f"    num_output_channels={channel_cfg_name}['num_output_channels'],\n"
            f"    num_joints={channel_cfg_name}['dataset_joints'],\n"
            f"    dataset_channel={channel_cfg_name}['dataset_channel'],\n"
            f"    inference_channel={channel_cfg_name}['inference_channel'],\n"
            f"    soft_nms=False,\n"
            f"    nms_thr=1.0,\n"
            f"    oks_thr=0.9,\n"
            f"    vis_thr=0.2,\n"
            f"    use_gt_bbox=True,\n"
            f"    det_bbox_thr=0.0,\n"
            f"    bbox_file='',\n"
            f"    max_num_joints={max_num_joints},\n"
            f"    dataset_idx={spec.dataset_idx},\n"
            f")\n"
        )
        if index > 0:
            associate_heads.append(
                "        dict(\n"
                "            type='TopdownHeatmapSimpleHead',\n"
                "            in_channels=1280,\n"
                "            num_deconv_layers=2,\n"
                "            num_deconv_filters=(256, 256),\n"
                "            num_deconv_kernels=(4, 4),\n"
                "            extra=dict(final_conv_kernel=1),\n"
                f"            out_channels={channel_cfg_name}['num_output_channels'],\n"
                "            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),\n"
                "        ),\n"
            )

        train_dataset_blocks.append(
            "        dict(\n"
            "            type='TopDownCocoDataset',\n"
            f"            ann_file=f'{{data_root}}/annotations/{spec.label_format}/person_keypoints_train.json',\n"
            "            img_prefix=f'{data_root}/train2017/',\n"
            f"            data_cfg={data_cfg_name},\n"
            "            pipeline=train_pipeline,\n"
            f"            dataset_info={dataset_info_name},\n"
            "        ),\n"
        )

    main_spec = specs[0]
    main_info_name = f"{main_spec.label_format.lower()}_dataset_info"
    main_data_cfg_name = f"{main_spec.label_format.lower()}_data_cfg"
    main_channel_cfg_name = f"{main_spec.label_format.lower()}_channel_cfg"

    config_text = (
        f"_base_ = ['{base_config.as_posix()}']\n\n"
        f"data_root = '{output_root.as_posix()}'\n"
        f"load_from = '{pretrained_checkpoint.as_posix()}'\n"
        f"work_dir = '{work_dir.as_posix()}'\n"
        f"total_epochs = {total_epochs}\n"
        f"evaluation = dict(interval={val_interval}, metric='mAP', save_best='AP')\n"
        "target_type = 'GaussianHeatmap'\n\n"
        f"{main_spec.label_format.lower()}_dataset_info = {pformat(dataset_infos[main_spec.label_format], width=100)}\n\n"
    )
    for spec in specs[1:]:
        config_text += (
            f"{spec.label_format.lower()}_dataset_info = "
            f"{pformat(dataset_infos[spec.label_format], width=100)}\n\n"
        )

    config_text += "\n".join(channel_cfg_texts) + "\n"
    config_text += "\n".join(data_cfg_texts) + "\n"
    config_text += (
        "train_pipeline = [\n"
        "    dict(type='LoadImageFromFile'),\n"
        "    dict(type='TopDownRandomFlip', flip_prob=0.5),\n"
        "    dict(type='TopDownHalfBodyTransform', num_joints_half_body=8, prob_half_body=0.3),\n"
        "    dict(type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),\n"
        "    dict(type='TopDownAffine', use_udp=True),\n"
        "    dict(type='ToTensor'),\n"
        "    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),\n"
        "    dict(type='TopDownGenerateTarget', sigma=2, encoding='UDP', target_type=target_type),\n"
        "    dict(\n"
        "        type='Collect',\n"
        "        keys=['img', 'target', 'target_weight'],\n"
        "        meta_keys=['image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'],\n"
        "    ),\n"
        "]\n\n"
        "val_pipeline = [\n"
        "    dict(type='LoadImageFromFile'),\n"
        "    dict(type='TopDownAffine', use_udp=True),\n"
        "    dict(type='ToTensor'),\n"
        "    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),\n"
        "    dict(\n"
        "        type='Collect',\n"
        "        keys=['img'],\n"
        "        meta_keys=['image_file', 'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'],\n"
        "    ),\n"
        "]\n\n"
        "test_pipeline = val_pipeline\n\n"
        "model = dict(\n"
        "    type='TopDownMoE',\n"
        "    pretrained=None,\n"
        "    backbone=dict(\n"
        "        type='ViTMoE',\n"
        "        img_size=(256, 192),\n"
        "        patch_size=16,\n"
        "        embed_dim=1280,\n"
        "        depth=32,\n"
        "        num_heads=16,\n"
        "        ratio=1,\n"
        "        use_checkpoint=False,\n"
        "        mlp_ratio=4,\n"
        "        qkv_bias=True,\n"
        "        drop_path_rate=0.55,\n"
        f"        num_expert={len(specs)},\n"
        "        part_features=320,\n"
        "    ),\n"
        "    keypoint_head=dict(\n"
        "        type='TopdownHeatmapSimpleHead',\n"
        "        in_channels=1280,\n"
        "        num_deconv_layers=2,\n"
        "        num_deconv_filters=(256, 256),\n"
        "        num_deconv_kernels=(4, 4),\n"
        "        extra=dict(final_conv_kernel=1),\n"
        f"        out_channels={main_channel_cfg_name}['num_output_channels'],\n"
        "        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),\n"
        "    ),\n"
        "    associate_keypoint_head=[\n"
    )
    config_text += "".join(associate_heads)
    config_text += (
        "    ],\n"
        "    train_cfg=dict(),\n"
        "    test_cfg=dict(\n"
        "        flip_test=True,\n"
        "        post_process='default',\n"
        "        shift_heatmap=False,\n"
        "        target_type=target_type,\n"
        "        modulate_kernel=11,\n"
        "        use_udp=True,\n"
        "    ),\n"
        ")\n\n"
        "data = dict(\n"
        f"    samples_per_gpu={samples_per_gpu},\n"
        f"    workers_per_gpu={workers_per_gpu},\n"
        f"    val_dataloader=dict(samples_per_gpu={max(1, samples_per_gpu // 2)}),\n"
        f"    test_dataloader=dict(samples_per_gpu={max(1, samples_per_gpu // 2)}),\n"
        "    train=[\n"
    )
    config_text += "".join(train_dataset_blocks)
    config_text += (
        "    ],\n"
        "    val=dict(\n"
        "        type='TopDownCocoDataset',\n"
        f"        ann_file=f'{{data_root}}/annotations/{main_spec.label_format}/person_keypoints_val.json',\n"
        "        img_prefix=f'{data_root}/val2017/',\n"
        f"        data_cfg={main_data_cfg_name},\n"
        "        pipeline=val_pipeline,\n"
        f"        dataset_info={main_info_name},\n"
        "    ),\n"
        "    test=dict(\n"
        "        type='TopDownCocoDataset',\n"
        f"        ann_file=f'{{data_root}}/annotations/{main_spec.label_format}/person_keypoints_test.json',\n"
        "        img_prefix=f'{data_root}/test2017/',\n"
        f"        data_cfg={main_data_cfg_name},\n"
        "        pipeline=test_pipeline,\n"
        f"        dataset_info={main_info_name},\n"
        "    ),\n"
        ")\n"
    )
    output_path.write_text(config_text, encoding="utf-8")


def prepare_dataset(
    args: argparse.Namespace,
    project_root: Path,
    dataset_root: Path,
    output_root: Path,
) -> None:
    label_formats = parse_formats(args.formats, args.main_format)
    specs = build_format_specs(label_formats)
    bundles = discover_dataset_bundles(
        dataset_root=dataset_root,
        label_formats=label_formats,
        label_dimension=args.label_dimension,
        label_reference=args.label_reference,
        max_videos=args.max_videos,
    )
    base_config = resolve_path(project_root, args.base_config)
    pretrained_checkpoint = resolve_path(project_root, args.pretrained_checkpoint)
    work_dir = resolve_path(project_root, args.work_dir)
    if not base_config.is_file():
        raise FileNotFoundError(f"Missing base config: {base_config}")
    if not pretrained_checkpoint.is_file():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {pretrained_checkpoint}")

    dataset_name = dataset_name_from_converted_root(dataset_root)
    frame_cache_root = output_root / "_frame_cache"
    annotations_root = output_root / "annotations"
    reports_root = output_root / "reports"
    generated_config = output_root / "generated_configs" / "swimxyz_vitposepp_huge.py"
    train_dir = output_root / "train2017"
    val_dir = output_root / "val2017"
    test_dir = output_root / "test2017"

    frame_cache_root.mkdir(parents=True, exist_ok=True)
    ensure_clean_dir(train_dir)
    ensure_clean_dir(val_dir)
    ensure_clean_dir(test_dir)
    ensure_clean_dir(annotations_root)
    ensure_clean_dir(reports_root)
    (output_root / "generated_configs").mkdir(parents=True, exist_ok=True)

    existing_index = collect_existing_frame_index(output_root, dataset_name)
    validation_counters = {"shared_keypoints": 0, "y_checked": 0, "y_flipped": 0}
    skip_counters = {"missing_frame_for_label": 0, "no_valid_keypoints": 0}
    exception_log_entries: list[str] = []

    all_records: list[FrameRecord] = []
    for bundle in bundles:
        all_records.extend(
            prepare_frame_records(
                bundle=bundle,
                specs=specs,
                cache_root=frame_cache_root,
                existing_index=existing_index,
                frame_step=args.frame_step,
                max_frames_per_video=args.max_frames_per_video,
                bbox_padding_ratio=args.bbox_padding_ratio,
                min_visible_keypoints=args.min_visible_keypoints,
                flip_y=args.flip_y,
                coherence_tolerance=args.coherence_tolerance,
                validation_counters=validation_counters,
                skip_counters=skip_counters,
                exception_log_entries=exception_log_entries,
            )
        )

    if not all_records:
        raise RuntimeError("No valid SwimXYZ frames were prepared for VitPose++.")

    all_records.sort(key=lambda item: (item.source_name, item.frame_index))
    train_records, val_records, test_records = split_frame_records(
        all_records,
        args.val_ratio,
        args.test_ratio,
    )

    split_map = {
        "train": (train_records, train_dir),
        "val": (val_records, val_dir),
        "test": (test_records, test_dir),
    }
    for _, (records, split_dir) in split_map.items():
        for record in records:
            hardlink_or_copy(record.cache_path, split_dir / record.image_name)

    for label_format, spec in specs.items():
        format_root = annotations_root / label_format
        format_root.mkdir(parents=True, exist_ok=True)
        (format_root / "person_keypoints_train.json").write_text(
            json.dumps(build_coco_json(train_records, spec), indent=2),
            encoding="utf-8",
        )
        (format_root / "person_keypoints_val.json").write_text(
            json.dumps(build_coco_json(val_records, spec), indent=2),
            encoding="utf-8",
        )
        (format_root / "person_keypoints_test.json").write_text(
            json.dumps(build_coco_json(test_records, spec), indent=2),
            encoding="utf-8",
        )

    write_vitposepp_config(
        output_path=generated_config,
        output_root=output_root,
        base_config=base_config,
        pretrained_checkpoint=pretrained_checkpoint,
        work_dir=work_dir,
        total_epochs=args.total_epochs,
        val_interval=args.val_interval,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        specs=[specs[label_format] for label_format in label_formats],
    )

    counts = {
        label_format: {
            "train": sum(1 for record in train_records if label_format in record.samples_by_format),
            "val": sum(1 for record in val_records if label_format in record.samples_by_format),
            "test": sum(1 for record in test_records if label_format in record.samples_by_format),
        }
        for label_format in label_formats
    }
    report = {
        "dataset_root": dataset_root.as_posix(),
        "output_root": output_root.as_posix(),
        "formats": label_formats,
        "videos": len(bundles),
        "shared_frames": len(all_records),
        "split_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "format_counts": counts,
        "validation": {
            "shared_keypoint_values_checked": validation_counters["shared_keypoints"],
            "y_coordinates_checked": validation_counters["y_checked"],
            "y_coordinates_flipped": validation_counters["y_flipped"],
            "flip_y_enabled": args.flip_y,
        },
        "skips": skip_counters,
    }
    (reports_root / "preparation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    log_path = annotations_root / "dataset_exceptions.log"
    if exception_log_entries:
        log_path.write_text("\n".join(exception_log_entries) + "\n", encoding="utf-8")
    else:
        log_path.write_text("No exceptions detected.\n", encoding="utf-8")

    print(f"Prepared SwimXYZ VitPose++ dataset: {dataset_name}")
    print(f"Converted dataset root: {dataset_root}")
    print(f"Dataset root: {output_root}")
    print(f"Shared frames: {len(all_records)}")
    print(f"Train/val/test frames: {len(train_records)}/{len(val_records)}/{len(test_records)}")
    for label_format in label_formats:
        format_counts = counts[label_format]
        print(
            f"{label_format} samples train/val/test: "
            f"{format_counts['train']}/{format_counts['val']}/{format_counts['test']}"
        )
    print(
        "Validated shared keypoint values / Y coordinates: "
        f"{validation_counters['shared_keypoints']} / {validation_counters['y_checked']}"
    )
    print(
        "Skipped samples (missing frame / invalid keypoints): "
        f"{skip_counters['missing_frame_for_label']} / {skip_counters['no_valid_keypoints']}"
    )
    print(f"Exception log: {log_path}")
    print(f"Generated config: {generated_config}")
    print("Training command:")
    print(
        "conda run -n vitpose python "
        f"{(project_root / 'src/vitpose_base/tools/train.py').as_posix()} "
        f"{generated_config.as_posix()}"
    )


def main() -> None:
    args = parse_args()
    project_root = resolve_path(Path.cwd(), args.project_root)
    dataset_root = resolve_path(project_root, args.dataset_root)
    dataset_roots = iter_converted_dataset_roots(dataset_root)

    if len(dataset_roots) > 1 and args.output_root:
        raise ValueError(
            "When processing multiple converted datasets, do not pass a shared "
            "--output-root. Let the script infer one '_train_vitposepp' directory per dataset."
        )

    for current_dataset_root in dataset_roots:
        output_root = build_output_root(project_root, current_dataset_root, args.output_root)
        prepare_dataset(args, project_root, current_dataset_root, output_root)


if __name__ == "__main__":
    main()
