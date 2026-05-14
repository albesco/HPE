from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

# Shared utilities for the consolidated SwimXYZ -> VitPose++ preparation flow.
# This module is the common implementation layer for:
# - parsing SwimXYZ label rows
# - mapping the selected SwimXYZ representation to COCO17
# - computing bbox, visibility and overlay images
# - split management and JSON export
#
# The active VitPose++ pipeline imports these helpers rather than
# reimplementing them, so behavior stays aligned across dataset generation and
# verification scripts.


# Default constants for in-script usage.
DEFAULT_PROJECT_ROOT = Path("/home/albertosco/HPE")
DEFAULT_DATASET_ROOT = "data/intermediate"
DEFAULT_OUTPUT_ROOT = ""
DEFAULT_BASE_CONFIG = (
    "src/vitpose_base/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/"
    "ViTPose_huge_coco_256x192.py"
)
DEFAULT_PRETRAINED_CHECKPOINT = "models/pose/coco.pth"
DEFAULT_WORK_DIR = "runs/vitpose_subset_xyz"
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

COCO_SKELETON = [
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
]

OVERLAY_KEYPOINT_COLORS = {
    "nose": (255, 200, 0),
    "left_eye": (0, 200, 255),
    "right_eye": (255, 150, 0),
    "left_ear": (0, 180, 255),
    "right_ear": (255, 120, 0),
    "left_shoulder": (0, 255, 0),
    "right_shoulder": (0, 128, 255),
    "left_elbow": (0, 220, 0),
    "right_elbow": (0, 100, 255),
    "left_wrist": (0, 190, 0),
    "right_wrist": (0, 80, 255),
    "left_hip": (0, 255, 80),
    "right_hip": (80, 140, 255),
    "left_knee": (0, 255, 140),
    "right_knee": (120, 160, 255),
    "left_ankle": (0, 255, 200),
    "right_ankle": (160, 180, 255),
}
OVERLAY_SKELETON_COLOR = (255, 255, 255)
ANOMALOUS_BBOX_CENTER_DISTANCE_RATIO = 1.5
ANOMALOUS_BBOX_IOU_THRESHOLD = 0.1


@dataclass
class DatasetEntry:
    video_path: Path
    labels_path: Path


@dataclass
class Sample:
    source_name: str
    frame_index: int
    image_path: Path
    width: int
    height: int
    bbox: list[float]
    keypoints: list[float]
    num_keypoints: int
    area: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a SwimXYZ dataset in COCO format for the consolidated pipeline."
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
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument(
        "--bbox-padding-ratio",
        type=float,
        default=None,
        help="Deprecated uniform bbox padding ratio. If set, applies to both axes.",
    )
    parser.add_argument("--bbox-padding-x-ratio", type=float, default=0.20)
    parser.add_argument("--bbox-padding-y-ratio", type=float, default=0.25)
    parser.add_argument("--bbox-min-padding-px", type=float, default=15.0)
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


def build_train_output_root(project_root: Path, dataset_root: Path, output_root_value: str) -> Path:
    if output_root_value:
        return resolve_path(project_root, output_root_value)

    dataset_name = dataset_name_from_converted_root(dataset_root)
    return (project_root / "data" / "intermediate" / dataset_name / "_train_vitpose").resolve()


def build_label_filename(
    label_format: str,
    label_dimension: str,
    label_reference: str,
) -> str:
    return f"{label_dimension}_{label_reference}_{label_format}.txt"


def discover_dataset_entries(
    dataset_root: Path,
    label_format: str,
    label_dimension: str,
    label_reference: str,
) -> list[DatasetEntry]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")

    label_filename = build_label_filename(label_format, label_dimension, label_reference)
    manifest_path = dataset_root / "manifest.json"
    entries: list[DatasetEntry] = []

    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in payload.get("items", []):
            video_rel_path = item.get("video_path") or item.get("video_file")
            if not video_rel_path:
                continue

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
                continue

            video_path = (dataset_root / video_rel_path).resolve()
            labels_path = (dataset_root / matching_label).resolve()
            if video_path.is_file() and labels_path.is_file():
                entries.append(DatasetEntry(video_path=video_path, labels_path=labels_path))
    else:
        for video_path in sorted(dataset_root.glob("*.webm")):
            labels_path = dataset_root / video_path.stem / label_format / label_filename
            if labels_path.is_file():
                entries.append(
                    DatasetEntry(
                        video_path=video_path.resolve(),
                        labels_path=labels_path.resolve(),
                    )
                )

    if not entries:
        raise RuntimeError(
            "No dataset entries discovered in "
            f"{dataset_root} for {label_format}/{label_dimension}_{label_reference}."
        )
    return entries


def parse_dataset_entries(args: argparse.Namespace, project_root: Path) -> list[DatasetEntry]:
    raw_entries = args.dataset_entry
    if not raw_entries:
        dataset_root = resolve_path(project_root, args.dataset_root)
        return discover_dataset_entries(
            dataset_root=dataset_root,
            label_format=args.label_format,
            label_dimension=args.label_dimension,
            label_reference=args.label_reference,
        )

    entries: list[DatasetEntry] = []
    for raw_entry in raw_entries:
        if "::" not in raw_entry:
            raise ValueError(
                "Each --dataset-entry must be formatted as video_path::labels_path"
            )
        video_value, labels_value = raw_entry.split("::", 1)
        entry = DatasetEntry(
            video_path=resolve_path(project_root, video_value),
            labels_path=resolve_path(project_root, labels_value),
        )
        if not entry.video_path.is_file():
            raise FileNotFoundError(f"Missing video file: {entry.video_path}")
        if not entry.labels_path.is_file():
            raise FileNotFoundError(f"Missing label file: {entry.labels_path}")
        entries.append(entry)
    return entries


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

            # SwimXYZ rows are systematically truncated only at the tail:
            # LEar plus foot/toe keypoints. Pad exactly those missing fields
            # so they remain explicit absences rather than shifting columns.
            values.extend([""] * (len(header) - len(values)))
        elif len(values) > len(header):
            raise ValueError(
                f"Invalid row length in {labels_path}: expected {len(header)}, got {len(values)}"
            )
        rows.append(dict(zip(header, values)))
    return rows


def keypoint_row_from_values(keypoint_order: list[str], values: list[str]) -> dict[str, str]:
    row = {}
    for index, keypoint_name in enumerate(keypoint_order):
        offset = index * 3
        row[f"{keypoint_name}.x"] = values[offset]
        row[f"{keypoint_name}.y"] = values[offset + 1]
        row[f"{keypoint_name}.z"] = values[offset + 2]
    return row


def parse_decimal(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value.replace(",", "."))


def to_image_y(y: float, height: int, flip_y: bool) -> float:
    return float(height) - float(y) if flip_y else float(y)


def build_keypoints_and_bbox(
    row: dict[str, str],
    width: int,
    height: int,
    bbox_padding_x_ratio: float,
    bbox_padding_y_ratio: float,
    bbox_min_padding_px: float,
    min_visible_keypoints: int,
    flip_y: bool,
) -> tuple[list[float], int, list[float], float] | None:
    # SwimXYZ stores x,y,z triples, but in our consolidated workflow z is not a
    # confidence score. Visibility is reconstructed from the horizontal image
    # position so that border-clamped points can be excluded consistently.
    keypoints: list[float] = []
    visible_points: list[tuple[float, float]] = []

    for body25_name, _ in BODY25_TO_COCO:
        x = parse_decimal(row.get(f"{body25_name}.x"))
        y = parse_decimal(row.get(f"{body25_name}.y"))

        # SwimXYZ provides x,y,z where z is a spatial coordinate (Blender 3D),
        # not a confidence/visibility score. Visibility is derived from x only:
        # - if x == 0 or x == max (image boundary), v = 0
        # - otherwise (0 < x < max), v = 2
        if x is None or y is None or math.isnan(x) or math.isnan(y):
            keypoints.extend([0.0, 0.0, 0.0])
            continue

        clipped_x = min(max(float(x), 0.0), float(width - 1))
        image_y = to_image_y(float(y), height, flip_y)
        clipped_y = min(max(image_y, 0.0), float(height - 1))

        max_x = float(width - 1)
        v = 0.0 if clipped_x <= 0.0 or clipped_x >= max_x else 2.0
        if v == 0.0:
            keypoints.extend([0.0, 0.0, 0.0])
            continue

        keypoints.extend([clipped_x, clipped_y, v])
        visible_points.append((clipped_x, clipped_y))

    if len(visible_points) < min_visible_keypoints:
        return None

    min_x = min(point[0] for point in visible_points)
    max_x = max(point[0] for point in visible_points)
    min_y = min(point[1] for point in visible_points)
    max_y = max(point[1] for point in visible_points)
    pad_x = max((max_x - min_x) * bbox_padding_x_ratio, bbox_min_padding_px)
    pad_y = max((max_y - min_y) * bbox_padding_y_ratio, bbox_min_padding_px)

    bbox_x = max(0.0, min_x - pad_x)
    bbox_y = max(0.0, min_y - pad_y)
    bbox_w = min(float(width) - bbox_x, (max_x - min_x) + 2 * pad_x)
    bbox_h = min(float(height) - bbox_y, (max_y - min_y) + 2 * pad_y)
    bbox = [round(bbox_x, 2), round(bbox_y, 2), round(bbox_w, 2), round(bbox_h, 2)]
    area = round(bbox_w * bbox_h, 2)

    return keypoints, len(visible_points), bbox, area


def bbox_iou(first_bbox: list[float], second_bbox: list[float]) -> float:
    first_x, first_y, first_w, first_h = first_bbox
    second_x, second_y, second_w, second_h = second_bbox

    first_x2 = first_x + first_w
    first_y2 = first_y + first_h
    second_x2 = second_x + second_w
    second_y2 = second_y + second_h

    inter_x1 = max(first_x, second_x)
    inter_y1 = max(first_y, second_y)
    inter_x2 = min(first_x2, second_x2)
    inter_y2 = min(first_y2, second_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = (first_w * first_h) + (second_w * second_h) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def detect_anomalous_bbox_shift(
    previous_bbox: list[float] | None,
    current_bbox: list[float],
) -> bool:
    # The goal here is not to reject every fast swimmer movement. We only flag
    # cases where the bbox suddenly jumps far away from the previous one and at
    # the same time stops overlapping it in any meaningful way.
    if previous_bbox is None:
        return False

    prev_x, prev_y, prev_w, prev_h = previous_bbox
    curr_x, curr_y, curr_w, curr_h = current_bbox
    prev_center_x = prev_x + (prev_w / 2.0)
    prev_center_y = prev_y + (prev_h / 2.0)
    curr_center_x = curr_x + (curr_w / 2.0)
    curr_center_y = curr_y + (curr_h / 2.0)
    center_distance = math.hypot(curr_center_x - prev_center_x, curr_center_y - prev_center_y)
    previous_diagonal = math.hypot(prev_w, prev_h)
    if previous_diagonal <= 0:
        return False

    return (
        center_distance > (previous_diagonal * ANOMALOUS_BBOX_CENTER_DISTANCE_RATIO)
        and bbox_iou(previous_bbox, current_bbox) < ANOMALOUS_BBOX_IOU_THRESHOLD
    )


def extract_samples_from_entry(
    entry: DatasetEntry,
    images_root: Path,
    frame_step: int,
    max_frames_per_video: int,
    bbox_padding_x_ratio: float,
    bbox_padding_y_ratio: float,
    bbox_min_padding_px: float,
    min_visible_keypoints: int,
    flip_y: bool,
) -> list[Sample]:
    label_rows = read_label_rows(entry.labels_path)
    capture = cv2.VideoCapture(str(entry.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {entry.video_path}")

    source_name = entry.video_path.stem.replace(",", "_")
    source_dir = images_root / source_name
    source_dir.mkdir(parents=True, exist_ok=True)

    samples: list[Sample] = []
    frame_index = 0
    saved_count = 0
    label_index = 0

    while True:
        ok, frame = capture.read()
        if not ok or label_index >= len(label_rows):
            break

        if frame_index % frame_step != 0:
            frame_index += 1
            label_index += 1
            continue

        height, width = frame.shape[:2]
        prepared = build_keypoints_and_bbox(
            row=label_rows[label_index],
            width=width,
            height=height,
            bbox_padding_x_ratio=bbox_padding_x_ratio,
            bbox_padding_y_ratio=bbox_padding_y_ratio,
            bbox_min_padding_px=bbox_min_padding_px,
            min_visible_keypoints=min_visible_keypoints,
            flip_y=flip_y,
        )
        if prepared is not None:
            keypoints, num_keypoints, bbox, area = prepared
            image_path = source_dir / f"{source_name}_{frame_index:06d}.jpg"
            cv2.imwrite(str(image_path), frame)
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
            saved_count += 1

            if max_frames_per_video > 0 and saved_count >= max_frames_per_video:
                break

        frame_index += 1
        label_index += 1

    capture.release()
    return samples


def split_samples(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    if not samples:
        return [], [], []

    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("val_ratio and test_ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1.")

    if len(samples) == 1:
        return samples, [], []
    if len(samples) == 2:
        return [samples[0]], [samples[1]], []

    test_count = int(round(len(samples) * test_ratio))
    val_count = int(round(len(samples) * val_ratio))

    if test_ratio > 0:
        test_count = max(1, test_count)
    if val_ratio > 0:
        val_count = max(1, val_count)

    if val_count + test_count >= len(samples):
        overflow = (val_count + test_count) - (len(samples) - 1)
        if test_count >= overflow and test_count > 0:
            test_count -= overflow
        else:
            val_count = max(0, val_count - overflow)

    train_count = len(samples) - val_count - test_count
    if train_count <= 0:
        raise ValueError("Split ratios leave no samples for training.")

    train_samples = samples[:train_count]
    val_end = train_count + val_count
    val_samples = samples[train_count:val_end]
    test_samples = samples[val_end:]
    return train_samples, val_samples, test_samples


def build_coco_json(samples: list[Sample], img_prefix: str) -> dict:
    images = []
    annotations = []
    for image_id, sample in enumerate(samples, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": sample.image_path.name,
                "width": sample.width,
                "height": sample.height,
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

    return {
        "info": {"description": "SwimXYZ converted for VitPose fine-tuning"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": [name for _, name in BODY25_TO_COCO],
                "skeleton": COCO_SKELETON,
            }
        ],
    }


def move_split_images(samples: list[Sample], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        target = destination_dir / sample.image_path.name
        shutil.move(str(sample.image_path), str(target))
        sample.image_path = target


def overlay_output_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_with-KP{image_path.suffix}")


def draw_sample_overlay(image, sample: Sample) -> None:
    # Overlay files are generated next to train/val/test images so we can audit
    # the final dataset exactly as it is consumed by VitPose++.
    keypoint_names = [name for _, name in BODY25_TO_COCO]

    for start_idx, end_idx in COCO_SKELETON:
        start_offset = (start_idx - 1) * 3
        end_offset = (end_idx - 1) * 3
        start_visibility = sample.keypoints[start_offset + 2]
        end_visibility = sample.keypoints[end_offset + 2]
        if start_visibility <= 0 or end_visibility <= 0:
            continue
        cv2.line(
            image,
            (
                int(round(sample.keypoints[start_offset])),
                int(round(sample.keypoints[start_offset + 1])),
            ),
            (
                int(round(sample.keypoints[end_offset])),
                int(round(sample.keypoints[end_offset + 1])),
            ),
            OVERLAY_SKELETON_COLOR,
            2,
            lineType=cv2.LINE_AA,
        )

    for index, keypoint_name in enumerate(keypoint_names):
        offset = index * 3
        x = sample.keypoints[offset]
        y = sample.keypoints[offset + 1]
        visibility = sample.keypoints[offset + 2]
        if visibility <= 0:
            continue
        cv2.circle(
            image,
            (int(round(x)), int(round(y))),
            4,
            OVERLAY_KEYPOINT_COLORS[keypoint_name],
            thickness=-1,
            lineType=cv2.LINE_AA,
        )


def write_split_overlay_images(samples: list[Sample]) -> None:
    for sample in samples:
        image = cv2.imread(str(sample.image_path))
        if image is None:
            raise RuntimeError(f"Unable to read split image for overlay: {sample.image_path}")
        draw_sample_overlay(image, sample)
        overlay_path = overlay_output_path(sample.image_path)
        ok = cv2.imwrite(str(overlay_path), image)
        if not ok:
            raise RuntimeError(f"Unable to write overlay image: {overlay_path}")


def write_training_config(
    output_path: Path,
    dataset_root: Path,
    base_config: Path,
    pretrained_checkpoint: Path,
    work_dir: Path,
    total_epochs: int,
    val_interval: int,
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_text = f"""_base_ = ['{base_config.as_posix()}']

data_root = '{dataset_root.as_posix()}'
load_from = '{pretrained_checkpoint.as_posix()}'
work_dir = '{work_dir.as_posix()}'
total_epochs = {total_epochs}
evaluation = dict(interval={val_interval}, metric='mAP', save_best='AP')

data = dict(
    samples_per_gpu={samples_per_gpu},
    workers_per_gpu={workers_per_gpu},
    val_dataloader=dict(samples_per_gpu={max(1, samples_per_gpu)}),
    test_dataloader=dict(samples_per_gpu={max(1, samples_per_gpu)}),
    train=dict(
        ann_file=f'{{data_root}}/annotations/person_keypoints_train.json',
        img_prefix=f'{{data_root}}/train2017/',
        data_cfg=dict(use_gt_bbox=True, bbox_file=''),
    ),
    val=dict(
        ann_file=f'{{data_root}}/annotations/person_keypoints_val.json',
        img_prefix=f'{{data_root}}/val2017/',
        data_cfg=dict(use_gt_bbox=True, bbox_file=''),
    ),
    test=dict(
        ann_file=f'{{data_root}}/annotations/person_keypoints_test.json',
        img_prefix=f'{{data_root}}/test2017/',
        data_cfg=dict(use_gt_bbox=True, bbox_file=''),
    ),
)
"""
    output_path.write_text(config_text, encoding="utf-8")


def prepare_dataset(
    args: argparse.Namespace,
    project_root: Path,
    dataset_root: Path,
    output_root: Path,
) -> None:
    scoped_args = argparse.Namespace(**vars(args))
    scoped_args.dataset_root = str(dataset_root)
    dataset_entries = parse_dataset_entries(scoped_args, project_root)
    base_config = resolve_path(project_root, args.base_config)
    pretrained_checkpoint = resolve_path(project_root, args.pretrained_checkpoint)
    work_dir = resolve_path(project_root, args.work_dir)

    if not base_config.is_file():
        raise FileNotFoundError(f"Missing base config: {base_config}")
    if not pretrained_checkpoint.is_file():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {pretrained_checkpoint}")

    images_root = output_root / "_tmp_images"
    train_dir = output_root / "train2017"
    val_dir = output_root / "val2017"
    test_dir = output_root / "test2017"
    annotations_dir = output_root / "annotations"
    generated_config = output_root / "generated_configs" / "swimxyz_vitpose_huge.py"

    if output_root.exists():
        shutil.rmtree(output_root)
    images_root.mkdir(parents=True, exist_ok=True)

    all_samples: list[Sample] = []
    for entry in dataset_entries:
        all_samples.extend(
            extract_samples_from_entry(
                entry=entry,
                images_root=images_root,
                frame_step=args.frame_step,
                max_frames_per_video=args.max_frames_per_video,
                bbox_padding_x_ratio=args.bbox_padding_ratio
                if args.bbox_padding_ratio is not None
                else args.bbox_padding_x_ratio,
                bbox_padding_y_ratio=args.bbox_padding_ratio
                if args.bbox_padding_ratio is not None
                else args.bbox_padding_y_ratio,
                bbox_min_padding_px=args.bbox_min_padding_px,
                min_visible_keypoints=args.min_visible_keypoints,
                flip_y=args.flip_y,
            )
        )

    if not all_samples:
        raise RuntimeError("No valid SwimXYZ samples were extracted.")

    all_samples.sort(key=lambda sample: (sample.source_name, sample.frame_index))
    train_samples, val_samples, test_samples = split_samples(
        all_samples,
        args.val_ratio,
        args.test_ratio,
    )
    move_split_images(train_samples, train_dir)
    move_split_images(val_samples, val_dir)
    move_split_images(test_samples, test_dir)
    write_split_overlay_images(train_samples)
    write_split_overlay_images(val_samples)
    write_split_overlay_images(test_samples)

    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / "person_keypoints_train.json").write_text(
        json.dumps(build_coco_json(train_samples, "train2017"), indent=2),
        encoding="utf-8",
    )
    (annotations_dir / "person_keypoints_val.json").write_text(
        json.dumps(build_coco_json(val_samples, "val2017"), indent=2),
        encoding="utf-8",
    )
    (annotations_dir / "person_keypoints_test.json").write_text(
        json.dumps(build_coco_json(test_samples, "test2017"), indent=2),
        encoding="utf-8",
    )

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

    shutil.rmtree(images_root, ignore_errors=True)

    print(
        "Prepared SwimXYZ dataset with shared consolidated utilities: "
        f"{dataset_name_from_converted_root(dataset_root)}"
    )
    print(f"Converted dataset root: {dataset_root}")
    print(f"Dataset root: {output_root}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
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

    if args.dataset_entry:
        output_root = build_train_output_root(project_root, dataset_root, args.output_root)
        prepare_dataset(args, project_root, dataset_root, output_root)
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
        prepare_dataset(args, project_root, current_dataset_root, output_root)


if __name__ == "__main__":
    main()
