from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


DEFAULT_PROJECT_ROOT = Path("/home/albertosco/HPE")
DEFAULT_DATASET_ROOT = "data/intermediate"
DEFAULT_ANNOTATION_FILE = ""

COCO_KEYPOINT_NAMES = [
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
]

COCO_SKELETON = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
]

KEYPOINT_COLORS = {
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

SKELETON_COLOR = (255, 255, 255)
BBOX_COLOR = (180, 180, 0)
LABEL_TEXT_COLOR = (255, 255, 255)
LABEL_BG_COLOR = (20, 20, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw COCO keypoints on images from a VitPose-ready dataset and save "
            "the result next to each original image with suffix _with-KP."
        )
    )
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root or collection root used to infer a VitPose training dataset.",
    )
    parser.add_argument("--annotation-file", default=DEFAULT_ANNOTATION_FILE)
    parser.add_argument(
        "--image-root",
        default="",
        help="Optional directory containing the images referenced by the annotation file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images to render. Use 0 to process all.",
    )
    parser.add_argument(
        "--image-name-contains",
        default="",
        help="Optional substring filter applied to image file names.",
    )
    parser.add_argument(
        "--draw-bbox",
        action="store_true",
        help="Also draw the COCO bounding box.",
    )
    parser.add_argument(
        "--hide-labels",
        dest="draw_labels",
        action="store_false",
        default=True,
        help="Do not draw keypoint names next to the markers.",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Render keypoints as image_height - y for debugging bottom-origin labels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_with-KP images if they already exist.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def infer_annotation_file(dataset_root: Path) -> Path:
    direct_candidate = dataset_root / "annotations" / "person_keypoints_train.json"
    if direct_candidate.is_file():
        return direct_candidate

    vitposepp_direct_candidate = dataset_root / "annotations" / "COCO" / "person_keypoints_train.json"
    if vitposepp_direct_candidate.is_file():
        return vitposepp_direct_candidate

    nested_candidates = sorted(
        child / "_train_vitpose" / "annotations" / "person_keypoints_train.json"
        for child in dataset_root.iterdir()
        if child.is_dir()
        and (child / "_train_vitpose" / "annotations" / "person_keypoints_train.json").is_file()
    )
    nested_vitposepp_candidates = sorted(
        child / "_train_vitposepp" / "annotations" / "COCO" / "person_keypoints_train.json"
        for child in dataset_root.iterdir()
        if child.is_dir()
        and (child / "_train_vitposepp" / "annotations" / "COCO" / "person_keypoints_train.json").is_file()
    )
    if len(nested_candidates) == 1:
        return nested_candidates[0]
    if len(nested_vitposepp_candidates) == 1:
        return nested_vitposepp_candidates[0]

    legacy_candidates = sorted(
        child / "annotations" / "person_keypoints_train.json"
        for child in dataset_root.iterdir()
        if child.is_dir()
        and child.name.endswith("_train_vitpose")
        and (child / "annotations" / "person_keypoints_train.json").is_file()
    )
    legacy_vitposepp_candidates = sorted(
        child / "annotations" / "COCO" / "person_keypoints_train.json"
        for child in dataset_root.iterdir()
        if child.is_dir()
        and child.name.endswith("_train_vitposepp")
        and (child / "annotations" / "COCO" / "person_keypoints_train.json").is_file()
    )
    if len(legacy_candidates) == 1:
        return legacy_candidates[0]
    if len(legacy_vitposepp_candidates) == 1:
        return legacy_vitposepp_candidates[0]

    candidates = (
        nested_candidates
        + nested_vitposepp_candidates
        + legacy_candidates
        + legacy_vitposepp_candidates
    )
    if not candidates:
        raise FileNotFoundError(
            "Unable to infer an annotation file from dataset root: "
            f"{dataset_root}. Pass --annotation-file explicitly."
        )

    raise ValueError(
        "Multiple train annotation files found under "
        f"{dataset_root}. Pass --annotation-file explicitly."
    )


def infer_image_root(annotation_path: Path) -> Path:
    if annotation_path.parent.name == "annotations":
        parent = annotation_path.parent.parent
    elif annotation_path.parent.parent.name == "annotations":
        parent = annotation_path.parent.parent.parent
    else:
        parent = annotation_path.parent.parent
    filename = annotation_path.name
    if "_train" in filename:
        return parent / "train2017"
    if "_val" in filename:
        return parent / "val2017"
    if "_test" in filename:
        return parent / "test2017"
    raise ValueError(
        "Unable to infer image root from annotation filename. "
        "Pass --image-root explicitly."
    )


def annotation_to_keypoints(
    annotation: dict,
    image_height: int,
    flip_y: bool,
) -> list[tuple[float, float, float]]:
    values = annotation["keypoints"]
    keypoints = []
    for index in range(0, len(values), 3):
        x = values[index]
        y = values[index + 1]
        visibility = values[index + 2]
        if flip_y and visibility > 0:
            y = float(image_height) - float(y)
        keypoints.append((x, y, visibility))
    return keypoints


def draw_keypoints(
    image,
    keypoints: list[tuple[float, float, float]],
    draw_labels: bool,
) -> None:
    for start_idx, end_idx in COCO_SKELETON:
        start = keypoints[start_idx]
        end = keypoints[end_idx]
        if start[2] > 0 and end[2] > 0:
            cv2.line(
                image,
                (int(round(start[0])), int(round(start[1]))),
                (int(round(end[0])), int(round(end[1]))),
                SKELETON_COLOR,
                2,
                lineType=cv2.LINE_AA,
            )

    for index, (x, y, visibility) in enumerate(keypoints):
        if visibility <= 0:
            continue
        name = COCO_KEYPOINT_NAMES[index]
        color = KEYPOINT_COLORS[name]
        anchor = (int(round(x)), int(round(y)))
        cv2.circle(
            image,
            anchor,
            4,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        if not draw_labels:
            continue

        label_anchor = (anchor[0] + 6, anchor[1] - 6)
        text_size, baseline = cv2.getTextSize(
            name,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            1,
        )
        top_left = (label_anchor[0] - 2, label_anchor[1] - text_size[1] - 2)
        bottom_right = (
            label_anchor[0] + text_size[0] + 2,
            label_anchor[1] + baseline + 2,
        )
        cv2.rectangle(
            image,
            top_left,
            bottom_right,
            LABEL_BG_COLOR,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image,
            name,
            label_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            LABEL_TEXT_COLOR,
            1,
            lineType=cv2.LINE_AA,
        )


def draw_bbox(image, bbox: list[float]) -> None:
    x, y, w, h = bbox
    start = (int(round(x)), int(round(y)))
    end = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(image, start, end, BBOX_COLOR, 2, lineType=cv2.LINE_AA)


def flipped_bbox(bbox: list[float], image_height: int) -> list[float]:
    x, y, w, h = bbox
    return [x, float(image_height) - float(y) - float(h), w, h]


def output_path_for(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_with-KP{image_path.suffix}")


def main() -> None:
    args = parse_args()
    project_root = resolve_path(Path.cwd(), args.project_root)
    dataset_root = resolve_path(project_root, args.dataset_root)
    annotation_path = (
        resolve_path(project_root, args.annotation_file)
        if args.annotation_file
        else infer_annotation_file(dataset_root)
    )
    image_root = (
        resolve_path(project_root, args.image_root)
        if args.image_root
        else infer_image_root(annotation_path)
    )

    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image root: {image_root}")

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images_by_id = {item["id"]: item for item in payload.get("images", [])}
    annotations_by_image_id: dict[int, list[dict]] = {}
    for annotation in payload.get("annotations", []):
        annotations_by_image_id.setdefault(annotation["image_id"], []).append(annotation)

    processed = 0
    skipped_existing = 0
    skipped_filter = 0
    for image_id, image_info in images_by_id.items():
        image_name = image_info["file_name"]
        if args.image_name_contains and args.image_name_contains not in image_name:
            skipped_filter += 1
            continue

        source_path = image_root / image_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source image: {source_path}")

        target_path = output_path_for(source_path)
        if target_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        image = cv2.imread(str(source_path))
        if image is None:
            raise RuntimeError(f"Unable to read image: {source_path}")

        for annotation in annotations_by_image_id.get(image_id, []):
            keypoints = annotation_to_keypoints(
                annotation=annotation,
                image_height=image_info["height"],
                flip_y=args.flip_y,
            )
            draw_keypoints(image, keypoints, args.draw_labels)
            if args.draw_bbox:
                bbox = (
                    flipped_bbox(annotation["bbox"], image_info["height"])
                    if args.flip_y
                    else annotation["bbox"]
                )
                draw_bbox(image, bbox)

        ok = cv2.imwrite(str(target_path), image)
        if not ok:
            raise RuntimeError(f"Unable to write annotated image: {target_path}")

        processed += 1
        if args.limit > 0 and processed >= args.limit:
            break

    print("Keypoint visualization completed.")
    print(f"Annotation file: {annotation_path}")
    print(f"Image root: {image_root}")
    print(f"Images written: {processed}")
    print(f"Skipped by filter: {skipped_filter}")
    print(f"Skipped existing: {skipped_existing}")


if __name__ == "__main__":
    main()
