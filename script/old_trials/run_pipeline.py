from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


DEFAULT_POSE_CONFIG = (
    "src/vitpose_base/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/"
    "coco/ViTPose_huge_coco_256x192.py"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".webm"}
POSE_KEYPOINTS = [
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
COCO_INDEX = {
    "Nose": 0,
    "LEye": 1,
    "REye": 2,
    "LEar": 3,
    "REar": 4,
    "LShoulder": 5,
    "RShoulder": 6,
    "LElbow": 7,
    "RElbow": 8,
    "LWrist": 9,
    "RWrist": 10,
    "LHip": 11,
    "RHip": 12,
    "LKnee": 13,
    "RKnee": 14,
    "LAnkle": 15,
    "RAnkle": 16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO person detection followed by ViTPose pose estimation."
    )
    parser.add_argument("--input", required=True, help="Input image, MP4, or WEBM path.")
    parser.add_argument("--project-root", default=".", help="Project root path.")
    parser.add_argument("--yolo-model", default="models/detection/yolo26x.pt")
    parser.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--pose-checkpoint", default="models/pose/coco.pth")
    parser.add_argument("--output-dir", default="data/output/pipeline")
    parser.add_argument("--intermediate-dir", default="data/intermediate/pipeline")
    parser.add_argument(
        "--annotation-file",
        default="",
        help=(
            "Optional COCO annotation file. If a matching bbox is present for an "
            "image, use it directly and skip YOLO for that image."
        ),
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clean-intermediate", action="store_true")
    return parser.parse_args()


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def ensure_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing))


def boxes_xyxy_to_xywh(boxes_xyxy) -> list[list[float]]:
    boxes_xywh = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = [float(v) for v in box]
        x = max(0.0, x1)
        y = max(0.0, y1)
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        boxes_xywh.append([x, y, w, h])
    return boxes_xywh


def make_coco_json(image_path: Path, boxes_xywh: list[list[float]], output_path: Path) -> int:
    image = Image.open(image_path)
    width, height = image.size

    annotations = []
    for idx, box in enumerate(boxes_xywh, start=1):
        x, y, w, h = box
        annotations.append(
            {
                "id": idx,
                "image_id": 1,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
        )

    coco = {
        "images": [
            {
                "id": 1,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        ],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, indent=2))
    return len(annotations)


def load_annotation_index(annotation_path: Optional[Path]) -> dict[str, list[list[float]]]:
    if annotation_path is None:
        return {}

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images_by_id = {item["id"]: item["file_name"] for item in payload.get("images", [])}
    boxes_by_name: dict[str, list[list[float]]] = {}

    for annotation in payload.get("annotations", []):
        image_name = images_by_id.get(annotation.get("image_id"))
        bbox = annotation.get("bbox")
        if image_name is None or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x, y, w, h = [float(value) for value in bbox]
        if w <= 0 or h <= 0:
            continue
        boxes_by_name.setdefault(Path(image_name).name, []).append([x, y, w, h])

    return boxes_by_name


def annotation_boxes_for_image(
    boxes_by_name: dict[str, list[list[float]]],
    image_path: Path,
) -> list[list[float]]:
    return [box[:] for box in boxes_by_name.get(image_path.name, [])]


def detect_people(model: YOLO, image_path: Path, conf: float):
    results = model.predict(
        source=str(image_path),
        classes=[0],
        conf=conf,
        save=False,
        verbose=False,
    )
    return results[0].boxes.xyxy.cpu().numpy()


def load_pose_model(vitpose_root: Path, pose_config: Path, pose_checkpoint: Path, device: str):
    sys.path.insert(0, str(vitpose_root))
    from mmpose.apis import init_pose_model

    return init_pose_model(str(pose_config), str(pose_checkpoint), device=device)


def run_vitpose(pose_model, image_path: Path, boxes_xywh: list[list[float]], output_image: Path):
    from mmpose.apis import inference_top_down_pose_model, vis_pose_result

    person_results = [{"bbox": np.array(box, dtype=np.float32)} for box in boxes_xywh]
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        str(image_path),
        person_results,
        bbox_thr=None,
        format="xywh",
    )
    output_image.parent.mkdir(parents=True, exist_ok=True)
    vis_pose_result(
        pose_model,
        str(image_path),
        pose_results,
        radius=4,
        thickness=1,
        out_file=str(output_image),
    )
    return pose_results


def save_bbox_crop(image_path: Path, box_xyxy, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Unable to read image for bbox crop: {image_path}")

    x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy]
    height, width = image.shape[:2]

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    crop = image[y1:y2, x1:x2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), crop)


def csv_header(include_frame: bool = False) -> str:
    columns = []
    if include_frame:
        columns.extend(["Frame", "Person"])
    for keypoint in POSE_KEYPOINTS:
        columns.extend([f"{keypoint}.x", f"{keypoint}.y", f"{keypoint}.z"])
    return ";".join(columns) + ";"


def format_decimal(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.2f}".replace(".", ",")


def average_keypoint(keypoints, left_name: str, right_name: str) -> Optional[Tuple[float, float, float]]:
    left = keypoints[COCO_INDEX[left_name]]
    right = keypoints[COCO_INDEX[right_name]]
    return (
        float((left[0] + right[0]) / 2),
        float((left[1] + right[1]) / 2),
        float((left[2] + right[2]) / 2),
    )


def get_keypoint(keypoints, name: str) -> Optional[Tuple[float, float, float]]:
    if name == "Neck":
        return average_keypoint(keypoints, "LShoulder", "RShoulder")
    if name == "MidHip":
        return average_keypoint(keypoints, "LHip", "RHip")
    if name not in COCO_INDEX:
        return None
    point = keypoints[COCO_INDEX[name]]
    return float(point[0]), float(point[1]), float(point[2])


def pose_results_to_csv_rows(pose_results, frame_index: Optional[int] = None) -> list[str]:
    rows = []
    for person_index, result in enumerate(pose_results, start=1):
        values = []
        if frame_index is not None:
            values.extend([str(frame_index), str(person_index)])
        keypoints = result["keypoints"]
        for name in POSE_KEYPOINTS:
            point = get_keypoint(keypoints, name)
            if point is None:
                values.extend(["", "", ""])
            else:
                values.extend(format_decimal(value) for value in point)
        rows.append(";".join(values) + ";")
    return rows


def write_csv(csv_path: Path, rows: list[str], include_frame: bool = False) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(csv_header(include_frame) + "\n" + "\n".join(rows) + ("\n" if rows else ""))


def process_image(
    image_path: Path,
    yolo_model: Optional[YOLO],
    pose_model,
    args: argparse.Namespace,
    paths: dict[str, Path],
    annotation_boxes_by_name: dict[str, list[list[float]]],
    frame_index: Optional[int] = None,
) -> int:
    coco_json = paths["intermediate_dir"] / f"{image_path.stem}_yolo_coco.json"
    boxes_xywh = annotation_boxes_for_image(annotation_boxes_by_name, image_path)
    if boxes_xywh:
        boxes_xyxy = np.array(
            [[x, y, x + w, y + h] for x, y, w, h in boxes_xywh],
            dtype=np.float32,
        )
    else:
        if yolo_model is None:
            raise RuntimeError(
                "YOLO model is required when no bounding box is available in the annotation file."
            )
        boxes_xyxy = detect_people(yolo_model, image_path, args.conf)
        boxes_xywh = boxes_xyxy_to_xywh(boxes_xyxy)
    count = make_coco_json(image_path, boxes_xywh, coco_json)

    output_image = paths["output_dir"] / image_path.name
    bbox_image = output_image.with_name(f"{output_image.stem}_bbox{output_image.suffix}")
    csv_path = output_image.with_suffix(".csv")

    if count:
        save_bbox_crop(image_path, boxes_xyxy[0], bbox_image)
        pose_results = run_vitpose(
            pose_model=pose_model,
            image_path=image_path,
            boxes_xywh=boxes_xywh,
            output_image=output_image,
        )
        rows = pose_results_to_csv_rows(pose_results, frame_index=frame_index)
        write_csv(csv_path, rows, include_frame=frame_index is not None)
    else:
        output_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, output_image)
        write_csv(csv_path, [], include_frame=frame_index is not None)

    if args.clean_intermediate:
        coco_json.unlink(missing_ok=True)

    return count


def extract_video_frames(video_path: Path, frames_dir: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_dir.mkdir(parents=True, exist_ok=True)

    index = 0
    with tqdm(total=frame_count or None, desc="Extracting frames", unit="frame") as progress:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.imwrite(str(frames_dir / f"frame_{index:06d}.jpg"), frame)
            index += 1
            progress.update(1)

    capture.release()
    return index, fps


def build_video(frames_dir: Path, output_video: Path, fps: float) -> None:
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No annotated frames found in {frames_dir}")

    first = cv2.imread(str(frames[0]))
    height, width = first.shape[:2]
    output_video.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame_path in tqdm(frames, desc="Writing video", unit="frame"):
        writer.write(cv2.imread(str(frame_path)))
    writer.release()


def process_video(
    video_path: Path,
    yolo_model: Optional[YOLO],
    pose_model,
    args: argparse.Namespace,
    paths: dict[str, Path],
    annotation_boxes_by_name: dict[str, list[list[float]]],
) -> None:
    work_dir = paths["intermediate_dir"] / video_path.stem
    frames_dir = work_dir / "frames"
    annotated_dir = work_dir / "annotated"

    if work_dir.exists():
        shutil.rmtree(work_dir)
    frames_total, fps = extract_video_frames(video_path, frames_dir)

    detected_frames = 0
    frames = sorted(frames_dir.glob("*.jpg"))
    video_rows = []
    for frame_path in tqdm(frames, desc="Running YOLO + ViTPose", unit="frame"):
        frame_output_dir = annotated_dir / frame_path.stem
        frame_paths = dict(paths)
        frame_paths["output_dir"] = frame_output_dir
        count = process_image(
            frame_path,
            yolo_model,
            pose_model,
            args,
            frame_paths,
            annotation_boxes_by_name,
            frame_index=int(frame_path.stem.split("_")[-1]),
        )
        if count:
            detected_frames += 1
            frame_csv = frame_output_dir / f"{frame_path.stem}.csv"
            if frame_csv.exists():
                video_rows.extend(frame_csv.read_text().splitlines()[1:])

    collected_dir = work_dir / "collected"
    collected_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in frames:
        annotated_frame = annotated_dir / frame_path.stem / frame_path.name
        source = annotated_frame if annotated_frame.exists() else frame_path
        shutil.copy2(source, collected_dir / frame_path.name)

    output_video = paths["output_dir"] / f"{video_path.stem}_pose.mp4"
    build_video(collected_dir, output_video, fps)
    write_csv(output_video.with_suffix(".csv"), video_rows, include_frame=True)

    if args.clean_intermediate:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"Frames: {frames_total}")
    print(f"Frames with detections: {detected_frames}")
    print(f"Output video: {output_video}")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    input_path = resolve_path(project_root, args.input)
    annotation_path = resolve_path(project_root, args.annotation_file) if args.annotation_file else None
    paths = {
        "output_dir": resolve_path(project_root, args.output_dir),
        "intermediate_dir": resolve_path(project_root, args.intermediate_dir),
        "pose_config": resolve_path(project_root, args.pose_config),
        "pose_checkpoint": resolve_path(project_root, args.pose_checkpoint),
        "vitpose_root": project_root / "src/vitpose_base",
        "yolo_model": resolve_path(project_root, args.yolo_model),
    }

    required_paths = [input_path, paths["pose_config"], paths["pose_checkpoint"], paths["vitpose_root"]]
    if annotation_path is not None:
        required_paths.append(annotation_path)
    ensure_paths(required_paths)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["intermediate_dir"].mkdir(parents=True, exist_ok=True)

    annotation_boxes_by_name = load_annotation_index(annotation_path)
    suffix = input_path.suffix.lower()
    needs_yolo = annotation_path is None
    if suffix in IMAGE_EXTENSIONS:
        needs_yolo = not annotation_boxes_for_image(annotation_boxes_by_name, input_path)
    else:
        # Video frames extracted on the fly usually do not match dataset annotation names.
        needs_yolo = True

    yolo_model = None
    if needs_yolo:
        ensure_paths([paths["yolo_model"]])
        yolo_model = YOLO(str(paths["yolo_model"]))
    pose_model = load_pose_model(
        paths["vitpose_root"],
        paths["pose_config"],
        paths["pose_checkpoint"],
        args.device,
    )

    if suffix in IMAGE_EXTENSIONS:
        boxes = process_image(
            input_path,
            yolo_model,
            pose_model,
            args,
            paths,
            annotation_boxes_by_name,
        )
        print(f"Persons processed: {boxes}")
        if annotation_boxes_for_image(annotation_boxes_by_name, input_path):
            print("Bounding boxes source: annotation file")
        else:
            print("Bounding boxes source: YOLO detection")
        print(f"Output directory: {paths['output_dir']}")
    elif suffix in VIDEO_EXTENSIONS:
        process_video(
            input_path,
            yolo_model,
            pose_model,
            args,
            paths,
            annotation_boxes_by_name,
        )
    else:
        raise ValueError(f"Unsupported input extension: {input_path.suffix}")


if __name__ == "__main__":
    main()
