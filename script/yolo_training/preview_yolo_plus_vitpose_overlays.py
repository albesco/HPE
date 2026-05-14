from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "script"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pose_overlay_utils import (
    bbox_xywh_to_xyxy,
    draw_bbox_xyxy,
    draw_prediction_keypoints,
    run_pose_prediction,
)


DEFAULT_YOLO_MODEL = (
    "runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox_aniso_20x25y_min15_5ep/weights/best.pt"
)
DEFAULT_VITPOSE_CONFIG = (
    "data/intermediate/Side_above_water/_train_vitposepp_swap_ears/generated_configs/"
    "swimxyz_vitposepp_huge_single_head_swap_ears.py"
)
DEFAULT_VITPOSE_CHECKPOINT = "runs/vitposepp_side_above_water_aniso_20x25_min15/epoch_1.pth"
DEFAULT_SOURCE_DIR = "data/intermediate/Side_above_water/_yolo_detection/images/test"
DEFAULT_OUTPUT_DIR = "data/output/preview/yolo_plus_vitpose_epoch1_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay YOLO bbox prediction + VitPose++ keypoint prediction on random images."
    )
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--vitpose-config", default=DEFAULT_VITPOSE_CONFIG)
    parser.add_argument("--vitpose-checkpoint", default=DEFAULT_VITPOSE_CHECKPOINT)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--keypoint-score-threshold", type=float, default=0.3)
    return parser.parse_args()


def image_paths(source_dir: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png"}
    return sorted(path for path in source_dir.iterdir() if path.suffix.lower() in suffixes)


def xyxy_to_xywh(xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def main() -> None:
    args = parse_args()

    yolo_model_path = Path(args.yolo_model).expanduser().resolve()
    vitpose_config_path = Path(args.vitpose_config).expanduser().resolve()
    vitpose_checkpoint_path = Path(args.vitpose_checkpoint).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not yolo_model_path.is_file():
        raise FileNotFoundError(f"Missing YOLO model: {yolo_model_path}")
    if not vitpose_config_path.is_file():
        raise FileNotFoundError(f"Missing VitPose config: {vitpose_config_path}")
    if not vitpose_checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing VitPose checkpoint: {vitpose_checkpoint_path}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source dir: {source_dir}")

    candidates = image_paths(source_dir)
    if len(candidates) < args.count:
        raise RuntimeError(f"Requested {args.count} images, found {len(candidates)} in {source_dir}")

    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: args.count]

    output_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(yolo_model_path))

    # Lazy import to keep ultralytics startup fast when not needed.
    from mmpose.apis import init_pose_model

    vitpose = init_pose_model(str(vitpose_config_path), str(vitpose_checkpoint_path), device="cuda:0")
    dataset_info = vitpose.cfg.data.get("val", {}).get("dataset_info", None)

    saved = 0
    for image_path in selected:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        # YOLO bbox prediction (best box by confidence).
        results = yolo.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            continue

        best_index = int(boxes.conf.argmax().item())
        bbox_xyxy = boxes.xyxy[best_index].detach().cpu().tolist()
        bbox_xywh = xyxy_to_xywh(bbox_xyxy)

        # VitPose++ keypoint prediction inside YOLO bbox.
        predicted_keypoints = run_pose_prediction(vitpose, image_path, bbox_xywh, dataset_info)

        overlay = image.copy()
        draw_bbox_xyxy(overlay, bbox_xyxy, (0, 0, 255))
        draw_prediction_keypoints(overlay, predicted_keypoints, args.keypoint_score_threshold)

        # Also draw the bbox passed to VitPose (xywh -> xyxy) in a different color for sanity.
        draw_bbox_xyxy(overlay, bbox_xywh_to_xyxy(bbox_xywh), (255, 0, 0))

        cv2.imwrite(str(output_dir / image_path.name), overlay)
        saved += 1

    print(f"Saved {saved} YOLO+VitPose overlays to: {output_dir}")


if __name__ == "__main__":
    main()
