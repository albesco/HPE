from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from pose_overlay_utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_DATASET_ROOT,
    bbox_xywh_to_xyxy,
    dataset_info_from_model,
    draw_bbox_xywh,
    draw_bbox_xyxy,
    draw_coco_keypoints,
    draw_prediction_keypoints,
    resolve_path,
    run_pose_prediction,
    split_paths,
)

DEFAULT_OUTPUT_DIR = "data/output/preview/comparate"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paired GT (_T) and prediction (_P) keypoint/bbox overlays for a COCO split."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images to process. Use 0 for the whole split.",
    )
    parser.add_argument("--prediction-score-threshold", type=float, default=0.2)
    return parser.parse_args()


def output_path(output_dir: Path, source_name: str, suffix: str) -> Path:
    source_path = Path(source_name)
    return output_dir / f"{source_path.stem}_{suffix}{source_path.suffix}"


def main() -> None:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir)
    ann_path, images_dir = split_paths(dataset_root, args.split)

    for required_file in (config_path, checkpoint_path, ann_path):
        if not required_file.is_file():
            raise FileNotFoundError(required_file)
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)

    from mmpose.apis import init_pose_model

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    anns_by_image_id: dict[int, list[dict]] = {}
    for ann in annotations:
        anns_by_image_id.setdefault(int(ann["image_id"]), []).append(ann)

    output_dir.mkdir(parents=True, exist_ok=True)

    model = init_pose_model(str(config_path), str(checkpoint_path), device="cuda:0")
    dataset_info = dataset_info_from_model(model)

    processed = 0
    for image_info in images:
        if args.limit > 0 and processed >= args.limit:
            break

        image_id = int(image_info["id"])
        image_name = image_info["file_name"]
        image_path = images_dir / image_name
        image_annotations = anns_by_image_id.get(image_id, [])
        if not image_path.is_file() or not image_annotations:
            continue

        ann = image_annotations[0]
        bbox_xywh = ann.get("bbox")
        label_keypoints = ann.get("keypoints")
        if not bbox_xywh or not label_keypoints:
            continue

        label_image = cv2.imread(str(image_path))
        pred_image = cv2.imread(str(image_path))
        if label_image is None or pred_image is None:
            continue

        draw_bbox_xywh(label_image, bbox_xywh, (0, 255, 0))
        draw_coco_keypoints(label_image, label_keypoints, color=(0, 255, 255), line_color=(255, 191, 0))

        predicted_keypoints = run_pose_prediction(model, image_path, bbox_xywh, dataset_info)
        pred_bbox = draw_prediction_keypoints(pred_image, predicted_keypoints, args.prediction_score_threshold)
        if pred_bbox is not None:
            draw_bbox_xyxy(pred_image, pred_bbox, (0, 0, 255))

        cv2.imwrite(str(output_path(output_dir, image_name, "T")), label_image)
        cv2.imwrite(str(output_path(output_dir, image_name, "P")), pred_image)
        processed += 1

    print(f"Saved {processed * 2} overlays for {processed} {args.split} images to: {output_dir}")


if __name__ == "__main__":
    main()
