from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from pose_overlay_utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_DATASET_ROOT,
    draw_bbox_xywh,
    draw_coco_keypoints,
    draw_prediction_keypoints,
    dataset_info_from_model,
    resolve_path,
    run_pose_prediction,
    split_paths,
)


DEFAULT_OUTPUT_DIR = "data/output/preview/gt_vs_pred_keypoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay GT and predicted keypoints on the same images using GT bboxes."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--prediction-score-threshold", type=float, default=0.2)
    return parser.parse_args()


def draw_legend(image: np.ndarray) -> None:
    cv2.rectangle(image, (12, 12), (460, 82), (0, 0, 0), -1)
    cv2.rectangle(image, (12, 12), (460, 82), (255, 255, 255), 1)
    cv2.putText(image, "GT bbox/keypoints: green", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(
        image,
        "Pred keypoints: magenta/orange",
        (24, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 255),
        2,
    )


def output_path(output_dir: Path, source_name: str) -> Path:
    source_path = Path(source_name)
    return output_dir / f"{source_path.stem}_GT_PRED{source_path.suffix}"


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
    annotations_by_image_id: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        annotations_by_image_id.setdefault(int(ann["image_id"]), []).append(ann)

    output_dir.mkdir(parents=True, exist_ok=True)

    model = init_pose_model(str(config_path), str(checkpoint_path), device="cuda:0")
    dataset_info = dataset_info_from_model(model)

    saved = 0
    for image_info in coco.get("images", []):
        if args.limit > 0 and saved >= args.limit:
            break

        image_id = int(image_info["id"])
        image_name = image_info["file_name"]
        image_path = images_dir / image_name
        image_annotations = annotations_by_image_id.get(image_id, [])
        if not image_path.is_file() or not image_annotations:
            continue

        ann = image_annotations[0]
        bbox_xywh = ann.get("bbox")
        gt_keypoints = ann.get("keypoints")
        if not bbox_xywh or not gt_keypoints:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        predicted_keypoints = run_pose_prediction(model, image_path, bbox_xywh, dataset_info)

        draw_bbox_xywh(image, bbox_xywh, (0, 255, 0))
        draw_coco_keypoints(
            image,
            gt_keypoints,
            color=(0, 255, 0),
            line_color=(0, 210, 0),
            draw_indices=True,
            index_offset=(-12, -8),
        )
        draw_prediction_keypoints(
            image,
            predicted_keypoints,
            args.prediction_score_threshold,
            draw_indices=True,
            index_offset=(7, 13),
        )
        draw_legend(image)

        cv2.imwrite(str(output_path(output_dir, image_name)), image)
        saved += 1

    print(f"Saved {saved} GT-vs-pred overlays to: {output_dir}")


if __name__ == "__main__":
    main()
