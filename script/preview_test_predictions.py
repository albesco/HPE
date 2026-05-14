from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2

from pose_overlay_utils import (
    bbox_xywh_to_xyxy,
    dataset_info_from_model,
    resolve_path,
    run_pose_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate prediction overlays (predicted keypoints) for N test images "
            "from a prepared SwimXYZ->VitPose++ dataset."
        )
    )
    parser.add_argument(
        "--config",
        default="data/intermediate/Side_above_water/_train_vitposepp_swap_ears/generated_configs/"
        "swimxyz_vitposepp_huge_single_head_swap_ears.py",
        help="MMPose/VitPose++ config to use for inference.",
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/vitposepp_single_head_subset_xyz_swap_ears/best_AP_epoch_10.pth",
        help="Checkpoint .pth to load.",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/intermediate/Side_above_water/_train_vitposepp_swap_ears",
        help="Prepared dataset root (contains test2017/ and annotations/).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("test", "val", "train"),
        help="Dataset split to sample images from.",
    )
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--output-dir", default="data/output/preview")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic ordering seed (shuffles candidates).",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle; take images in dataset order (useful for 'first N').",
    )
    return parser.parse_args()


def coco_paths(dataset_root: Path, split: str) -> tuple[Path, Path]:
    ann_name = f"person_keypoints_{split}.json"
    images_dir = dataset_root / f"{split}2017"
    ann_path = dataset_root / "annotations" / ann_name
    return ann_path, images_dir


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    dataset_root = resolve_path(args.dataset_root)
    output_dir = resolve_path(args.output_dir)

    split = args.split
    ann_path, images_dir = coco_paths(dataset_root, split)
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing COCO annotation file: {ann_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    from mmpose.apis import init_pose_model, vis_pose_result

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    if not images:
        raise RuntimeError(f"No images found in {ann_path}")
    if not annotations:
        raise RuntimeError(f"No annotations found in {ann_path}")

    anns_by_image_id: dict[int, list[dict]] = {}
    for ann in annotations:
        anns_by_image_id.setdefault(int(ann["image_id"]), []).append(ann)

    images_sorted = list(images)
    if not args.no_shuffle:
        # Deterministic shuffle without importing numpy.
        rng = (args.seed * 1103515245 + 12345) & 0x7FFFFFFF
        for idx in range(len(images_sorted) - 1, 0, -1):
            rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
            jdx = rng % (idx + 1)
            images_sorted[idx], images_sorted[jdx] = images_sorted[jdx], images_sorted[idx]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_subdir = output_dir / f"side_above_water_pred_{split}_{timestamp}"
    out_subdir.mkdir(parents=True, exist_ok=True)

    model = init_pose_model(str(config_path), str(checkpoint_path), device="cuda:0")
    dataset_info = dataset_info_from_model(model)

    saved = 0
    for img_info in images_sorted:
        if saved >= args.num_images:
            break
        image_id = int(img_info["id"])
        file_name = img_info["file_name"]
        image_path = images_dir / file_name
        if not image_path.is_file():
            continue

        img_anns = anns_by_image_id.get(image_id, [])
        if not img_anns:
            continue

        # Use GT bbox for the (single) person.
        ann = img_anns[0]
        bbox_xywh = ann.get("bbox", None)
        if not bbox_xywh or len(bbox_xywh) != 4:
            continue
        predicted_keypoints = run_pose_prediction(model, image_path, bbox_xywh, dataset_info)
        bbox_xyxy = bbox_xywh_to_xyxy(bbox_xywh)

        pose_results = [
            {
                "bbox": bbox_xyxy,
                "keypoints": predicted_keypoints,
            }
        ]

        # Load original BGR image, draw predicted keypoints.
        img = cv2.imread(str(image_path))
        vis_img = vis_pose_result(
            model,
            img,
            pose_results,
            radius=3,
            thickness=2,
            show=False,
            dataset_info=dataset_info,
        )

        out_path = out_subdir / f"pred_{saved:02d}__{image_path.name}"
        cv2.imwrite(str(out_path), vis_img)
        saved += 1

    if saved == 0:
        raise RuntimeError("No preview images were generated (no valid test samples found).")

    print(f"Saved {saved} preview overlays to: {out_subdir}")


if __name__ == "__main__":
    main()
