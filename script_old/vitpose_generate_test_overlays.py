from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
YOLO_SCRIPT_DIR = SCRIPT_DIR / 'yolo_training'
if str(YOLO_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_SCRIPT_DIR))

from pose_overlay_utils import dataset_info_from_model, run_pose_prediction
from yolo_detection_utils import select_top_detection_box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data/intermediate/Side_above_water/_train_canonical"
DEFAULT_CONFIG = PROJECT_ROOT / "data/intermediate/Side_above_water/_train_canonical/generated_configs/swimxyz_vitposepp_huge_grid_winner_resume.py"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs/vitposepp_side_above_water_grid_winner_resume/best_AP_epoch_24.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/intermediate/Side_above_water/_train_canonical/reports/test_overlays/vitposepp_grid_winner_best_AP_epoch_24"
DEFAULT_YOLO_MODEL = PROJECT_ROOT / "runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VitPose++ keypoint overlays for the test split.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def ann_and_image_paths(dataset_root: Path) -> tuple[Path, Path]:
    return dataset_root / "annotations" / "person_keypoints_test.json", dataset_root / "test2017"


def xyxy_to_xywh(bbox_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def main() -> None:
    args = parse_args()
    ann_path, images_dir = ann_and_image_paths(args.dataset_root.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from mmpose.apis import init_pose_model
    from mmpose.core.visualization import imshow_keypoints
    import torch

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = init_pose_model(str(args.config.resolve()), str(args.checkpoint.resolve()), device=device)
    yolo = YOLO(str(args.yolo_model.resolve()))
    dataset_info = dataset_info_from_model(model)

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    anns_by_image_id: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image_id.setdefault(int(ann["image_id"]), []).append(ann)

    images = sorted(coco.get("images", []), key=lambda item: int(item["id"]))
    metadata = []
    for index, image_info in enumerate(images, start=1):
        image_id = int(image_info["id"])
        image_path = images_dir / image_info["file_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        pose_results = []
        yolo_results = yolo.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
        boxes = yolo_results[0].boxes if yolo_results else None
        selected_box = select_top_detection_box(boxes)
        if selected_box is not None:
            bbox_xyxy = selected_box.xyxy
            bbox_xywh = xyxy_to_xywh(bbox_xyxy)
            predicted_keypoints = run_pose_prediction(model, image_path, bbox_xywh, dataset_info)
            pose_results.append(predicted_keypoints)

        if pose_results:
            overlay = imshow_keypoints(
                image,
                pose_results,
                skeleton=dataset_info.skeleton if dataset_info is not None else None,
                pose_kpt_color=dataset_info.pose_kpt_color if dataset_info is not None else None,
                pose_link_color=dataset_info.pose_link_color if dataset_info is not None else None,
                radius=3,
                thickness=2,
            )
        else:
            overlay = image

        out_path = output_dir / image_info["file_name"]
        cv2.imwrite(str(out_path), overlay)
        metadata.append(
            {
                "image_id": image_id,
                "file_name": image_info["file_name"],
                "num_people": len(pose_results),
                "bbox_source": "yolo26x_detection_top1",
                "output_path": str(out_path),
            }
        )
        if index % 100 == 0:
            print(f"Processed {index}/{len(images)} test images", flush=True)

    manifest_path = output_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "manifest": str(manifest_path), "num_images": len(metadata)}, indent=2))


if __name__ == "__main__":
    main()
