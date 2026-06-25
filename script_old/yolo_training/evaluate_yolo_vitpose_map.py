from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from yolo_detection_utils import select_top_detection_box
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "script"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pose_overlay_utils import (  # noqa: E402
    bbox_xywh_to_xyxy,
    dataset_info_from_model,
    draw_bbox_xyxy,
    run_pose_prediction,
)

DEFAULT_DATASET_ROOT = "data/intermediate/Side_above_water/_train_canonical"
DEFAULT_YOLO_MODEL = "runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt"
DEFAULT_VITPOSE_CONFIG = (
    "data/intermediate/Side_above_water/_train_canonical/generated_configs/"
    "swimxyz_vitposepp_huge.py"
)
DEFAULT_VITPOSE_CHECKPOINT = "runs/vitposepp_side_above_water_aniso_20x25_min15/best_AP_epoch_35.pth"
DEFAULT_OUTPUT_ROOT = "data/output/experiments/YoloVitPose_mAP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO bbox + VitPose++ keypoints with COCO mAP.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--vitpose-config", default=DEFAULT_VITPOSE_CONFIG)
    parser.add_argument("--vitpose-checkpoint", default=DEFAULT_VITPOSE_CHECKPOINT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--overlay-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--keypoint-score-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def xyxy_to_xywh(xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def coco_paths(dataset_root: Path, split: str) -> tuple[Path, Path]:
    return dataset_root / "annotations" / f"person_keypoints_{split}.json", dataset_root / f"{split}2017"


def category_id(coco: dict) -> int:
    categories = coco.get("categories") or []
    for cat in categories:
        if cat.get("name") == "person":
            return int(cat.get("id", 1))
    return int(categories[0].get("id", 1)) if categories else 1


def best_yolo_box(yolo: YOLO, image_path: Path, imgsz: int, conf: float) -> tuple[list[float], float] | None:
    results = yolo.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)
    boxes = results[0].boxes if results else None
    if boxes is None or len(boxes) == 0:
        return None
    selected_box = select_top_detection_box(boxes)
    if selected_box is None:
        return None
    return selected_box.xyxy, selected_box.confidence


def prediction_score(keypoints: np.ndarray, bbox_score: float, threshold: float) -> float:
    valid = keypoints[:, 2] >= threshold
    if not np.any(valid):
        return 0.0
    return float(bbox_score * float(keypoints[valid, 2].mean()))


def flatten_keypoints(keypoints: np.ndarray) -> list[float]:
    flat: list[float] = []
    for x, y, score in keypoints:
        flat.extend([float(x), float(y), float(score)])
    return flat


def save_overlay(
    model,
    image_path: Path,
    output_path: Path,
    yolo_bbox_xyxy: list[float],
    predicted_keypoints: np.ndarray,
    dataset_info,
) -> None:
    from mmpose.apis import vis_pose_result

    image = cv2.imread(str(image_path))
    if image is None:
        return
    pose_results = [{"bbox": yolo_bbox_xyxy, "keypoints": predicted_keypoints}]
    overlay = vis_pose_result(
        model,
        image,
        pose_results,
        radius=3,
        thickness=2,
        show=False,
        dataset_info=dataset_info,
    )
    draw_bbox_xyxy(overlay, yolo_bbox_xyxy, (0, 0, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)

def evaluate_coco(ann_path: Path, results_path: Path) -> dict[str, float]:
    coco_gt = COCO(str(ann_path))
    coco_dt = coco_gt.loadRes(str(results_path))
    evaluator = COCOeval(coco_gt, coco_dt, "keypoints")
    evaluator.params.useSegm = None
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ["AP", "AP50", "AP75", "AP_M", "AP_L", "AR", "AR50", "AR75", "AR_M", "AR_L"]
    return {name: float(value) for name, value in zip(names, evaluator.stats)}


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    ann_path, images_dir = coco_paths(dataset_root, args.split)
    yolo_model_path = Path(args.yolo_model).expanduser().resolve()
    vitpose_config_path = Path(args.vitpose_config).expanduser().resolve()
    vitpose_checkpoint_path = Path(args.vitpose_checkpoint).expanduser().resolve()

    for required in [ann_path, yolo_model_path, vitpose_config_path, vitpose_checkpoint_path]:
        if not required.is_file():
            raise FileNotFoundError(required)
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root).expanduser().resolve() / f"{args.split}_{timestamp}"
    overlay_dir = run_dir / "overlays"
    run_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    cat_id = category_id(coco)
    images = sorted(coco.get("images", []), key=lambda item: int(item["id"]))
    anns_by_image_id: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image_id.setdefault(int(ann["image_id"]), []).append(ann)

    rng = random.Random(args.seed)
    overlay_ids = {int(img["id"]) for img in rng.sample(images, min(args.overlay_count, len(images)))}

    yolo = YOLO(str(yolo_model_path))
    from mmpose.apis import init_pose_model

    vitpose = init_pose_model(str(vitpose_config_path), str(vitpose_checkpoint_path), device=args.device)
    dataset_info = dataset_info_from_model(vitpose)

    result_records: list[dict] = []
    failures: list[dict] = []
    overlay_saved = 0

    for index, img_info in enumerate(images, start=1):
        image_id = int(img_info["id"])
        image_path = images_dir / img_info["file_name"]
        if not image_path.is_file():
            failures.append({"image_id": image_id, "file_name": img_info["file_name"], "reason": "missing_image"})
            continue

        detected = best_yolo_box(yolo, image_path, imgsz=args.imgsz, conf=args.conf)
        if detected is None:
            failures.append({"image_id": image_id, "file_name": img_info["file_name"], "reason": "no_yolo_detection"})
            continue

        bbox_xyxy, bbox_score = detected
        bbox_xywh = xyxy_to_xywh(bbox_xyxy)
        predicted_keypoints = run_pose_prediction(vitpose, image_path, bbox_xywh, dataset_info)
        score = prediction_score(predicted_keypoints, bbox_score, args.keypoint_score_threshold)
        result_records.append(
            {
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": bbox_xywh,
                "keypoints": flatten_keypoints(predicted_keypoints),
                "score": score,
            }
        )

        if image_id in overlay_ids and overlay_saved < args.overlay_count:
            save_overlay(
                model=vitpose,
                image_path=image_path,
                output_path=overlay_dir / f"pred_{overlay_saved:02d}__{image_path.name}",
                yolo_bbox_xyxy=bbox_xyxy,
                predicted_keypoints=predicted_keypoints,
                dataset_info=dataset_info,
            )
            overlay_saved += 1

        if index % 100 == 0:
            print(f"Processed {index}/{len(images)} images; detections={len(result_records)}; failures={len(failures)}", flush=True)

    results_path = run_dir / "yolo_vitpose_keypoints_results.json"
    results_path.write_text(json.dumps(result_records), encoding="utf-8")
    failures_path = run_dir / "failures.json"
    failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    metrics = evaluate_coco(ann_path, results_path)
    summary = {
        "experiment": "YoloVitPose_mAP",
        "split": args.split,
        "dataset_root": str(dataset_root),
        "annotation_file": str(ann_path),
        "images_dir": str(images_dir),
        "yolo_model": str(yolo_model_path),
        "vitpose_config": str(vitpose_config_path),
        "vitpose_checkpoint": str(vitpose_checkpoint_path),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "num_images": len(images),
        "num_predictions": len(result_records),
        "num_failures": len(failures),
        "overlay_dir": str(overlay_dir),
        "results_json": str(results_path),
        "failures_json": str(failures_path),
        "metrics": metrics,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
