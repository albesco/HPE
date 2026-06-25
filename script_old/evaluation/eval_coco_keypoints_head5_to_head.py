#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

KEYPOINT_NAMES = [
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
SIGMAS = np.array(
    [
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    ],
    dtype=float,
)
HEAD_IDXS = [0, 1, 2, 3, 4]
BODY_IDXS = list(range(5, 17))
OUTPUT_NAMES = ["head"] + [KEYPOINT_NAMES[idx] for idx in BODY_IDXS]
HEAD_SIGMA = float(SIGMAS[HEAD_IDXS].mean())
OUTPUT_SIGMAS = np.array([HEAD_SIGMA, *SIGMAS[BODY_IDXS].tolist()], dtype=float)
OUTPUT_SKELETON = [
    [1, 2],
    [1, 3],
    [2, 3],
    [2, 8],
    [3, 9],
    [8, 10],
    [9, 11],
    [2, 4],
    [3, 5],
    [4, 5],
    [4, 6],
    [5, 7],
    [6, 8],
    [7, 9],
]
STAT_NAMES = ["AP", "AP50", "AP75", "AP_M", "AP_L", "AR", "AR50", "AR75", "AR_M", "AR_L"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate COCO keypoint mAP after replacing the first 5 head keypoints with one synthetic head keypoint."
    )
    parser.add_argument("--gt", required=True, help="COCO GT annotation JSON")
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="Prediction input as label=/path/to/file.json. Repeat for multiple models.",
    )
    parser.add_argument("--out-json", help="Optional output JSON path")
    parser.add_argument(
        "--save-prepared-dir",
        help="Optional directory where transformed GT/prediction JSON files are saved",
    )
    return parser.parse_args()


def parse_pred_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Invalid --pred value: {raw!r}. Expected label=/path/to/file.json")
    label, path_str = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Missing label in --pred value: {raw!r}")
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return label, path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def xyxy_to_xywh(bbox_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def detect_prediction_format(predictions: Any) -> str:
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Prediction JSON must be a non-empty list")
    sample = predictions[0]
    if isinstance(sample, dict) and "image_id" in sample and "keypoints" in sample:
        return "coco_results"
    if isinstance(sample, dict) and "image" in sample and "predictions" in sample:
        return "yolo_raw"
    raise ValueError(f"Unsupported prediction format. Sample keys: {sorted(sample.keys())}")


def convert_yolo_raw_to_coco_results(
    predictions: list[dict[str, Any]],
    images_by_name: dict[str, dict[str, Any]],
    category_id: int,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for image_record in predictions:
        image_name = image_record["image"]
        image_info = images_by_name.get(image_name)
        if image_info is None:
            raise KeyError(f"Prediction image not found in GT images: {image_name}")
        image_id = int(image_info["id"])
        for pred in image_record.get("predictions", []):
            keypoints = np.asarray(pred["keypoints"], dtype=float)
            if keypoints.shape != (17, 3):
                raise ValueError(f"Expected keypoints shape (17, 3), got {keypoints.shape} for {image_name}")
            converted.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": xyxy_to_xywh(pred["bbox_xyxy"]),
                    "score": float(pred["score"]),
                    "keypoints": keypoints.reshape(-1).tolist(),
                }
            )
    return converted


def synthesize_head_gt(keypoints: np.ndarray) -> np.ndarray:
    head = keypoints[HEAD_IDXS]
    visible = head[:, 2] > 0
    if not np.any(visible):
        return np.array([0.0, 0.0, 0.0], dtype=float)
    visible_head = head[visible]
    return np.array(
        [
            float(visible_head[:, 0].mean()),
            float(visible_head[:, 1].mean()),
            2.0,
        ],
        dtype=float,
    )


def synthesize_head_pred(keypoints: np.ndarray) -> np.ndarray:
    head = keypoints[HEAD_IDXS]
    return np.array(
        [
            float(head[:, 0].mean()),
            float(head[:, 1].mean()),
            float(head[:, 2].mean()),
        ],
        dtype=float,
    )


def replace_head_flat(keypoints_flat: list[float], is_gt: bool) -> list[float]:
    keypoints = np.asarray(keypoints_flat, dtype=float).reshape(17, 3)
    head = synthesize_head_gt(keypoints) if is_gt else synthesize_head_pred(keypoints)
    merged = np.vstack([head, keypoints[BODY_IDXS]])
    return merged.reshape(-1).tolist()


def filter_category(category: dict[str, Any]) -> dict[str, Any]:
    filtered = deepcopy(category)
    filtered["keypoints"] = OUTPUT_NAMES
    filtered["skeleton"] = OUTPUT_SKELETON
    return filtered


def transform_gt(gt_data: dict[str, Any]) -> dict[str, Any]:
    transformed = deepcopy(gt_data)
    transformed["categories"] = [filter_category(cat) for cat in gt_data.get("categories", [])]
    transformed_annotations = []
    for ann in gt_data.get("annotations", []):
        updated = deepcopy(ann)
        updated["keypoints"] = replace_head_flat(ann["keypoints"], is_gt=True)
        keypoints = np.asarray(updated["keypoints"], dtype=float).reshape(-1, 3)
        updated["num_keypoints"] = int(np.count_nonzero(keypoints[:, 2] > 0))
        transformed_annotations.append(updated)
    transformed["annotations"] = transformed_annotations
    return transformed


def transform_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transformed = []
    for pred in predictions:
        updated = deepcopy(pred)
        updated["keypoints"] = replace_head_flat(pred["keypoints"], is_gt=False)
        transformed.append(updated)
    return transformed


def evaluate(gt_path: Path, pred_path: Path) -> dict[str, float]:
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pred_path))
    evaluator = COCOeval(coco_gt, coco_dt, "keypoints", OUTPUT_SIGMAS)
    evaluator.params.useSegm = None
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {name: float(value) for name, value in zip(STAT_NAMES, evaluator.stats)}


def main() -> None:
    args = parse_args()
    gt_path = Path(args.gt).expanduser().resolve()
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)

    pred_specs = [parse_pred_arg(item) for item in args.pred]
    gt_data = load_json(gt_path)
    category_id = int(gt_data["categories"][0]["id"])
    images_by_name = {image["file_name"]: image for image in gt_data["images"]}

    transformed_gt = transform_gt(gt_data)

    save_prepared_dir = Path(args.save_prepared_dir).expanduser().resolve() if args.save_prepared_dir else None
    if save_prepared_dir is not None:
        save_prepared_dir.mkdir(parents=True, exist_ok=True)
    work_dir = save_prepared_dir if save_prepared_dir is not None else Path("/tmp")
    transformed_gt_path = work_dir / "gt_head5_to_head.json"
    transformed_gt_path.write_text(json.dumps(transformed_gt), encoding="utf-8")

    output: dict[str, Any] = {
        "variant": "head5_to_head",
        "merged_keypoints": KEYPOINT_NAMES[:5],
        "output_keypoints": OUTPUT_NAMES,
        "head_sigma": HEAD_SIGMA,
        "sigmas": OUTPUT_SIGMAS.tolist(),
        "gt": str(gt_path),
        "models": {},
    }

    for label, pred_path in pred_specs:
        raw_predictions = load_json(pred_path)
        pred_format = detect_prediction_format(raw_predictions)
        if pred_format == "coco_results":
            coco_predictions = raw_predictions
        else:
            coco_predictions = convert_yolo_raw_to_coco_results(raw_predictions, images_by_name, category_id)

        transformed_predictions = transform_predictions(coco_predictions)
        transformed_pred_path = work_dir / f"{label}_head5_to_head_predictions.json"
        transformed_pred_path.write_text(json.dumps(transformed_predictions), encoding="utf-8")

        metrics = evaluate(transformed_gt_path, transformed_pred_path)
        output["models"][label] = {
            "input_path": str(pred_path),
            "input_format": pred_format,
            "prepared_prediction_path": str(transformed_pred_path),
            "metrics": metrics,
        }

    rendered = json.dumps(output, indent=2)
    print(rendered)
    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
