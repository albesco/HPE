#!/usr/bin/env python3
"""Grid-evaluate ViTPose++ huge with GT top-down bboxes on train/val only.

Example dry-run:

    python script/grid-eval/grid_eval_VitPose.py \
      --train-data data/intermediate/SUW_frames/_VitPosePP/annotations/person_keypoints_train.json \
      --val-data data/intermediate/SUW_frames/_VitPosePP/annotations/person_keypoints_val.json \
      --output-dir runs/grid-eval/vitpose \
      --experiment-name smoke_dry_run \
      --dry-run

The requested crop-size convention is width x height. The generated MMPose
``data_cfg.image_size`` uses ``[width, height]`` and the ViT backbone
``img_size`` uses ``(height, width)``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

from grid_eval_common import (
    cartesian_grid,
    choose_best,
    command_to_text,
    config_name,
    formatted_value,
    grid_from_cli_params,
    grid_from_file,
    metric_value,
    parse_float,
    read_status,
    repo_root,
    resolve_path,
    run_logged_subprocess,
    update_status,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


DEFAULT_PRETRAINED_CHECKPOINT = "models/pose/wholebody.pth"
DEFAULT_OUTPUT_DIR = "runs/grid-eval/vitpose"
DEFAULT_GRID = {
    "lr": [0.00067, 0.00100],
    "crop_size": ["384x128", "512x128"],
}
DEFAULT_CONFIGS = [
    {"lr": 0.00067, "crop_size": "384x128"},
    {"lr": 0.00100, "crop_size": "384x128"},
    {"lr": 0.00067, "crop_size": "512x128"},
    {"lr": 0.00100, "crop_size": "512x128"},
]
METRIC_ALIASES = {
    "map50-95": "keypoint_ap_50_95",
    "map50_95": "keypoint_ap_50_95",
    "map": "keypoint_ap_50_95",
    "ap": "keypoint_ap_50_95",
    "keypoint_ap": "keypoint_ap_50_95",
    "keypoint_ap_50_95": "keypoint_ap_50_95",
    "ap50": "keypoint_ap50",
    "keypoint_ap50": "keypoint_ap50",
    "ap75": "keypoint_ap75",
    "keypoint_ap75": "keypoint_ap75",
    "ar": "keypoint_ar",
    "keypoint_ar": "keypoint_ar",
}
SUMMARY_FIELDS = [
    "config",
    "status",
    "lr",
    "crop_size",
    "crop_width",
    "crop_height",
    "best_epoch",
    "final_epoch",
    "keypoint_ap_50_95",
    "keypoint_ap50",
    "keypoint_ap75",
    "keypoint_ar",
    "mean_keypoint_error",
    "distal_keypoint_metrics",
    "crop_border_keypoint_pct",
    "best_checkpoint",
    "latest_checkpoint",
    "final_epoch_checkpoint",
    "run_dir",
    "config_path",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a train/val-only ViTPose++ huge grid-evaluation with GT bboxes.")
    parser.add_argument("--train-data", help="COCO train annotation JSON, preferably _VitPosePP/annotations/person_keypoints_train.json.")
    parser.add_argument("--val-data", help="COCO validation annotation JSON, preferably _VitPosePP/annotations/person_keypoints_val.json.")
    parser.add_argument("--test-data", help="Optional COCO test annotation JSON; recorded but not used for selection.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5, help="Recorded for consistency; current MMPose training runs the requested epochs.")
    parser.add_argument("--metric", default="mAP50-95")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto", help="auto or a GPU id/cuda:N. CPU is not supported by the existing train launcher.")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--list-grid", action="store_true")
    parser.add_argument("--grid-param", action="append", default=[])
    parser.add_argument("--grid-json")
    parser.add_argument("--grid-yaml")
    parser.add_argument("--base-config", help="Reference VitPose++ config. Default: inferred from train-data dataset root when possible.")
    parser.add_argument("--pretrained-checkpoint", default=DEFAULT_PRETRAINED_CHECKPOINT)
    parser.add_argument("--train-images", help="Optional train image directory override.")
    parser.add_argument("--val-images", help="Optional validation image directory override.")
    parser.add_argument("--test-images", help="Optional test image directory override; not used for selection.")
    parser.add_argument("--conda-env", default="vitpose")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--status-interval", type=int, default=20)
    return parser.parse_args()


def selected_grid(args: argparse.Namespace) -> dict[str, list[Any]]:
    sources = [bool(args.grid_param), bool(args.grid_json), bool(args.grid_yaml)]
    if sum(sources) > 1:
        raise ValueError("Use only one grid source: --grid-param, --grid-json, or --grid-yaml")
    if args.grid_param:
        return grid_from_cli_params(args.grid_param)
    if args.grid_json:
        return grid_from_file(resolve_path(args.grid_json))
    if args.grid_yaml:
        return grid_from_file(resolve_path(args.grid_yaml))
    return DEFAULT_GRID


def validate_vitpose_grid(grid: dict[str, list[Any]]) -> None:
    required = {"lr", "crop_size"}
    missing = sorted(required - set(grid))
    if missing:
        raise ValueError(f"ViTPose grid is missing required parameter(s): {', '.join(missing)}")
    unsupported = sorted(set(grid) - required)
    if unsupported:
        raise ValueError(f"ViTPose grid may vary only lr and crop_size; unsupported parameter(s): {', '.join(unsupported)}")
    for crop in grid["crop_size"]:
        parse_crop_size(crop)


def parse_crop_size(value: Any) -> tuple[int, int]:
    text = str(value)
    match = re.fullmatch(r"(\d+)x(\d+)", text)
    if not match:
        raise ValueError(f"Invalid crop_size {value!r}; expected WxH, e.g. 384x128")
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0 or width % 4 or height % 4:
        raise ValueError(f"Invalid crop_size {value!r}; width and height must be positive and divisible by 4")
    return width, height


def crop_value(value: Any) -> str:
    width, height = parse_crop_size(value)
    return f"{width}x{height}"


def vitpose_config_name(index: int, params: dict[str, Any]) -> str:
    return f"cfg_{index:02d}_lr_{formatted_value('lr', params['lr'])}_crop_{crop_value(params['crop_size'])}"


def build_config_list(grid: dict[str, list[Any]], max_runs: int | None) -> list[tuple[str, dict[str, Any]]]:
    raw_configs = DEFAULT_CONFIGS if grid == DEFAULT_GRID else cartesian_grid(grid)
    normalized = []
    for params in raw_configs:
        normalized.append({"lr": params["lr"], "crop_size": crop_value(params["crop_size"])})
    configs = [(vitpose_config_name(index, params), params) for index, params in enumerate(normalized, start=1)]
    if max_runs is not None:
        if max_runs < 1:
            raise ValueError("--max-runs must be >= 1")
        return configs[:max_runs]
    return configs


def normalize_metric_name(metric: str) -> str:
    normalized = metric.strip().lower()
    if normalized not in METRIC_ALIASES:
        choices = ", ".join(sorted(METRIC_ALIASES))
        raise ValueError(f"Unsupported --metric {metric!r}; choices: {choices}")
    return METRIC_ALIASES[normalized]


def cli_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.absolute()
    return (repo_root() / path).absolute()


def split_from_ann(path: Path) -> str:
    name = path.name.lower()
    for split in ("train", "val", "test"):
        if split in name:
            return split
    raise ValueError(f"Cannot infer split from annotation path: {path}")


def dataset_root_from_ann(path: Path) -> Path | None:
    parts = path.parts
    if "_VitPosePP" in parts:
        index = parts.index("_VitPosePP")
        return Path(*parts[:index])
    if "_train_canonical" in parts:
        index = parts.index("_train_canonical")
        return Path(*parts[:index])
    return None


def infer_image_dir(annotation_path: Path, override: str | None) -> Path:
    if override:
        return cli_path(override)
    split = split_from_ann(annotation_path)
    dataset_root = dataset_root_from_ann(annotation_path)
    if dataset_root is None:
        raise ValueError(f"Cannot infer image directory from annotation path; pass explicit image override: {annotation_path}")
    return dataset_root / "_train_canonical" / f"{split}2017"


def resolve_annotation(value: str) -> Path:
    path = cli_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Missing COCO annotation JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("images"), list) or not isinstance(payload.get("annotations"), list):
        raise ValueError(f"Annotation file is not COCO-like: {path}")
    return path


def count_visible_keypoints(annotation_path: Path) -> dict[str, int]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations", [])
    visible = 0
    total = 0
    for ann in annotations:
        keypoints = ann.get("keypoints", [])
        for index in range(2, len(keypoints), 3):
            total += 1
            if keypoints[index] > 0:
                visible += 1
    return {"annotations": len(annotations), "keypoints": total, "visible_keypoints": visible}


def validate_split(annotation_path: Path, images_dir: Path, split_name: str) -> dict[str, Any]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing {split_name} image directory: {images_dir}")
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_count = len(payload.get("images", []))
    annotation_count = len(payload.get("annotations", []))
    if image_count == 0 or annotation_count == 0:
        raise ValueError(f"Empty COCO split {split_name}: {annotation_path}")
    counts = count_visible_keypoints(annotation_path)
    return {
        "split": split_name,
        "annotation_file": annotation_path.as_posix(),
        "images_dir": images_dir.as_posix(),
        "image_count": image_count,
        "annotation_count": annotation_count,
        **counts,
        "bbox_source": "GT bbox from COCO annotations",
    }


def infer_base_config(train_annotation: Path, explicit: str | None) -> Path:
    if explicit:
        return resolve_path(explicit)
    dataset_root = dataset_root_from_ann(train_annotation)
    if dataset_root is not None:
        candidate = dataset_root / "_VitPosePP" / "generated_configs" / "swimxyz_vitposepp_huge.py"
        if candidate.is_file():
            return candidate.resolve()
    return resolve_path("data/intermediate/Side_above_water/_VitPosePP/generated_configs/swimxyz_vitposepp_huge.py")


def write_effective_config(
    config_path: Path,
    base_config: Path,
    pretrained_checkpoint: Path,
    run_dir: Path,
    params: dict[str, Any],
    args: argparse.Namespace,
    train_ann: Path,
    val_ann: Path,
    test_ann: Path | None,
    train_images: Path,
    val_images: Path,
    test_images: Path | None,
) -> None:
    crop_width, crop_height = parse_crop_size(params["crop_size"])
    heatmap_width = crop_width // 4
    heatmap_height = crop_height // 4
    test_ann_text = (test_ann or val_ann).as_posix()
    test_images_text = (test_images or val_images).as_posix()
    workers_per_gpu = args.workers if args.workers is not None else 2
    workers_line = f"    workers_per_gpu={workers_per_gpu},\n"
    seed_text = f"seed = {args.seed}\n" if args.seed is not None else ""
    config_text = f"""_base_ = ['{base_config.as_posix()}']

# Generated by script/grid-eval/grid_eval_VitPose.py.
# Crop convention: requested crop_size is width x height.
# MMPose TopDown data_cfg.image_size is [width, height].
# ViT backbone img_size is (height, width).

load_from = '{pretrained_checkpoint.as_posix()}'
work_dir = '{run_dir.as_posix()}'
total_epochs = {args.epochs}
{seed_text}
optimizer = dict(lr={float(params['lr']):.5f})
checkpoint_config = dict(interval=1, max_keep_ckpts=1, create_symlink=True)
evaluation = dict(interval=1, metric='mAP', save_best='AP')

model = dict(backbone=dict(img_size=({crop_height}, {crop_width})))

dataset_data_cfg = dict(
    image_size=[{crop_width}, {crop_height}],
    heatmap_size=[{heatmap_width}, {heatmap_height}],
    num_output_channels=17,
    num_joints=17,
    dataset_channel=[list(range(17))],
    inference_channel=list(range(17)),
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    max_num_joints=17,
    dataset_idx=0,
)

data = dict(
    _delete_=True,
    samples_per_gpu=4,
{workers_line}    val_dataloader=dict(samples_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=4),
    train=dict(
        type='TopDownCocoDataset',
        ann_file='{train_ann.as_posix()}',
        img_prefix='{train_images.as_posix()}/',
        data_cfg=dataset_data_cfg,
        pipeline=train_pipeline,
        dataset_info=coco_dataset_info,
    ),
    val=dict(
        type='TopDownCocoDataset',
        ann_file='{val_ann.as_posix()}',
        img_prefix='{val_images.as_posix()}/',
        data_cfg=dataset_data_cfg,
        pipeline=val_pipeline,
        dataset_info=coco_dataset_info,
    ),
    test=dict(
        type='TopDownCocoDataset',
        ann_file='{test_ann_text}',
        img_prefix='{test_images_text}/',
        data_cfg=dataset_data_cfg,
        pipeline=test_pipeline,
        dataset_info=coco_dataset_info,
    ),
)
"""
    write_text(config_path, config_text)


def checkpoint_paths(run_dir: Path) -> dict[str, str | None]:
    best_paths = sorted(run_dir.glob("best_*.pth"), key=lambda path: epoch_number(path.name))
    latest = latest_checkpoint(run_dir)
    return {
        "best": best_paths[-1].as_posix() if best_paths else None,
        "latest": (run_dir / "latest.pth").as_posix() if (run_dir / "latest.pth").is_file() else None,
        "final_epoch": latest.as_posix() if latest is not None else None,
    }


def epoch_number(name: str) -> int:
    match = re.search(r"epoch_(\d+)", name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(run_dir: Path) -> Path | None:
    latest = run_dir / "latest.pth"
    if latest.exists():
        return latest
    epoch_paths = sorted(run_dir.glob("epoch_*.pth"), key=lambda path: epoch_number(path.name))
    return epoch_paths[-1] if epoch_paths else None


def read_validation_metrics(run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for log_json in sorted(run_dir.glob("*.log.json")):
        for line in log_json.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("mode") == "val":
                rows.append(row)
    if not rows:
        return {}

    def ap_value(row: dict[str, Any]) -> float:
        value = parse_float(row.get("AP"), float("-inf"))
        return value if value is not None else float("-inf")

    best = max(rows, key=lambda row: (ap_value(row), parse_float(row.get("AP .75"), float("-inf")) or float("-inf")))
    final = max(rows, key=lambda row: int(row.get("epoch", 0)))
    return {
        "best_epoch": best.get("epoch"),
        "final_epoch": final.get("epoch"),
        "keypoint_ap_50_95": parse_float(best.get("AP")),
        "keypoint_ap50": parse_float(best.get("AP .5")),
        "keypoint_ap75": parse_float(best.get("AP .75")),
        "keypoint_ar": parse_float(best.get("AR")),
        "mean_keypoint_error": None,
        "distal_keypoint_metrics": None,
        "crop_border_keypoint_pct": None,
        "epochs": [
            {
                "epoch": row.get("epoch"),
                "AP": parse_float(row.get("AP")),
                "AP50": parse_float(row.get("AP .5")),
                "AP75": parse_float(row.get("AP .75")),
                "AR": parse_float(row.get("AR")),
            }
            for row in rows
        ],
    }


def row_for_config(run_dir: Path, cfg_name: str, params: dict[str, Any]) -> dict[str, Any]:
    status = read_status(run_dir)
    metrics = read_validation_metrics(run_dir)
    checkpoints = checkpoint_paths(run_dir)
    crop_width, crop_height = parse_crop_size(params["crop_size"])
    row: dict[str, Any] = {
        "config": cfg_name,
        "status": status.get("status", "pending"),
        "lr": params.get("lr"),
        "crop_size": crop_value(params.get("crop_size")),
        "crop_width": crop_width,
        "crop_height": crop_height,
        "run_dir": run_dir.as_posix(),
        "config_path": (run_dir / "effective_config.py").as_posix(),
        "error": status.get("error"),
        "best_checkpoint": checkpoints["best"],
        "latest_checkpoint": checkpoints["latest"],
        "final_epoch_checkpoint": checkpoints["final_epoch"],
    }
    for field in SUMMARY_FIELDS:
        row.setdefault(field, metrics.get(field))
    return row


def best_ranking(metric_name: str) -> Any:
    def ranking(row: dict[str, Any]) -> tuple[float, float, float, int]:
        mean_error = metric_value(row, "mean_keypoint_error",) if row.get("mean_keypoint_error") is not None else float("inf")
        return (
            metric_value(row, metric_name),
            metric_value(row, "keypoint_ap75"),
            -mean_error,
            -int(row.get("crop_width") or 10**9),
        )

    return ranking


def write_report(experiment_dir: Path, rows: list[dict[str, Any]], best: dict[str, Any] | None, metric_name: str) -> None:
    lines = [
        "# ViTPose++ huge grid-evaluation",
        "",
        "Purpose: select a ViTPose++ huge configuration under GT-bbox top-down conditions for swimmer keypoint estimation.",
        "",
        "Selection split: validation only. The test split is not used for hyperparameter selection.",
        "",
        f"Primary metric: `{metric_name}` (Keypoint AP@[OKS 0.50:0.95] by default).",
        "",
        "Selection priority: Keypoint AP@[OKS 0.50:0.95], then AP75, then lower mean keypoint error if available, then crop 384x128.",
        "",
        "BBox policy: train and validation use GT bboxes from COCO annotations with `use_gt_bbox=True` and `bbox_file=''`; detector-predicted bboxes are not used.",
        "",
        "Crop convention: requested crop_size is width x height. Generated configs write MMPose `data_cfg.image_size=[width, height]` and ViT backbone `img_size=(height, width)`. Thus 384x128 becomes image_size=[384,128] and backbone img_size=(128,384).",
        "",
        "Checkpoint policy: `checkpoint_config.interval=1`, `max_keep_ckpts=1`, `create_symlink=True`, and `evaluation.save_best='AP'`; this keeps best plus latest/final rather than retained intermediate epoch checkpoints.",
        "",
        "Invariant settings: optimizer type, scheduler, weight decay, layer-wise LR decay, augmentation, head parameters, and heatmap generation pipeline are inherited from the base config except for learning rate, crop/heatmap size, total epochs, validation cadence, GT bbox data_cfg, and checkpoint retention.",
        "",
        "Unavailable secondary metrics: mean keypoint error, distal-keypoint metrics, and crop-border keypoint percentage are not emitted by standard MMPose validation logs; they remain null unless a separate evaluator is added. Values are not inferred.",
        "",
        "## Best Config",
        "",
    ]
    if best is None:
        lines.append("No completed configuration is available yet.")
    else:
        lines.extend(
            [
                f"- Config: `{best.get('config')}`",
                f"- Keypoint AP50-95: `{best.get('keypoint_ap_50_95')}`",
                f"- AP50: `{best.get('keypoint_ap50')}`",
                f"- AP75: `{best.get('keypoint_ap75')}`",
                f"- AR: `{best.get('keypoint_ar')}`",
                f"- Best checkpoint: `{best.get('best_checkpoint')}`",
                f"- Latest checkpoint: `{best.get('latest_checkpoint')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Configs",
            "",
            "| Config | lr | crop | Status | Best Epoch | AP | AP50 | AP75 | AR | Best checkpoint |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.get('config')} | {row.get('lr')} | {row.get('crop_size')} | {row.get('status')} | "
            f"{row.get('best_epoch')} | {row.get('keypoint_ap_50_95')} | {row.get('keypoint_ap50')} | "
            f"{row.get('keypoint_ap75')} | {row.get('keypoint_ar')} | {row.get('best_checkpoint')} |"
        )
    write_text(experiment_dir / "report.md", "\n".join(lines) + "\n")


def write_experiment_summaries(experiment_dir: Path, configs: list[tuple[str, dict[str, Any]]], metric_name: str) -> dict[str, Any] | None:
    rows = [row_for_config(experiment_dir / cfg_name, cfg_name, params) for cfg_name, params in configs]
    best = choose_best(rows, best_ranking(metric_name))
    write_csv(experiment_dir / "summary.csv", rows, SUMMARY_FIELDS)
    write_json(experiment_dir / "summary.json", {"updated_at": utc_now(), "configs": rows})
    write_json(experiment_dir / "best_config.json", best or {"status": "unavailable"})
    if best and best.get("run_dir"):
        command_path = Path(str(best["run_dir"])) / "command.txt"
        write_text(experiment_dir / "best_config_command.txt", command_path.read_text(encoding="utf-8") if command_path.exists() else "")
    else:
        write_text(experiment_dir / "best_config_command.txt", "")
    write_report(experiment_dir, rows, best, metric_name)
    return best


def device_to_gpu_id(device: str) -> str | None:
    if device == "auto":
        return None
    if device == "cpu":
        raise ValueError("The existing VitPose train.py launcher does not expose a CPU mode; use a GPU id such as 0 or cuda:0")
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return device


def build_command(args: argparse.Namespace, config_path: Path, run_dir: Path, resume_from: Path | None) -> list[str]:
    command = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        "src/vitpose_base/tools/train.py",
        config_path.as_posix(),
        "--work-dir",
        run_dir.as_posix(),
        "--log-interval",
        str(args.log_interval),
        "--status-file",
        (run_dir / "training_status.txt").as_posix(),
        "--status-interval",
        str(args.status_interval),
    ]
    gpu_id = device_to_gpu_id(args.device)
    if gpu_id is not None:
        command.extend(["--gpu-id", str(gpu_id)])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if resume_from is not None:
        command.extend(["--resume-from", resume_from.as_posix()])
    return command


def should_skip(run_dir: Path, overwrite: bool) -> tuple[bool, str | None]:
    if overwrite:
        return False, None
    status = read_status(run_dir)
    if status.get("status") == "completed":
        return True, "completed"
    return False, None


def run_config(
    args: argparse.Namespace,
    experiment_dir: Path,
    configs: list[tuple[str, dict[str, Any]]],
    cfg_name: str,
    params: dict[str, Any],
    metric_name: str,
    base_config: Path,
    pretrained_checkpoint: Path,
    train_ann: Path,
    val_ann: Path,
    test_ann: Path | None,
    train_images: Path,
    val_images: Path,
    test_images: Path | None,
) -> None:
    run_dir = experiment_dir / cfg_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "effective_config.py"
    resume_from = latest_checkpoint(run_dir) if args.resume else None
    write_effective_config(config_path, base_config, pretrained_checkpoint, run_dir, params, args, train_ann, val_ann, test_ann, train_images, val_images, test_images)
    command = build_command(args, config_path, run_dir, resume_from)
    write_text(run_dir / "command.txt", command_to_text(command) + "\n")
    crop_width, crop_height = parse_crop_size(params["crop_size"])
    write_json(
        run_dir / "args.json",
        {
            "config": cfg_name,
            "params": params,
            "epochs": args.epochs,
            "patience": args.patience,
            "metric": args.metric,
            "base_config": base_config.as_posix(),
            "pretrained_checkpoint": pretrained_checkpoint.as_posix(),
            "effective_config": config_path.as_posix(),
            "resume_from": resume_from.as_posix() if resume_from else None,
            "dry_run": args.dry_run,
            "command": command,
            "crop_convention": {
                "requested": "width x height",
                "mmpose_data_cfg_image_size": [crop_width, crop_height],
                "vit_backbone_img_size": [crop_height, crop_width],
                "heatmap_size": [crop_width // 4, crop_height // 4],
            },
            "bbox_policy": "GT bbox from COCO annotations; use_gt_bbox=True; bbox_file=''",
        },
    )

    skip, reason = should_skip(run_dir, args.overwrite)
    if skip:
        write_experiment_summaries(experiment_dir, configs, metric_name)
        print(f"skip {cfg_name}: {reason}")
        return

    if args.dry_run:
        update_status(run_dir, {"status": "pending", "dry_run": True, "config": cfg_name, "command": command})
        write_experiment_summaries(experiment_dir, configs, metric_name)
        print(f"dry-run {cfg_name}: {command_to_text(command)}")
        return

    if not pretrained_checkpoint.is_file() and resume_from is None:
        raise FileNotFoundError(f"Missing pretrained checkpoint: {pretrained_checkpoint}")
    if shutil.which("conda") is None:
        raise RuntimeError("conda is not available on PATH")

    log_path = run_dir / "stdout_stderr.log"
    update_status(run_dir, {"status": "running", "started_at": utc_now(), "dry_run": False, "resume_from": resume_from.as_posix() if resume_from else None, "config": cfg_name, "command": command, "log_path": log_path.as_posix()})
    write_experiment_summaries(experiment_dir, configs, metric_name)
    return_code = run_logged_subprocess(command, log_path)
    metrics = read_validation_metrics(run_dir)
    write_json(run_dir / "validation_metrics.json", metrics)
    checkpoints = checkpoint_paths(run_dir)
    if return_code == 0:
        update_status(run_dir, {"status": "completed", "completed_at": utc_now(), "exit_code": return_code, "metrics": metrics, "checkpoints": checkpoints})
    else:
        update_status(run_dir, {"status": "failed", "failed_at": utc_now(), "exit_code": return_code, "metrics": metrics, "checkpoints": checkpoints, "error": f"Training command exited with code {return_code}"})
    write_experiment_summaries(experiment_dir, configs, metric_name)


def main() -> None:
    args = parse_args()
    metric_name = normalize_metric_name(args.metric)
    grid = selected_grid(args)
    validate_vitpose_grid(grid)
    configs = build_config_list(grid, args.max_runs)

    if args.list_grid:
        print(json.dumps({"grid": grid, "configs": [{"name": name, "params": params} for name, params in configs]}, indent=2))
        return

    if not args.train_data or not args.val_data:
        raise ValueError("--train-data and --val-data are required unless --list-grid is used")

    train_ann = resolve_annotation(args.train_data)
    val_ann = resolve_annotation(args.val_data)
    test_ann = resolve_annotation(args.test_data) if args.test_data else None
    train_images = infer_image_dir(train_ann, args.train_images)
    val_images = infer_image_dir(val_ann, args.val_images)
    test_images = infer_image_dir(test_ann, args.test_images) if test_ann else None
    base_config = infer_base_config(train_ann, args.base_config)
    pretrained_checkpoint = resolve_path(args.pretrained_checkpoint)
    if not base_config.is_file():
        raise FileNotFoundError(f"Missing base config: {base_config}")

    split_report = {
        "train": validate_split(train_ann, train_images, "train"),
        "val": validate_split(val_ann, val_images, "val"),
        "test": validate_split(test_ann, test_images, "test") if test_ann and test_images else None,
    }
    experiment_dir = resolve_path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        experiment_dir / "grid_effective.json",
        {
            "updated_at": utc_now(),
            "script": Path(__file__).resolve().as_posix(),
            "repo_root": repo_root().as_posix(),
            "selection_split": "val",
            "test_data_policy": "recorded only; not used for hyperparameter selection",
            "grid": grid,
            "configs": [{"name": name, "params": params} for name, params in configs],
            "split_report": split_report,
            "base_config": base_config.as_posix(),
            "pretrained_checkpoint": pretrained_checkpoint.as_posix(),
            "defaults_not_overridden": "Optimizer type, scheduler, weight decay, layer-wise LR decay, augmentation, head parameters and pipelines are inherited from the base config except lr, crop/heatmap size, total epochs, GT bbox data_cfg, validation cadence and checkpoint retention.",
            "crop_convention": "requested crop_size is width x height; MMPose image_size=[width,height]; ViT backbone img_size=(height,width)",
            "bbox_policy": "train and validation use GT bbox from COCO annotations; use_gt_bbox=True; bbox_file=''",
            "checkpoint_policy": "checkpoint interval=1; max_keep_ckpts=1; create_symlink=True; save_best=AP",
        },
    )

    write_experiment_summaries(experiment_dir, configs, metric_name)
    for cfg_name, params in configs:
        run_config(args, experiment_dir, configs, cfg_name, params, metric_name, base_config, pretrained_checkpoint, train_ann, val_ann, test_ann, train_images, val_images, test_images)
    write_experiment_summaries(experiment_dir, configs, metric_name)


if __name__ == "__main__":
    main()
