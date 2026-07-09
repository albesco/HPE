#!/usr/bin/env python3
"""Grid-evaluate YOLO26x-Detection on train/validation splits only.

Example dry-run:

    python script/grid-eval/grid_eval_Yolo26x-Detection.py \
      --train-data data/intermediate/SUW_frames/_Yolo26x_detection/images/train \
      --val-data data/intermediate/SUW_frames/_Yolo26x_detection/images/val \
      --output-dir runs/grid-eval/yolo26x_detection \
      --experiment-name smoke_dry_run \
      --dry-run

Heavy runs should be launched inside tmux by the caller. The script never uses
the test split to choose hyperparameters; ``--test-data`` is recorded only as
metadata unless a future evaluation mode explicitly consumes it.
"""

from __future__ import annotations

import argparse
import csv
import json
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


DEFAULT_MODEL = "models/detection/yolo26x.pt"
DEFAULT_OUTPUT_DIR = "runs/grid-eval/yolo26x_detection"
DEFAULT_GRID = {
    "lr0": [0.00067, 0.00100],
    "imgsz": [640, 768],
}
DEFAULT_CONFIGS = [
    {"lr0": 0.00067, "imgsz": 640},
    {"lr0": 0.00100, "imgsz": 640},
    {"lr0": 0.00067, "imgsz": 768},
    {"lr0": 0.00100, "imgsz": 768},
]
METRIC_ALIASES = {
    "map50-95": "map50_95",
    "map50_95": "map50_95",
    "map": "map50_95",
    "recall": "recall",
    "precision": "precision",
    "map50": "map50",
    "ap75": "ap75",
    "mean_iou": "mean_iou",
    "visible_keypoint_coverage": "visible_keypoint_coverage",
}
SUMMARY_FIELDS = [
    "config",
    "status",
    "lr0",
    "imgsz",
    "epoch",
    "precision",
    "recall",
    "map50",
    "map50_95",
    "ap75",
    "mean_iou",
    "visible_keypoint_coverage",
    "train_box_loss",
    "val_box_loss",
    "best_checkpoint",
    "last_checkpoint",
    "run_dir",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a train/val-only YOLO26x-Detection grid-evaluation."
    )
    parser.add_argument("--train-data", help="YOLO train images directory.")
    parser.add_argument("--val-data", help="YOLO validation images directory.")
    parser.add_argument("--test-data", help="Optional YOLO test images directory; recorded but not used for selection.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--metric", default="mAP50-95", help="Fallback selection metric after detector-priority metrics.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--list-grid", action="store_true")
    parser.add_argument("--grid-param", action="append", default=[], help="Repeatable name=value1,value2 grid definition.")
    parser.add_argument("--grid-json")
    parser.add_argument("--grid-yaml")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conda-env", default="vitpose")
    return parser.parse_args()


def selected_grid(args: argparse.Namespace) -> dict[str, list[Any]]:
    sources = [
        bool(args.grid_param),
        bool(args.grid_json),
        bool(args.grid_yaml),
    ]
    if sum(sources) > 1:
        raise ValueError("Use only one grid source: --grid-param, --grid-json, or --grid-yaml")
    if args.grid_param:
        return grid_from_cli_params(args.grid_param)
    if args.grid_json:
        return grid_from_file(resolve_path(args.grid_json))
    if args.grid_yaml:
        return grid_from_file(resolve_path(args.grid_yaml))
    return DEFAULT_GRID


def validate_detection_grid(grid: dict[str, list[Any]]) -> None:
    required = {"lr0", "imgsz"}
    missing = sorted(required - set(grid))
    if missing:
        raise ValueError(f"YOLO26x-Detection grid is missing required parameter(s): {', '.join(missing)}")
    unsupported = sorted(set(grid) - required)
    if unsupported:
        raise ValueError(
            "YOLO26x-Detection grid may vary only lr0 and imgsz; "
            f"unsupported parameter(s): {', '.join(unsupported)}"
        )


def image_files(images_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def infer_labels_dir(images_dir: Path) -> Path:
    parts = images_dir.parts
    if "images" in parts:
        index = len(parts) - 1 - list(reversed(parts)).index("images")
        return Path(*parts[:index], "labels", *parts[index + 1 :])
    return images_dir.parent.parent / "labels" / images_dir.name


def cli_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.absolute()
    return (repo_root() / path).absolute()


def validate_split(images_dir: Path, split_name: str) -> dict[str, Any]:
    labels_dir = infer_labels_dir(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing {split_name} images directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing inferred {split_name} labels directory: {labels_dir}")
    images = image_files(images_dir)
    missing = [image.stem for image in images if not (labels_dir / f"{image.stem}.txt").is_file()]
    if not images:
        raise RuntimeError(f"No images found for {split_name}: {images_dir}")
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"Missing {len(missing)} {split_name} labels; examples: {preview}")
    label_count = len(list(labels_dir.glob("*.txt")))
    return {
        "split": split_name,
        "images_dir": images_dir.as_posix(),
        "labels_dir": labels_dir.as_posix(),
        "image_count": len(images),
        "label_count": label_count,
    }


def write_data_yaml(experiment_dir: Path, train_data: Path, val_data: Path, test_data: Path | None) -> Path:
    yaml_path = experiment_dir / "dataset_yamls" / "yolo26x_detection_train_val.yaml"
    lines = [
        f"train: {train_data.as_posix()}",
        f"val: {val_data.as_posix()}",
    ]
    if test_data is not None:
        lines.append(f"test: {test_data.as_posix()}")
    lines.extend(
        [
            "names:",
            "  0: swimmer",
            "",
        ]
    )
    write_text(yaml_path, "\n".join(lines))
    return yaml_path


def normalize_metric_name(metric: str) -> str:
    normalized = metric.strip().lower()
    if normalized not in METRIC_ALIASES:
        choices = ", ".join(sorted(METRIC_ALIASES))
        raise ValueError(f"Unsupported --metric {metric!r}; choices: {choices}")
    return METRIC_ALIASES[normalized]


def read_results_csv(results_path: Path) -> dict[str, Any]:
    if not results_path.exists():
        return {}
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    normalized_rows = [{key.strip(): value for key, value in row.items()} for row in rows]
    last_row = max(normalized_rows, key=lambda row: parse_float(row.get("epoch"), -1.0) or -1.0)
    return {
        "epoch": int(parse_float(last_row.get("epoch"), 0.0) or 0),
        "precision": parse_float(last_row.get("metrics/precision(B)")),
        "recall": parse_float(last_row.get("metrics/recall(B)")),
        "map50": parse_float(last_row.get("metrics/mAP50(B)")),
        "map50_95": parse_float(last_row.get("metrics/mAP50-95(B)")),
        "ap75": None,
        "mean_iou": None,
        "visible_keypoint_coverage": None,
        "train_box_loss": parse_float(last_row.get("train/box_loss")),
        "val_box_loss": parse_float(last_row.get("val/box_loss")),
    }


def checkpoint_paths(run_dir: Path) -> dict[str, str | None]:
    weights_dir = run_dir / "weights"
    return {
        "best": (weights_dir / "best.pt").as_posix() if (weights_dir / "best.pt").is_file() else None,
        "last": (weights_dir / "last.pt").as_posix() if (weights_dir / "last.pt").is_file() else None,
    }


def row_for_config(run_dir: Path, cfg_name: str, params: dict[str, Any]) -> dict[str, Any]:
    status = read_status(run_dir)
    metrics = read_results_csv(run_dir / "results.csv")
    checkpoints = checkpoint_paths(run_dir)
    row: dict[str, Any] = {
        "config": cfg_name,
        "status": status.get("status", "pending"),
        "lr0": params.get("lr0"),
        "imgsz": params.get("imgsz"),
        "run_dir": run_dir.as_posix(),
        "error": status.get("error"),
        "best_checkpoint": checkpoints["best"],
        "last_checkpoint": checkpoints["last"],
    }
    for field in SUMMARY_FIELDS:
        row.setdefault(field, metrics.get(field))
    return row


def best_ranking(metric_name: str) -> Any:
    def ranking(row: dict[str, Any]) -> tuple[float, float, float, int]:
        secondary = max(metric_value(row, "ap75"), metric_value(row, "mean_iou"))
        return (
            metric_value(row, "recall"),
            secondary,
            metric_value(row, metric_name),
            -int(row.get("imgsz") or 10**9),
        )

    return ranking


def write_report(experiment_dir: Path, rows: list[dict[str, Any]], best: dict[str, Any] | None, metric_name: str) -> None:
    lines = [
        "# YOLO26x-Detection grid-evaluation",
        "",
        "Purpose: consolidate a swimmer detector configuration for the YOLO26x-Detection -> ViTPose++ pipeline.",
        "",
        "Selection split: validation only. The test split is not used for hyperparameter selection.",
        "",
        f"Fallback metric: `{metric_name}`.",
        "",
        "Selection priority: recall, then AP75/mean IoU when available, then fallback metric, then imgsz=640.",
        "",
        "Unavailable metrics: AP75, mean predicted-vs-GT IoU, and visible-keypoint coverage are not emitted by standard Ultralytics train `results.csv`; they remain null unless an evaluator is added. Values are not inferred.",
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
                f"- Recall: `{best.get('recall')}`",
                f"- mAP50-95: `{best.get('map50_95')}`",
                f"- Best checkpoint: `{best.get('best_checkpoint')}`",
                f"- Last checkpoint: `{best.get('last_checkpoint')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Configs",
            "",
            "| Config | lr0 | imgsz | Status | Recall | mAP50 | mAP50-95 | AP75 | Mean IoU | KP Coverage |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.get('config')} | {row.get('lr0')} | {row.get('imgsz')} | {row.get('status')} | "
            f"{row.get('recall')} | {row.get('map50')} | {row.get('map50_95')} | "
            f"{row.get('ap75')} | {row.get('mean_iou')} | {row.get('visible_keypoint_coverage')} |"
        )
    write_text(experiment_dir / "report.md", "\n".join(lines) + "\n")


def write_experiment_summaries(
    experiment_dir: Path,
    configs: list[tuple[str, dict[str, Any]]],
    metric_name: str,
) -> dict[str, Any] | None:
    rows = [row_for_config(experiment_dir / cfg_name, cfg_name, params) for cfg_name, params in configs]
    best = choose_best(rows, best_ranking(metric_name))
    write_csv(experiment_dir / "summary.csv", rows, SUMMARY_FIELDS)
    write_json(experiment_dir / "summary.json", {"updated_at": utc_now(), "configs": rows})
    write_json(experiment_dir / "best_config.json", best or {"status": "unavailable"})
    if best and best.get("run_dir"):
        command_path = Path(str(best["run_dir"])) / "command.txt"
        write_text(
            experiment_dir / "best_config_command.txt",
            command_path.read_text(encoding="utf-8") if command_path.exists() else "",
        )
    else:
        write_text(experiment_dir / "best_config_command.txt", "")
    write_report(experiment_dir, rows, best, metric_name)
    return best


def build_command(args: argparse.Namespace, params: dict[str, Any], data_yaml: Path, run_dir: Path, resume: bool) -> list[str]:
    model_path = run_dir / "weights" / "last.pt" if resume else resolve_path(args.model)
    command = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "yolo",
        "detect",
        "train",
        f"model={model_path.as_posix()}",
        f"data={data_yaml.as_posix()}",
        f"epochs={args.epochs}",
        f"patience={args.patience}",
        f"lr0={formatted_value('lr0', params['lr0'])}",
        f"imgsz={params['imgsz']}",
        f"project={experiment_dir_from_run(run_dir).as_posix()}",
        f"name={run_dir.name}",
        "exist_ok=True",
        "save=True",
        "save_period=-1",
        "val=True",
        "split=val",
        "plots=True",
    ]
    if args.device != "auto":
        command.append(f"device={args.device}")
    if args.workers is not None:
        command.append(f"workers={args.workers}")
    if args.seed is not None:
        command.append(f"seed={args.seed}")
    if resume:
        command.append("resume=True")
    return command


def experiment_dir_from_run(run_dir: Path) -> Path:
    return run_dir.parent


def should_skip(run_dir: Path, overwrite: bool) -> tuple[bool, str | None]:
    if overwrite:
        return False, None
    status = read_status(run_dir)
    state = status.get("status")
    if state == "completed":
        return True, "completed"
    return False, None


def run_config(
    args: argparse.Namespace,
    experiment_dir: Path,
    configs: list[tuple[str, dict[str, Any]]],
    cfg_name: str,
    params: dict[str, Any],
    data_yaml: Path,
    metric_name: str,
) -> None:
    run_dir = experiment_dir / cfg_name
    run_dir.mkdir(parents=True, exist_ok=True)
    skip, reason = should_skip(run_dir, args.overwrite)
    resume = args.resume and (run_dir / "weights" / "last.pt").is_file() and not skip
    command = build_command(args, params, data_yaml, run_dir, resume=resume)
    write_text(run_dir / "command.txt", command_to_text(command) + "\n")
    write_json(
        run_dir / "args.json",
        {
            "config": cfg_name,
            "params": params,
            "epochs": args.epochs,
            "patience": args.patience,
            "metric": args.metric,
            "model": resolve_path(args.model).as_posix(),
            "data_yaml": data_yaml.as_posix(),
            "resume": resume,
            "dry_run": args.dry_run,
            "command": command,
        },
    )

    if skip:
        write_experiment_summaries(experiment_dir, configs, metric_name)
        print(f"skip {cfg_name}: {reason}")
        return

    if args.dry_run:
        update_status(
            run_dir,
            {
                "status": "pending",
                "dry_run": True,
                "config": cfg_name,
                "command": command,
            },
        )
        write_experiment_summaries(experiment_dir, configs, metric_name)
        print(f"dry-run {cfg_name}: {command_to_text(command)}")
        return

    if not resolve_path(args.model).is_file() and not resume:
        raise FileNotFoundError(f"Missing YOLO model checkpoint: {resolve_path(args.model)}")
    if shutil.which("conda") is None:
        raise RuntimeError("conda is not available on PATH")

    log_path = run_dir / "stdout_stderr.log"
    update_status(
        run_dir,
        {
            "status": "running",
            "started_at": utc_now(),
            "dry_run": False,
            "resume": resume,
            "config": cfg_name,
            "command": command,
            "log_path": log_path.as_posix(),
        },
    )
    write_experiment_summaries(experiment_dir, configs, metric_name)
    return_code = run_logged_subprocess(command, log_path)
    metrics = read_results_csv(run_dir / "results.csv")
    checkpoints = checkpoint_paths(run_dir)
    if return_code == 0:
        update_status(
            run_dir,
            {
                "status": "completed",
                "completed_at": utc_now(),
                "exit_code": return_code,
                "metrics": metrics,
                "checkpoints": checkpoints,
            },
        )
    else:
        update_status(
            run_dir,
            {
                "status": "failed",
                "failed_at": utc_now(),
                "exit_code": return_code,
                "metrics": metrics,
                "checkpoints": checkpoints,
                "error": f"Training command exited with code {return_code}",
            },
        )
    write_experiment_summaries(experiment_dir, configs, metric_name)


def build_config_list(grid: dict[str, list[Any]], max_runs: int | None) -> list[tuple[str, dict[str, Any]]]:
    if grid == DEFAULT_GRID:
        raw_configs = DEFAULT_CONFIGS
    else:
        raw_configs = cartesian_grid(grid)
    configs = [(config_name(index, params), params) for index, params in enumerate(raw_configs, start=1)]
    if max_runs is not None:
        if max_runs < 1:
            raise ValueError("--max-runs must be >= 1")
        return configs[:max_runs]
    return configs


def main() -> None:
    args = parse_args()
    metric_name = normalize_metric_name(args.metric)
    grid = selected_grid(args)
    validate_detection_grid(grid)
    configs = build_config_list(grid, args.max_runs)

    if args.list_grid:
        print(json.dumps({"grid": grid, "configs": [{"name": name, "params": params} for name, params in configs]}, indent=2))
        return

    if not args.train_data or not args.val_data:
        raise ValueError("--train-data and --val-data are required unless --list-grid is used")

    experiment_dir = resolve_path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    train_data = cli_path(args.train_data)
    val_data = cli_path(args.val_data)
    test_data = cli_path(args.test_data) if args.test_data else None
    split_report = {
        "train": validate_split(train_data, "train"),
        "val": validate_split(val_data, "val"),
        "test": validate_split(test_data, "test") if test_data else None,
    }
    data_yaml = write_data_yaml(experiment_dir, train_data, val_data, test_data)
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
            "defaults_not_overridden": "All YOLO parameters not emitted in command.txt remain framework/model defaults.",
        },
    )

    write_experiment_summaries(experiment_dir, configs, metric_name)
    for cfg_name, params in configs:
        run_config(args, experiment_dir, configs, cfg_name, params, data_yaml, metric_name)
    write_experiment_summaries(experiment_dir, configs, metric_name)


if __name__ == "__main__":
    main()
