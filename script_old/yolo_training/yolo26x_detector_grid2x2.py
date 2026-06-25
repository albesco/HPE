#!/usr/bin/env python3
"""Run a local 2x2 YOLO26x detector hyperparameter search on train/val only.

The search consolidates the one-class swimmer detector used by the
YOLO26x-detector -> VitPose++ huge pipeline. It varies only ``lr0`` and
``imgsz`` across four 5-epoch runs, writes one run directory per config, and
updates ``summary.csv``, ``summary.json``, ``report.md``, and
``best_config.json`` after each config.

Recommended heavy-run usage:

    tmux new-session -d -s yolo26x_detector_grid2x2 \
      'cd /home/albertosco/HPE && python script/yolo_training/yolo26x_detector_grid2x2.py'

Dry-run / static setup check:

    python script/yolo_training/yolo26x_detector_grid2x2.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models/detection/yolo26x.pt"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data/intermediate/Side_above_water/_train_canonical"
DEFAULT_LABEL_ROOT = REPO_ROOT / "data/intermediate/Side_above_water/_Yolo26x_detection"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/hparam_search/yolo26x_detector"
GRID = (
    (1, 0.00067, 640),
    (2, 0.00100, 640),
    (3, 0.00067, 768),
    (4, 0.00100, 768),
)
METRIC_FIELDS = (
    "precision",
    "recall",
    "map50",
    "map50_95",
    "ap75",
    "mean_iou",
    "visible_keypoint_coverage",
    "train_box_loss",
    "val_box_loss",
)


@dataclass(frozen=True)
class SearchConfig:
    index: int
    lr0: float
    imgsz: int

    @property
    def name(self) -> str:
        return f"cfg_{self.index:02d}_lr0_{self.lr0:.5f}_imgsz_{self.imgsz}"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 2x2 YOLO26x detector lr0/imgsz search on train/val only."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL.as_posix())
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT.as_posix())
    parser.add_argument("--label-root", default=DEFAULT_LABEL_ROOT.as_posix())
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--conda-env", default="vitpose")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-epoch-checkpoints", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Create dirs/reports without training.")
    parser.add_argument("--rerun-failed", action="store_true", help="Rerun configs with failed status.")
    parser.add_argument("--rerun-running", action="store_true", help="Treat running statuses as stale.")
    parser.add_argument(
        "--launch-tmux",
        action="store_true",
        help="Start this grid search in a detached tmux session and exit.",
    )
    parser.add_argument("--tmux-session", default="yolo26x_detector_grid2x2")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "failed", "error": f"Invalid JSON in {path}"}


def image_files(root: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def reset_view_dir(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def symlink_file_or_check(link_path: Path, target_path: Path) -> None:
    if not target_path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {target_path}")
    if link_path.is_symlink():
        existing = link_path.resolve()
        if existing != target_path.resolve():
            raise RuntimeError(f"Unexpected symlink target: {link_path} -> {existing}")
        return
    if link_path.exists():
        raise RuntimeError(f"Refusing to replace existing non-symlink path: {link_path}")
    link_path.symlink_to(target_path)


def prepare_split_view(view_root: Path, split: str, image_root: Path, label_root: Path) -> dict[str, int]:
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"Missing label directory: {label_root}")

    view_image_root = view_root / "images" / split
    view_label_root = view_root / "labels" / split
    reset_view_dir(view_image_root)
    reset_view_dir(view_label_root)

    images = image_files(image_root)
    missing_labels = []
    for image_path in images:
        label_path = label_root / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels.append(image_path.stem)
            continue
        symlink_file_or_check(view_image_root / image_path.name, image_path)
        symlink_file_or_check(view_label_root / label_path.name, label_path)

    image_count = len(image_files(view_image_root))
    label_count = len(list(view_label_root.glob("*.txt")))
    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        raise RuntimeError(f"Missing {len(missing_labels)} {split} labels; examples: {preview}")
    if image_count == 0 or label_count == 0 or image_count != label_count:
        raise RuntimeError(
            f"Invalid {split} dataset view: images={image_count}, labels={label_count}"
        )
    return {"images": image_count, "labels": label_count}


def prepare_dataset_view(output_root: Path, dataset_root: Path, label_root: Path) -> Path:
    view_root = output_root / "dataset_view"
    summary = {
        "train": prepare_split_view(
            view_root,
            "train",
            dataset_root / "train2017",
            label_root / "labels" / "train",
        ),
        "val": prepare_split_view(
            view_root,
            "val",
            dataset_root / "val2017",
            label_root / "labels" / "val",
        ),
    }
    write_json(view_root / "dataset_view_report.json", summary)

    data_yaml = view_root / "swimxyz_side_above_water_yolo26x_detection_train_val.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {view_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: swimmer",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def config_command(
    args: argparse.Namespace,
    config: SearchConfig,
    data_yaml: Path,
    run_dir: Path,
    resume: bool,
) -> list[str]:
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
        f"imgsz={config.imgsz}",
        f"batch={args.batch}",
        f"device={args.device}",
        f"workers={args.workers}",
        f"patience={args.patience}",
        f"project={run_dir.parent.as_posix()}",
        f"name={run_dir.name}",
        "exist_ok=True",
        "save=True",
        "save_period=1",
        "val=True",
        "split=val",
        "plots=True",
        f"seed={args.seed}",
    ]
    if resume:
        command.append("resume=True")
    else:
        command.append(f"lr0={config.lr0:.5f}")
    return command


def read_results_csv(results_path: Path) -> dict[str, Any]:
    if not results_path.exists():
        return {}

    with results_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    normalized_rows = [{key.strip(): value for key, value in row.items()} for row in rows]
    last_row = max(normalized_rows, key=lambda row: parse_float(row.get("epoch"), -1.0))
    return {
        "epoch": int(parse_float(last_row.get("epoch"), 0.0)),
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


def parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def checkpoint_paths(run_dir: Path) -> dict[str, str | None]:
    weights_dir = run_dir / "weights"
    return {
        "best": (weights_dir / "best.pt").as_posix() if (weights_dir / "best.pt").exists() else None,
        "last": (weights_dir / "last.pt").as_posix() if (weights_dir / "last.pt").exists() else None,
    }


def prune_epoch_checkpoints(weights_dir: Path, keep: int) -> None:
    if keep < 0 or not weights_dir.is_dir():
        return
    epoch_paths: list[tuple[int, Path]] = []
    for path in weights_dir.glob("epoch*.pt"):
        stem = path.stem
        number = stem.replace("epoch", "", 1)
        if number.isdigit():
            epoch_paths.append((int(number), path))
    epoch_paths.sort()
    for _epoch, path in epoch_paths[:-keep] if keep else epoch_paths:
        path.unlink()


def status_path(run_dir: Path) -> Path:
    return run_dir / "status.json"


def update_status(run_dir: Path, payload: dict[str, Any]) -> None:
    current = read_json(status_path(run_dir))
    current.update(payload)
    current["updated_at"] = utc_now()
    write_json(status_path(run_dir), current)


def config_row(config: SearchConfig, run_dir: Path) -> dict[str, Any]:
    status = read_json(status_path(run_dir))
    metrics = read_results_csv(run_dir / "results.csv")
    row: dict[str, Any] = {
        "config": config.name,
        "index": config.index,
        "lr0": config.lr0,
        "imgsz": config.imgsz,
        "status": status.get("status", "pending"),
        "run_dir": run_dir.as_posix(),
        "epoch": metrics.get("epoch"),
        "best_checkpoint": checkpoint_paths(run_dir)["best"],
        "last_checkpoint": checkpoint_paths(run_dir)["last"],
        "error": status.get("error"),
    }
    for field in METRIC_FIELDS:
        row[field] = metrics.get(field)
    return row


def metric_value(row: dict[str, Any], key: str) -> float:
    value = parse_float(row.get(key))
    return value if value is not None else float("-inf")


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [row for row in rows if row.get("status") == "completed"]
    if not completed:
        return None

    def ranking(row: dict[str, Any]) -> tuple[float, float, float, int]:
        secondary = max(metric_value(row, "ap75"), metric_value(row, "mean_iou"))
        return (
            metric_value(row, "recall"),
            secondary,
            metric_value(row, "map50_95"),
            -int(row["imgsz"]),
        )

    return max(completed, key=ranking)


def write_summary(output_root: Path, configs: list[SearchConfig]) -> None:
    rows = [config_row(config, output_root / config.name) for config in configs]
    fieldnames = [
        "config",
        "index",
        "lr0",
        "imgsz",
        "status",
        "epoch",
        *METRIC_FIELDS,
        "best_checkpoint",
        "last_checkpoint",
        "run_dir",
        "error",
    ]
    summary_csv = output_root / "summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = choose_best(rows)
    write_json(output_root / "summary.json", {"updated_at": utc_now(), "configs": rows})
    write_json(output_root / "best_config.json", best or {"status": "unavailable"})
    write_report(output_root, rows, best)


def write_report(output_root: Path, rows: list[dict[str, Any]], best: dict[str, Any] | None) -> None:
    lines = [
        "# YOLO26x Detector 2x2 Hyperparameter Search",
        "",
        "Purpose: consolidate the YOLO26x one-class swimmer detector as the bbox component for the YOLO26x-detector -> VitPose++ huge pipeline.",
        "",
        "Selection split: validation only. The test split is not used for hyperparameter choice.",
        "",
        "Selection priority: recall, then AP75 or mean IoU when available, then mAP50-95, then imgsz=640 for simpler operation.",
        "",
        "Unavailable metrics: AP75, mean predicted-vs-GT IoU, and visible-keypoint coverage are not emitted by the standard Ultralytics train `results.csv`; they remain null unless a separate evaluator is added. Values are not inferred.",
        "",
        "## Best Config",
        "",
    ]
    if best is None:
        lines.append("No completed configuration is available yet.")
    else:
        lines.extend(
            [
                f"- Config: `{best['config']}`",
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
            "| {config} | {lr0:.5f} | {imgsz} | {status} | {recall} | {map50} | {map50_95} | {ap75} | {mean_iou} | {visible_keypoint_coverage} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Usage",
            "",
            "Run the full search inside tmux:",
            "",
            "```bash",
            "tmux new-session -d -s yolo26x_detector_grid2x2 'cd /home/albertosco/HPE && python script/yolo_training/yolo26x_detector_grid2x2.py'",
            "```",
            "",
            "Run a dry-run setup check:",
            "",
            "```bash",
            "python script/yolo_training/yolo26x_detector_grid2x2.py --dry-run",
            "```",
            "",
        ]
    )
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def should_skip(run_dir: Path, args: argparse.Namespace) -> tuple[bool, str | None]:
    status = read_json(status_path(run_dir))
    state = status.get("status")
    if state == "completed":
        return True, "completed"
    if state == "failed" and not args.rerun_failed:
        return True, "failed"
    if state == "running" and not args.rerun_running:
        return True, "running"
    return False, None


def run_config(args: argparse.Namespace, config: SearchConfig, data_yaml: Path) -> None:
    output_root = resolve_path(args.output_root)
    run_dir = output_root / config.name
    run_dir.mkdir(parents=True, exist_ok=True)

    skip, reason = should_skip(run_dir, args)
    resume = (run_dir / "weights" / "last.pt").exists() and not skip
    command = config_command(args, config, data_yaml, run_dir, resume=resume)
    command_text = " ".join(command)

    (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")
    write_json(
        run_dir / "training_args.json",
        {
            **asdict(config),
            "name": config.name,
            "epochs": args.epochs,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "patience": args.patience,
            "seed": args.seed,
            "model": args.model,
            "data": data_yaml.as_posix(),
            "run_dir": run_dir.as_posix(),
            "resume": resume,
            "command": command,
        },
    )

    if skip:
        print(f"skip {config.name}: {reason}")
        write_summary(output_root, search_configs())
        return

    if args.dry_run:
        update_status(
            run_dir,
            {
                "status": "pending",
                "config": config.name,
                "dry_run": True,
                "command": command,
            },
        )
        print(f"dry-run {config.name}: {command_text}")
        write_summary(output_root, search_configs())
        return

    log_path = run_dir / "stdout_stderr.log"
    update_status(
        run_dir,
        {
            "status": "running",
            "config": config.name,
            "started_at": utc_now(),
            "dry_run": False,
            "resume": resume,
            "log_path": log_path.as_posix(),
            "command": command,
        },
    )
    write_summary(output_root, search_configs())

    with log_path.open("ab") as log_file:
        log_file.write(f"\n# Started {utc_now()}\n".encode("utf-8"))
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        return_code = process.wait()
        log_file.write(f"\n# Finished {utc_now()} exit_code={return_code}\n".encode("utf-8"))

    metrics = read_results_csv(run_dir / "results.csv")
    if return_code == 0:
        prune_epoch_checkpoints(run_dir / "weights", args.keep_epoch_checkpoints)
        update_status(
            run_dir,
            {
                "status": "completed",
                "completed_at": utc_now(),
                "exit_code": return_code,
                "metrics": metrics,
                "checkpoints": checkpoint_paths(run_dir),
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
                "checkpoints": checkpoint_paths(run_dir),
                "error": f"Training command exited with code {return_code}",
            },
        )
    write_summary(output_root, search_configs())


def search_configs() -> list[SearchConfig]:
    return [SearchConfig(index=index, lr0=lr0, imgsz=imgsz) for index, lr0, imgsz in GRID]


def launch_in_tmux(args: argparse.Namespace) -> None:
    if os.environ.get("TMUX"):
        return
    script_path = Path(__file__).resolve()
    forwarded = [arg for arg in sys.argv[1:] if arg != "--launch-tmux"]
    inner_command = " ".join(
        [
            "cd",
            shlex.quote(REPO_ROOT.as_posix()),
            "&&",
            "python",
            shlex.quote(script_path.as_posix()),
            *[shlex.quote(arg) for arg in forwarded],
        ]
    )
    command = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        args.tmux_session,
        inner_command,
    ]
    subprocess.run(command, check=True)
    print(f"Launched tmux session: {args.tmux_session}")
    print(f"Attach with: tmux attach -t {args.tmux_session}")
    raise SystemExit(0)


def main() -> None:
    args = parse_args()
    if args.launch_tmux:
        launch_in_tmux(args)

    output_root = resolve_path(args.output_root)
    dataset_root = resolve_path(args.dataset_root)
    label_root = resolve_path(args.label_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not resolve_path(args.model).exists():
        raise FileNotFoundError(f"Missing detector model: {resolve_path(args.model)}")
    if shutil.which("conda") is None and not args.dry_run:
        raise RuntimeError("conda is not available on PATH")

    data_yaml = prepare_dataset_view(output_root, dataset_root, label_root)
    write_json(
        output_root / "search_args.json",
        {
            "created_or_updated_at": utc_now(),
            "grid": [asdict(config) | {"name": config.name} for config in search_configs()],
            "dataset_root": dataset_root.as_posix(),
            "label_root": label_root.as_posix(),
            "data_yaml": data_yaml.as_posix(),
            "output_root": output_root.as_posix(),
            "uses_test_split_for_selection": False,
        },
    )

    write_summary(output_root, search_configs())
    for config in search_configs():
        run_config(args, config, data_yaml)
    write_summary(output_root, search_configs())
    print(f"Search artifacts: {output_root}")


if __name__ == "__main__":
    main()
