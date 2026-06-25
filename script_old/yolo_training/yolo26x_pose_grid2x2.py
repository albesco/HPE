#!/usr/bin/env python3
"""Run a local 2x2 YOLO26x-pose hyperparameter search on train/val only.

The search varies only ``lr0`` and ``imgsz`` across four 5-epoch runs.
It does not use the test split for hyperparameter selection.

Recommended heavy-run usage:

    tmux new-session -d -s yolo26x_pose_grid2x2 \
      'cd /home/albertosco/HPE && python script/yolo_training/yolo26x_pose_grid2x2.py'

Dry-run / static setup check:

    python script/yolo_training/yolo26x_pose_grid2x2.py --dry-run
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
DEFAULT_MODEL = "yolo26x-pose.pt"
DEFAULT_DATA_YAML = (
    REPO_ROOT
    / "data/intermediate/Side_above_water/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/hparam_search/yolo26x_pose"
EPOCHS = 5
GRID = (
    (1, 0.00067, 640),
    (2, 0.00100, 640),
    (3, 0.00067, 768),
    (4, 0.00100, 768),
)
INVARIANT_TRAIN_ARGS = {
    "task": "pose",
    "mode": "train",
    "batch": 1,
    "device": "0",
    "workers": 2,
    "patience": 30,
    "save": True,
    "save_period": -1,
    "val": True,
    "split": "val",
    "plots": True,
    "verbose": True,
    "seed": 0,
    "deterministic": True,
    "amp": True,
    "exist_ok": True,
    "optimizer": "AdamW",
}
METRIC_FIELDS = (
    "keypoint_ap_50_95",
    "keypoint_ap50",
    "keypoint_ap75",
    "keypoint_ar",
    "pose_precision",
    "pose_recall",
    "box_map50",
    "box_map50_95",
    "train_box_loss",
    "train_pose_loss",
    "train_kobj_loss",
    "train_rle_loss",
    "val_box_loss",
    "val_pose_loss",
    "val_kobj_loss",
    "val_rle_loss",
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
        description="Run a 2x2 YOLO26x-pose lr0/imgsz search on train/val only."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA_YAML.as_posix())
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--conda-env", default="vitpose")
    parser.add_argument("--rerun-failed", action="store_true", help="Rerun configs with failed status.")
    parser.add_argument("--rerun-running", action="store_true", help="Treat running statuses as stale.")
    parser.add_argument("--dry-run", action="store_true", help="Create dirs/reports without training.")
    parser.add_argument(
        "--launch-tmux",
        action="store_true",
        help="Start this grid search in a detached tmux session and exit.",
    )
    parser.add_argument("--tmux-session", default="yolo26x_pose_grid2x2")
    return parser.parse_args()


def resolve_existing_path(value: str) -> Path:
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


def parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def search_configs() -> list[SearchConfig]:
    return [SearchConfig(index=index, lr0=lr0, imgsz=imgsz) for index, lr0, imgsz in GRID]


def status_path(run_dir: Path) -> Path:
    return run_dir / "status.json"


def update_status(run_dir: Path, payload: dict[str, Any]) -> None:
    current = read_json(status_path(run_dir))
    current.update(payload)
    current["updated_at"] = utc_now()
    write_json(status_path(run_dir), current)


def checkpoint_paths(run_dir: Path) -> dict[str, str | None]:
    weights_dir = run_dir / "weights"
    return {
        "best": (weights_dir / "best.pt").as_posix() if (weights_dir / "best.pt").exists() else None,
        "last": (weights_dir / "last.pt").as_posix() if (weights_dir / "last.pt").exists() else None,
    }


def read_results_csv(results_path: Path) -> dict[str, Any]:
    if not results_path.exists():
        return {}
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        return {}

    def key_metric(row: dict[str, Any]) -> tuple[float, float]:
        return (
            parse_float(row.get("metrics/mAP50-95(P)"), float("-inf")) or float("-inf"),
            parse_float(row.get("epoch"), float("-inf")) or float("-inf"),
        )

    best_row = max(rows, key=key_metric)
    final_row = max(rows, key=lambda row: parse_float(row.get("epoch"), -1.0) or -1.0)
    return {
        "best_epoch": int(parse_float(best_row.get("epoch"), 0.0) or 0),
        "final_epoch": int(parse_float(final_row.get("epoch"), 0.0) or 0),
        "keypoint_ap_50_95": parse_float(best_row.get("metrics/mAP50-95(P)")),
        "keypoint_ap50": parse_float(best_row.get("metrics/mAP50(P)")),
        "keypoint_ap75": None,
        "keypoint_ar": None,
        "pose_precision": parse_float(best_row.get("metrics/precision(P)")),
        "pose_recall": parse_float(best_row.get("metrics/recall(P)")),
        "box_map50": parse_float(best_row.get("metrics/mAP50(B)")),
        "box_map50_95": parse_float(best_row.get("metrics/mAP50-95(B)")),
        "train_box_loss": parse_float(best_row.get("train/box_loss")),
        "train_pose_loss": parse_float(best_row.get("train/pose_loss")),
        "train_kobj_loss": parse_float(best_row.get("train/kobj_loss")),
        "train_rle_loss": parse_float(best_row.get("train/rle_loss")),
        "val_box_loss": parse_float(best_row.get("val/box_loss")),
        "val_pose_loss": parse_float(best_row.get("val/pose_loss")),
        "val_kobj_loss": parse_float(best_row.get("val/kobj_loss")),
        "val_rle_loss": parse_float(best_row.get("val/rle_loss")),
    }


def config_command(args: argparse.Namespace, config: SearchConfig, run_dir: Path) -> list[str]:
    invariant = INVARIANT_TRAIN_ARGS
    return [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "yolo",
        "pose",
        "train",
        f"model={args.model}",
        f"data={resolve_existing_path(args.data).as_posix()}",
        f"epochs={EPOCHS}",
        f"imgsz={config.imgsz}",
        f"lr0={config.lr0:.5f}",
        f"batch={invariant['batch']}",
        f"device={invariant['device']}",
        f"workers={invariant['workers']}",
        f"patience={invariant['patience']}",
        f"save={invariant['save']}",
        f"save_period={invariant['save_period']}",
        f"val={invariant['val']}",
        f"split={invariant['split']}",
        f"plots={invariant['plots']}",
        f"verbose={invariant['verbose']}",
        f"seed={invariant['seed']}",
        f"deterministic={invariant['deterministic']}",
        f"amp={invariant['amp']}",
        f"exist_ok={invariant['exist_ok']}",
        f"optimizer={invariant['optimizer']}",
        f"project={run_dir.parent.as_posix()}",
        f"name={run_dir.name}",
    ]


def metric_value(row: dict[str, Any], key: str) -> float:
    value = parse_float(row.get(key))
    return value if value is not None else float("-inf")


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [row for row in rows if row.get("status") == "completed"]
    if not completed:
        return None

    def ranking(row: dict[str, Any]) -> tuple[float, float, float, int]:
        return (
            metric_value(row, "keypoint_ap_50_95"),
            metric_value(row, "keypoint_ap75"),
            metric_value(row, "keypoint_ar"),
            -int(row["imgsz"]),
        )

    best = max(completed, key=ranking)
    best["selection_reason"] = (
        "Selected by highest validation Keypoint AP@[OKS 0.50:0.95]; ties use AP75, "
        "then AR, then prefer imgsz=640. Test split was not used."
    )
    return best


def config_row(config: SearchConfig, run_dir: Path) -> dict[str, Any]:
    status = read_json(status_path(run_dir))
    metrics = read_results_csv(run_dir / "results.csv")
    row: dict[str, Any] = {
        "config": config.name,
        "index": config.index,
        "lr0": config.lr0,
        "imgsz": config.imgsz,
        "status": status.get("status", "pending"),
        "best_epoch": metrics.get("best_epoch"),
        "final_epoch": metrics.get("final_epoch"),
        "best_checkpoint": checkpoint_paths(run_dir)["best"],
        "last_checkpoint": checkpoint_paths(run_dir)["last"],
        "run_dir": run_dir.as_posix(),
        "error": status.get("error"),
    }
    for field in METRIC_FIELDS:
        row[field] = metrics.get(field)
    return row


def write_summary(output_root: Path, configs: list[SearchConfig]) -> None:
    rows = [config_row(config, output_root / config.name) for config in configs]
    fieldnames = [
        "config",
        "index",
        "lr0",
        "imgsz",
        "status",
        "best_epoch",
        "final_epoch",
        *METRIC_FIELDS,
        "best_checkpoint",
        "last_checkpoint",
        "run_dir",
        "error",
    ]
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = choose_best(rows)
    write_json(output_root / "summary.json", {"updated_at": utc_now(), "configs": rows})
    write_json(output_root / "best_config.json", best or {"status": "unavailable"})
    write_report(output_root, rows, best)


def write_report(output_root: Path, rows: list[dict[str, Any]], best: dict[str, Any] | None) -> None:
    lines = [
        "# YOLO26x-Pose 2x2 Hyperparameter Search",
        "",
        "Purpose: select YOLO26x-pose fine-tuning dynamics and spatial resolution for the Side_above_water swimmer pose task.",
        "",
        "Selection split: validation only. The test split is not used for hyperparameter choice.",
        "",
        "Grid: lr0 in {0.00067, 0.00100}; imgsz in {640, 768}; each run trains for 5 epochs.",
        "",
        "Invariant training settings: batch=1, workers=2, patience=30, save_period=-1, deterministic=True, amp=True, default Ultralytics augmentation/loss/NMS settings.",
        "",
        "Optimizer note: historical logs show optimizer=auto ignores lr0. This search sets optimizer=AdamW, the optimizer type selected automatically in prior YOLO26x-pose logs, so lr0 is actually applied.",
        "",
        "Unavailable metrics: AP75, COCO AR, mean per-keypoint error, and distal-keypoint metrics are not emitted by standard Ultralytics train results.csv; they remain null unless a separate evaluator is added.",
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
                f"- Validation Keypoint AP@[OKS 0.50:0.95]: `{best.get('keypoint_ap_50_95')}`",
                f"- Validation Keypoint AP50: `{best.get('keypoint_ap50')}`",
                f"- Best checkpoint: `{best.get('best_checkpoint')}`",
                f"- Last checkpoint: `{best.get('last_checkpoint')}`",
                f"- Reason: {best.get('selection_reason')}",
            ]
        )

    lines.extend(
        [
            "",
            "## Configs",
            "",
            "| Config | lr0 | imgsz | Status | Best Epoch | KP AP50-95 | KP AP50 | KP AP75 | KP AR | Box mAP50 | Box mAP50-95 |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {config} | {lr0:.5f} | {imgsz} | {status} | {best_epoch} | {keypoint_ap_50_95} | {keypoint_ap50} | {keypoint_ap75} | {keypoint_ar} | {box_map50} | {box_map50_95} |".format(
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
            "tmux new-session -d -s yolo26x_pose_grid2x2 'cd /home/albertosco/HPE && python script/yolo_training/yolo26x_pose_grid2x2.py'",
            "```",
            "",
            "Run a dry-run setup check:",
            "",
            "```bash",
            "python script/yolo_training/yolo26x_pose_grid2x2.py --dry-run",
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


def write_run_inputs(run_dir: Path, args: argparse.Namespace, config: SearchConfig, command: list[str]) -> None:
    data_yaml = resolve_existing_path(args.data)
    shutil.copy2(data_yaml, run_dir / "data.yaml")
    (run_dir / "command.txt").write_text(" ".join(shlex.quote(part) for part in command) + "\n", encoding="utf-8")
    write_json(
        run_dir / "training_args.json",
        {
            **asdict(config),
            "name": config.name,
            "epochs": EPOCHS,
            "model": args.model,
            "data": data_yaml.as_posix(),
            "run_dir": run_dir.as_posix(),
            "resume": False,
            "varied_hyperparameters": ["lr0", "imgsz"],
            "invariant_train_args": INVARIANT_TRAIN_ARGS,
            "command": command,
            "uses_test_split_for_selection": False,
        },
    )


def run_config(args: argparse.Namespace, config: SearchConfig, output_root: Path) -> None:
    run_dir = output_root / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    command = config_command(args, config, run_dir)
    write_run_inputs(run_dir, args, config, command)

    skip, reason = should_skip(run_dir, args)
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
                "resume": False,
            },
        )
        print(f"dry-run {config.name}: {' '.join(shlex.quote(part) for part in command)}")
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
            "resume": False,
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
    write_json(run_dir / "validation_metrics.json", metrics)
    if return_code == 0:
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


def validate_static_setup(args: argparse.Namespace, output_root: Path) -> None:
    data_yaml = resolve_existing_path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing YOLO26x-pose data YAML: {data_yaml}")
    if Path(args.model).is_absolute() and not Path(args.model).exists():
        raise FileNotFoundError(f"Missing YOLO26x-pose pretrained weights: {args.model}")
    if args.model.startswith("/") is False and "/" in args.model:
        model_path = resolve_existing_path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Missing YOLO26x-pose pretrained weights: {model_path}")

    configs = search_configs()
    command_payloads = []
    for config in configs:
        command = config_command(args, config, output_root / config.name)
        payload = {
            "config": config.name,
            "lr0": config.lr0,
            "imgsz": config.imgsz,
            "constant_args": {
                item.split("=", 1)[0]: item.split("=", 1)[1]
                for item in command
                if "=" in item and not item.startswith(("lr0=", "imgsz=", "project=", "name="))
            },
        }
        command_payloads.append(payload)

    reference = command_payloads[0]["constant_args"]
    for payload in command_payloads[1:]:
        if payload["constant_args"] != reference:
            raise RuntimeError("Static setup check failed: arguments other than lr0/imgsz vary.")

    write_json(
        output_root / "static_validation.json",
        {
            "validated_at": utc_now(),
            "varied_hyperparameters": ["lr0", "imgsz"],
            "constant_args": reference,
            "configs": command_payloads,
            "epochs_per_run": EPOCHS,
            "uses_test_split_for_selection": False,
            "resume_from_checkpoint": False,
            "checkpoint_policy": "Keep best.pt and last.pt/latest only; do not save periodic epoch checkpoints.",
        },
    )


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
    subprocess.run(["tmux", "new-session", "-d", "-s", args.tmux_session, inner_command], check=True)
    print(f"Launched tmux session: {args.tmux_session}")
    print(f"Attach with: tmux attach -t {args.tmux_session}")
    raise SystemExit(0)


def main() -> None:
    args = parse_args()
    if args.launch_tmux:
        launch_in_tmux(args)

    output_root = resolve_existing_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    validate_static_setup(args, output_root)
    if shutil.which("conda") is None and not args.dry_run:
        raise RuntimeError("conda is not available on PATH")

    write_json(
        output_root / "search_args.json",
        {
            "created_or_updated_at": utc_now(),
            "grid": [asdict(config) | {"name": config.name} for config in search_configs()],
            "data": resolve_existing_path(args.data).as_posix(),
            "model": args.model,
            "output_root": output_root.as_posix(),
            "epochs_per_run": EPOCHS,
            "selection_metric": "validation_keypoint_ap_50_95",
            "selection_tiebreakers": ["keypoint_ap75", "keypoint_ar", "prefer_imgsz_640"],
            "uses_test_split_for_selection": False,
            "resume_from_checkpoint": False,
            "optimizer_note": "optimizer=AdamW is set because prior logs show optimizer=auto ignores lr0.",
            "checkpoint_policy": "save_period=-1; keep best.pt and last.pt/latest only, with no periodic epoch checkpoints.",
        },
    )

    write_summary(output_root, search_configs())
    for config in search_configs():
        run_config(args, config, output_root)
    write_summary(output_root, search_configs())
    print(f"Search artifacts: {output_root}")


if __name__ == "__main__":
    main()
