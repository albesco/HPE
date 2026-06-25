#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
from pathlib import Path


GRID_RUN = Path("runs/hparam_search/yolo26x_pose/cfg_04_lr0_0.00100_imgsz_768")
FINAL_RUN = Path("runs/pose/runs/yolo26x_pose_side_above_water/yolo26x-pose-incremental-from-cfg04")
DATA_YAML = Path("data/intermediate/Side_above_water/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml")
OVERLAY_SCRIPT = Path("script/yolo_training/render_yolo_pose_overlays.py")
EVAL_SCRIPT = Path("script/yolo_training/evaluate_yolo_pose_split.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final YOLO26x-Pose training report, with optional test-per-checkpoint eval."
    )
    parser.add_argument("--grid-run", type=Path, default=GRID_RUN)
    parser.add_argument("--final-run", type=Path, default=FINAL_RUN)
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--render-overlays", action="store_true")
    parser.add_argument("--overlay-max-images", type=int, default=0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def float_or_empty(value: str | None) -> float | str:
    if value is None or value == "":
        return ""
    return float(value)


def checkpoint_for_final_epoch(final_run: Path, epoch: int) -> Path:
    return final_run / "weights" / f"epoch{epoch - 1}.pt"


def build_rows(grid_run: Path, final_run: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for grid_row in read_csv(grid_run / "results.csv"):
        grid_epoch = int(float(grid_row["epoch"]))
        checkpoint = grid_run / "weights" / "best.pt" if grid_epoch == 5 else None
        rows.append(
            {
                "epoch_total": grid_epoch,
                "stage": "grid_cfg04",
                "epoch_in_stage": grid_epoch,
                "checkpoint": checkpoint.as_posix() if checkpoint and checkpoint.exists() else "",
                "train_pose_loss": float_or_empty(grid_row.get("train/pose_loss")),
                "val_pose_loss": float_or_empty(grid_row.get("val/pose_loss")),
                "val_pose_map50_95": float_or_empty(grid_row.get("metrics/mAP50-95(P)")),
                "test_pose_map50_95": "",
            }
        )

    for final_row in read_csv(final_run / "results.csv"):
        final_epoch = int(float(final_row["epoch"]))
        checkpoint = checkpoint_for_final_epoch(final_run, final_epoch)
        rows.append(
            {
                "epoch_total": 5 + final_epoch,
                "stage": "incremental_cfg04",
                "epoch_in_stage": final_epoch,
                "checkpoint": checkpoint.as_posix() if checkpoint.exists() else "",
                "train_pose_loss": float_or_empty(final_row.get("train/pose_loss")),
                "val_pose_loss": float_or_empty(final_row.get("val/pose_loss")),
                "val_pose_map50_95": float_or_empty(final_row.get("metrics/mAP50-95(P)")),
                "test_pose_map50_95": "",
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch_total",
        "stage",
        "epoch_in_stage",
        "checkpoint",
        "train_pose_loss",
        "val_pose_loss",
        "val_pose_map50_95",
        "test_pose_map50_95",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_metric_csv(path: Path) -> float:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No metric rows in {path}")
    return float(rows[0]["pose_map50_95"])


def eval_checkpoint(args: argparse.Namespace, checkpoint: str, metric_csv: Path) -> float:
    if metric_csv.exists():
        return read_metric_csv(metric_csv)

    cmd = [
        "conda",
        "run",
        "-n",
        "vitpose",
        "python",
        EVAL_SCRIPT.as_posix(),
        "--model",
        checkpoint,
        "--data",
        args.data.as_posix(),
        "--split",
        "test",
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--out-csv",
        metric_csv.as_posix(),
    ]
    metric_csv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    return read_metric_csv(metric_csv)


def add_test_metrics(args: argparse.Namespace, rows: list[dict[str, object]], report_dir: Path) -> None:
    for row in rows:
        checkpoint = str(row["checkpoint"])
        if not checkpoint:
            continue
        metric_csv = report_dir / "test_metrics_by_checkpoint" / f"epoch_total_{int(row['epoch_total']):03d}.csv"
        row["test_pose_map50_95"] = eval_checkpoint(args, checkpoint, metric_csv)


def plot(rows: list[dict[str, object]], out_path: Path, y_key: str, title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    points = [
        (int(row["epoch_total"]), float(row[y_key]))
        for row in rows
        if row.get(y_key) not in ("", None) and not math.isnan(float(row[y_key]))
    ]
    if not points:
        return
    x_values, y_values = zip(*points)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, marker="o", linewidth=1.8)
    plt.xlabel("Epoch totale (1-5 grid cfg04, poi train incrementale)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def render_overlays(args: argparse.Namespace, final_run: Path, report_dir: Path) -> None:
    output_dir = report_dir / "test_overlays_best"
    if output_dir.exists() and any(output_dir.iterdir()):
        return
    cmd = [
        "conda",
        "run",
        "-n",
        "vitpose",
        "python",
        OVERLAY_SCRIPT.as_posix(),
        "--model",
        (final_run / "weights" / "best.pt").as_posix(),
        "--source",
        "data/intermediate/Side_above_water/_Yolo26x_pose/images/test",
        "--output-dir",
        output_dir.as_posix(),
        "--imgsz",
        str(args.imgsz),
        "--device",
        args.device,
        "--max-images",
        str(args.overlay_max_images),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    report_dir = args.final_run / "reports" / "final_training"
    rows = build_rows(args.grid_run, args.final_run)

    if args.evaluate_test:
        add_test_metrics(args, rows, report_dir)

    out_csv = report_dir / "loss_pose_map50_95_by_epoch.csv"
    write_csv(out_csv, rows)
    plot(rows, report_dir / "loss_pose_by_epoch.png", "train_pose_loss", "YOLO26x-Pose train Pose loss", "train/pose_loss")
    plot(
        rows,
        report_dir / "test_map50_95_by_epoch.png",
        "test_pose_map50_95",
        "YOLO26x-Pose test mAP50-95 by checkpoint",
        "test Pose mAP50-95",
    )
    plot(
        rows,
        report_dir / "val_map50_95_by_epoch.png",
        "val_pose_map50_95",
        "YOLO26x-Pose validation mAP50-95 by epoch",
        "val Pose mAP50-95",
    )

    if args.render_overlays:
        render_overlays(args, args.final_run, report_dir)

    print(f"Wrote: {out_csv}")
    print(f"Report directory: {report_dir}")


if __name__ == "__main__":
    main()
