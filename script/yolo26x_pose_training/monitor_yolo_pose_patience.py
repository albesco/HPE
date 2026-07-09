#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import shutil
import time
from pathlib import Path


EPOCH_RE = re.compile(r"^epoch(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor Ultralytics YOLO pose training for min-delta patience and checkpoint pruning."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--metric", default="metrics/mAP50-95(P)")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--keep-last", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--status-json", required=True)
    parser.add_argument("--best-checkpoint", default="")
    return parser.parse_args()


def read_rows(results_csv: Path) -> list[dict[str, str]]:
    if not results_csv.is_file():
        return []
    with results_csv.open(newline="") as handle:
        return list(csv.DictReader(handle))


def epoch_from_checkpoint(path: Path) -> int | None:
    match = EPOCH_RE.match(path.name)
    return int(match.group(1)) if match else None


def prune_checkpoints(
    weights_dir: Path,
    best_epoch: int | None,
    latest_epoch: int | None,
    keep_last: int,
) -> list[str]:
    if not weights_dir.is_dir() or latest_epoch is None:
        return []
    keep_last = keep_last if keep_last > 0 else 1
    keep_epochs = set(range(max(0, latest_epoch - keep_last + 1), latest_epoch + 1))
    if best_epoch is not None:
        keep_epochs.add(best_epoch)

    removed: list[str] = []
    for path in sorted(weights_dir.glob("epoch*.pt")):
        epoch = epoch_from_checkpoint(path)
        if epoch is None or epoch in keep_epochs:
            continue
        path.unlink()
        removed.append(path.as_posix())
    return removed


def checkpoint_for_epoch(weights_dir: Path, csv_epoch: int) -> Path | None:
    candidates = [
        weights_dir / f"epoch{max(0, csv_epoch - 1)}.pt",
        weights_dir / f"epoch{csv_epoch}.pt",
        weights_dir / "last.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def copy_best_checkpoint(weights_dir: Path, csv_epoch: int, best_checkpoint: Path) -> str | None:
    source = checkpoint_for_epoch(weights_dir, csv_epoch)
    if source is None:
        return None
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, best_checkpoint)
    return source.as_posix()


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stop_training(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    results_csv = run_dir / "results.csv"
    weights_dir = run_dir / "weights"
    status_json = Path(args.status_json).expanduser().resolve()
    best_checkpoint = (
        Path(args.best_checkpoint).expanduser().resolve()
        if args.best_checkpoint
        else weights_dir / "best_map50_95_pose.pt"
    )

    best_metric: float | None = None
    best_epoch: int | None = None
    best_source_checkpoint: str | None = None
    stale_epochs = 0
    stopped = False
    processed_rows = 0

    while True:
        rows = read_rows(results_csv)
        for latest in rows[processed_rows:]:
            latest_epoch = int(float(latest["epoch"]))
            metric = float(latest[args.metric])
            improved = best_metric is None or metric > best_metric + args.min_delta
            if improved:
                best_metric = metric
                best_epoch = latest_epoch
                best_source_checkpoint = copy_best_checkpoint(
                    weights_dir,
                    latest_epoch,
                    best_checkpoint,
                )
                stale_epochs = 0
            else:
                stale_epochs += 1

            keep_last = args.keep_last if args.keep_last > 0 else args.patience
            removed = prune_checkpoints(weights_dir, best_epoch, latest_epoch, keep_last)
            status = {
                "phase": "monitoring",
                "latest_epoch": latest_epoch,
                "metric": args.metric,
                "latest_metric": metric,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "best_checkpoint": best_checkpoint.as_posix(),
                "best_source_checkpoint": best_source_checkpoint,
                "stale_epochs": stale_epochs,
                "patience": args.patience,
                "min_epochs": args.min_epochs,
                "keep_last": keep_last,
                "min_delta": args.min_delta,
                "removed_checkpoints": removed,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            write_status(status_json, status)

            if latest_epoch >= args.min_epochs and stale_epochs >= args.patience:
                status["phase"] = "stopping"
                status["reason"] = "metric_plateau"
                write_status(status_json, status)
                stop_training(args.pid)
                stopped = True
                break

        processed_rows = len(rows)

        try:
            os.kill(args.pid, 0)
        except ProcessLookupError:
            phase = "stopped_by_monitor" if stopped else "training_process_finished"
            write_status(
                status_json,
                {
                    "phase": phase,
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "best_checkpoint": best_checkpoint.as_posix(),
                    "best_source_checkpoint": best_source_checkpoint,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            break

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
