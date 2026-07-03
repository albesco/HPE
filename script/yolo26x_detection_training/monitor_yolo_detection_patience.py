#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


EPOCH_RE = re.compile(r"^epoch(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor Ultralytics YOLO detection training with min-delta patience."
    )
    parser.add_argument("--run-dir", required=True, help="YOLO run directory containing results.csv")
    parser.add_argument("--metric", default="metrics/mAP50-95(B)")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.007)
    parser.add_argument("--keep-last", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--pid", type=int, help="Optional training process PID to stop")
    parser.add_argument("--tmux-session", help="Optional tmux session to stop")
    parser.add_argument("--kill-tmux", action="store_true", help="Kill tmux session instead of sending C-c")
    parser.add_argument("--status-json", help="Output monitor status JSON")
    return parser.parse_args()


def read_rows(results_csv: Path) -> list[dict[str, str]]:
    if not results_csv.is_file():
        return []
    with results_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({key.strip(): value for key, value in row.items()})
        return rows


def checkpoint_epoch(path: Path) -> int | None:
    match = EPOCH_RE.match(path.name)
    return int(match.group(1)) if match else None


def prune_epoch_checkpoints(
    weights_dir: Path,
    best_epoch: int | None,
    latest_epoch: int | None,
    keep_last: int,
) -> list[str]:
    if latest_epoch is None or not weights_dir.is_dir():
        return []

    keep_last = max(1, keep_last)
    keep_epochs = set(range(max(0, latest_epoch - keep_last + 1), latest_epoch + 1))
    if best_epoch is not None:
        keep_epochs.add(best_epoch)

    removed: list[str] = []
    for checkpoint in sorted(weights_dir.glob("epoch*.pt")):
        epoch = checkpoint_epoch(checkpoint)
        if epoch is None or epoch in keep_epochs:
            continue
        checkpoint.unlink()
        removed.append(checkpoint.as_posix())
    return removed


def write_status(status_json: Path, payload: dict[str, Any]) -> None:
    status_json.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def pid_is_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_pid(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        os.kill(pid, signal.SIGTERM)


def stop_tmux(session: str, kill_tmux: bool) -> None:
    command = ["tmux", "kill-session", "-t", session] if kill_tmux else ["tmux", "send-keys", "-t", session, "C-c"]
    subprocess.run(command, check=False)


def stop_training(args: argparse.Namespace) -> None:
    if args.pid is not None:
        stop_pid(args.pid)
    if args.tmux_session:
        stop_tmux(args.tmux_session, args.kill_tmux)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    results_csv = run_dir / "results.csv"
    weights_dir = run_dir / "weights"
    status_json = (
        Path(args.status_json).expanduser().resolve()
        if args.status_json
        else run_dir / "monitor_status.json"
    )

    best_metric: float | None = None
    best_epoch: int | None = None
    stale_epochs = 0
    last_seen_epoch: int | None = None

    write_status(
        status_json,
        {
            "phase": "waiting_for_results",
            "run_dir": run_dir.as_posix(),
            "metric": args.metric,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "keep_last": args.keep_last,
        },
    )

    while True:
        rows = read_rows(results_csv)
        if rows:
            latest = rows[-1]
            latest_epoch = int(float(latest["epoch"]))
            if latest_epoch != last_seen_epoch:
                if args.metric not in latest:
                    raise KeyError(f"Metric column not found: {args.metric}")

                metric_value = float(latest[args.metric])
                improved = best_metric is None or metric_value >= best_metric + args.min_delta
                if improved:
                    best_metric = metric_value
                    best_epoch = latest_epoch
                    stale_epochs = 0
                else:
                    stale_epochs += 1

                removed = prune_epoch_checkpoints(
                    weights_dir=weights_dir,
                    best_epoch=best_epoch,
                    latest_epoch=latest_epoch,
                    keep_last=args.keep_last,
                )
                status = {
                    "phase": "monitoring",
                    "latest_epoch": latest_epoch,
                    "metric": args.metric,
                    "latest_metric": metric_value,
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "stale_epochs": stale_epochs,
                    "patience": args.patience,
                    "min_delta": args.min_delta,
                    "keep_last": args.keep_last,
                    "removed_checkpoints": removed,
                }
                write_status(status_json, status)
                last_seen_epoch = latest_epoch

                if stale_epochs >= args.patience:
                    status["phase"] = "stopping"
                    status["reason"] = "min_delta_patience_exhausted"
                    write_status(status_json, status)
                    stop_training(args)
                    break

        if args.pid is not None and not pid_is_alive(args.pid):
            write_status(
                status_json,
                {
                    "phase": "training_process_finished",
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                },
            )
            break

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
