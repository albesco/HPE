#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

EPOCH_RE = re.compile(r"epoch_(\d+)\.pth$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor VitPose++ validation AP and enforce patience.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--session-name", default="")
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--metric", default="AP")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--keep-last", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--status-file", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--kill-session", action="store_true")
    return parser.parse_args()


def parse_status(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def read_val_rows(work_dir: Path) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for log_path in sorted(work_dir.glob("*.log.json")):
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("mode") != "val":
                continue
            epoch = payload.get("epoch")
            if epoch is None:
                continue
            try:
                epoch_num = int(epoch)
            except (TypeError, ValueError):
                continue
            row: dict[str, Any] = {"epoch": epoch_num}
            for key, value in payload.items():
                if key in {"epoch", "iter", "mode", "memory", "data_time", "time"}:
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    continue
            rows[epoch_num] = row
    return [rows[key] for key in sorted(rows)]


def epoch_from_checkpoint(path: Path) -> int | None:
    match = EPOCH_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def prune_checkpoints(work_dir: Path, best_epoch: int | None, latest_epoch: int | None, keep_last: int) -> list[str]:
    if latest_epoch is None:
        return []
    if keep_last <= 0:
        keep_last = 1
    keep_epochs = set(range(max(1, latest_epoch - keep_last + 1), latest_epoch + 1))
    if best_epoch is not None:
        keep_epochs.add(best_epoch)

    removed: list[str] = []
    for checkpoint in sorted(work_dir.glob("epoch_*.pth")):
        epoch = epoch_from_checkpoint(checkpoint)
        if epoch is None or epoch in keep_epochs:
            continue
        checkpoint.unlink(missing_ok=True)
        removed.append(checkpoint.as_posix())
    return removed


def patience_state(rows: list[dict[str, Any]], metric_name: str, min_delta: float) -> dict[str, Any]:
    best_metric = float("-inf")
    best_epoch = None
    stale_evals = 0
    metric_rows = 0
    last_metric = None
    last_epoch = None
    for row in rows:
        metric = row.get(metric_name)
        epoch = row["epoch"]
        if metric is None:
            continue
        metric_rows += 1
        last_metric = metric
        last_epoch = epoch
        if metric > best_metric + min_delta:
            best_metric = metric
            best_epoch = epoch
            stale_evals = 0
        else:
            stale_evals += 1
    return {
        "metric": metric_name,
        "best_metric": None if best_epoch is None else best_metric,
        "best_epoch": best_epoch,
        "last_metric": last_metric,
        "last_epoch": last_epoch,
        "num_evals": metric_rows,
        "stale_evals": stale_evals,
    }


def session_exists(name: str) -> bool:
    if not name:
        return False
    result = subprocess.run(["tmux", "has-session", "-t", name], check=False)
    return result.returncode == 0


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_process_group(pid: int) -> None:
    if pid <= 0:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def write_output(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    status_file = Path(args.status_file).expanduser().resolve() if args.status_file else None
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    while True:
        rows = read_val_rows(work_dir)
        status = parse_status(status_file)
        state = patience_state(rows, args.metric, args.min_delta)
        keep_last = args.keep_last if args.keep_last > 0 else args.patience
        removed = prune_checkpoints(work_dir, state["best_epoch"], state["last_epoch"], keep_last)
        payload = {
            "work_dir": work_dir.as_posix(),
            "session_name": args.session_name,
            "pid": args.pid,
            "patience": args.patience,
            "keep_last": keep_last,
            "min_delta": args.min_delta,
            "training_phase": status.get("phase"),
            "state": state,
            "removed_checkpoints": removed,
            "updated_at": int(time.time()),
        }

        if status.get("phase") == "finished":
            payload["result"] = "training_finished"
            write_output(output_path, payload)
            return

        if args.pid > 0 and not process_exists(args.pid):
            payload["result"] = "process_missing"
            write_output(output_path, payload)
            return

        if args.session_name and args.pid <= 0 and not session_exists(args.session_name):
            payload["result"] = "session_missing"
            write_output(output_path, payload)
            return

        if state["num_evals"] > 0 and state["stale_evals"] >= args.patience:
            payload["result"] = "patience_exhausted"
            write_output(output_path, payload)
            if args.pid > 0:
                stop_process_group(args.pid)
            if args.kill_session and args.session_name:
                subprocess.run(["tmux", "kill-session", "-t", args.session_name], check=False)
            return

        write_output(output_path, payload)
        time.sleep(max(5, args.poll_interval))


if __name__ == "__main__":
    main()
