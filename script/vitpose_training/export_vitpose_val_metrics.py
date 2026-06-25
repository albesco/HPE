#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXCLUDE_KEYS = {"epoch", "iter", "mode", "memory", "time", "data_time"}
PREFERRED_COLUMNS = [
    "epoch",
    "AP",
    "AP .5",
    "AP .75",
    "AP (M)",
    "AP (L)",
    "AR",
    "AR .5",
    "AR .75",
    "AR (M)",
    "AR (L)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VitPose++ validation metrics by epoch from MMPose .log.json files.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def load_rows(work_dir: Path) -> list[dict[str, float]]:
    by_epoch: dict[int, dict[str, float]] = {}
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
            row: dict[str, float] = {"epoch": float(epoch_num)}
            for key, value in payload.items():
                if key in EXCLUDE_KEYS:
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    continue
            by_epoch[epoch_num] = row
    return [by_epoch[key] for key in sorted(by_epoch)]


def column_order(rows: list[dict[str, float]]) -> list[str]:
    seen: set[str] = set()
    for row in rows:
        seen.update(row.keys())
    ordered = [column for column in PREFERRED_COLUMNS if column in seen]
    extras = sorted(column for column in seen if column not in ordered)
    return ordered + extras


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    rows = load_rows(work_dir)
    if not rows:
        raise SystemExit(f"No validation rows found under {work_dir}")

    columns = column_order(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            serializable = {key: (int(value) if key == "epoch" else value) for key, value in row.items()}
            writer.writerow(serializable)

    print(out_csv)


if __name__ == "__main__":
    main()
