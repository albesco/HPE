#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FIELDNAMES = [
    'epoch', 'AP', 'AP50', 'AP75', 'AP_M', 'AP_L', 'AR', 'AR50', 'AR75', 'AR_M', 'AR_L', 'val_loss'
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export YOLO detection validation metrics and plots by epoch.')
    parser.add_argument('--results-csv', required=True)
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--start-epoch', type=int, default=1)
    return parser.parse_args()


def clean_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value for key, value in row.items()}


def to_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ''):
        return None
    return float(value)


def val_loss(row: dict[str, str]) -> float | None:
    losses = [to_float(row, key) for key in ('val/box_loss', 'val/cls_loss', 'val/dfl_loss')]
    values = [value for value in losses if value is not None]
    if not values:
        return None
    return sum(values)


def read_rows(path: Path, start_epoch: int) -> list[dict[str, Any]]:
    with path.open(newline='') as handle:
        rows = [clean_row(row) for row in csv.DictReader(handle)]

    exported: list[dict[str, Any]] = []
    for row in rows:
        local_epoch = int(float(row['epoch']))
        exported.append({
            'epoch': start_epoch + local_epoch - 1,
            'AP': to_float(row, 'metrics/mAP50-95(B)'),
            'AP50': to_float(row, 'metrics/mAP50(B)'),
            'AP75': '',
            'AP_M': '',
            'AP_L': '',
            'AR': '',
            'AR50': '',
            'AR75': '',
            'AR_M': '',
            'AR_L': '',
            'val_loss': val_loss(row),
        })
    return exported


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict[str, Any]], key: str, output: Path, title: str, ylabel: str) -> None:
    points = [(row['epoch'], row[key]) for row in rows if row[key] not in (None, '')]
    if not points:
        return
    x_values, y_values = zip(*points)
    plt.figure(figsize=(10, 5.6))
    plt.plot(x_values, y_values, marker='o', label=key)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    results_csv = Path(args.results_csv).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not results_csv.is_file():
        raise FileNotFoundError(f'Missing results CSV: {results_csv}')

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(results_csv, args.start_epoch)
    write_csv(out_csv, rows)
    plot_metric(rows, 'val_loss', output_dir / 'loss_validation_by_epoch.png', 'YOLO26x-Detection validation loss by epoch', 'validation loss')
    plot_metric(rows, 'AP', output_dir / 'map50_95_validation_by_epoch.png', 'YOLO26x-Detection validation mAP50-95 by epoch', 'AP / mAP50-95')
    print(f'Wrote validation metrics: {out_csv}')
    print(f'Wrote plots: {output_dir}')


if __name__ == '__main__':
    main()
