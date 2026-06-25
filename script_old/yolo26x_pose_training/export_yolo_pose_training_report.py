#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOSS_COLUMNS = [
    ('train/pose_loss', 'train pose loss'),
    ('val/pose_loss', 'val pose loss'),
    ('train/box_loss', 'train box loss'),
    ('val/box_loss', 'val box loss'),
]
MAP_COLUMNS = [
    ('metrics/mAP50-95(P)', 'pose mAP50-95'),
    ('metrics/mAP50(P)', 'pose mAP50'),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export YOLO pose training metrics and plots from results.csv.')
    parser.add_argument('--results-csv', required=True)
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def load_rows(results_csv: Path) -> list[dict[str, str]]:
    with results_csv.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float | None:
    if value is None or value == '':
        return None
    try:
        return float(value)
    except ValueError:
        return None


def save_csv(rows: list[dict[str, str]], out_csv: Path) -> None:
    if not rows:
        raise SystemExit(f'No rows found in {out_csv}')
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_series(rows: list[dict[str, str]], columns: list[tuple[str, str]], output_path: Path, title: str, ylabel: str) -> None:
    epochs = [int(float(row['epoch'])) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for column, label in columns:
        values = [as_float(row.get(column)) for row in rows]
        if not any(value is not None for value in values):
            continue
        cleaned_epochs = [epoch for epoch, value in zip(epochs, values) if value is not None]
        cleaned_values = [value for value in values if value is not None]
        ax.plot(cleaned_epochs, cleaned_values, marker='o', markersize=3, linewidth=1.5, label=label)
        plotted = True
    if not plotted:
        raise SystemExit(f'No plottable columns found for {output_path.name}')
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_csv = Path(args.results_csv).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    rows = load_rows(results_csv)
    save_csv(rows, out_csv)
    plot_series(rows, LOSS_COLUMNS, output_dir / 'loss_by_epoch.png', 'YOLO26x-Pose loss by epoch', 'loss')
    plot_series(rows, MAP_COLUMNS, output_dir / 'map50_95_by_epoch.png', 'YOLO26x-Pose mAP by epoch', 'mAP')
    print(out_csv)
    print(output_dir / 'loss_by_epoch.png')
    print(output_dir / 'map50_95_by_epoch.png')


if __name__ == '__main__':
    main()
