from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ScalarSeries:
    name: str
    epochs: list[int]
    values: list[float]


EPOCH_LINE_RE = re.compile(
    r"Epoch\s+\[(?P<epoch>\d+)\]\s*\[(?P<iter>\d+)/(?P<total_iter>\d+)\]\s+"
)
SCALAR_KV_RE = re.compile(r"(?P<key>[\w/.-]+):\s+(?P<value>-?\d+(?:\.\d+)?(?:e-?\d+)?)")
EVAL_AP_LINE_RE = re.compile(
    r"Epoch\(val\)\s+\[(?P<epoch>\d+)\]\[\d+\].*?AP:\s+(?P<ap>-?\d+(?:\.\d+)?(?:e-?\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot VitPose++ loss and mAP curves from MMPose logs.")
    parser.add_argument(
        "--log-file",
        action="append",
        default=[],
        help="Path to a .log file produced by src/vitpose_base/tools/train.py. Repeatable.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where plots will be saved.")
    parser.add_argument(
        "--timestamp",
        default="",
        help="Timestamp suffix for output files (default: current UTC, YYYYmmdd_HHMMSS).",
    )
    return parser.parse_args()


def parse_training_scalars(log_text: str) -> dict[str, ScalarSeries]:
    series: dict[str, ScalarSeries] = {}
    last_epoch: int | None = None

    for line in log_text.splitlines():
        match = EPOCH_LINE_RE.search(line)
        if not match:
            continue
        epoch = int(match.group("epoch"))
        last_epoch = epoch

        for kv in SCALAR_KV_RE.finditer(line):
            key = kv.group("key")
            raw_value = kv.group("value")
            try:
                value = float(raw_value)
            except ValueError:
                continue
            if key in {"Epoch", "iter"}:
                continue
            if key not in series:
                series[key] = ScalarSeries(name=key, epochs=[], values=[])
            series[key].epochs.append(epoch)
            series[key].values.append(value)

    if last_epoch is None:
        raise RuntimeError("No training epoch lines found in log file.")
    return series


def average_by_epoch(series: ScalarSeries, name: str) -> ScalarSeries:
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    for epoch, value in zip(series.epochs, series.values):
        totals[epoch] = totals.get(epoch, 0.0) + value
        counts[epoch] = counts.get(epoch, 0) + 1
    epochs = sorted(totals)
    values = [totals[epoch] / counts[epoch] for epoch in epochs]
    return ScalarSeries(name=name, epochs=epochs, values=values)


def parse_eval_ap(log_text: str) -> ScalarSeries | None:
    values_by_epoch: dict[int, float] = {}
    for line in log_text.splitlines():
        match = EVAL_AP_LINE_RE.search(line)
        if not match:
            continue
        values_by_epoch[int(match.group("epoch"))] = float(match.group("ap"))

    if not values_by_epoch:
        return None
    epochs = sorted(values_by_epoch)
    values = [values_by_epoch[epoch] for epoch in epochs]
    return ScalarSeries(name="mAP validation", epochs=epochs, values=values)


def marker_points(series: ScalarSeries, prefer_max: bool) -> tuple[tuple[int, float], tuple[int, float]]:
    ranked = list(zip(series.epochs, series.values))
    best = max(ranked, key=lambda item: item[1]) if prefer_max else min(ranked, key=lambda item: item[1])
    last = ranked[-1]
    return best, last


def save_metric_plot(
    series: ScalarSeries,
    output_path: Path,
    ylabel: str,
    title: str,
    prefer_max: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series.epochs, series.values, marker="o", markersize=3, linewidth=1.5, color="#3b82f6")

    best, last = marker_points(series, prefer_max=prefer_max)
    ax.scatter([best[0]], [best[1]], s=38, color="#16a34a", zorder=4, label=f"best epoch {best[0]}")
    ax.scatter([last[0]], [last[1]], s=38, color="#dc2626", zorder=4, label=f"last epoch {last[0]}")

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_summary_csv(loss_series: ScalarSeries, ap_series: ScalarSeries | None, output_path: Path) -> None:
    ap_by_epoch = dict(zip(ap_series.epochs, ap_series.values)) if ap_series is not None else {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "avg_loss", "AP"])
        for epoch, loss_value in zip(loss_series.epochs, loss_series.values):
            writer.writerow([epoch, loss_value, ap_by_epoch.get(epoch, "")])


def main() -> None:
    args = parse_args()
    if not args.log_file:
        raise SystemExit("--log-file is required (repeatable).")
    output_dir = Path(args.output_dir)
    timestamp = args.timestamp.strip()
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    log_text_chunks: list[str] = []
    for log_file in args.log_file:
        log_path = Path(log_file)
        log_text_chunks.append(log_path.read_text(encoding="utf-8", errors="replace"))
    log_text = "\n".join(log_text_chunks)

    scalars = parse_training_scalars(log_text)
    if "loss" not in scalars:
        raise RuntimeError("No loss scalar found in log file.")

    loss_epoch_avg = average_by_epoch(scalars["loss"], name="loss epoch avg")
    eval_ap = parse_eval_ap(log_text)

    save_metric_plot(
        loss_epoch_avg,
        output_dir / f"loss_epoch_avg__{timestamp}.png",
        ylabel="avg loss",
        title="VitPose++ training loss by epoch",
        prefer_max=False,
    )

    if eval_ap is not None:
        save_metric_plot(
            eval_ap,
            output_dir / f"mAP_validation__{timestamp}.png",
            ylabel="mAP / AP",
            title="VitPose++ validation mAP by epoch",
            prefer_max=True,
        )

    save_summary_csv(loss_epoch_avg, eval_ap, output_dir / f"loss_map_summary__{timestamp}.csv")
    print(f"Saved metric plots to: {output_dir}")


if __name__ == "__main__":
    main()
