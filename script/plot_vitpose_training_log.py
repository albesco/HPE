from __future__ import annotations

import argparse
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
EVAL_AP_RE = re.compile(r"AP:\s+(?P<ap>-?\d+(?:\.\d+)?(?:e-?\d+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from an MMPose TextLoggerHook log.")
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


def parse_eval_ap(log_text: str) -> ScalarSeries | None:
    epochs: list[int] = []
    values: list[float] = []
    current_epoch: int | None = None

    for line in log_text.splitlines():
        match = re.search(r"Epoch\((?P<epoch>\d+)\)", line)
        if match:
            current_epoch = int(match.group("epoch"))

        ap_match = EVAL_AP_RE.search(line)
        if ap_match and current_epoch is not None:
            epochs.append(current_epoch)
            values.append(float(ap_match.group("ap")))

    if not epochs:
        return None
    return ScalarSeries(name="AP", epochs=epochs, values=values)


def save_series_plot(series: ScalarSeries, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series.epochs, series.values, marker="o", linewidth=1.5)
    ax.set_title(series.name)
    ax.set_xlabel("epoch")
    ax.set_ylabel(series.name)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


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
    eval_ap = parse_eval_ap(log_text)

    for name, scalar in sorted(scalars.items(), key=lambda item: item[0]):
        safe_name = name.replace("/", "_").replace(" ", "_")
        save_series_plot(scalar, output_dir / f"{safe_name}__{timestamp}.png")

    if eval_ap is not None:
        save_series_plot(eval_ap, output_dir / f"eval_AP__{timestamp}.png")

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
