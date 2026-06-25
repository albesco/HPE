from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Series:
    x: list[float]
    y: list[float]


EPOCH_ITER_RE = re.compile(r"Epoch\s+\[(?P<epoch>\d+)\]\s*\[(?P<iter>\d+)/(?P<total_iter>\d+)\]")
LOSS_RE = re.compile(r"(?:^|\s)loss:\s+(?P<loss>-?\d+(?:\.\d+)?(?:e-?\d+)?)")
EVAL_AP_RE = re.compile(
    r"Epoch\(val\)\s+\[(?P<epoch>\d+)\]\[\d+\].*?AP:\s+(?P<ap>-?\d+(?:\.\d+)?(?:e-?\d+)?)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot VitPose++ loss and mAP with checkpoint epoch markers.")
    p.add_argument(
        "--log-file",
        action="append",
        default=[],
        help="Training log file(s). Repeatable; concatenated in order.",
    )
    p.add_argument(
        "--work-dir",
        default="runs/vitposepp_side_above_water_aniso_20x25_min15",
        help="Work dir containing epoch_*.pth, best_AP_epoch_*.pth, latest.pth.",
    )
    p.add_argument(
        "--output-dir",
        default="data/intermediate/Side_above_water/_train_canonical/reports/training_plots",
    )
    p.add_argument(
        "--timestamp",
        default="",
        help="Suffix for output filenames (default: UTC YYYYmmdd_HHMMSS).",
    )
    return p.parse_args()


def read_logs(log_files: list[str]) -> str:
    chunks: list[str] = []
    for lf in log_files:
        chunks.append(Path(lf).read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def parse_loss_points(log_text: str) -> tuple[Series, dict[int, float]]:
    xs: list[float] = []
    ys: list[float] = []
    last_loss_in_epoch: dict[int, float] = {}

    for line in log_text.splitlines():
        m = EPOCH_ITER_RE.search(line)
        if not m:
            continue
        lm = LOSS_RE.search(line)
        if not lm:
            continue
        epoch = int(m.group("epoch"))
        iter_in_epoch = int(m.group("iter"))
        iters_per_epoch = int(m.group("total_iter"))
        loss = float(lm.group("loss"))

        # x as fractional epoch gives all points across all epochs.
        frac = (iter_in_epoch / iters_per_epoch) if iters_per_epoch else 0.0
        xs.append(epoch + frac)
        ys.append(loss)
        last_loss_in_epoch[epoch] = loss

    if not xs:
        raise RuntimeError("No (epoch,iter,loss) points found in logs.")
    return Series(x=xs, y=ys), last_loss_in_epoch


def parse_map_points(log_text: str) -> Series:
    by_epoch: dict[int, float] = {}
    for line in log_text.splitlines():
        m = EVAL_AP_RE.search(line)
        if not m:
            continue
        by_epoch[int(m.group("epoch"))] = float(m.group("ap"))
    if not by_epoch:
        return Series(x=[], y=[])
    epochs = sorted(by_epoch)
    return Series(x=[float(e) for e in epochs], y=[by_epoch[e] for e in epochs])


def checkpoint_epochs(work_dir: Path) -> tuple[set[int], int | None, int | None]:
    ckpt_epochs: set[int] = set()

    for p in work_dir.glob("epoch_*.pth"):
        m = re.search(r"epoch_(\d+)\.pth$", p.name)
        if m:
            ckpt_epochs.add(int(m.group(1)))

    best_epoch: int | None = None
    for p in work_dir.glob("best_AP_epoch_*.pth"):
        m = re.search(r"best_AP_epoch_(\d+)\.pth$", p.name)
        if m:
            ep = int(m.group(1))
            ckpt_epochs.add(ep)
            if best_epoch is None or ep > best_epoch:
                # This is *best file name*, not necessarily best score; still useful.
                best_epoch = ep

    last_epoch: int | None = None
    latest = work_dir / "latest.pth"
    if latest.is_symlink():
        target = latest.resolve().name
        m = re.search(r"epoch_(\d+)\.pth$", target)
        if m:
            last_epoch = int(m.group(1))
            ckpt_epochs.add(last_epoch)

    return ckpt_epochs, best_epoch, last_epoch


def plot_loss(loss: Series, loss_last_by_epoch: dict[int, float], ckpt_epochs: set[int], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build an epoch-average curve so the plot matches loss_epoch_avg__*.png style.
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    last_point_x: float | None = None
    last_point_y: float | None = None

    for x, y in zip(loss.x, loss.y):
        epoch = int(x)
        totals[epoch] = totals.get(epoch, 0.0) + y
        counts[epoch] = counts.get(epoch, 0) + 1
        last_point_x = x
        last_point_y = y

    epochs = sorted(totals)
    values = [totals[e] / counts[e] for e in epochs]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.plot(epochs, values, marker="o", markersize=3, linewidth=1.5, color="#3b82f6")
    ax.set_title("VitPose++ training loss by epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("avg loss")
    ax.grid(True, alpha=0.3)

    # Overlay checkpoint epochs as dots at the epoch-average loss when available.
    val_by_epoch = dict(zip(epochs, values))
    ckpt_x = [ep for ep in sorted(ckpt_epochs) if ep in val_by_epoch]
    ckpt_y = [val_by_epoch[ep] for ep in ckpt_x]
    if ckpt_x:
        ax.scatter(ckpt_x, ckpt_y, s=18, color="#111827", alpha=0.75, label="checkpoint epoch")

    # Best/last markers: on loss, best is min.
    best_i = min(range(len(values)), key=lambda i: values[i])
    ax.scatter([epochs[best_i]], [values[best_i]], s=42, color="#16a34a", zorder=5, label=f"best epoch {epochs[best_i]}")

    # 'last' is the latest logged point; map it to its epoch-average value for a stable marker.
    if last_point_x is not None and last_point_y is not None:
        last_epoch = int(last_point_x)
        last_y = val_by_epoch.get(last_epoch)
        if last_y is not None:
            ax.scatter([last_epoch], [last_y], s=42, color="#f59e0b", zorder=5, label=f"last epoch {last_epoch}")

    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_map(m: Series, ckpt_epochs: set[int], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5), dpi=160)
    if m.x:
        # mAP is only available at validation epochs; plot as a simple piecewise line
        # using integer epoch values (no fractional x-axis).
        ax.plot(m.x, m.y, linewidth=1.5, color="#2563eb")

        # Overlay checkpoint epochs as dots at those epochs if mAP exists for that epoch.
        map_by_epoch = {int(x): y for x, y in zip(m.x, m.y)}
        ckpt_x = [float(ep) for ep in sorted(ckpt_epochs) if ep in map_by_epoch]
        ckpt_y = [map_by_epoch[int(ep)] for ep in ckpt_x]
        if ckpt_x:
            ax.scatter(ckpt_x, ckpt_y, s=18, color="#111827", alpha=0.75, label="checkpoint epoch")

        best_i = max(range(len(m.y)), key=lambda i: m.y[i])
        ax.scatter([m.x[best_i]], [m.y[best_i]], s=42, color="#16a34a", zorder=5, label="best (max mAP)")
        ax.scatter([m.x[-1]], [m.y[-1]], s=42, color="#f59e0b", zorder=5, label="last")
    else:
        ax.text(0.5, 0.5, "No validation AP points found in logs", ha="center", va="center")

    ax.set_title("VitPose++ validation mAP/AP (all available points)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP / AP")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.log_file:
        raise SystemExit("--log-file is required (repeatable).")

    output_dir = Path(args.output_dir)
    ts = args.timestamp.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    log_text = read_logs(args.log_file)
    loss_series, loss_last_by_epoch = parse_loss_points(log_text)
    map_series = parse_map_points(log_text)

    work_dir = Path(args.work_dir)
    ckpt_epochs, _best_epoch, _last_epoch = checkpoint_epochs(work_dir)

    plot_loss(
        loss_series,
        loss_last_by_epoch,
        ckpt_epochs,
        output_dir / f"loss_all_points__{ts}.png",
    )
    plot_map(
        map_series,
        ckpt_epochs,
        output_dir / f"mAP_all_points__{ts}.png",
    )

    print(f"Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()
