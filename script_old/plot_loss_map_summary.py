from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Row:
    epoch: int
    avg_loss: float
    samples: int
    ap: float | None


AP_RE = re.compile(r"Epoch\\(val\\) \\[(?P<epoch>\\d+)\\]\\[\\d+\\]\\s+AP:\\s+(?P<ap>[0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training avg loss + validation AP from a CSV summary and a log.")
    parser.add_argument(
        "--csv",
        default="data/intermediate/Side_above_water/_train_canonical/reports/training_plots/"
        "loss_map_summary__20260515_145601.csv",
        help="CSV with columns: epoch,avg_loss,samples,AP (AP may be empty).",
    )
    parser.add_argument(
        "--log",
        default="runs/vitposepp_side_above_water_aniso_20x25_min15/20260514_191100.log",
        help="Training log to scrape Epoch(val) AP lines from (fills missing AP entries).",
    )
    parser.add_argument(
        "--out",
        default="data/intermediate/Side_above_water/_train_canonical/reports/training_plots/"
        "loss_mAP_combined__AUTO.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[Row]:
    rows: list[Row] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            epoch = int(raw["epoch"])
            avg_loss = float(raw["avg_loss"])
            samples = int(raw["samples"])
            ap_raw = (raw.get("AP", "") or "").strip()
            ap = float(ap_raw) if ap_raw else None
            rows.append(Row(epoch=epoch, avg_loss=avg_loss, samples=samples, ap=ap))
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    return rows


def scrape_ap_from_log(log_path: Path) -> dict[int, float]:
    ap_by_epoch: dict[int, float] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = AP_RE.search(line)
        if not match:
            continue
        ap_by_epoch[int(match["epoch"])] = float(match["ap"])
    return ap_by_epoch


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    log_path = Path(args.log)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)
    ap_by_epoch = scrape_ap_from_log(log_path) if log_path.is_file() else {}

    epochs = [r.epoch for r in rows]
    losses = [r.avg_loss for r in rows]
    aps: list[float | None] = []
    for r in rows:
        ap = r.ap
        if ap is None and r.epoch in ap_by_epoch:
            ap = ap_by_epoch[r.epoch]
        aps.append(ap)

    ap_epochs = [e for e, ap in zip(epochs, aps) if ap is not None]
    ap_values = [ap for ap in aps if ap is not None]

    # Avoid matplotlib dependency; draw a lightweight chart with OpenCV.
    import cv2
    import numpy as np

    width, height = 1400, 700
    margin_l, margin_r, margin_t, margin_b = 90, 90, 70, 90
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    img = np.full((height, width, 3), 255, dtype=np.uint8)

    min_epoch, max_epoch = min(epochs), max(epochs)
    min_loss, max_loss = min(losses), max(losses)
    min_ap, max_ap = (min(ap_values), max(ap_values)) if ap_values else (0.0, 1.0)

    loss_pad = (max_loss - min_loss) * 0.05 if max_loss > min_loss else 1e-6
    ap_pad = (max_ap - min_ap) * 0.05 if max_ap > min_ap else 1e-6
    min_loss -= loss_pad
    max_loss += loss_pad
    min_ap -= ap_pad
    max_ap += ap_pad

    def x_of(epoch: int) -> int:
        if max_epoch == min_epoch:
            return margin_l
        return int(margin_l + (epoch - min_epoch) / (max_epoch - min_epoch) * plot_w)

    def y_of_loss(loss: float) -> int:
        if max_loss == min_loss:
            return margin_t + plot_h
        return int(margin_t + (max_loss - loss) / (max_loss - min_loss) * plot_h)

    def y_of_ap(ap: float) -> int:
        if max_ap == min_ap:
            return margin_t + plot_h
        return int(margin_t + (max_ap - ap) / (max_ap - min_ap) * plot_h)

    # Axes + grid
    cv2.rectangle(img, (margin_l, margin_t), (margin_l + plot_w, margin_t + plot_h), (0, 0, 0), 2)
    for i in range(1, 5):
        y = margin_t + int(i * plot_h / 5)
        cv2.line(img, (margin_l, y), (margin_l + plot_w, y), (220, 220, 220), 1)
    for i in range(1, 5):
        x = margin_l + int(i * plot_w / 5)
        cv2.line(img, (x, margin_t), (x, margin_t + plot_h), (220, 220, 220), 1)

    # Title
    cv2.putText(
        img,
        "VitPose++ training summary (avg loss + val AP)",
        (margin_l, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # Loss (orange)
    loss_color = (30, 120, 240)
    loss_pts = np.array([(x_of(e), y_of_loss(l)) for e, l in zip(epochs, losses)], dtype=np.int32)
    cv2.polylines(img, [loss_pts], isClosed=False, color=loss_color, thickness=3)

    # AP (blue)
    ap_color = (240, 80, 30)
    if ap_epochs:
        ap_pts = np.array([(x_of(e), y_of_ap(a)) for e, a in zip(ap_epochs, ap_values)], dtype=np.int32)
        cv2.polylines(img, [ap_pts], isClosed=False, color=ap_color, thickness=3)
        for (x, y), a in zip(ap_pts.tolist(), ap_values):
            cv2.circle(img, (x, y), 6, ap_color, -1)
            cv2.putText(
                img,
                f"{a:.4f}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                ap_color,
                1,
                cv2.LINE_AA,
            )

    # Labels + legend
    cv2.putText(img, "epoch", (margin_l + plot_w // 2 - 40, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "avg train loss", (margin_l, height - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, loss_color, 2)
    cv2.putText(img, "val AP (mAP)", (margin_l + 240, height - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ap_color, 2)
    cv2.putText(img, "loss (left axis)", (margin_l, height - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, loss_color, 2)
    cv2.putText(img, "AP (right axis)", (width - margin_r - 230, height - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ap_color, 2)

    if not cv2.imwrite(str(out_path), img):
        raise RuntimeError(f"Failed to write: {out_path}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
