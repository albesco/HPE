from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmpose.apis import single_gpu_test
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
from mmpose.utils import setup_multi_processes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "data/intermediate/Side_above_water/_train_canonical/generated_configs/swimxyz_vitposepp_huge_grid_winner_resume.py"
DEFAULT_GRID_LOG = PROJECT_ROOT / "runs/hparam_search/vitposepp_huge/cfg_02_lr_0.00100_crop_384x128/20260527_123734.log.json"
DEFAULT_GRID_CHECKPOINT = PROJECT_ROOT / "runs/hparam_search/vitposepp_huge/cfg_02_lr_0.00100_crop_384x128/best_AP_epoch_5.pth"
DEFAULT_RESUME_LOG = PROJECT_ROOT / "runs/vitposepp_side_above_water_grid_winner_resume/20260528_061439.log.json"
DEFAULT_RESUME_DIR = PROJECT_ROOT / "runs/vitposepp_side_above_water_grid_winner_resume"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data/intermediate/Side_above_water/_train_canonical/reports/training_plots"


@dataclass(frozen=True)
class Phase:
    name: str
    log_json: Path
    overall_epoch_offset: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VitPose++ epoch report with test-set mAP50-95.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--grid-log", type=Path, default=DEFAULT_GRID_LOG)
    parser.add_argument("--grid-checkpoint", type=Path, default=DEFAULT_GRID_CHECKPOINT)
    parser.add_argument("--resume-log", type=Path, default=DEFAULT_RESUME_LOG)
    parser.add_argument("--resume-dir", type=Path, default=DEFAULT_RESUME_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def epoch_loss_rows(phase: Phase) -> list[dict[str, Any]]:
    rows = []
    grouped: dict[int, list[float]] = {}
    for item in load_jsonl(phase.log_json):
        if item.get("mode") != "train":
            continue
        epoch = int(item["epoch"])
        loss = item.get("main_stream_loss", item.get("loss"))
        if loss is None:
            continue
        grouped.setdefault(epoch, []).append(float(loss))

    for local_epoch in sorted(grouped):
        values = grouped[local_epoch]
        rows.append(
            {
                "phase": phase.name,
                "phase_epoch": local_epoch,
                "overall_epoch": phase.overall_epoch_offset + local_epoch,
                "loss_pose_epoch_avg": sum(values) / len(values),
            }
        )
    return rows


def build_test_cfg(config_path: Path, work_dir: Path) -> Config:
    cfg = Config.fromfile(str(config_path))
    setup_multi_processes(cfg)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.work_dir = str(work_dir)
    if not hasattr(cfg, "gpu_ids"):
        cfg.gpu_ids = [0]
    os.makedirs(cfg.work_dir, exist_ok=True)
    return cfg


def evaluate_checkpoint(config_path: Path, checkpoint_path: Path, work_dir: Path) -> dict[str, float]:
    cfg = build_test_cfg(config_path, work_dir)
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    loader_cfg: dict[str, Any] = dict(seed=cfg.get("seed"), drop_last=False, dist=False)
    test_loader_cfg: dict[str, Any] = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **dict(workers_per_gpu=1),
        **dict(samples_per_gpu=1),
        **cfg.data.get("test_dataloader", {}),
    }
    test_loader_cfg["samples_per_gpu"] = 1
    test_loader_cfg["workers_per_gpu"] = 1
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    model = build_posenet(cfg.model)
    load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    outputs = single_gpu_test(model, data_loader)
    eval_config = cfg.get("evaluation", {}).copy()
    eval_config.update(dict(metric=["mAP"]))
    metrics = dataset.evaluate(outputs, cfg.work_dir, **eval_config)
    return {
        "AP": float(metrics.get("AP")),
        "AP50": float(metrics.get("AP .5", metrics.get("AP50"))),
        "AP75": float(metrics.get("AP .75", metrics.get("AP75"))),
        "AR": float(metrics.get("AR")),
    }


def available_resume_checkpoints(resume_dir: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in sorted(resume_dir.glob("epoch_*.pth")):
        try:
            local_epoch = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((local_epoch, path))
    return checkpoints


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "overall_epoch",
        "phase",
        "phase_epoch",
        "loss_pose_epoch_avg",
        "test_mAP50_95",
        "test_AP50",
        "test_AP75",
        "test_AR",
        "checkpoint_path",
        "test_metrics_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_loss(path: Path, rows: list[dict[str, Any]]) -> None:
    loss_rows = [row for row in rows if row["loss_pose_epoch_avg"] not in ("", None)]
    plt.figure(figsize=(10, 5))
    plt.plot(
        [row["overall_epoch"] for row in loss_rows],
        [row["loss_pose_epoch_avg"] for row in loss_rows],
        marker="o",
        linewidth=1.5,
    )
    plt.xlabel("Overall epoch")
    plt.ylabel("Loss-Pose (train epoch avg)")
    plt.title("VitPose++ Loss-Pose by epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_map(path: Path, rows: list[dict[str, Any]]) -> None:
    map_rows = [row for row in rows if row["test_mAP50_95"] not in ("", None)]
    plt.figure(figsize=(10, 5))
    plt.plot(
        [row["overall_epoch"] for row in map_rows],
        [row["test_mAP50_95"] for row in map_rows],
        marker="o",
        linewidth=1.5,
    )
    plt.xlabel("Overall epoch")
    plt.ylabel("Test mAP50-95")
    plt.title("VitPose++ Test mAP50-95 by available checkpoint epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    test_metrics_dir = report_dir / "test_metrics_vitpose_grid_winner_resume"
    eval_work_dir = report_dir / "tmp_eval_workdir"
    eval_work_dir.mkdir(parents=True, exist_ok=True)

    phases = [
        Phase(name="grid_cfg02", log_json=args.grid_log.resolve(), overall_epoch_offset=0),
        Phase(name="winner_resume", log_json=args.resume_log.resolve(), overall_epoch_offset=5),
    ]

    rows_by_epoch: dict[int, dict[str, Any]] = {}
    for phase in phases:
        for row in epoch_loss_rows(phase):
            rows_by_epoch[row["overall_epoch"]] = {
                **row,
                "test_mAP50_95": "",
                "test_AP50": "",
                "test_AP75": "",
                "test_AR": "",
                "checkpoint_path": "",
                "test_metrics_json": "",
            }

    checkpoint_specs = [(5, "grid_cfg02", 5, args.grid_checkpoint.resolve())]
    checkpoint_specs.extend(
        (5 + local_epoch, "winner_resume", local_epoch, path.resolve())
        for local_epoch, path in available_resume_checkpoints(args.resume_dir.resolve())
    )

    for overall_epoch, phase_name, phase_epoch, checkpoint_path in checkpoint_specs:
        metrics_json = test_metrics_dir / f"epoch_{overall_epoch:02d}__{phase_name}.json"
        metrics = evaluate_checkpoint(args.config.resolve(), checkpoint_path, eval_work_dir / f"epoch_{overall_epoch:02d}")
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics_json.write_text(
            json.dumps(
                {
                    "overall_epoch": overall_epoch,
                    "phase": phase_name,
                    "phase_epoch": phase_epoch,
                    "checkpoint_path": str(checkpoint_path),
                    "metrics": metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        row = rows_by_epoch.setdefault(
            overall_epoch,
            {
                "overall_epoch": overall_epoch,
                "phase": phase_name,
                "phase_epoch": phase_epoch,
                "loss_pose_epoch_avg": "",
                "test_mAP50_95": "",
                "test_AP50": "",
                "test_AP75": "",
                "test_AR": "",
                "checkpoint_path": "",
                "test_metrics_json": "",
            },
        )
        row.update(
            {
                "test_mAP50_95": metrics["AP"],
                "test_AP50": metrics["AP50"],
                "test_AP75": metrics["AP75"],
                "test_AR": metrics["AR"],
                "checkpoint_path": str(checkpoint_path),
                "test_metrics_json": str(metrics_json),
            }
        )

    rows = [rows_by_epoch[key] for key in sorted(rows_by_epoch)]
    csv_path = report_dir / "loss_pose_test_map_by_epoch__grid_winner_resume.csv"
    loss_png = report_dir / "loss_pose_by_epoch__grid_winner_resume.png"
    map_png = report_dir / "test_map50_95_by_epoch__grid_winner_resume.png"
    write_csv(csv_path, rows)
    plot_loss(loss_png, rows)
    plot_map(map_png, rows)

    summary = {
        "csv": str(csv_path),
        "loss_png": str(loss_png),
        "test_map_png": str(map_png),
        "test_metrics_dir": str(test_metrics_dir),
        "evaluated_overall_epochs": [spec[0] for spec in checkpoint_specs],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
