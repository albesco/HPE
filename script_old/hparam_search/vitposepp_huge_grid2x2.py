#!/usr/bin/env python3
"""Run a local 2x2 VitPose++ huge hyperparameter search on train/val only.

The search varies only learning rate and crop size across four 5-epoch runs.
It uses the canonical Side_above_water COCO17 dataset and never touches the
held-out test split for hyperparameter selection.

Recommended heavy-run usage:

    tmux new-session -d -s vitposepp_huge_grid2x2 \
      'cd /home/albertosco/HPE && python script/hparam_search/vitposepp_huge_grid2x2.py'

Dry-run / static setup check:

    python script/hparam_search/vitposepp_huge_grid2x2.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG = REPO_ROOT / "data/intermediate/Side_above_water/_train_canonical/generated_configs/swimxyz_vitposepp_huge.py"
DEFAULT_PRETRAINED_CHECKPOINT = REPO_ROOT / "models/pose/wholebody.pth"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/hparam_search/vitposepp_huge"
EPOCHS = 5
EVAL_INTERVAL = 1
CHECKPOINT_INTERVAL = 1
LOG_INTERVAL = 20
STATUS_INTERVAL = 20
GRID = (
    (1, 0.00067, 384, 128),
    (2, 0.00100, 384, 128),
    (3, 0.00067, 512, 128),
    (4, 0.00100, 512, 128),
)
METRIC_FIELDS = (
    "keypoint_ap_50_95",
    "keypoint_ap50",
    "keypoint_ap75",
    "keypoint_ar",
    "mean_keypoint_error",
    "distal_keypoint_error",
    "failure_rate",
    "crop_border_keypoint_pct",
    "best_epoch",
    "final_epoch",
)


@dataclass(frozen=True)
class SearchConfig:
    index: int
    lr: float
    crop_width: int
    crop_height: int

    @property
    def name(self) -> str:
        return f"cfg_{self.index:02d}_lr_{self.lr:.5f}_crop_{self.crop_width}x{self.crop_height}"

    @property
    def heatmap_width(self) -> int:
        return self.crop_width // 4

    @property
    def heatmap_height(self) -> int:
        return self.crop_height // 4


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 2x2 VitPose++ huge lr/crop-size search on train/val only."
    )
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG.as_posix())
    parser.add_argument("--pretrained-checkpoint", default=DEFAULT_PRETRAINED_CHECKPOINT.as_posix())
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--conda-env", default="vitpose")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--yolo-detector-model", default="", help="Optional designated YOLO detector checkpoint path (for provenance only; not used during VitPose++ training).")
    parser.add_argument("--yolo-detector-config", default="", help="Optional designated YOLO detector config name/id (for provenance only).")
    parser.add_argument("--rerun-failed", action="store_true", help="Rerun failed configs; resume from latest.pth if present.")
    parser.add_argument("--rerun-running", action="store_true", help="Treat running statuses as stale; resume from latest.pth if present.")
    parser.add_argument("--dry-run", action="store_true", help="Create dirs/configs/reports without training.")
    parser.add_argument(
        "--launch-tmux",
        action="store_true",
        help="Start this grid search in a detached tmux session and exit.",
    )
    parser.add_argument("--tmux-session", default="vitposepp_huge_grid2x2")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "failed", "error": f"Invalid JSON in {path}"}


def parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def search_configs() -> list[SearchConfig]:
    return [SearchConfig(index=index, lr=lr, crop_width=width, crop_height=height) for index, lr, width, height in GRID]


def status_path(run_dir: Path) -> Path:
    return run_dir / "status.json"


def update_status(run_dir: Path, payload: dict[str, Any]) -> None:
    current = read_json(status_path(run_dir))
    current.update(payload)
    current["updated_at"] = utc_now()
    write_json(status_path(run_dir), current)


def write_effective_config(config_path: Path, base_config: Path, pretrained_checkpoint: Path, run_dir: Path, config: SearchConfig) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""_base_ = ['{base_config.as_posix()}']

# Generated by script/hparam_search/vitposepp_huge_grid2x2.py.
# Crop convention: MMPose TopDown data_cfg.image_size is [width, height].
# The ViT backbone img_size uses (height, width), matching existing 256x192 configs.

load_from = '{pretrained_checkpoint.as_posix()}'
work_dir = '{run_dir.as_posix()}'
total_epochs = {EPOCHS}

optimizer = dict(lr={config.lr:.5f})
checkpoint_config = dict(interval={CHECKPOINT_INTERVAL}, max_keep_ckpts=1, create_symlink=True)
evaluation = dict(interval={EVAL_INTERVAL}, metric='mAP', save_best='AP')

model = dict(backbone=dict(img_size=({config.crop_height}, {config.crop_width})))

dataset_data_cfg = dict(
    image_size=[{config.crop_width}, {config.crop_height}],
    heatmap_size=[{config.heatmap_width}, {config.heatmap_height}],
    num_output_channels=17,
    num_joints=17,
    dataset_channel=[list(range(17))],
    inference_channel=list(range(17)),
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    max_num_joints=17,
    dataset_idx=0,
)

data = dict(
    train=dict(data_cfg=dataset_data_cfg),
    val=dict(data_cfg=dataset_data_cfg),
    test=dict(data_cfg=dataset_data_cfg),
)
"""
    config_path.write_text(text, encoding="utf-8")


def config_command(args: argparse.Namespace, config_path: Path, run_dir: Path, resume_from: Path | None) -> list[str]:
    command = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        "src/vitpose_base/tools/train.py",
        config_path.as_posix(),
        "--work-dir",
        run_dir.as_posix(),
        "--log-interval",
        str(LOG_INTERVAL),
        "--status-file",
        (run_dir / "training_status.txt").as_posix(),
        "--status-interval",
        str(STATUS_INTERVAL),
        "--gpu-id",
        str(args.gpu_id),
    ]
    if resume_from is not None:
        command.extend(["--resume-from", resume_from.as_posix()])
    return command


def epoch_number(value: str) -> int:
    match = re.search(r"epoch_(\d+)", value)
    return int(match.group(1)) if match else -1


def latest_checkpoint(run_dir: Path) -> Path | None:
    latest = run_dir / "latest.pth"
    if latest.exists():
        return latest
    epoch_paths = sorted(run_dir.glob("epoch_*.pth"), key=lambda path: epoch_number(path.name))
    if epoch_paths:
        return epoch_paths[-1]
    return None


def checkpoint_paths(run_dir: Path) -> dict[str, str | None]:
    best_paths = sorted(run_dir.glob("best_*.pth"), key=lambda path: epoch_number(path.name))
    latest = latest_checkpoint(run_dir)
    return {
        "best": best_paths[-1].as_posix() if best_paths else None,
        "latest": (run_dir / "latest.pth").as_posix() if (run_dir / "latest.pth").exists() else None,
        "final_epoch": latest.as_posix() if latest is not None else None,
    }


def read_validation_metrics(run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for log_json in sorted(run_dir.glob("*.log.json")):
        for line in log_json.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("mode") == "val":
                rows.append(row)
    if not rows:
        return {
            "epochs": [],
            "best_epoch": None,
            "final_epoch": None,
            "keypoint_ap_50_95": None,
            "keypoint_ap50": None,
            "keypoint_ap75": None,
            "keypoint_ar": None,
            "mean_keypoint_error": None,
            "distal_keypoint_error": None,
            "failure_rate": None,
            "crop_border_keypoint_pct": None,
        }

    def ap_value(row: dict[str, Any]) -> float:
        value = parse_float(row.get("AP"), float("-inf"))
        return value if value is not None else float("-inf")

    best = max(rows, key=lambda row: (ap_value(row), parse_float(row.get("AP .75"), float("-inf")) or float("-inf")))
    final = max(rows, key=lambda row: int(row.get("epoch", 0)))
    return {
        "epochs": [
            {
                "epoch": row.get("epoch"),
                "AP": parse_float(row.get("AP")),
                "AP50": parse_float(row.get("AP .5")),
                "AP75": parse_float(row.get("AP .75")),
                "AP_M": parse_float(row.get("AP (M)")),
                "AP_L": parse_float(row.get("AP (L)")),
                "AR": parse_float(row.get("AR")),
            }
            for row in rows
        ],
        "best_epoch": best.get("epoch"),
        "final_epoch": final.get("epoch"),
        "keypoint_ap_50_95": parse_float(best.get("AP")),
        "keypoint_ap50": parse_float(best.get("AP .5")),
        "keypoint_ap75": parse_float(best.get("AP .75")),
        "keypoint_ar": parse_float(best.get("AR")),
        "mean_keypoint_error": None,
        "distal_keypoint_error": None,
        "failure_rate": None,
        "crop_border_keypoint_pct": None,
    }


def metric_value(row: dict[str, Any], key: str, missing: float = float("-inf")) -> float:
    value = parse_float(row.get(key))
    return value if value is not None else missing


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [row for row in rows if row.get("status") == "completed"]
    if not completed:
        return None

    def ranking(row: dict[str, Any]) -> tuple[float, float, float, int]:
        mean_error = metric_value(row, "mean_keypoint_error", float("inf"))
        return (
            metric_value(row, "keypoint_ap_50_95"),
            metric_value(row, "keypoint_ap75"),
            -mean_error,
            -int(row["crop_width"]),
        )

    best = dict(max(completed, key=ranking))
    best["selection_reason"] = (
        "Selected by highest validation Keypoint AP@[OKS 0.50:0.95]; ties use AP75, "
        "then lower mean keypoint error if available, then prefer crop 384x128 for lower complexity. "
        "The test split was not used."
    )
    return best


def config_row(config: SearchConfig, run_dir: Path) -> dict[str, Any]:
    status = read_json(status_path(run_dir))
    metrics = read_validation_metrics(run_dir)
    row: dict[str, Any] = {
        "config": config.name,
        "index": config.index,
        "lr": config.lr,
        "crop_width": config.crop_width,
        "crop_height": config.crop_height,
        "status": status.get("status", "pending"),
        "run_dir": run_dir.as_posix(),
        "config_path": (run_dir / "effective_config.py").as_posix(),
        "best_checkpoint": checkpoint_paths(run_dir)["best"],
        "latest_checkpoint": checkpoint_paths(run_dir)["latest"],
        "final_epoch_checkpoint": checkpoint_paths(run_dir)["final_epoch"],
        "error": status.get("error"),
    }
    for field in METRIC_FIELDS:
        row[field] = metrics.get(field)
    return row


def write_summary(output_root: Path, configs: list[SearchConfig]) -> None:
    rows = [config_row(config, output_root / config.name) for config in configs]
    fieldnames = [
        "config",
        "index",
        "lr",
        "crop_width",
        "crop_height",
        "status",
        *METRIC_FIELDS,
        "best_checkpoint",
        "latest_checkpoint",
        "final_epoch_checkpoint",
        "run_dir",
        "config_path",
        "error",
    ]
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = choose_best(rows)
    write_json(output_root / "summary.json", {"updated_at": utc_now(), "configs": rows})
    write_json(output_root / "best_config.json", best or {"status": "unavailable"})
    write_report(output_root, rows, best)


def render_value(value: Any) -> str:
    return "" if value is None else str(value)


def write_report(output_root: Path, rows: list[dict[str, Any]], best: dict[str, Any] | None) -> None:
    lines = [
        "# VitPose++ Huge 2x2 Hyperparameter Search",
        "",
        "Purpose: choose VitPose++ huge learning rate and horizontal crop size for the top-down YOLO26x-detector -> VitPose++ huge pipeline.",
        "",
        "Selection split: validation only. The test split is not used for hyperparameter choice.",
        "",
        "Dataset: canonical Side_above_water COCO17 top-down dataset under `data/intermediate/Side_above_water/_train_canonical/` with split counts train=18181, val=5195, test=2597.",
        "",
        "BBox convention: training/validation use the existing padded GT bboxes (`use_gt_bbox=True`). Detector-produced validation bbox files are `not reconstructible from workspace files`, so no additional box expansion is introduced.",
        "Designated detector (provenance only): set via --yolo-detector-config/--yolo-detector-model; not used for VitPose++ training.",
        "",
        "Crop convention: MMPose TopDown `data_cfg.image_size` is `[width, height]`. The requested crop sizes are therefore written as `image_size=[384, 128]` and `image_size=[512, 128]`; the ViT backbone `img_size` is written as `(height, width)`.",
        "",
        "Grid: lr in {0.00067, 0.00100}; crop in {384x128, 512x128}; each run trains for 5 epochs from the pretrained `models/pose/wholebody.pth` weights.",
        "",
        "Invariant training settings: AdamW optimizer type, scheduler, weight decay, layer-wise decay, augmentation, head parameters, canonical COCO17 labels, batch/workers, and all other reference VitPose++ huge settings are inherited from the baseline config unless required for crop size, 5-epoch training, validation cadence, or checkpoint retention.",
        "",
        "Checkpoint policy: `checkpoint_config.interval=1`, `max_keep_ckpts=1`, `create_symlink=True`, and `evaluation.save_best='AP'`. MMCV uses the periodic hook to maintain `latest.pth`, while `max_keep_ckpts=1` prevents retained intermediate epoch checkpoints.",
        "",
        "Unavailable secondary metrics: mean keypoint error, distal keypoint errors, failure rate, and crop-border keypoint percentage are not emitted by standard MMPose train validation logs; they remain null unless a separate evaluator is added.",
        "",
        "## Best Config",
        "",
    ]
    if best is None:
        lines.append("No completed configuration is available yet.")
    else:
        lines.extend(
            [
                f"- Config: `{best['config']}`",
                f"- Validation Keypoint AP@[OKS 0.50:0.95]: `{best.get('keypoint_ap_50_95')}`",
                f"- Validation AP50: `{best.get('keypoint_ap50')}`",
                f"- Validation AP75: `{best.get('keypoint_ap75')}`",
                f"- Validation AR: `{best.get('keypoint_ar')}`",
                f"- Best checkpoint: `{best.get('best_checkpoint')}`",
                f"- Latest checkpoint: `{best.get('latest_checkpoint')}`",
                f"- Reason: {best.get('selection_reason')}",
            ]
        )

    lines.extend(
        [
            "",
            "## Configs",
            "",
            "| Config | lr | crop | Status | Best Epoch | AP | AP50 | AP75 | AR | Best checkpoint |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {config} | {lr:.5f} | {crop_width}x{crop_height} | {status} | {best_epoch} | {ap} | {ap50} | {ap75} | {ar} | {best_ckpt} |".format(
                config=row["config"],
                lr=row["lr"],
                crop_width=row["crop_width"],
                crop_height=row["crop_height"],
                status=row["status"],
                best_epoch=render_value(row.get("best_epoch")),
                ap=render_value(row.get("keypoint_ap_50_95")),
                ap50=render_value(row.get("keypoint_ap50")),
                ap75=render_value(row.get("keypoint_ap75")),
                ar=render_value(row.get("keypoint_ar")),
                best_ckpt=render_value(row.get("best_checkpoint")),
            )
        )
    lines.extend(
        [
            "",
            "## Usage",
            "",
            "Run the full search inside tmux:",
            "",
            "```bash",
            "tmux new-session -d -s vitposepp_huge_grid2x2 'cd /home/albertosco/HPE && python script/hparam_search/vitposepp_huge_grid2x2.py'",
            "```",
            "",
            "Run a dry-run setup check:",
            "",
            "```bash",
            "python script/hparam_search/vitposepp_huge_grid2x2.py --dry-run",
            "```",
            "",
        ]
    )
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def should_skip(run_dir: Path, args: argparse.Namespace) -> tuple[bool, str | None]:
    if args.dry_run:
        return False, None
    status = read_json(status_path(run_dir))
    state = status.get("status")
    if state == "completed":
        return True, "completed"
    if state == "failed" and not args.rerun_failed:
        return True, "failed"
    if state == "running" and not args.rerun_running:
        return True, "running"
    return False, None


def write_run_inputs(run_dir: Path, args: argparse.Namespace, config: SearchConfig, command: list[str], resume_from: Path | None) -> None:
    (run_dir / "command.txt").write_text(" ".join(shlex.quote(part) for part in command) + "\n", encoding="utf-8")
    write_json(
        run_dir / "training_args.json",
        {
            **asdict(config),
            "name": config.name,
            "epochs": EPOCHS,
            "base_config": resolve_path(args.base_config).as_posix(),
            "pretrained_checkpoint": resolve_path(args.pretrained_checkpoint).as_posix(),
            "run_dir": run_dir.as_posix(),
            "resume_from": resume_from.as_posix() if resume_from else None,
            "varied_hyperparameters": ["optimizer.lr", "crop_size"],
            "crop_convention": {
                "requested": "width x height",
                "mmpose_data_cfg_image_size": "[width, height]",
                "vit_backbone_img_size": "(height, width)",
            },
            "uses_test_split_for_selection": False,
            "bbox_source": "padded GT bboxes from canonical COCO17 annotations; detector bbox file is not documented yet",
            "checkpoint_policy": "checkpoint interval=1, max_keep_ckpts=1, create_symlink=True, save_best=AP",
            "command": command,
        },
    )


def run_config(args: argparse.Namespace, config: SearchConfig, output_root: Path) -> None:
    run_dir = output_root / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "effective_config.py"
    base_config = resolve_path(args.base_config)
    pretrained_checkpoint = resolve_path(args.pretrained_checkpoint)
    write_effective_config(config_path, base_config, pretrained_checkpoint, run_dir, config)

    skip, reason = should_skip(run_dir, args)
    if skip:
        print(f"skip {config.name}: {reason}")
        write_summary(output_root, search_configs())
        return

    resume_from = latest_checkpoint(run_dir)
    command = config_command(args, config_path, run_dir, resume_from)
    write_run_inputs(run_dir, args, config, command, resume_from)

    if args.dry_run:
        update_status(
            run_dir,
            {
                "status": "pending",
                "config": config.name,
                "dry_run": True,
                "command": command,
                "resume_from": resume_from.as_posix() if resume_from else None,
            },
        )
        print(f"dry-run {config.name}: {' '.join(shlex.quote(part) for part in command)}")
        write_summary(output_root, search_configs())
        return

    log_path = run_dir / "stdout_stderr.log"
    update_status(
        run_dir,
        {
            "status": "running",
            "config": config.name,
            "started_at": utc_now(),
            "dry_run": False,
            "resume_from": resume_from.as_posix() if resume_from else None,
            "log_path": log_path.as_posix(),
            "command": command,
        },
    )
    write_summary(output_root, search_configs())

    with log_path.open("ab") as log_file:
        log_file.write(f"\n# Started {utc_now()}\n".encode("utf-8"))
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
        return_code = process.wait()
        log_file.write(f"\n# Finished {utc_now()} exit_code={return_code}\n".encode("utf-8"))

    metrics = read_validation_metrics(run_dir)
    write_json(run_dir / "validation_metrics.json", metrics)
    if return_code == 0:
        update_status(
            run_dir,
            {
                "status": "completed",
                "completed_at": utc_now(),
                "exit_code": return_code,
                "metrics": metrics,
                "checkpoints": checkpoint_paths(run_dir),
            },
        )
    else:
        update_status(
            run_dir,
            {
                "status": "failed",
                "failed_at": utc_now(),
                "exit_code": return_code,
                "metrics": metrics,
                "checkpoints": checkpoint_paths(run_dir),
                "error": f"Training command exited with code {return_code}",
            },
        )
    write_summary(output_root, search_configs())


def validate_static_setup(args: argparse.Namespace, output_root: Path) -> None:
    base_config = resolve_path(args.base_config)
    pretrained_checkpoint = resolve_path(args.pretrained_checkpoint)
    if not base_config.exists():
        raise FileNotFoundError(f"Missing VitPose++ base config: {base_config}")
    if not pretrained_checkpoint.exists():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {pretrained_checkpoint}")

    configs = search_configs()
    payloads: list[dict[str, Any]] = []
    for config in configs:
        run_dir = output_root / config.name
        effective_config = run_dir / "effective_config.py"
        payloads.append(
            {
                "config": config.name,
                "lr": config.lr,
                "crop_width": config.crop_width,
                "crop_height": config.crop_height,
                "image_size": [config.crop_width, config.crop_height],
                "heatmap_size": [config.heatmap_width, config.heatmap_height],
                "backbone_img_size": [config.crop_height, config.crop_width],
                "effective_config": effective_config.as_posix(),
            }
        )

    write_json(
        output_root / "static_validation.json",
        {
            "validated_at": utc_now(),
            "varied_hyperparameters": ["optimizer.lr", "crop_size"],
            "crop_related_fields": ["dataset_data_cfg.image_size", "dataset_data_cfg.heatmap_size", "model.backbone.img_size"],
            "allowed_operational_differences": ["work_dir", "run_dir", "config_path", "command resume_from if same-run latest.pth exists"],
            "invariant_settings": {
                "base_config": base_config.as_posix(),
                "pretrained_checkpoint": pretrained_checkpoint.as_posix(),
                "epochs": EPOCHS,
                "evaluation_interval": EVAL_INTERVAL,
                "checkpoint_interval": CHECKPOINT_INTERVAL,
                "checkpoint_max_keep_ckpts": 1,
                "checkpoint_create_symlink": True,
                "uses_test_split_for_selection": False,
                "bbox_source": "padded GT bbox in canonical train/val annotations",
                "crop_convention": "MMPose image_size=[width,height], ViT backbone img_size=(height,width)",
            },
            "configs": payloads,
        },
    )


def launch_in_tmux(args: argparse.Namespace) -> None:
    if os.environ.get("TMUX"):
        return
    script_path = Path(__file__).resolve()
    forwarded = [arg for arg in sys.argv[1:] if arg != "--launch-tmux"]
    inner_command = " ".join(
        [
            "cd",
            shlex.quote(REPO_ROOT.as_posix()),
            "&&",
            "python",
            shlex.quote(script_path.as_posix()),
            *[shlex.quote(arg) for arg in forwarded],
        ]
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", args.tmux_session, inner_command], check=True)
    print(f"Launched tmux session: {args.tmux_session}")
    print(f"Attach with: tmux attach -t {args.tmux_session}")
    raise SystemExit(0)


def write_search_args(args: argparse.Namespace, output_root: Path) -> None:
    write_json(
        output_root / "search_args.json",
        {
            "created_or_updated_at": utc_now(),
            "grid": [asdict(config) | {"name": config.name} for config in search_configs()],
            "base_config": resolve_path(args.base_config).as_posix(),
            "pretrained_checkpoint": resolve_path(args.pretrained_checkpoint).as_posix(),
            "output_root": output_root.as_posix(),
            "epochs_per_run": EPOCHS,
            "selection_metric": "validation_keypoint_ap_50_95",
            "selection_tiebreakers": ["keypoint_ap75", "mean_keypoint_error", "prefer_crop_384x128"],
            "uses_test_split_for_selection": False,
            "detector_bbox_file": "not reconstructible from workspace files",
            "yolo_detector_model": resolve_path(args.yolo_detector_model).as_posix() if args.yolo_detector_model else "",
            "yolo_detector_config": args.yolo_detector_config,
            "checkpoint_policy": "checkpoint interval=1; max_keep_ckpts=1; save_best=AP; keep latest.pth; retain no intermediate epoch checkpoints",
            "crop_convention": "requested crop is width x height; MMPose data_cfg.image_size=[width,height]; backbone.img_size=(height,width)",
        },
    )


def main() -> None:
    args = parse_args()
    if args.launch_tmux:
        launch_in_tmux(args)

    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    validate_static_setup(args, output_root)
    if shutil.which("conda") is None and not args.dry_run:
        raise RuntimeError("conda is not available on PATH")

    write_search_args(args, output_root)
    write_summary(output_root, search_configs())
    for config in search_configs():
        run_config(args, config, output_root)
    write_summary(output_root, search_configs())
    print(f"Search artifacts: {output_root}")


if __name__ == "__main__":
    main()
