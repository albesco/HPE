from __future__ import annotations

import argparse
import importlib
from pathlib import Path


REQUIRED_IMPORTS = [
    "torch",
    "numpy",
    "cv2",
    "PIL",
    "tqdm",
    "ultralytics",
    "mmcv",
    "mmpose",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the HPE runtime environment.")
    parser.add_argument("--project-root", default=".", help="Project root path.")
    parser.add_argument("--yolo-model", default="models/detection/yolo26x.pt")
    parser.add_argument("--pose-checkpoint", default="models/pose/coco.pth")
    parser.add_argument(
        "--pose-config",
        default=(
            "src/vitpose_base/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/"
            "coco/ViTPose_huge_coco_256x192.py"
        ),
    )
    return parser.parse_args()


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()

    print(f"Project root: {project_root}")

    for module_name in REQUIRED_IMPORTS:
        importlib.import_module(module_name)
        print(f"import ok: {module_name}")

    import torch

    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")

    required_paths = [
        resolve(project_root, args.yolo_model),
        resolve(project_root, args.pose_checkpoint),
        resolve(project_root, args.pose_config),
        project_root / "src/vitpose_base",
        project_root / "script/run_pipeline.py",
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"path ok: {path}")

    print("Smoke test completed.")


if __name__ == "__main__":
    main()
