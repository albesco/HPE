from __future__ import annotations

import argparse
import re
from pathlib import Path


EPOCH_CHECKPOINT_RE = re.compile(r"^epoch(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep the latest N Ultralytics periodic epoch checkpoints."
    )
    parser.add_argument("--weights-dir", required=True, help="Ultralytics run weights directory.")
    parser.add_argument("--keep", type=int, default=3, help="Number of epoch*.pt files to retain.")
    parser.add_argument("--dry-run", action="store_true", help="Print removals without deleting files.")
    return parser.parse_args()


def epoch_number(path: Path) -> int | None:
    match = EPOCH_CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else None


def main() -> None:
    args = parse_args()
    weights_dir = Path(args.weights_dir).expanduser().resolve()
    if args.keep < 0:
        raise ValueError("--keep must be non-negative")
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Missing weights directory: {weights_dir}")

    epoch_checkpoints = []
    for path in weights_dir.glob("epoch*.pt"):
        epoch = epoch_number(path)
        if epoch is not None:
            epoch_checkpoints.append((epoch, path))
    epoch_checkpoints.sort()

    to_remove = epoch_checkpoints[:-args.keep] if args.keep else epoch_checkpoints
    retained = epoch_checkpoints[-args.keep:] if args.keep else []

    for _epoch, path in to_remove:
        print(f"remove {path}")
        if not args.dry_run:
            path.unlink()
    for _epoch, path in retained:
        print(f"keep {path}")


if __name__ == "__main__":
    main()
