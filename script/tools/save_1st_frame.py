from __future__ import annotations

import argparse
from pathlib import Path

import cv2


DEFAULT_PROJECT_ROOT = Path("/home/albertosco/HPE")
DEFAULT_OUTPUT_DIR = "data/output/preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Salva il primo frame di un video.")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("video_input", help="Percorso del file video.")
    parser.add_argument(
        "--output",
        default="",
        help="Percorso del file PNG di output. Se omesso usa data/output/preview/<video>_0.png",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def default_output_path(project_root: Path, video_input: Path) -> Path:
    return (project_root / DEFAULT_OUTPUT_DIR / f"{video_input.stem}_0.png").resolve()


def salva_primo_frame(video_input: Path, output_path: Path) -> None:
    if not video_input.is_file():
        raise FileNotFoundError(f"Errore: Il file '{video_input}' non esiste.")

    cap = cv2.VideoCapture(str(video_input))
    if not cap.isOpened():
        raise RuntimeError(f"Errore: Impossibile aprire il file video '{video_input}'.")

    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError("Errore: Impossibile leggere il frame dal video.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    print(f"Successo: Primo frame salvato come '{output_path}'")


if __name__ == "__main__":
    args = parse_args()
    project_root = resolve_path(Path.cwd(), args.project_root)
    video_input = resolve_path(project_root, args.video_input)
    output_path = (
        resolve_path(project_root, args.output)
        if args.output
        else default_output_path(project_root, video_input)
    )
    salva_primo_frame(video_input, output_path)
