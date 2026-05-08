from __future__ import annotations

import argparse
from pathlib import Path

import cv2


DEFAULT_PROJECT_ROOT = Path("/home/albertosco/HPE")
DEFAULT_OUTPUT_DIR = "data/output/preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Salva tutti i frame di un video.")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("video_input", help="Percorso del file video.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory di output. Se omessa usa data/output/preview/<video>/",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def default_output_dir(project_root: Path, video_input: Path) -> Path:
    return (project_root / DEFAULT_OUTPUT_DIR / video_input.stem).resolve()


def salva_tutti_i_frame(video_input: Path, output_dir: Path) -> None:
    if not video_input.is_file():
        raise FileNotFoundError(f"Errore: Il file '{video_input}' non esiste.")

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_input))
    if not cap.isOpened():
        raise RuntimeError(f"Errore: Impossibile aprire il file video '{video_input}'.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Inizio estrazione: {total_frames} frame totali stimati.")

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        filename = f"frame_{str(frame_count).zfill(5)}.png"
        output_file_path = output_dir / filename
        cv2.imwrite(str(output_file_path), frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Progresso: {frame_count} frame salvati...")

    cap.release()
    print(f"Completato: {frame_count} frame estratti in '{output_dir}'.")


if __name__ == "__main__":
    args = parse_args()
    project_root = resolve_path(Path.cwd(), args.project_root)
    video_input = resolve_path(project_root, args.video_input)
    output_dir = (
        resolve_path(project_root, args.output_dir)
        if args.output_dir
        else default_output_dir(project_root, video_input)
    )
    salva_tutti_i_frame(video_input, output_dir)
