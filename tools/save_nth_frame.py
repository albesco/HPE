from __future__ import annotations

import argparse
from pathlib import Path

import cv2


DEFAULT_PROJECT_ROOT = Path("/home/albertosco/HPE")
DEFAULT_OUTPUT_DIR = "data/output/preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Salva un frame specifico di un video.")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("video_input", help="Percorso del file video.")
    parser.add_argument("frame_index", type=int, help="Indice del frame da salvare.")
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Percorso del file PNG di output. Se omesso usa "
            "data/output/preview/<video>_frame_<indice>.png"
        ),
    )
    return parser.parse_args()


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def default_output_path(project_root: Path, video_input: Path, frame_index: int) -> Path:
    return (project_root / DEFAULT_OUTPUT_DIR / f"{video_input.stem}_frame_{frame_index}.png").resolve()


def salva_frame_specifico(video_input: Path, frame_to_save: int, output_path: Path) -> None:
    if not video_input.is_file():
        raise FileNotFoundError(f"Errore: Il file '{video_input}' non esiste.")

    cap = cv2.VideoCapture(str(video_input))
    if not cap.isOpened():
        raise RuntimeError(f"Errore: Impossibile aprire il file video '{video_input}'.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_to_save >= total_frames:
        cap.release()
        raise ValueError(
            f"Errore: Il frame richiesto ({frame_to_save}) eccede il totale dei frame ({total_frames})."
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_save)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(f"Errore: Impossibile leggere il frame {frame_to_save}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    print(f"Successo: Frame {frame_to_save} salvato come '{output_path}'")


if __name__ == "__main__":
    args = parse_args()
    project_root = resolve_path(Path.cwd(), args.project_root)
    video_input = resolve_path(project_root, args.video_input)
    output_path = (
        resolve_path(project_root, args.output)
        if args.output
        else default_output_path(project_root, video_input, args.frame_index)
    )
    salva_frame_specifico(video_input, args.frame_index, output_path)
