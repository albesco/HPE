from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


DEFAULT_FRAME_COUNT = 15000
DEFAULT_MAX_OK_FRAME_COUNT = 299
DEFAULT_FRAME_WIDTH = 1920
DEFAULT_FRAME_HEIGHT = 1080
DEFAULT_INPUT = Path("data/input/Freestyle_part2.zip/Side_above_water/")
DEFAULT_OUTPUT_DIR = Path("data/intermediate/SAW/")
DEFAULT_LABELS_ZIP = Path("data/input/Freestyle_labels.zip")
DEFAULT_LABELS_ROOT = "Freestyle/Side_above_water"
DEFAULT_FRAME_EXTENSION = ".jpg"
DEFAULT_JPEG_QUALITY = 95
MAX_JPEG_QSCALE = 1
TARGET_LABEL_RELATIVE_PATH = "COCO/2D_cam.txt"
TARGET_LABEL_SUFFIX = "__COCO__2D_cam.txt"
MANIFEST_FILENAME = "manifest.json"
FRAME_FILE_RE = re.compile(r"^(?P<video>.+?)(?:__frame_|_)(?P<idx>\d{6})\.(png|jpg|jpeg)$", re.IGNORECASE)


@dataclass(frozen=True)
class InputSpec:
    zip_path: Path
    internal_root: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a random subset of valid frames from a SwimXYZ ZIP dataset, "
            "optionally keep only samples whose keypoints are inside the frame, "
            "and write image-label pairs plus a manifest."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Combined input path in the form <zip-file>.zip/<internal/dataset/path>. "
            f"Defaults to {DEFAULT_INPUT}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for extracted image-label pairs. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--labels-zip",
        type=Path,
        default=DEFAULT_LABELS_ZIP,
        help=(
            "ZIP containing COCO/2D_cam.txt labels. If the file does not exist, "
            "the input ZIP is used as a fallback. "
            f"Defaults to {DEFAULT_LABELS_ZIP}."
        ),
    )
    parser.add_argument(
        "--labels-root",
        default=DEFAULT_LABELS_ROOT,
        help=(
            "Root path inside the labels ZIP used to derive label paths from video paths. "
            f"Defaults to {DEFAULT_LABELS_ROOT}."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_FRAME_COUNT,
        help=f"Number of image-label pairs to export. Defaults to {DEFAULT_FRAME_COUNT}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible global frame sampling.",
    )
    parser.add_argument(
        "--max-ok-frames",
        type=int,
        default=DEFAULT_MAX_OK_FRAME_COUNT,
        help=(
            "Maximum number of annotation rows considered from the beginning of each video. "
            f"Defaults to {DEFAULT_MAX_OK_FRAME_COUNT}."
        ),
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help=f"Frame width used for keypoint bounds checking. Defaults to {DEFAULT_FRAME_WIDTH}.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=DEFAULT_FRAME_HEIGHT,
        help=f"Frame height used for keypoint bounds checking. Defaults to {DEFAULT_FRAME_HEIGHT}.",
    )
    parser.add_argument(
        "--filter-kp-inside-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only frames whose keypoints are entirely inside the image. Enabled by default.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(6, (os.cpu_count() or 2) - 1)),
        help="Number of parallel worker processes. Defaults to min(6, CPU count - 1).",
    )
    parser.add_argument(
        "--hwaccel",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="ffmpeg hardware acceleration mode. Use cuda to target NVIDIA GPUs such as V100. Defaults to auto.",
    )
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="ffmpeg executable path or command name. Defaults to ffmpeg from PATH.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Optional parent directory for temporary extracted videos.",
    )
    parser.add_argument(
        "--prune-invalid-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove previously exported files that no longer pass the keypoint filter. Enabled by default.",
    )
    return parser.parse_args()


def split_zip_internal_path(combined_path: Path) -> InputSpec:
    parts = combined_path.as_posix().split("/")
    zip_index = next((idx for idx, part in enumerate(parts) if part.lower().endswith(".zip")), None)
    if zip_index is None:
        raise ValueError("--input must contain a .zip file followed by an optional internal path.")
    zip_path = Path("/".join(parts[: zip_index + 1]))
    internal_root = "/".join(part for part in parts[zip_index + 1 :] if part)
    return InputSpec(zip_path=zip_path, internal_root=internal_root.strip("/"))


def ffmpeg_install_hint() -> str:
    if platform.system().lower() == "linux":
        return (
            "Install ffmpeg with one of these commands: "
            "sudo apt update && sudo apt install -y ffmpeg; "
            "or conda install -c conda-forge ffmpeg; "
            "or mamba install -c conda-forge ffmpeg."
        )
    return "Install ffmpeg and add it to PATH, or pass its path through --ffmpeg."


def resolve_ffmpeg_command(explicit_value: str | None) -> str:
    candidates = [explicit_value] if explicit_value else ["ffmpeg"]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return str(candidate_path)
        if shutil.which(candidate):
            return candidate
    raise FileNotFoundError("Unable to find ffmpeg. " + ffmpeg_install_hint())


def cuda_available(ffmpeg_cmd: str) -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        gpu_result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if gpu_result.returncode != 0:
            return False
        hwaccel_result = subprocess.run([ffmpeg_cmd, "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=10)
        return hwaccel_result.returncode == 0 and "cuda" in hwaccel_result.stdout.lower()
    except (OSError, subprocess.TimeoutExpired):
        return False


def normalize_zip_member(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def sanitize_name(relative_path: str) -> str:
    return normalize_zip_member(relative_path).replace("/", "__")


def convert_filename(old_filename: str) -> str:
    name, ext = Path(old_filename).stem, Path(old_filename).suffix
    replacements = {
        "Freestyle__Side_above_water__": "FSAW_",
        "Side_above_water__": "SAW_",
        "Swimmer_Skin_": "Skin_",
        "Water_Quantity_": "Water_Q_",
        "Height_": "Hght_",
        "Lighting_rotx_": "Light_rx_",
        "Speed_": "Spd_",
        "position_": "pos_",
        "__frame_": "_",
    }
    for old_text, new_text in replacements.items():
        name = name.replace(old_text, new_text)
    name = name.replace(",", "_")
    return name + ext


def get_legacy_video_output_stem(entry: dict) -> str:
    video_name = sanitize_name(entry["video_relative_path"])
    return Path(video_name).stem


def get_video_output_stem(entry: dict) -> str:
    return Path(convert_filename(get_legacy_video_output_stem(entry))).stem


def get_frame_output_name(entry: dict, frame_idx: int) -> str:
    legacy_name = f"{get_legacy_video_output_stem(entry)}__frame_{frame_idx:06d}{DEFAULT_FRAME_EXTENSION}"
    return convert_filename(legacy_name)


def get_label_output_name(entry: dict, frame_idx: int) -> str:
    legacy_name = f"{get_legacy_video_output_stem(entry)}__frame_{frame_idx:06d}{TARGET_LABEL_SUFFIX}"
    return convert_filename(legacy_name)


def parse_label_lines(label_text: str) -> tuple[str, list[str]]:
    lines = [line.strip() for line in label_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Label file is empty.")
    return lines[0], lines[1:]


def parse_decimal(value: str) -> float:
    return float(value.replace(",", "."))


def row_has_all_keypoints_inside_frame(row: str, frame_width: int, frame_height: int) -> bool:
    values = [value.strip() for value in row.split(";") if value.strip()]
    if len(values) < 3 or len(values) % 3 != 0:
        return False
    for index in range(0, len(values), 3):
        try:
            x = parse_decimal(values[index])
            y = parse_decimal(values[index + 1])
        except ValueError:
            return False
        if not (0.0 <= x < frame_width and 0.0 <= y < frame_height):
            return False
    return True


def label_candidates(video_member: str, input_root: str, labels_root: str) -> list[str]:
    video_path = PurePosixPath(normalize_zip_member(video_member))
    video_without_ext = video_path.with_suffix("").as_posix()
    input_root = normalize_zip_member(input_root)
    labels_root = normalize_zip_member(labels_root)
    candidates = [
        f"Freestyle/{video_without_ext}/{TARGET_LABEL_RELATIVE_PATH}",
        f"{video_without_ext}/{TARGET_LABEL_RELATIVE_PATH}",
    ]
    if input_root and video_without_ext.startswith(input_root + "/"):
        rest = video_without_ext[len(input_root) + 1 :]
        candidates.append(f"{labels_root}/{rest}/{TARGET_LABEL_RELATIVE_PATH}")
    elif labels_root:
        candidates.append(f"{labels_root}/{video_path.stem}/{TARGET_LABEL_RELATIVE_PATH}")
    seen: set[str] = set()
    return [candidate for candidate in candidates if not (candidate in seen or seen.add(candidate))]


def build_valid_entries(
    input_spec: InputSpec,
    labels_zip_path: Path,
    labels_root: str,
    max_ok_frames: int,
) -> list[dict]:
    if max_ok_frames <= 0:
        raise ValueError("--max-ok-frames must be greater than zero.")
    if not input_spec.zip_path.exists():
        raise FileNotFoundError(f"Input ZIP not found: {input_spec.zip_path}")

    effective_labels_zip_path = labels_zip_path if labels_zip_path.exists() else input_spec.zip_path
    valid_entries: list[dict] = []

    with ZipFile(input_spec.zip_path) as video_zip, ZipFile(effective_labels_zip_path) as labels_zip:
        video_members = set(video_zip.namelist())
        label_members = set(labels_zip.namelist())
        root = normalize_zip_member(input_spec.internal_root)
        root_prefix = f"{root}/" if root else ""
        candidate_videos = sorted(
            member for member in video_members
            if member.startswith(root_prefix) and member.lower().endswith(".webm")
        )

        for video_member in candidate_videos:
            label_member = next(
                (candidate for candidate in label_candidates(video_member, root, labels_root) if candidate in label_members),
                None,
            )
            if label_member is None:
                continue
            header, rows = parse_label_lines(labels_zip.read(label_member).decode("utf-8"))
            usable_frame_count = min(max_ok_frames, len(rows))
            if usable_frame_count <= 0:
                continue
            valid_entries.append(
                {
                    "video_zip_path": str(input_spec.zip_path),
                    "labels_zip_path": str(effective_labels_zip_path),
                    "video_relative_path": video_member,
                    "label_member": label_member,
                    "label_header": header,
                    "label_rows": rows,
                    "usable_frame_count": usable_frame_count,
                }
            )
    return valid_entries


def get_existing_frame_keys(output_dir: Path) -> set[tuple[str, int]]:
    existing_keys: set[tuple[str, int]] = set()
    if not output_dir.exists():
        return existing_keys
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        match = FRAME_FILE_RE.match(path.name)
        if match:
            video_stem = Path(convert_filename(match.group("video"))).stem
            existing_keys.add((video_stem, int(match.group("idx"))))
    return existing_keys


def get_frame_and_label_output_paths(output_dir: Path, entry: dict, frame_idx: int) -> tuple[Path, Path]:
    return output_dir / get_frame_output_name(entry, frame_idx), output_dir / get_label_output_name(entry, frame_idx)


def prune_invalid_existing_outputs(
    entries: list[dict],
    output_dir: Path,
    frame_width: int,
    frame_height: int,
    filter_kp_inside_frame: bool,
    prune_invalid_existing: bool,
) -> int:
    if not filter_kp_inside_frame or not prune_invalid_existing:
        return 0
    removed_count = 0
    for entry in entries:
        for frame_idx in range(entry["usable_frame_count"]):
            if row_has_all_keypoints_inside_frame(entry["label_rows"][frame_idx], frame_width, frame_height):
                continue
            frame_output_path, label_output_path = get_frame_and_label_output_paths(output_dir, entry, frame_idx)
            if frame_output_path.exists():
                frame_output_path.unlink()
                removed_count += 1
            if label_output_path.exists():
                label_output_path.unlink()
                removed_count += 1
    return removed_count


def build_available_frames(
    entries: list[dict],
    output_dir: Path,
    frame_width: int,
    frame_height: int,
    filter_kp_inside_frame: bool,
) -> list[dict]:
    existing_frame_keys = get_existing_frame_keys(output_dir)
    available_frames: list[dict] = []
    for entry in entries:
        video_stem = get_video_output_stem(entry)
        for frame_idx in range(entry["usable_frame_count"]):
            if (video_stem, frame_idx) in existing_frame_keys:
                continue
            if filter_kp_inside_frame and not row_has_all_keypoints_inside_frame(entry["label_rows"][frame_idx], frame_width, frame_height):
                continue
            available_frames.append({"entry": entry, "frame_idx": frame_idx})
    return available_frames


def select_frames(available_frames: list[dict], count: int, seed: int | None) -> list[dict]:
    if count <= 0:
        raise ValueError("--count must be greater than zero.")
    if count > len(available_frames):
        raise ValueError(f"Requested {count} frames, but only {len(available_frames)} eligible frames are available.")
    rng = random.Random(seed)
    return rng.sample(available_frames, count)


def extract_zip_member(archive: ZipFile, member_name: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member_name) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def ffmpeg_command(
    ffmpeg_cmd: str,
    video_path: Path,
    frame_idx: int,
    output_path: Path,
    use_cuda: bool,
) -> list[str]:
    command = [ffmpeg_cmd, "-y"]
    if use_cuda:
        command.extend(["-hwaccel", "cuda"])
    command.extend(
        [
            "-i", str(video_path),
            "-vf", f"select=eq(n\\,{frame_idx})",
            "-vframes", "1",
            "-q:v", str(MAX_JPEG_QSCALE),
            "-qmin", str(MAX_JPEG_QSCALE),
            "-qmax", str(MAX_JPEG_QSCALE),
            "-pix_fmt", "yuvj444p",
            str(output_path),
        ]
    )
    return command


def run_ffmpeg_extract_frame(
    ffmpeg_cmd: str,
    video_path: Path,
    frame_idx: int,
    output_path: Path,
    hwaccel: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    use_cuda = hwaccel == "cuda"
    command = ffmpeg_command(ffmpeg_cmd, video_path, frame_idx, output_path, use_cuda)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return
    if hwaccel == "cuda":
        raise RuntimeError(f"ffmpeg failed with CUDA for frame {frame_idx} in {video_path.name}: {result.stderr.strip()}")
    raise RuntimeError(f"ffmpeg failed for frame {frame_idx} in {video_path.name}: {result.stderr.strip()}")


def write_frame_label(output_path: Path, header: str, row: str) -> None:
    output_path.write_text(f"{header}\n{row}\n", encoding="utf-8")


def process_video_group(args: tuple) -> list[dict]:
    (
        video_relative_path,
        items,
        output_dir,
        ffmpeg_cmd,
        hwaccel,
        tmp_parent,
    ) = args
    output_dir = Path(output_dir)
    tmp_parent_path = Path(tmp_parent) if tmp_parent else None
    written_items: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="swimxyz_frames_", dir=tmp_parent_path) as tmp_dir:
        entry = items[0]["entry"]
        tmp_video_path = Path(tmp_dir) / Path(video_relative_path).name
        with ZipFile(entry["video_zip_path"]) as video_zip:
            extract_zip_member(video_zip, video_relative_path, tmp_video_path)
        for item in sorted(items, key=lambda value: value["frame_idx"]):
            frame_idx = item["frame_idx"]
            frame_output_name = get_frame_output_name(entry, frame_idx)
            label_output_name = get_label_output_name(entry, frame_idx)
            frame_output_path = output_dir / frame_output_name
            label_output_path = output_dir / label_output_name
            run_ffmpeg_extract_frame(ffmpeg_cmd, tmp_video_path, frame_idx, frame_output_path, hwaccel)
            write_frame_label(label_output_path, entry["label_header"], entry["label_rows"][frame_idx])
            written_items.append(
                {
                    "video_relative_path": entry["video_relative_path"],
                    "label_member": entry["label_member"],
                    "frame_idx": frame_idx,
                    "frame_file": frame_output_name,
                    "label_file": label_output_name,
                }
            )
    return written_items


def write_selected_frames(
    selected_frames: list[dict],
    output_dir: Path,
    ffmpeg_cmd: str,
    hwaccel: str,
    workers: int,
    tmp_dir: Path | None,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped_frames: dict[str, list[dict]] = defaultdict(list)
    for item in selected_frames:
        grouped_frames[item["entry"]["video_relative_path"]].append(item)

    worker_count = max(1, min(workers, len(grouped_frames)))
    tasks = [
        (video_relative_path, items, str(output_dir), ffmpeg_cmd, hwaccel, str(tmp_dir) if tmp_dir else None)
        for video_relative_path, items in grouped_frames.items()
    ]
    written_items: list[dict] = []
    completed = 0
    total = len(selected_frames)

    if worker_count == 1:
        for task in tasks:
            chunk = process_video_group(task)
            written_items.extend(chunk)
            completed += len(chunk)
            print(f"[{completed}/{total}] {(completed / total) * 100:5.1f}%", flush=True)
        return written_items

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(process_video_group, task) for task in tasks]
        for future in as_completed(futures):
            chunk = future.result()
            written_items.extend(chunk)
            completed += len(chunk)
            print(f"[{completed}/{total}] {(completed / total) * 100:5.1f}%", flush=True)
    return written_items


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {"items": []}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def update_manifest(manifest_path: Path, output_dir: Path, manifest: dict, written_items: list[dict], count: int, max_ok_frames: int) -> None:
    items = manifest.setdefault("items", [])
    tracked_keys = {(item["video_relative_path"], item["frame_idx"]) for item in items}
    for item in written_items:
        key = (item["video_relative_path"], item["frame_idx"])
        if key in tracked_keys:
            continue
        items.append(item)
        tracked_keys.add(key)
    items.sort(key=lambda item: (item["video_relative_path"], item["frame_idx"]))
    manifest["output_dir"] = str(output_dir)
    manifest["requested_count"] = count
    manifest["max_ok_frames"] = max_ok_frames
    manifest["item_count"] = len(items)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_spec = split_zip_internal_path(args.input)
    ffmpeg_cmd = resolve_ffmpeg_command(args.ffmpeg)
    requested_hwaccel = args.hwaccel
    if requested_hwaccel == "auto":
        effective_hwaccel = "cuda" if cuda_available(ffmpeg_cmd) else "cpu"
    else:
        effective_hwaccel = requested_hwaccel
        if effective_hwaccel == "cuda" and not cuda_available(ffmpeg_cmd):
            raise RuntimeError("CUDA ffmpeg acceleration was requested, but no usable NVIDIA/CUDA ffmpeg setup was detected.")

    valid_entries = build_valid_entries(input_spec, args.labels_zip, args.labels_root, args.max_ok_frames)
    removed_invalid_outputs = prune_invalid_existing_outputs(
        valid_entries,
        args.output_dir,
        args.frame_width,
        args.frame_height,
        args.filter_kp_inside_frame,
        args.prune_invalid_existing,
    )
    available_frames = build_available_frames(
        valid_entries,
        args.output_dir,
        args.frame_width,
        args.frame_height,
        args.filter_kp_inside_frame,
    )
    selected_frames = select_frames(available_frames, args.count, args.seed)

    print(f"Input ZIP: {input_spec.zip_path}")
    print(f"Input internal root: {input_spec.internal_root or '/'}")
    print(f"Output dir: {args.output_dir}")
    print(f"Valid videos with labels: {len(valid_entries)}")
    print(f"Removed invalid existing files: {removed_invalid_outputs}")
    print(f"Eligible new frames: {len(available_frames)}")
    print(f"Selected frames: {len(selected_frames)}")
    print(f"Workers: {args.workers}")
    print(f"ffmpeg hwaccel: {effective_hwaccel}")
    print(f"KP inside-frame filter: {args.filter_kp_inside_frame}")

    written_items = write_selected_frames(
        selected_frames=selected_frames,
        output_dir=args.output_dir,
        ffmpeg_cmd=ffmpeg_cmd,
        hwaccel=effective_hwaccel,
        workers=args.workers,
        tmp_dir=args.tmp_dir,
    )
    manifest_path = args.output_dir / MANIFEST_FILENAME
    manifest = load_manifest(manifest_path)
    update_manifest(manifest_path, args.output_dir, manifest, written_items, args.count, args.max_ok_frames)
    print(f"Written frames: {len(written_items)}")
    print(f"Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
