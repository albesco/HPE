from __future__ import annotations

import argparse
from pathlib import Path

import cv2

# This helper is intentionally independent from the training split pipeline.
# We use it when we need to inspect frame/label alignment directly on a source
# SwimXYZ video before generating a full dataset, or when we want to compare a
# few frames after changing the mapping rules.
from prepare_swimxyz_vitposepp_utils import (
    BODY25_TO_COCO,
    COCO_SKELETON,
    build_keypoints_and_bbox,
    read_label_rows,
)


KEYPOINT_COLORS = {
    "nose": (255, 200, 0),
    "left_eye": (0, 200, 255),
    "right_eye": (255, 150, 0),
    "left_ear": (0, 180, 255),
    "right_ear": (255, 120, 0),
    "left_shoulder": (0, 255, 0),
    "right_shoulder": (0, 128, 255),
    "left_elbow": (0, 220, 0),
    "right_elbow": (0, 100, 255),
    "left_wrist": (0, 190, 0),
    "right_wrist": (0, 80, 255),
    "left_hip": (0, 255, 80),
    "right_hip": (80, 140, 255),
    "left_knee": (0, 255, 140),
    "right_knee": (120, 160, 255),
    "left_ankle": (0, 255, 200),
    "right_ankle": (160, 180, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render SwimXYZ labels (mapped to COCO17 for the consolidated VitPose++ flow) on specific "
            "frames of a given video, to verify frame/label alignment."
        )
    )
    parser.add_argument("--video", required=True, help="Path to the .webm video.")
    parser.add_argument("--labels", required=True, help="Path to the SwimXYZ label .txt file.")
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        help="Frame indices to render (0-based, aligned to label row indices).",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Render all readable frames that have a corresponding label row.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where PNGs will be written.")
    parser.add_argument("--flip-y", action="store_true", default=True)
    parser.add_argument("--no-flip-y", dest="flip_y", action="store_false")
    parser.add_argument("--bbox-padding-ratio", type=float, default=0.05)
    parser.add_argument("--min-visible-keypoints", type=int, default=1)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument(
        "--points-only",
        action="store_true",
        help="Draw only keypoint markers (no skeleton lines).",
    )
    parser.add_argument(
        "--draw-bbox",
        action="store_true",
        help="Draw the bounding box computed from visible keypoints.",
    )
    return parser.parse_args()


def _draw_skeleton(
    image,
    keypoints: list[tuple[float, float, float]],
    thickness: int,
) -> None:
    for a, b in COCO_SKELETON:
        ax, ay, av = keypoints[a - 1]
        bx, by, bv = keypoints[b - 1]
        if av <= 0 or bv <= 0:
            continue
        cv2.line(
            image,
            (int(round(ax)), int(round(ay))),
            (int(round(bx)), int(round(by))),
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def _draw_points(
    image,
    keypoints: list[tuple[float, float, float]],
    radius: int,
) -> None:
    names = [name for _, name in BODY25_TO_COCO]
    for (x, y, v), name in zip(keypoints, names):
        if v <= 0:
            continue
        color = KEYPOINT_COLORS.get(name, (200, 200, 200))
        cv2.circle(image, (int(round(x)), int(round(y))), radius, color, -1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    labels_path = Path(args.labels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_rows = read_label_rows(labels_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    # The script supports both modes we used during debugging:
    # - targeted frame probes
    # - rendering of the full video sequence
    if args.all_frames:
        wanted = list(range(len(label_rows)))
    elif args.frames:
        wanted = sorted(set(args.frames))
    else:
        raise ValueError("Pass --frames or --all-frames.")

    wanted_set = set(wanted)
    max_wanted = max(wanted)

    frame_index = 0
    while frame_index <= max_wanted:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index in wanted_set:
            if frame_index >= len(label_rows):
                raise RuntimeError(
                    f"Missing label row for frame {frame_index}: labels have {len(label_rows)} rows"
                )
            height, width = frame.shape[:2]
            prepared = build_keypoints_and_bbox(
                row=label_rows[frame_index],
                width=width,
                height=height,
                bbox_padding_ratio=args.bbox_padding_ratio,
                min_visible_keypoints=args.min_visible_keypoints,
                flip_y=args.flip_y,
            )
            if prepared is None:
                raise RuntimeError(f"No valid keypoints for frame {frame_index}")
            keypoints_flat, _, bbox, _ = prepared
            keypoints = []
            for i in range(0, len(keypoints_flat), 3):
                keypoints.append((keypoints_flat[i], keypoints_flat[i + 1], keypoints_flat[i + 2]))

            rendered = frame.copy()
            if not args.points_only:
                _draw_skeleton(rendered, keypoints, thickness=args.thickness)
            _draw_points(rendered, keypoints, radius=args.radius)
            if args.draw_bbox:
                x, y, w, h = bbox
                cv2.rectangle(
                    rendered,
                    (int(round(x)), int(round(y))),
                    (int(round(x + w)), int(round(y + h))),
                    (180, 180, 0),
                    args.thickness,
                    cv2.LINE_AA,
                )

            out_path = output_dir / f"{video_path.stem}_frame_{frame_index:06d}.png"
            cv2.imwrite(str(out_path), rendered)

        frame_index += 1

    cap.release()


if __name__ == "__main__":
    main()
