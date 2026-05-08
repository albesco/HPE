from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from prepare_swimxyz_vitpose_train import (
    DEFAULT_PROJECT_ROOT,
    BODY25_TO_COCO,
    DatasetEntry,
    Sample,
    build_coco_json,
    build_keypoints_and_bbox,
    dataset_name_from_converted_root,
    detect_anomalous_bbox_shift,
    iter_converted_dataset_roots,
    move_split_images,
    parse_dataset_entries,
    read_label_rows,
    resolve_path,
    split_samples,
    write_split_overlay_images,
)


DEFAULT_DATASET_ROOT = "data/intermediate"
DEFAULT_OUTPUT_ROOT = ""
DEFAULT_BASE_CONFIG = (
    "src/vitpose_base/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/"
    "vitPose+_huge_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py"
)
DEFAULT_PRETRAINED_CHECKPOINT = "models/pose/wholebody.pth"
DEFAULT_WORK_DIR = "runs/vitposepp_single_head_subset_xyz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a SwimXYZ COCO17 dataset and a single-head VitPose++ config. "
            "This keeps the verified SwimXYZ->COCO17 keypoint mapping while using "
            "the stronger VitPose++ backbone/checkpoint."
        )
    )
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing converted subset_xyz videos and labels.",
    )
    parser.add_argument(
        "--dataset-entry",
        action="append",
        default=[],
        help="Pair formatted as video_path::labels_path. Repeatable.",
    )
    parser.add_argument("--label-format", default="COCO")
    parser.add_argument("--label-dimension", default="2D")
    parser.add_argument("--label-reference", default="cam")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--pretrained-checkpoint", default=DEFAULT_PRETRAINED_CHECKPOINT)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--bbox-padding-ratio", type=float, default=0.05)
    parser.add_argument("--min-visible-keypoints", type=int, default=4)
    parser.add_argument(
        "--flip-y",
        dest="flip_y",
        action="store_true",
        default=True,
        help="Convert SwimXYZ bottom-origin y coordinates to image top-origin coordinates.",
    )
    parser.add_argument(
        "--no-flip-y",
        dest="flip_y",
        action="store_false",
        help="Keep label y coordinates unchanged.",
    )
    parser.add_argument("--samples-per-gpu", type=int, default=4)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--total-epochs", type=int, default=30)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    return parser.parse_args()


def build_vitposepp_output_root(
    project_root: Path,
    dataset_root: Path,
    output_root_value: str,
) -> Path:
    if output_root_value:
        return resolve_path(project_root, output_root_value)

    dataset_name = dataset_name_from_converted_root(dataset_root)
    return (project_root / "data" / "intermediate" / dataset_name / "_train_vitposepp").resolve()


def extract_samples_from_entry(
    entry: DatasetEntry,
    images_root: Path,
    frame_step: int,
    max_frames_per_video: int,
    bbox_padding_ratio: float,
    min_visible_keypoints: int,
    flip_y: bool,
    skip_counters: dict[str, int],
    exception_log_entries: list[str],
) -> list[Sample]:
    label_rows = read_label_rows(entry.labels_path)
    capture = cv2.VideoCapture(str(entry.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {entry.video_path}")

    source_name = entry.video_path.stem.replace(",", "_")
    source_dir = images_root / source_name
    source_dir.mkdir(parents=True, exist_ok=True)

    samples: list[Sample] = []
    frame_index = 0
    saved_count = 0
    label_index = 0
    stopped_by_frame_limit = False
    previous_bbox: list[float] | None = None
    anomaly_logged = False

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if label_index >= len(label_rows):
            skip_counters["missing_label_for_frame"] += 1
            exception_log_entries.append(
                f"{entry.video_path.name} | frame_index={frame_index} | missing_label_for_frame"
            )
            break

        if frame_index % frame_step != 0:
            frame_index += 1
            label_index += 1
            continue

        height, width = frame.shape[:2]
        prepared = build_keypoints_and_bbox(
            row=label_rows[label_index],
            width=width,
            height=height,
            bbox_padding_ratio=bbox_padding_ratio,
            min_visible_keypoints=min_visible_keypoints,
            flip_y=flip_y,
        )
        if prepared is not None:
            keypoints, num_keypoints, bbox, area = prepared
            if detect_anomalous_bbox_shift(previous_bbox, bbox):
                if not anomaly_logged:
                    exception_log_entries.append(
                        f"{entry.video_path.name} | anomalous_bbox_shift_detected"
                    )
                    anomaly_logged = True
                exception_log_entries.append(
                    f"{entry.video_path.name} | frame_index={frame_index} | anomalous_bbox_shift | previous_bbox={previous_bbox} | current_bbox={bbox}"
                )
            image_path = source_dir / f"{source_name}_{frame_index:06d}.jpg"
            cv2.imwrite(str(image_path), frame)
            samples.append(
                Sample(
                    source_name=source_name,
                    frame_index=frame_index,
                    image_path=image_path,
                    width=width,
                    height=height,
                    bbox=bbox,
                    keypoints=keypoints,
                    num_keypoints=num_keypoints,
                    area=area,
                )
            )
            previous_bbox = bbox
            saved_count += 1

            if max_frames_per_video > 0 and saved_count >= max_frames_per_video:
                stopped_by_frame_limit = True
                break
        else:
            skip_counters["no_valid_keypoints"] += 1

        frame_index += 1
        label_index += 1

    capture.release()

    if not stopped_by_frame_limit and label_index < len(label_rows):
        for missing_label_index in range(label_index, len(label_rows)):
            skip_counters["missing_frame_for_label"] += 1
            exception_log_entries.append(
                f"{entry.video_path.name} | frame_index={missing_label_index} | missing_frame_for_label"
            )

    return samples


def write_vitposepp_single_head_config(
    output_path: Path,
    dataset_root: Path,
    base_config: Path,
    pretrained_checkpoint: Path,
    work_dir: Path,
    total_epochs: int,
    val_interval: int,
    checkpoint_interval: int,
    samples_per_gpu: int,
    workers_per_gpu: int,
    project_root: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coco_dataset_config = project_root / "src/vitpose_base/configs/_base_/datasets/coco.py"
    config_text = f"""_base_ = ['{base_config.as_posix()}']

import runpy as _runpy

data_root = '{dataset_root.as_posix()}'
load_from = '{pretrained_checkpoint.as_posix()}'
work_dir = '{work_dir.as_posix()}'
total_epochs = {total_epochs}

checkpoint_config = dict(interval={checkpoint_interval}, create_symlink=False)
evaluation = dict(interval={val_interval}, metric='mAP', save_best='AP')

model = dict(associate_keypoint_head=[])

coco_dataset_info = _runpy.run_path('{coco_dataset_config.as_posix()}')['dataset_info']
del _runpy

target_type = 'GaussianHeatmap'
dataset_data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
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

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(type='TopDownHalfBodyTransform', num_joints_half_body=8, prob_half_body=0.3),
    dict(type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2, encoding='UDP', target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'
        ],
    ),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs', 'dataset_idx'
        ],
    ),
]

test_pipeline = val_pipeline

data = dict(
    _delete_=True,
    samples_per_gpu={samples_per_gpu},
    workers_per_gpu={workers_per_gpu},
    val_dataloader=dict(samples_per_gpu={max(1, samples_per_gpu)}),
    test_dataloader=dict(samples_per_gpu={max(1, samples_per_gpu)}),
    train=dict(
        type='TopDownCocoDataset',
        ann_file=f'{{data_root}}/annotations/person_keypoints_train.json',
        img_prefix=f'{{data_root}}/train2017/',
        data_cfg=dataset_data_cfg,
        pipeline=train_pipeline,
        dataset_info=coco_dataset_info,
    ),
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{{data_root}}/annotations/person_keypoints_val.json',
        img_prefix=f'{{data_root}}/val2017/',
        data_cfg=dataset_data_cfg,
        pipeline=val_pipeline,
        dataset_info=coco_dataset_info,
    ),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{{data_root}}/annotations/person_keypoints_test.json',
        img_prefix=f'{{data_root}}/test2017/',
        data_cfg=dataset_data_cfg,
        pipeline=test_pipeline,
        dataset_info=coco_dataset_info,
    ),
)
"""
    output_path.write_text(config_text, encoding="utf-8")


def prepare_dataset(
    args: argparse.Namespace,
    project_root: Path,
    dataset_root: Path,
    output_root: Path,
) -> None:
    scoped_args = argparse.Namespace(**vars(args))
    scoped_args.dataset_root = str(dataset_root)
    dataset_entries = parse_dataset_entries(scoped_args, project_root)
    if args.max_videos > 0:
        dataset_entries = dataset_entries[: args.max_videos]

    base_config = resolve_path(project_root, args.base_config)
    pretrained_checkpoint = resolve_path(project_root, args.pretrained_checkpoint)
    work_dir = resolve_path(project_root, args.work_dir)

    if not base_config.is_file():
        raise FileNotFoundError(f"Missing base config: {base_config}")
    if not pretrained_checkpoint.is_file():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {pretrained_checkpoint}")

    images_root = output_root / "_tmp_images"
    train_dir = output_root / "train2017"
    val_dir = output_root / "val2017"
    test_dir = output_root / "test2017"
    annotations_dir = output_root / "annotations"
    reports_dir = output_root / "reports"
    generated_config = output_root / "generated_configs" / "swimxyz_vitposepp_huge_single_head.py"

    if output_root.exists():
        shutil.rmtree(output_root)
    images_root.mkdir(parents=True, exist_ok=True)

    skip_counters = {
        "missing_frame_for_label": 0,
        "missing_label_for_frame": 0,
        "no_valid_keypoints": 0,
    }
    exception_log_entries: list[str] = []

    all_samples: list[Sample] = []
    for entry in dataset_entries:
        all_samples.extend(
            extract_samples_from_entry(
                entry=entry,
                images_root=images_root,
                frame_step=args.frame_step,
                max_frames_per_video=args.max_frames_per_video,
                bbox_padding_ratio=args.bbox_padding_ratio,
                min_visible_keypoints=args.min_visible_keypoints,
                flip_y=args.flip_y,
                skip_counters=skip_counters,
                exception_log_entries=exception_log_entries,
            )
        )

    if not all_samples:
        raise RuntimeError("No valid SwimXYZ samples were extracted.")

    all_samples.sort(key=lambda sample: (sample.source_name, sample.frame_index))
    train_samples, val_samples, test_samples = split_samples(
        all_samples,
        args.val_ratio,
        args.test_ratio,
    )
    move_split_images(train_samples, train_dir)
    move_split_images(val_samples, val_dir)
    move_split_images(test_samples, test_dir)
    write_split_overlay_images(train_samples)
    write_split_overlay_images(val_samples)
    write_split_overlay_images(test_samples)

    annotations_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / "person_keypoints_train.json").write_text(
        json.dumps(build_coco_json(train_samples, "train2017"), indent=2),
        encoding="utf-8",
    )
    (annotations_dir / "person_keypoints_val.json").write_text(
        json.dumps(build_coco_json(val_samples, "val2017"), indent=2),
        encoding="utf-8",
    )
    (annotations_dir / "person_keypoints_test.json").write_text(
        json.dumps(build_coco_json(test_samples, "test2017"), indent=2),
        encoding="utf-8",
    )

    log_path = annotations_dir / "dataset_exceptions.log"
    if exception_log_entries:
        log_path.write_text("\n".join(exception_log_entries) + "\n", encoding="utf-8")
    else:
        log_path.write_text("No exceptions detected.\n", encoding="utf-8")

    report = {
        "dataset_root": dataset_root.as_posix(),
        "output_root": output_root.as_posix(),
        "label_format": args.label_format,
        "keypoint_encoding": [name for _, name in BODY25_TO_COCO],
        "videos": len(dataset_entries),
        "samples": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
            "total": len(all_samples),
        },
        "skips": skip_counters,
        "flip_y_enabled": args.flip_y,
    }
    (reports_dir / "preparation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    write_vitposepp_single_head_config(
        output_path=generated_config,
        dataset_root=output_root,
        base_config=base_config,
        pretrained_checkpoint=pretrained_checkpoint,
        work_dir=work_dir,
        total_epochs=args.total_epochs,
        val_interval=args.val_interval,
        checkpoint_interval=args.checkpoint_interval,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        project_root=project_root,
    )

    shutil.rmtree(images_root, ignore_errors=True)

    print(f"Prepared SwimXYZ dataset for single-head VitPose++: {dataset_name_from_converted_root(dataset_root)}")
    print(f"Converted dataset root: {dataset_root}")
    print(f"Dataset root: {output_root}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(
        "Skipped samples (missing frame / missing label / invalid keypoints): "
        f"{skip_counters['missing_frame_for_label']} / "
        f"{skip_counters['missing_label_for_frame']} / "
        f"{skip_counters['no_valid_keypoints']}"
    )
    print(f"Exception log: {log_path}")
    print(f"Generated config: {generated_config}")
    print("Training command:")
    print(
        "conda run -n vitpose python "
        f"{(project_root / 'src/vitpose_base/tools/train.py').as_posix()} "
        f"{generated_config.as_posix()}"
    )


def main() -> None:
    args = parse_args()
    project_root = resolve_path(Path.cwd(), args.project_root)
    dataset_root = resolve_path(project_root, args.dataset_root)

    if args.dataset_entry:
        output_root = build_vitposepp_output_root(project_root, dataset_root, args.output_root)
        prepare_dataset(args, project_root, dataset_root, output_root)
        return

    dataset_roots = iter_converted_dataset_roots(dataset_root)
    if len(dataset_roots) > 1 and args.output_root:
        raise ValueError(
            "When processing multiple converted datasets, do not pass a shared "
            "--output-root. Let the script infer one '_train_vitposepp' directory per dataset."
        )
    for current_dataset_root in dataset_roots:
        output_root = build_vitposepp_output_root(
            project_root,
            current_dataset_root,
            args.output_root,
        )
        prepare_dataset(args, project_root, current_dataset_root, output_root)


if __name__ == "__main__":
    main()
