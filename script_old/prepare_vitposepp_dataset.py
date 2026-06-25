from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_CANONICAL_ROOT = "data/intermediate/Side_above_water/_train_canonical"
DEFAULT_OUTPUT_ROOT = "data/intermediate/Side_above_water/_VitPosePP"
DEFAULT_BASE_CONFIG = (
    "src/vitpose_base/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/"
    "vitPose+_huge_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py"
)
DEFAULT_PRETRAINED_CHECKPOINT = "models/pose/wholebody.pth"
DEFAULT_WORK_DIR = "runs/vitposepp_side_above_water_aniso_20x25_min15"
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export canonical SwimXYZ COCO annotations into a VitPose++ label/config "
            "dataset. This exporter is label-only and does not copy, link, resize, "
            "or modify images."
        )
    )
    parser.add_argument("--canonical-root", default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--pretrained-checkpoint", default=DEFAULT_PRETRAINED_CHECKPOINT)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--total-epochs", type=int, default=30)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--samples-per-gpu", type=int, default=4)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def copy_annotations(canonical_root: Path, output_root: Path) -> dict[str, dict[str, int]]:
    annotation_root = output_root / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        source = canonical_root / "annotations" / f"person_keypoints_{split}.json"
        if not source.is_file():
            raise FileNotFoundError(f"Missing annotation file: {source}")
        destination = annotation_root / source.name
        shutil.copy2(source, destination)
        payload = json.loads(destination.read_text(encoding="utf-8"))
        summary[split] = {
            "images": len(payload.get("images", [])),
            "annotations": len(payload.get("annotations", [])),
        }
    return summary


def write_vitposepp_config(
    output_path: Path,
    canonical_root: Path,
    output_root: Path,
    base_config: Path,
    pretrained_checkpoint: Path,
    work_dir: Path,
    total_epochs: int,
    val_interval: int,
    checkpoint_interval: int,
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> None:
    project_root = Path.cwd().resolve()
    coco_dataset_config = project_root / "src/vitpose_base/configs/_base_/datasets/coco.py"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_text = f"""_base_ = ['{base_config.as_posix()}']

import runpy as _runpy

canonical_image_root = '{canonical_root.as_posix()}'
label_root = '{output_root.as_posix()}'
load_from = '{pretrained_checkpoint.as_posix()}'
work_dir = '{work_dir.as_posix()}'
total_epochs = {total_epochs}

checkpoint_config = dict(interval={checkpoint_interval}, max_keep_ckpts=3, create_symlink=True)
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
    dict(type='Collect', keys=['img', 'target', 'target_weight'], meta_keys=[
        'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
        'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'
    ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='Collect', keys=['img'], meta_keys=[
        'image_file', 'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'
    ]),
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
        ann_file=f'{{label_root}}/annotations/person_keypoints_train.json',
        img_prefix=f'{{canonical_image_root}}/train2017/',
        data_cfg=dataset_data_cfg,
        pipeline=train_pipeline,
        dataset_info=coco_dataset_info,
    ),
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{{label_root}}/annotations/person_keypoints_val.json',
        img_prefix=f'{{canonical_image_root}}/val2017/',
        data_cfg=dataset_data_cfg,
        pipeline=val_pipeline,
        dataset_info=coco_dataset_info,
    ),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{{label_root}}/annotations/person_keypoints_test.json',
        img_prefix=f'{{canonical_image_root}}/test2017/',
        data_cfg=dataset_data_cfg,
        pipeline=test_pipeline,
        dataset_info=coco_dataset_info,
    ),
)
"""
    output_path.write_text(config_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    canonical_root = resolve_path(args.canonical_root)
    output_root = resolve_path(args.output_root)
    base_config = resolve_path(args.base_config)
    pretrained_checkpoint = resolve_path(args.pretrained_checkpoint)
    work_dir = resolve_path(args.work_dir)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists, pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_summary = copy_annotations(canonical_root, output_root)
    config_path = output_root / "generated_configs" / "swimxyz_vitposepp_huge.py"
    write_vitposepp_config(
        output_path=config_path,
        canonical_root=canonical_root,
        output_root=output_root,
        base_config=base_config,
        pretrained_checkpoint=pretrained_checkpoint,
        work_dir=work_dir,
        total_epochs=args.total_epochs,
        val_interval=args.val_interval,
        checkpoint_interval=args.checkpoint_interval,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
    )

    summary = {
        "source_dataset_root": canonical_root.as_posix(),
        "output_root": output_root.as_posix(),
        "config": config_path.as_posix(),
        "image_operations": "none",
        "splits": split_summary,
    }
    (output_root / "preparation_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"VitPose++ config: {config_path}")


if __name__ == "__main__":
    main()
