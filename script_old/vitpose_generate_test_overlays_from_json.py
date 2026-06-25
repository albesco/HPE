from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from pose_overlay_utils import dataset_info_from_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / 'data/intermediate/Side_above_water_VideoTest2/_train_canonical/subsets/random300_seed20260603'
DEFAULT_PREDICTIONS = PROJECT_ROOT / 'runs/vitposepp_video_test2_random300_eval_best_AP_epoch_24/result_keypoints.json'
DEFAULT_CONFIG = PROJECT_ROOT / 'data/intermediate/Side_above_water/_train_canonical/generated_configs/swimxyz_vitposepp_huge_grid_winner_resume.py'
DEFAULT_CHECKPOINT = PROJECT_ROOT / 'runs/vitposepp_side_above_water_grid_winner_resume/best_AP_epoch_24.pth'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'data/intermediate/Side_above_water_VideoTest2/_train_canonical/reports/test_overlays/vitposepp_video_test2_random300_best_AP_epoch_24'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Render VitPose++ overlays from saved predictions JSON.')
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--predictions-json', type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    parser.add_argument('--device', default='auto')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    annotations_path = dataset_root / 'annotations' / f'person_keypoints_{args.split}.json'
    images_dir = dataset_root / f'{args.split}2017'
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from mmpose.apis import init_pose_model
    from mmpose.core.visualization import imshow_keypoints
    import torch

    device = args.device
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = init_pose_model(str(args.config.resolve()), str(args.checkpoint.resolve()), device=device)
    dataset_info = dataset_info_from_model(model)

    coco = json.loads(annotations_path.read_text(encoding='utf-8'))
    images = {int(item['id']): item for item in coco.get('images', [])}
    annotations_by_image = {}
    for ann in coco.get('annotations', []):
        annotations_by_image.setdefault(int(ann['image_id']), []).append(ann)

    preds = json.loads(args.predictions_json.read_text(encoding='utf-8'))
    preds_by_image = {int(item['image_id']): item for item in preds}

    manifest = []
    for index, image_id in enumerate(sorted(images), start=1):
        image_info = images[image_id]
        image_path = images_dir / image_info['file_name']
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        pred = preds_by_image.get(image_id)
        pose_results = []
        ann_bbox = None
        for ann in annotations_by_image.get(image_id, []):
            bbox = ann.get('bbox')
            if bbox:
                ann_bbox = bbox
                break
        if pred is not None and ann_bbox is not None:
            keypoints = pred['keypoints']
            if isinstance(keypoints, list) and keypoints and not isinstance(keypoints[0], list):
                keypoints = [keypoints[index:index + 3] for index in range(0, len(keypoints), 3)]
            pose_results.append(keypoints)

        overlay = image
        if pose_results:
            overlay = imshow_keypoints(
                image,
                pose_results,
                skeleton=dataset_info.skeleton if dataset_info is not None else None,
                pose_kpt_color=dataset_info.pose_kpt_color if dataset_info is not None else None,
                pose_link_color=dataset_info.pose_link_color if dataset_info is not None else None,
                radius=3,
                thickness=2,
            )

        out_path = output_dir / image_info['file_name']
        cv2.imwrite(str(out_path), overlay)
        manifest.append({'image_id': image_id, 'file_name': image_info['file_name'], 'output_path': str(out_path)})
        if index % 100 == 0:
            print(f'Processed {index}/{len(images)} images', flush=True)

    (output_dir / '_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'output_dir': str(output_dir), 'num_images': len(manifest)}, indent=2))


if __name__ == '__main__':
    main()
