from pathlib import Path

import cv2
import numpy as np

from script.prepare_swimxyz_vitposepp_utils import build_keypoints_and_bbox


def test_build_keypoints_and_bbox_allows_zero_min_visible_keypoints(tmp_path: Path) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), image)

    row = {
        "Nose.x": "0",
        "Nose.y": "0",
        "Nose.z": "0",
        "Neck.x": "0",
        "Neck.y": "0",
        "Neck.z": "0",
        "RShoulder.x": "0",
        "RShoulder.y": "0",
        "RShoulder.z": "0",
        "RElbow.x": "0",
        "RElbow.y": "0",
        "RElbow.z": "0",
        "RWrist.x": "0",
        "RWrist.y": "0",
        "RWrist.z": "0",
        "LShoulder.x": "0",
        "LShoulder.y": "0",
        "LShoulder.z": "0",
        "LElbow.x": "0",
        "LElbow.y": "0",
        "LElbow.z": "0",
        "LWrist.x": "0",
        "LWrist.y": "0",
        "LWrist.z": "0",
        "RHip.x": "0",
        "RHip.y": "0",
        "RHip.z": "0",
        "RKnee.x": "0",
        "RKnee.y": "0",
        "RKnee.z": "0",
        "RAnkle.x": "0",
        "RAnkle.y": "0",
        "RAnkle.z": "0",
        "LHip.x": "0",
        "LHip.y": "0",
        "LHip.z": "0",
        "LKnee.x": "0",
        "LKnee.y": "0",
        "LKnee.z": "0",
        "LAnkle.x": "0",
        "LAnkle.y": "0",
        "LAnkle.z": "0",
        "REye.x": "0",
        "REye.y": "0",
        "REye.z": "0",
        "REar.x": "0",
        "REar.y": "0",
        "REar.z": "0",
        "LEye.x": "0",
        "LEye.y": "0",
        "LEye.z": "0",
        "LEar.x": "0",
        "LEar.y": "0",
        "LEar.z": "0",
    }

    prepared = build_keypoints_and_bbox(
        row=row,
        width=4,
        height=4,
        bbox_padding_x_ratio=0.2,
        bbox_padding_y_ratio=0.25,
        bbox_min_padding_px=15.0,
        min_visible_keypoints=0,
        flip_y=True,
    )

    assert prepared is not None
    keypoints, num_keypoints, bbox, area = prepared
    assert num_keypoints == 0
    assert bbox == [0.0, 0.0, 4.0, 4.0]
    assert area == 16.0
    assert len(keypoints) == 17 * 3
