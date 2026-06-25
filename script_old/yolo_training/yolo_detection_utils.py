from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionBox:
    xyxy: list[float]
    confidence: float
    area: float


def bbox_area_xyxy(xyxy: list[float] | np.ndarray) -> float:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_top_detection_box(boxes) -> DetectionBox | None:
    """Select one YOLO box by confidence, then area for ties."""
    if boxes is None or len(boxes) == 0:
        return None

    xyxy_values = boxes.xyxy.detach().cpu().numpy()
    conf_values = boxes.conf.detach().cpu().numpy()

    best_index = 0
    best_conf = float(conf_values[0])
    best_area = bbox_area_xyxy(xyxy_values[0])

    for index in range(1, len(conf_values)):
        confidence = float(conf_values[index])
        area = bbox_area_xyxy(xyxy_values[index])
        if confidence > best_conf or (confidence == best_conf and area > best_area):
            best_index = index
            best_conf = confidence
            best_area = area

    return DetectionBox(
        xyxy=[float(value) for value in xyxy_values[best_index]],
        confidence=best_conf,
        area=best_area,
    )
