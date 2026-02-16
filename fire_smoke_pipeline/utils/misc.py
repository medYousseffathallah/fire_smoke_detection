from typing import List


def score_detections(boxes, scores) -> float:
    import numpy as np

    if boxes.size == 0:
        return 0.0
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    areas = np.clip(areas, 1.0, None)
    area_norm = areas / np.max(areas)
    return float(np.max(scores * area_norm))


def ensure_class_index(class_ids, class_names: List[str]) -> int:
    if len(class_names) == 0:
        return 0
    idx = int(class_ids)
    if idx < 0:
        return 0
    if idx >= len(class_names):
        return len(class_names) - 1
    return idx

