from typing import Tuple

import cv2
import numpy as np


def letterbox(image: np.ndarray, new_size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = image.shape[:2]
    r = min(new_size / w, new_size / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    dw, dh = (new_size - new_w) // 2, (new_size - new_h) // 2
    canvas[dh:dh + new_h, dw:dw + new_w] = resized
    return canvas, r, (dw, dh)


def map_boxes(
    boxes: np.ndarray, ratio: float, pad: Tuple[int, int], original_shape: Tuple[int, int]
) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    dw, dh = pad
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= ratio
    h, w = original_shape
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes

