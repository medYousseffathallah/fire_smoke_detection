from typing import Optional

import cv2
import numpy as np


class MotionScorer:
    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None

    def score(self, frame: np.ndarray, boxes: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray
        if boxes.size == 0:
            return float(np.mean(diff)) / 255.0
        motion_scores = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(diff.shape[1] - 1, x2)
            y2 = min(diff.shape[0] - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = diff[y1:y2, x1:x2]
            motion_scores.append(float(np.mean(roi)) / 255.0)
        return float(np.max(motion_scores)) if motion_scores else float(np.mean(diff)) / 255.0

