import queue
import threading

import cv2
import numpy as np

from ..config.schema import ModelConfig, TemporalConfig
from .preprocessing import letterbox, map_boxes
from ..models.yolov5_onnx import YoloV5nOnnx
from ..models.yolov5_torch import YoloV5nTorch
from ..temporal.decision import TemporalDecision
from ..utils.misc import score_detections
from ..utils.motion import MotionScorer


class InferenceWorker(threading.Thread):
    def __init__(
        self,
        model_cfg: ModelConfig,
        temporal_cfg: TemporalConfig,
        frame_queue: queue.Queue,
        stop_event: threading.Event,
        display: bool,
    ):
        super().__init__(daemon=True)
        self.model_cfg = model_cfg
        self.temporal_cfg = temporal_cfg
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.display = display
        self.model = self._build_model(model_cfg)
        self.temporal = TemporalDecision(temporal_cfg)
        self.motion = MotionScorer()

    def _build_model(self, cfg: ModelConfig):
        if cfg.backend == "onnx":
            return YoloV5nOnnx(cfg)
        if cfg.backend == "torch":
            return YoloV5nTorch(cfg)
        raise ValueError("Unsupported backend")

    def run(self):
        while not self.stop_event.is_set():
            try:
                ts, frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if getattr(self.model, "expects_raw", False):
                boxes, scores, class_ids = self.model(frame)
            else:
                resized, ratio, pad = letterbox(frame, self.model_cfg.input_size)
                input_tensor = resized[:, :, ::-1].transpose(2, 0, 1)
                input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32) / 255.0
                input_tensor = input_tensor[None, ...]
                boxes, scores, class_ids = self.model(input_tensor)
                boxes = map_boxes(boxes, ratio, pad, frame.shape[:2])

            score = score_detections(boxes, scores)
            motion_score = self.motion.score(frame, boxes)
            motion_ok = (not self.temporal_cfg.motion_gate) or (motion_score >= self.temporal_cfg.motion_threshold)
            alert = self.temporal.update(score, motion_ok)

            if self.display:
                overlay = frame.copy()
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.astype(int)
                    color = (0, 0, 255) if alert else (0, 255, 0)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    # Safety check for class index bounds
                    class_idx = int(class_ids[i])
                    if 0 <= class_idx < len(self.model_cfg.class_names):
                        class_name = self.model_cfg.class_names[class_idx]
                    else:
                        class_name = f"class_{class_idx}"
                    label = f"{class_name}:{scores[i]:.2f}"
                    cv2.putText(
                        overlay,
                        label,
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )
                status = "ALERT" if alert else "CLEAR"
                cv2.putText(
                    overlay,
                    status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("fire_smoke_detection", overlay)
                if cv2.waitKey(1) & 0xFF == 27:
                    self.stop_event.set()
            else:
                if alert:
                    print(f"{ts:.3f} ALERT score={score:.3f} motion={motion_score:.3f}")
