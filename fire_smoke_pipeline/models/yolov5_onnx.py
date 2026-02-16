from typing import Tuple

import numpy as np

from ..config.schema import ModelConfig
from ..inference.postprocess import nms, xywh_to_xyxy


class YoloV5nOnnx:
    def __init__(self, cfg: ModelConfig):
        import onnxruntime as ort

        if not cfg.onnx_providers:
            raise ValueError("model.onnx_providers is required for ONNX backend.")
        providers = cfg.onnx_providers
        self.session = ort.InferenceSession(cfg.weights_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.cfg = cfg

    def __call__(self, input_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = self.session.run(None, {self.input_name: input_tensor})
        pred = outputs[0]
        pred = pred[0]
        scores = pred[:, 4:5] * pred[:, 5:]
        class_ids = np.argmax(scores, axis=1)
        class_scores = scores[np.arange(scores.shape[0]), class_ids]
        mask = class_scores >= self.cfg.conf_thres
        if not np.any(mask):
            return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
        pred = pred[mask]
        class_ids = class_ids[mask]
        class_scores = class_scores[mask]
        boxes = xywh_to_xyxy(pred[:, :4])
        keep = nms(boxes, class_scores, self.cfg.iou_thres)
        return boxes[keep], class_scores[keep], class_ids[keep]
