from typing import Tuple
import os

import numpy as np

from ..config.schema import ModelConfig
from ..inference.postprocess import nms


class YoloV5nTorch:
    def __init__(self, cfg: ModelConfig):
        import torch

        self.torch = torch
        self.expects_raw = False
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(base_dir, ".ultralytics"))
        model = None
        try:
            from ultralytics import YOLO

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model = YOLO(cfg.weights_path)
            self.expects_raw = True
        except Exception:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            os.environ.setdefault("YOLOV5_REQUIREMENTS", "0")
            os.environ.setdefault("YOLOv5_REQUIREMENTS", "0")
            repos = [
                "ultralytics/yolov5",
                "ultralytics/yolov5:v7.0",
                "ultralytics/yolov5:v6.2",
                "ultralytics/yolov5:v6.0",
            ]
            last_error = None
            for repo in repos:
                try:
                    model = torch.hub.load(
                        repo,
                        "custom",
                        path=cfg.weights_path,
                        force_reload=True,
                        trust_repo=True,
                    )
                    break
                except Exception as exc:
                    last_error = exc
            if model is None:
                raise last_error
            model.to(self.device)
            if cfg.use_half and self.device.type == "cuda":
                model.half()
            model.eval()
        self.model = model
        self.cfg = cfg

    def __call__(self, input_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.expects_raw:
            results = self.model.predict(
                source=input_tensor,
                imgsz=self.cfg.input_size,
                conf=self.cfg.conf_thres,
                iou=self.cfg.iou_thres,
                device=self.device,
                verbose=False,
            )
            if not results:
                return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
            boxes_obj = results[0].boxes
            if boxes_obj is None or boxes_obj.xyxy is None:
                return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
            boxes = boxes_obj.xyxy.cpu().numpy()
            scores = boxes_obj.conf.cpu().numpy()
            class_ids = boxes_obj.cls.cpu().numpy().astype(np.int32)
            return boxes, scores, class_ids
        torch = self.torch
        tensor = torch.from_numpy(input_tensor).to(self.device)
        if self.cfg.use_half and self.device.type == "cuda":
            tensor = tensor.half()
        with torch.no_grad():
            pred = self.model(tensor, size=self.cfg.input_size)[0]
        pred = pred.cpu().numpy()
        if pred.shape[0] == 0:
            return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
        boxes = pred[:, :4]
        scores = pred[:, 4]
        class_ids = pred[:, 5].astype(np.int32)
        keep = scores >= self.cfg.conf_thres
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        keep = nms(boxes, scores, self.cfg.iou_thres)
        return boxes[keep], scores[keep], class_ids[keep]
