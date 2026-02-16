import argparse
import json
import os as _os
import time
import threading
import queue
from dataclasses import dataclass
from typing import List, Optional, Tuple
import sys
from pathlib import Path

import numpy as np
import cv2

# Add the fire_smoke_pipeline package to Python path
package_path = str(Path(__file__).parent / "fire_smoke_pipeline")
if package_path not in sys.path:
    sys.path.insert(0, package_path)


@dataclass
class CameraConfig:
    source_type: str
    device_index: int
    file_path: str
    rtsp_url: str
    http_url: str
    width: int
    height: int
    fps: int
    use_gstreamer: bool
    latency_ms: int
    csi_sensor_id: int
    rtsp_decoder: str


@dataclass
class ModelConfig:
    backend: str
    weights_path: str
    input_size: int
    conf_thres: float
    iou_thres: float
    class_names: List[str]
    use_half: bool
    onnx_providers: Optional[List[str]]


@dataclass
class TemporalConfig:
    ema_alpha: float
    on_threshold: float
    off_threshold: float
    on_frames: int
    off_frames: int
    motion_gate: bool
    motion_threshold: float
    high_conf_bypass: float


def build_gstreamer_pipeline(cfg: CameraConfig) -> str:
    if cfg.source_type == "csi":
        return (
            f"nvarguscamerasrc sensor-id={cfg.csi_sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={cfg.width}, height={cfg.height}, framerate={cfg.fps}/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        )
    if cfg.source_type == "rtsp":
        decoder = "avdec_h264"
        if cfg.rtsp_decoder == "jetson-hw":
            decoder = "nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx"
        return (
            f"rtspsrc location={cfg.rtsp_url} latency={cfg.latency_ms} protocols=tcp ! "
            f"rtph264depay ! h264parse ! {decoder} ! "
            "videoconvert ! video/x-raw, format=BGR, "
            f"width={cfg.width}, height={cfg.height}, framerate={cfg.fps}/1 ! "
            "appsink drop=true sync=false max-buffers=1"
        )
    if cfg.source_type == "http":
        return (
            f"souphttpsrc location={cfg.http_url} ! decodebin ! "
            "videoconvert ! video/x-raw, format=BGR, "
            f"width={cfg.width}, height={cfg.height}, framerate={cfg.fps}/1 ! "
            "appsink drop=true sync=false max-buffers=1"
        )
    raise ValueError("Unsupported GStreamer source type")


def build_capture_source(cfg: CameraConfig):
    if cfg.source_type == "usb":
        return cfg.device_index
    if cfg.source_type == "file":
        return cfg.file_path
    if cfg.source_type == "rtsp":
        return build_gstreamer_pipeline(cfg) if cfg.use_gstreamer else cfg.rtsp_url
    if cfg.source_type == "http":
        return build_gstreamer_pipeline(cfg) if cfg.use_gstreamer else cfg.http_url
    if cfg.source_type == "csi":
        return build_gstreamer_pipeline(cfg)
    raise ValueError("Unsupported source type")


def letterbox(image: np.ndarray, new_size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = image.shape[:2]
    r = min(new_size / w, new_size / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    dw, dh = (new_size - new_w) // 2, (new_size - new_h) // 2
    canvas[dh:dh + new_h, dw:dw + new_w] = resized
    return canvas, r, (dw, dh)


def xywh_to_xyxy(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


class YoloV5nOnnx:
    def __init__(self, cfg: ModelConfig):
        import onnxruntime as ort
        if cfg.onnx_providers:
            providers = cfg.onnx_providers
        else:
            available = set(ort.get_available_providers())
            providers = []
            if "TensorrtExecutionProvider" in available:
                providers.append("TensorrtExecutionProvider")
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
            if not providers:
                providers = ["CPUExecutionProvider"]
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


class YoloV5nTorch:
    def __init__(self, cfg: ModelConfig):
        import torch
        self.torch = torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=cfg.weights_path)
        self.model.to(self.device)
        if cfg.use_half and self.device.type == "cuda":
            self.model.half()
        self.model.eval()
        self.cfg = cfg

    def __call__(self, input_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


class TemporalDecision:
    def __init__(self, cfg: TemporalConfig):
        self.cfg = cfg
        self.ema = 0.0
        self.on_count = 0
        self.off_count = 0
        self.alert = False

    def update(self, score: float, motion_ok: bool) -> bool:
        self.ema = self.cfg.ema_alpha * score + (1 - self.cfg.ema_alpha) * self.ema
        gated = self.ema if (motion_ok or self.ema >= self.cfg.high_conf_bypass) else 0.0
        if gated >= self.cfg.on_threshold:
            self.on_count += 1
            self.off_count = 0
        elif gated < self.cfg.off_threshold:
            self.off_count += 1
            self.on_count = 0
        if not self.alert and self.on_count >= self.cfg.on_frames:
            self.alert = True
        if self.alert and self.off_count >= self.cfg.off_frames:
            self.alert = False
        return self.alert


class FrameGrabber(threading.Thread):
    def __init__(self, cfg: CameraConfig, frame_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.frame_queue = frame_queue
        self.stop_event = stop_event

    def run(self):
        source = build_capture_source(self.cfg)
        api_preference = cv2.CAP_GSTREAMER if self.cfg.use_gstreamer else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, api_preference)
        if not cap.isOpened():
            self.stop_event.set()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            ts = time.monotonic()
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put((ts, frame))
        cap.release()


class InferenceWorker(threading.Thread):
    def __init__(
        self,
        model_cfg: ModelConfig,
        temporal_cfg: TemporalConfig,
        frame_queue: queue.Queue,
        stop_event: threading.Event,
        display: bool,
        max_frames: int,
    ):
        super().__init__(daemon=True)
        self.model_cfg = model_cfg
        self.temporal_cfg = temporal_cfg
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.display = display
        self.max_frames = max(0, int(max_frames))
        self.model = self._build_model(model_cfg)
        self.temporal = TemporalDecision(temporal_cfg)
        self.prev_gray: Optional[np.ndarray] = None

    def _build_model(self, cfg: ModelConfig):
        if cfg.backend == "onnx":
            return YoloV5nOnnx(cfg)
        if cfg.backend == "torch":
            return YoloV5nTorch(cfg)
        raise ValueError("Unsupported backend")

    def _motion_score(self, frame: np.ndarray, boxes: np.ndarray) -> float:
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

    def _score_detections(self, boxes: np.ndarray, scores: np.ndarray) -> float:
        if boxes.size == 0:
            return 0.0
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        areas = np.clip(areas, 1.0, None)
        area_norm = areas / np.max(areas)
        return float(np.max(scores * area_norm))

    def _map_boxes(self, boxes: np.ndarray, ratio: float, pad: Tuple[int, int], original_shape: Tuple[int, int]) -> np.ndarray:
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

    def run(self):
        processed = 0
        while not self.stop_event.is_set():
            try:
                ts, frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            resized, ratio, pad = letterbox(frame, self.model_cfg.input_size)
            input_tensor = resized[:, :, ::-1].transpose(2, 0, 1)
            input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32) / 255.0
            input_tensor = input_tensor[None, ...]
            boxes, scores, class_ids = self.model(input_tensor)
            boxes = self._map_boxes(boxes, ratio, pad, frame.shape[:2])
            score = self._score_detections(boxes, scores)
            motion_score = self._motion_score(frame, boxes)
            motion_ok = (not self.temporal_cfg.motion_gate) or (motion_score >= self.temporal_cfg.motion_threshold)
            alert = self.temporal.update(score, motion_ok)
            if self.display:
                overlay = frame.copy()
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.astype(int)
                    color = (0, 0, 255) if alert else (0, 255, 0)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    label = f"{self.model_cfg.class_names[class_ids[i]]}:{scores[i]:.2f}"
                    cv2.putText(overlay, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                status = "ALERT" if alert else "CLEAR"
                cv2.putText(overlay, f"{status} score={score:.2f} motion={motion_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("fire_smoke_detection", overlay)
                if cv2.waitKey(1) & 0xFF == 27:
                    self.stop_event.set()
            else:
                if alert:
                    print(f"{ts:.3f} ALERT score={score:.3f} motion={motion_score:.3f}")
            processed += 1
            if self.max_frames and processed >= self.max_frames:
                self.stop_event.set()


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _flatten_config(data: dict) -> dict:
    flat = {}
    for key in ("camera", "model", "temporal", "runtime"):
        section = data.get(key)
        if isinstance(section, dict):
            flat.update(section)
    for k, v in data.items():
        if k not in ("camera", "model", "temporal", "runtime") and not isinstance(v, dict):
            flat[k] = v
    return flat


def _build_parser(defaults: Optional[dict] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    if defaults:
        parser.set_defaults(**defaults)
    parser.add_argument("--source-type", choices=["usb", "csi", "rtsp", "http", "file"], default="usb")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--file-path", type=str, default="")
    parser.add_argument("--rtsp-url", type=str, default="")
    parser.add_argument("--http-url", type=str, default="")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--use-gstreamer", dest="use_gstreamer", action="store_true")
    parser.add_argument("--no-gstreamer", dest="use_gstreamer", action="store_false")
    parser.add_argument("--latency-ms", type=int, default=80)
    parser.add_argument("--csi-sensor-id", type=int, default=0)
    parser.add_argument("--rtsp-decoder", choices=["cpu", "jetson-hw"], default="cpu")
    parser.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    parser.add_argument("--weights-path", type=str, default="")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.35)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--class-names", type=str, default="fire,smoke")
    parser.add_argument("--use-half", dest="use_half", action="store_true")
    parser.add_argument("--no-half", dest="use_half", action="store_false")
    parser.add_argument("--onnx-providers", type=str, default="")
    parser.add_argument("--ema-alpha", type=float, default=0.6)
    parser.add_argument("--on-threshold", type=float, default=0.6)
    parser.add_argument("--off-threshold", type=float, default=0.4)
    parser.add_argument("--on-frames", type=int, default=3)
    parser.add_argument("--off-frames", type=int, default=5)
    parser.add_argument("--motion-gate", dest="motion_gate", action="store_true")
    parser.add_argument("--no-motion-gate", dest="motion_gate", action="store_false")
    parser.add_argument("--motion-threshold", type=float, default=0.08)
    parser.add_argument("--high-conf-bypass", type=float, default=0.85)
    parser.add_argument("--display", dest="display", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args()
    defaults = {}
    if pre_args.config:
        defaults = _flatten_config(_load_config(pre_args.config))
    parser = _build_parser(defaults=defaults)
    args = parser.parse_args()
    if not args.weights_path:
        parser.error("--weights-path is required (or set model.weights_path in --config).")
    return args


def main():
    args = parse_args()
    onnx_providers = [p.strip() for p in args.onnx_providers.split(",") if p.strip()] or None
    camera_cfg = CameraConfig(
        source_type=args.source_type,
        device_index=args.device_index,
        file_path=args.file_path,
        rtsp_url=args.rtsp_url,
        http_url=args.http_url,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_gstreamer=args.use_gstreamer,
        latency_ms=args.latency_ms,
        csi_sensor_id=args.csi_sensor_id,
        rtsp_decoder=args.rtsp_decoder,
    )
    model_cfg = ModelConfig(
        backend=args.backend,
        weights_path=args.weights_path,
        input_size=args.input_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        class_names=[c.strip() for c in args.class_names.split(",") if c.strip()],
        use_half=args.use_half,
        onnx_providers=onnx_providers,
    )
    temporal_cfg = TemporalConfig(
        ema_alpha=args.ema_alpha,
        on_threshold=args.on_threshold,
        off_threshold=args.off_threshold,
        on_frames=args.on_frames,
        off_frames=args.off_frames,
        motion_gate=args.motion_gate,
        motion_threshold=args.motion_threshold,
        high_conf_bypass=args.high_conf_bypass,
    )
    frame_queue = queue.Queue(maxsize=max(1, int(args.queue_size)))
    stop_event = threading.Event()
    grabber = FrameGrabber(camera_cfg, frame_queue, stop_event)
    worker = InferenceWorker(model_cfg, temporal_cfg, frame_queue, stop_event, args.display, args.max_frames)
    grabber.start()
    worker.start()
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
    worker.join()
    grabber.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
