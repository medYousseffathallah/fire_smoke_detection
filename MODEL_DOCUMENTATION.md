**Overview**
- This document describes the fire/smoke detection model and runtime pipeline in this repository, including model wrappers, preprocessing, postprocessing, temporal alerting, and configuration flow.
- The system is a YOLOv5-based detector wrapped for ONNXRuntime or PyTorch/Ultralytics inference, then gated with motion and temporal smoothing to produce an alert signal.
- Primary entry points: the standalone script [fire_smoke_pipeline.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline.py) and the package modules under [fire_smoke_pipeline](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline).

**Features**
- Dual inference backends: ONNXRuntime and PyTorch/Ultralytics via [YoloV5nOnnx](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/models/yolov5_onnx.py#L9-L35) and [YoloV5nTorch](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/models/yolov5_torch.py#L10-L91).
- Standard YOLO-style preprocessing: letterbox resize, RGB conversion, CHW layout, and 0–1 normalization via [letterbox](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/preprocessing.py#L7-L15).
- YOLO postprocessing with confidence filtering and NMS via [nms](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/postprocess.py#L15-L38).
- Temporal alert logic with EMA smoothing and hysteresis via [TemporalDecision](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/temporal/decision.py#L4-L25).
- Motion-based gating to reduce false positives via [MotionScorer](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/utils/motion.py#L7-L31).

**Model Architecture**
- Base model: YOLOv5n-style detector with custom weights loaded from model.weights_path.
- ONNX backend: The ONNX output is expected to be a tensor with columns [x, y, w, h, obj_conf, class_probs...]. The code computes class scores as obj_conf * class_prob and applies NMS ([YoloV5nOnnx.__call__](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/models/yolov5_onnx.py#L20-L35)).
- Torch/Ultralytics backend:
- If Ultralytics YOLO is available, it consumes raw frames (expects_raw=True) and uses model.predict with imgsz, conf, and iou ([YoloV5nTorch.__call__](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/models/yolov5_torch.py#L57-L75)).
- If not, it falls back to torch.hub.load from Ultralytics YOLOv5 repositories and expects a preprocessed tensor ([YoloV5nTorch.__init__](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/models/yolov5_torch.py#L10-L55)).

**Input/Output Formats**
- Input frames:
- Raw frames are np.ndarray BGR images (H x W x 3) from OpenCV capture.
- Preprocessed model input:
- Shape: (1, 3, input_size, input_size)
- Type: float32
- Range: [0, 1]
- Channels: RGB
- Preprocessing steps: letterbox resize → BGR→RGB → HWC→CHW → normalize ([InferenceWorker.run](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/worker.py#L42-L58)).
- Model output:
- boxes: np.ndarray of shape (N, 4) in xyxy format (pixel coordinates).
- scores: np.ndarray of shape (N,) confidence scores.
- class_ids: np.ndarray of shape (N,) class indices.
- Alert output:
- Boolean alert from temporal logic, emitted to the UI overlay or logs ([InferenceWorker.run](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/worker.py#L59-L101)).

**Preprocessing Requirements**
- Use the built-in letterbox to preserve aspect ratio and pad to a square canvas filled with 114 ([letterbox](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/preprocessing.py#L7-L15)).
- Convert BGR → RGB before passing to ONNX/YOLOv5 tensor inference ([InferenceWorker.run](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/worker.py#L52-L55)).
- Normalize pixel values by 1/255.0.

**Performance Metrics**
- The repository does not include explicit evaluation metrics such as mAP, precision, or recall. No training or evaluation scripts are present in the codebase to compute these metrics.
- The runtime uses a detection score derived from confidence multiplied by normalized bounding box area to feed the temporal alert logic ([score_detections](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/utils/misc.py#L4-L12)). This is not a standard evaluation metric but a runtime decision signal.

**Dataset**
- Dataset link: https://huggingface.co/datasets/medyoussef/fire-smoke-hardnegatives-int8/tree/main
- Visible structure from the dataset repository:
- fire_smoke_hardnegatives_complete.zip (~3.17 GB)
- calibration_subset_int8.zip (~146 MB)
- data_combined.yaml (small YAML file)
- Data structure and labels:
- The dataset repository exposes a YAML metadata file (data_combined.yaml) and zipped archives. The exact internal folder layout and annotation format are not available from the repository listing alone.
- To confirm structure, extract the zip archives locally and inspect the YAML file. The presence of data_combined.yaml suggests a YOLO-style dataset configuration, but the exact keys and paths are not visible via the listing.
- Training methodology:
- No training scripts are present in this repo. The model wrappers assume YOLOv5-style weights (.pt) or an ONNX export (.onnx) compatible with YOLOv5 output decoding.
- If this dataset is used for training elsewhere, the training setup is not documented in this repository, so developers should consult the dataset YAML file and any external training scripts.

**Usage**
- CLI usage (standalone script):
- The main script exposes CLI arguments for camera input, model settings, and temporal behavior via [parse_args](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline.py#L447-L458).
- Example:

```bash
python fire_smoke_pipeline.py ^
  --backend onnx ^
  --weights-path "C:\path\to\best.onnx" ^
  --onnx-providers "CUDAExecutionProvider,CPUExecutionProvider" ^
  --class-names "fire,smoke" ^
  --display
```

- YAML config usage:
- Use [example_config.yaml](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/example_config.yaml) or [rtsp_config.yaml](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/rtsp_config.yaml) as templates.
- Configs are validated by [load_config](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/config/loader.py#L14-L28) and _validate_cfg ([link](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/config/loader.py#L106-L124)).

**API Endpoints/Methods**
- Model wrappers
- YoloV5nOnnx.__call__(input_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
- Inputs: preprocessed tensor (1, 3, input_size, input_size) float32
- Outputs: boxes, scores, class_ids
- Reference: [YoloV5nOnnx](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/models/yolov5_onnx.py#L9-L35)
- YoloV5nTorch.__call__(input_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
- Inputs: raw frame if expects_raw=True, otherwise preprocessed tensor
- Outputs: boxes, scores, class_ids
- Reference: [YoloV5nTorch](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/models/yolov5_torch.py#L10-L91)

- Preprocessing
- letterbox(image, new_size) -> (image, ratio, (dw, dh))
- Reference: [preprocessing.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/preprocessing.py#L7-L15)
- map_boxes(boxes, ratio, pad, original_shape) -> boxes
- Reference: [preprocessing.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/preprocessing.py#L18-L32)

- Postprocessing
- xywh_to_xyxy(x) -> y
- Reference: [postprocess.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/postprocess.py#L6-L12)
- nms(boxes, scores, iou_thres) -> List[int]
- Reference: [postprocess.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/postprocess.py#L15-L38)

- Temporal logic
- TemporalDecision.update(score, motion_ok) -> bool
- Reference: [decision.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/temporal/decision.py#L4-L25)

**Parameters**
- ModelConfig ([schema.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/config/schema.py#L21-L30))
- backend: onnx or torch
- weights_path: path to .onnx or .pt weights
- input_size: square inference size (e.g., 640)
- conf_thres: confidence threshold in [0,1]
- iou_thres: NMS IoU threshold in [0,1]
- class_names: list of class labels (e.g., ["fire", "smoke"])
- use_half: enable FP16 on CUDA
- onnx_providers: list of ONNXRuntime providers when backend is ONNX

- TemporalConfig ([schema.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/config/schema.py#L33-L42))
- ema_alpha: EMA smoothing factor
- on_threshold / off_threshold: hysteresis thresholds
- on_frames / off_frames: consecutive frames for state changes
- motion_gate: enable motion gating
- motion_threshold: motion gate threshold
- high_conf_bypass: allow high score to bypass motion gate

**Examples**
- Using the inference worker directly:

```python
import queue
import threading
import numpy as np
import cv2

from fire_smoke_pipeline.config.schema import ModelConfig, TemporalConfig
from fire_smoke_pipeline.inference.worker import InferenceWorker

model_cfg = ModelConfig(
    backend="onnx",
    weights_path="C:/path/to/best.onnx",
    input_size=640,
    conf_thres=0.35,
    iou_thres=0.45,
    class_names=["fire", "smoke"],
    use_half=False,
    onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

temporal_cfg = TemporalConfig(
    ema_alpha=0.6,
    on_threshold=0.6,
    off_threshold=0.4,
    on_frames=3,
    off_frames=5,
    motion_gate=False,
    motion_threshold=0.08,
    high_conf_bypass=0.85,
)

frame_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()
worker = InferenceWorker(model_cfg, temporal_cfg, frame_queue, stop_event, display=True)
worker.start()

cap = cv2.VideoCapture(0)
try:
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            continue
        frame_queue.put((0.0, frame))
finally:
    stop_event.set()
    worker.join()
    cap.release()
```

**Notes**
- The model expects YOLOv5-style output layout for ONNX weights. If your ONNX export differs, postprocessing will break.
- class_names must match the order used during training. Misalignment causes mislabeled overlays.
- Motion gating is optional but recommended in dynamic environments to reduce false alerts.
- The runtime score is not a performance metric; it is a signal used for alerting.

**Troubleshooting**
- ONNX backend fails with provider error: Ensure model.onnx_providers is set and matches available providers on the system ([ModelConfig validation](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/config/loader.py#L106-L116)).
- No detections: Verify weights_path and class_names; check conf_thres is not too high.
- Incorrect box scaling: Ensure preprocessing uses letterbox and postprocessing uses map_boxes with correct ratio and pad ([preprocessing.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/preprocessing.py#L7-L32)).
- Slow inference: Consider ONNXRuntime with CUDA or reduce input_size.
