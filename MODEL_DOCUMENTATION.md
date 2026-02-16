**Model Information (Fire Smoke Detection - PC Pipeline)**

**Overview**

- This document describes the fire/smoke detection models used by the PC pipeline, including model artifacts, class mapping, and pipeline loading behavior.
- The system runs YOLOv5-style detection through ONNXRuntime or PyTorch/Ultralytics, followed by temporal and motion gating to produce an alert signal.
- Primary entry points: the standalone script [fire_smoke_pipeline.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline.py) and the package modules under [fire_smoke_pipeline](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline).

**Features**

- Supports ONNXRuntime inference via [YoloV5nOnnx](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/models/yolov5_onnx.py#L9-L35).
- Supports PyTorch/Ultralytics inference via [YoloV5nTorch](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/models/yolov5_torch.py#L10-L91).
- Provides class-name mapping through config class_names and uses class_ids to index labels in overlays.
- Produces detections (bounding boxes + confidence + class id) and an alert signal from temporal logic.

**Model Inventory**

- Model A: firesmokev1 (current)
- Files (in repository root)

| Artifact        | Path                                                                            | Size (bytes) | SHA256                                                           |
| --------------- | ------------------------------------------------------------------------------- | ------------ | ---------------------------------------------------------------- |
| PyTorch weights | c:\Users\youss\OneDrive\Desktop\youssef\fire_smoke_detection\firesmokev1.pt     | 18356737     | 4CB2C67655AC73F08FE87E0F371501C0FCC2AD5D6D7D5D394D2ED4AE0DDA9749 |
| ONNX export     | c:\Users\youss\OneDrive\Desktop\youssef\fire_smoke_detection\firesmokev1.onnx   | 22241225     | 53E81D564160239D3C6A63AAEC452B62547E70C4B77D7CA520C79BA04456CCC9 |
| TensorRT engine | c:\Users\youss\OneDrive\Desktop\youssef\fire_smoke_detection\firesmokev1.engine | 7829538      | 8E62C5EA61C6D9EB5380AC54E6FBB70A14B7C0080E258AAEECF641EE7A23FA2E |

- Model B: oldversion (legacy)
- Files (in repository root)

| Artifact        | Path                                                                         | Size (bytes) | SHA256                                                           |
| --------------- | ---------------------------------------------------------------------------- | ------------ | ---------------------------------------------------------------- |
| PyTorch weights | c:\Users\youss\OneDrive\Desktop\youssef\fire_smoke_detection\oldversion.pt   | 5249858      | A87CBE94E2F36971539BDDD11CCD9CB81377B16001A2E2AD2940B7D6FB859016 |
| ONNX export     | c:\Users\youss\OneDrive\Desktop\youssef\fire_smoke_detection\oldversion.onnx | 10264174     | 898EE47C6E0DC2578C158858D70B537F93B5B854098E4406869935DA83C3D3BF |

**Model Sources**

- No download scripts or upstream URLs are present in this repository for these model files.
- All listed artifacts are local files referenced by configuration or manual selection.

**Class Labels**

- The pipeline maps detection class_ids to labels via ModelConfig.class_names.
- Configured class_names examples:
- example_config.yaml uses "fire,smoke" ([example_config.yaml](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/example_config.yaml#L14-L26)).
- rtsp_config.yaml uses "smoke,fire" ([rtsp_config.yaml](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/rtsp_config.yaml#L14-L26)).
- The order of class_names must match the training label order. If class_names order differs, class_id mapping will be wrong.

**Model Loading and YOLO Output Format**

- ONNX loading uses ONNXRuntime and expects a YOLOv5-style output tensor. The code multiplies objectness by class probability and applies NMS ([YoloV5nOnnx.**call**](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/models/yolov5_onnx.py#L20-L35)).
- PyTorch/Ultralytics loading attempts Ultralytics YOLO first, then falls back to torch.hub YOLOv5 ([YoloV5nTorch.**init**](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/models/yolov5_torch.py#L10-L55)).
- Model outputs are normalized into:
- boxes: np.ndarray (N, 4) in xyxy pixel coordinates
- scores: np.ndarray (N,) confidence scores
- class_ids: np.ndarray (N,) class indices
- The pipeline uses class_names[class_id] to render labels ([InferenceWorker.run](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/worker.py#L64-L85)).

**Input/Output Formats**

- Input frames are np.ndarray BGR images (H x W x 3) from OpenCV capture.
- Preprocessed model input:
- Shape: (1, 3, input_size, input_size)
- Type: float32
- Range: [0, 1]
- Channels: RGB
- Preprocessing steps: letterbox resize → BGR→RGB → HWC→CHW → normalize ([InferenceWorker.run](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/worker.py#L42-L58)).
- Alert output is a boolean computed by EMA smoothing and hysteresis ([TemporalDecision.update](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/temporal/decision.py#L12-L25)).

**Preprocessing Requirements**

- Use the built-in letterbox to preserve aspect ratio and pad to a square canvas filled with 114 ([letterbox](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/preprocessing.py#L7-L15)).
- Convert BGR → RGB before passing to ONNX/YOLOv5 tensor inference ([InferenceWorker.run](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/worker.py#L52-L55)).
- Normalize pixel values by 1/255.0.

**Performance Metrics**

- This repository does not include explicit evaluation metrics such as mAP, precision, or recall.
- The runtime uses a detection score derived from confidence multiplied by normalized bounding box area to feed the temporal alert logic ([score_detections](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/utils/misc.py#L4-L12)). This is not a standard evaluation metric.

**Dataset**

- Dataset link: https://huggingface.co/datasets/medyoussef/fire-smoke-hardnegatives-int8/tree/main
- data_combined.yaml link: https://huggingface.co/datasets/medyoussef/fire-smoke-hardnegatives-int8/resolve/main/data_combined.yaml
- Visible structure from the dataset repository:
- fire_smoke_hardnegatives_complete.zip (~3.17 GB)
- calibration_subset_int8.zip (~146 MB)
- data_combined.yaml (small YAML file)
- Data structure and labels:
- The dataset repository exposes a YAML metadata file and zipped archives. The exact internal folder layout and annotation format are not available from the repository listing alone.
- To confirm structure, extract the zip archives locally and inspect data_combined.yaml. The presence of this file suggests a YOLO-style dataset configuration.

**Usage**

- Standalone script usage (JSON config):
- The script [fire_smoke_pipeline.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline.py) accepts --config pointing to a JSON file and falls back to CLI flags ([parse_args](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline.py#L447-L458)).
- Example:

```bash
python fire_smoke_pipeline.py ^
  --backend onnx ^
  --weights-path "C:\Users\youss\OneDrive\Desktop\youssef\fire_smoke_detection\firesmokev1.onnx" ^
  --onnx-providers "CUDAExecutionProvider,CPUExecutionProvider" ^
  --class-names "fire,smoke" ^
  --display
```

- Package config usage (YAML loader):
- The YAML loader lives in [config/loader.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/config/loader.py#L14-L28) and validates [ModelConfig](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/config/schema.py#L21-L30).
- Example YAML templates: [example_config.yaml](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/example_config.yaml) and [rtsp_config.yaml](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/rtsp_config.yaml).

**API Endpoints/Methods**

- `YoloV5nOnnx.__call__(input_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
- Inputs: preprocessed tensor (1, 3, input_size, input_size) float32
- Outputs: boxes, scores, class_ids
- Reference: [yolov5_onnx.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/models/yolov5_onnx.py#L9-L35)

- `YoloV5nTorch.__call__(input_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
- Inputs: raw frame if expects_raw=True, otherwise preprocessed tensor
- Outputs: boxes, scores, class_ids
- Reference: [yolov5_torch.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/models/yolov5_torch.py#L10-L91)

- `letterbox(image, new_size) -> (image, ratio, (dw, dh))`
- Reference: [preprocessing.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/preprocessing.py#L7-L15)

- `map_boxes(boxes, ratio, pad, original_shape) -> boxes`
- Reference: [preprocessing.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/preprocessing.py#L18-L32)

- `TemporalDecision.update(score, motion_ok) -> bool`
- Reference: [decision.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/temporal/decision.py#L4-L25)

**Parameters**

- ModelConfig ([schema.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/config/schema.py#L21-L30))
- backend: onnx or torch
- weights_path: path to .onnx or .pt weights
- input_size: square inference size (e.g., 640)
- conf_thres: confidence threshold in [0,1]
- iou_thres: NMS IoU threshold in [0,1]
- class_names: list of class labels (order must match training)
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

- Run one frame through the preprocessing path and model:

```python
import cv2
import numpy as np
from fire_smoke_pipeline.inference.preprocessing import letterbox
from fire_smoke_pipeline.models.yolov5_onnx import YoloV5nOnnx
from fire_smoke_pipeline.config.schema import ModelConfig

cfg = ModelConfig(
    backend="onnx",
    weights_path=r"C:\Users\youss\OneDrive\Desktop\youssef\fire_smoke_detection\firesmokev1.onnx",
    input_size=640,
    conf_thres=0.35,
    iou_thres=0.45,
    class_names=["fire", "smoke"],
    use_half=False,
    onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

model = YoloV5nOnnx(cfg)
frame = cv2.imread(r"C:\path\to\image.jpg")
resized, ratio, pad = letterbox(frame, cfg.input_size)
input_tensor = resized[:, :, ::-1].transpose(2, 0, 1)
input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32) / 255.0
input_tensor = input_tensor[None, ...]
boxes, scores, class_ids = model(input_tensor)
```

**Notes**

- The TensorRT engine file is not loaded by the current Python pipeline.
- class_names must match the order used during training to avoid mislabeled outputs.
- The repository does not include training scripts or evaluation reports for these models.

**Troubleshooting**

- ONNX backend fails with provider error: Ensure model.onnx_providers is set and matches available providers on the system ([ModelConfig validation](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/config/loader.py#L106-L116)).
- No detections: Verify weights_path and class_names; check conf_thres is not too high.
- Incorrect box scaling: Ensure preprocessing uses letterbox and postprocessing uses map_boxes with correct ratio and pad ([preprocessing.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/fire_smoke_detection/fire_smoke_pipeline/inference/preprocessing.py#L7-L32)).
