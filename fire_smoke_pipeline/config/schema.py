from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    weights_path: str
    input_size: int
    conf_thres: float
    iou_thres: float
    class_names: List[str]
    use_half: bool
    onnx_providers: Optional[List[str]]


@dataclass(frozen=True)
class TemporalConfig:
    ema_alpha: float
    on_threshold: float
    off_threshold: float
    on_frames: int
    off_frames: int
    motion_gate: bool
    motion_threshold: float
    high_conf_bypass: float


@dataclass(frozen=True)
class RuntimeConfig:
    display: bool
    queue_size: int


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    model: ModelConfig
    temporal: TemporalConfig
    runtime: RuntimeConfig

