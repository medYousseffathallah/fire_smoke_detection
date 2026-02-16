from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml

from fire_smoke_pipeline.config.schema import AppConfig, CameraConfig, ModelConfig, RuntimeConfig, TemporalConfig

T = TypeVar("T")


def load_config(config_path: str) -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML must be a mapping/dict.")
    cfg = AppConfig(
        camera=_parse_camera(_require_mapping(raw, "camera")),
        model=_parse_model(_require_mapping(raw, "model")),
        temporal=_parse_temporal(_require_mapping(raw, "temporal")),
        runtime=_parse_runtime(_require_mapping(raw, "runtime")),
    )
    _validate_cfg(cfg)
    return cfg


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _require_mapping(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in data:
        raise ValueError(f"Missing required config section: {key}")
    value = data[key]
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping/dict.")
    return value


def _parse_model(data: Dict[str, Any]) -> ModelConfig:
    values = dict(data)
    values["class_names"] = _parse_class_names(values.get("class_names"))
    providers = values.get("onnx_providers")
    if providers is None:
        values["onnx_providers"] = None
    else:
        if isinstance(providers, str):
            providers_list = [p.strip() for p in providers.split(",") if p.strip()]
        elif isinstance(providers, list):
            providers_list = [str(p).strip() for p in providers if str(p).strip()]
        else:
            raise ValueError("model.onnx_providers must be a list or comma-separated string or null.")
        values["onnx_providers"] = providers_list
    return _parse_dataclass(ModelConfig, values)


def _parse_camera(data: Dict[str, Any]) -> CameraConfig:
    return _parse_dataclass(CameraConfig, data)


def _parse_temporal(data: Dict[str, Any]) -> TemporalConfig:
    return _parse_dataclass(TemporalConfig, data)


def _parse_runtime(data: Dict[str, Any]) -> RuntimeConfig:
    return _parse_dataclass(RuntimeConfig, data)


def _parse_class_names(value: Any) -> List[str]:
    if value is None:
        raise ValueError("model.class_names is required.")
    if isinstance(value, str):
        names = [c.strip() for c in value.split(",") if c.strip()]
        if not names:
            raise ValueError("model.class_names must not be empty.")
        return names
    if isinstance(value, list):
        names = [str(c).strip() for c in value if str(c).strip()]
        if not names:
            raise ValueError("model.class_names must not be empty.")
        return names
    raise ValueError("model.class_names must be a list or comma-separated string.")


def _parse_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping for {cls.__name__}.")
    required = {f.name for f in fields(cls)}
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required fields for {cls.__name__}: {', '.join(sorted(missing))}")
    extra = [k for k in data.keys() if k not in required]
    if extra:
        raise ValueError(f"Unknown fields for {cls.__name__}: {', '.join(sorted(extra))}")
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        kwargs[f.name] = data[f.name]
    return cls(**kwargs)


def _validate_cfg(cfg: AppConfig) -> None:
    if cfg.camera.source_type not in {"usb", "csi", "rtsp", "http", "file"}:
        raise ValueError("camera.source_type must be one of: usb, csi, rtsp, http, file")
    if cfg.camera.rtsp_decoder not in {"cpu", "jetson-hw"}:
        raise ValueError("camera.rtsp_decoder must be one of: cpu, jetson-hw")
    if cfg.model.backend not in {"onnx", "torch"}:
        raise ValueError("model.backend must be one of: onnx, torch")
    if not cfg.model.weights_path:
        raise ValueError("model.weights_path must not be empty")
    if cfg.model.backend == "onnx" and not cfg.model.onnx_providers:
        raise ValueError("model.onnx_providers is required when model.backend is onnx")
    if cfg.model.input_size <= 0:
        raise ValueError("model.input_size must be > 0")
    if cfg.model.conf_thres < 0.0 or cfg.model.conf_thres > 1.0:
        raise ValueError("model.conf_thres must be in [0, 1]")
    if cfg.model.iou_thres < 0.0 or cfg.model.iou_thres > 1.0:
        raise ValueError("model.iou_thres must be in [0, 1]")
    if cfg.runtime.queue_size <= 0:
        raise ValueError("runtime.queue_size must be > 0")
