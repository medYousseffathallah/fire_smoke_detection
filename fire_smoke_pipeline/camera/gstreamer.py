from ..config.schema import CameraConfig


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

