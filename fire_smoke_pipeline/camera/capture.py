import queue
import threading
import time

import cv2

from .gstreamer import build_capture_source
from ..config.schema import CameraConfig


class FrameGrabber(threading.Thread):
    def __init__(self, cfg: CameraConfig, frame_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.frame_queue = frame_queue
        self.stop_event = stop_event

    def run(self):
        source = build_capture_source(self.cfg)
        api_preference = cv2.CAP_GSTREAMER if self.cfg.use_gstreamer else cv2.CAP_ANY
        last_log = 0.0
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(source, api_preference)
            if not cap.isOpened():
                now = time.monotonic()
                if now - last_log > 5:
                    print("Camera source not opened. Retrying...")
                    last_log = now
                time.sleep(0.5)
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
            failure_count = 0
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    failure_count += 1
                    if failure_count >= 30:
                        now = time.monotonic()
                        if now - last_log > 5:
                            print("Camera stream interrupted. Reconnecting...")
                            last_log = now
                        break
                    time.sleep(0.01)
                    continue
                failure_count = 0
                ts = time.monotonic()
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put((ts, frame))
            cap.release()
