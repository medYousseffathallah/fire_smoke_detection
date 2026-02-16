import argparse
import queue
import threading
import time
from dataclasses import replace

import cv2

from fire_smoke_pipeline.camera.capture import FrameGrabber
from fire_smoke_pipeline.config.loader import load_config
from fire_smoke_pipeline.inference.worker import InferenceWorker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-path", type=str, help="Override model path from config")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Override model path if provided via command line
    if args.model_path:
        cfg = replace(cfg, model=replace(cfg.model, weights_path=args.model_path))

    frame_queue = queue.Queue(maxsize=max(1, int(cfg.runtime.queue_size)))
    stop_event = threading.Event()

    grabber = FrameGrabber(cfg.camera, frame_queue, stop_event)
    worker = InferenceWorker(cfg.model, cfg.temporal, frame_queue, stop_event, cfg.runtime.display)

    grabber.start()
    worker.start()
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
    worker.join()
    grabber.join()
    if cfg.runtime.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
