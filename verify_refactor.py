import numpy as np

import fire_smoke_pipeline as legacy

from fire_smoke_pipeline.inference.postprocess import nms as nms_new
from fire_smoke_pipeline.inference.postprocess import xywh_to_xyxy as xywh_to_xyxy_new
from fire_smoke_pipeline.inference.preprocessing import letterbox as letterbox_new
from fire_smoke_pipeline.temporal.decision import TemporalDecision as TemporalDecisionNew
from fire_smoke_pipeline.utils.motion import MotionScorer as MotionScorerNew


def _assert_close(a, b, atol=0.0, rtol=0.0):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not np.allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol):
            raise AssertionError("Arrays differ.")
    else:
        if abs(a - b) > atol + rtol * abs(b):
            raise AssertionError("Scalars differ.")


def test_letterbox():
    rng = np.random.default_rng(0)
    frame = (rng.random((321, 517, 3)) * 255).astype(np.uint8)
    out_old, r_old, pad_old = legacy.letterbox(frame, 640)
    out_new, r_new, pad_new = letterbox_new(frame, 640)
    _assert_close(out_old, out_new)
    _assert_close(r_old, r_new)
    if pad_old != pad_new:
        raise AssertionError("Pad differs.")


def test_postprocess():
    rng = np.random.default_rng(1)
    xywh = rng.random((100, 4)).astype(np.float32) * 640
    boxes_old = legacy.xywh_to_xyxy(xywh)
    boxes_new = xywh_to_xyxy_new(xywh)
    _assert_close(boxes_old, boxes_new)

    scores = rng.random((100,)).astype(np.float32)
    keep_old = legacy.nms(boxes_old, scores, 0.45)
    keep_new = nms_new(boxes_new, scores, 0.45)
    if keep_old != keep_new:
        raise AssertionError("NMS keep differs.")


def test_temporal_and_motion():
    temporal_cfg = legacy.TemporalConfig(
        ema_alpha=0.6,
        on_threshold=0.6,
        off_threshold=0.4,
        on_frames=3,
        off_frames=5,
        motion_gate=True,
        motion_threshold=0.08,
        high_conf_bypass=0.85,
    )
    old = legacy.TemporalDecision(temporal_cfg)
    new = TemporalDecisionNew(temporal_cfg)

    scores = [0.0, 0.2, 0.7, 0.7, 0.7, 0.3, 0.0, 0.0]
    motions = [False, True, True, True, True, True, True, True]
    for s, m in zip(scores, motions):
        a_old = old.update(s, m)
        a_new = new.update(s, m)
        if a_old != a_new:
            raise AssertionError("Temporal decision differs.")
        _assert_close(old.ema, new.ema, atol=0.0, rtol=0.0)
        if old.on_count != new.on_count or old.off_count != new.off_count or old.alert != new.alert:
            raise AssertionError("Temporal internal state differs.")

    rng = np.random.default_rng(2)
    frame1 = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    frame2 = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    boxes = np.array([[10, 10, 100, 120]], dtype=np.float32)

    def motion_score_reference(prev_gray, frame, boxes):
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            return 0.0, gray
        diff = cv2.absdiff(gray, prev_gray)
        if boxes.size == 0:
            return float(np.mean(diff)) / 255.0, gray
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
        return float(np.max(motion_scores)) if motion_scores else float(np.mean(diff)) / 255.0, gray

    prev = None
    old_score1, prev = motion_score_reference(prev, frame1, boxes)
    old_score2, prev = motion_score_reference(prev, frame2, boxes)

    motion_new = MotionScorerNew()
    new_score1 = motion_new.score(frame1, boxes)
    new_score2 = motion_new.score(frame2, boxes)
    _assert_close(old_score1, new_score1)
    _assert_close(old_score2, new_score2)


def main():
    test_letterbox()
    test_postprocess()
    test_temporal_and_motion()
    print("OK")


if __name__ == "__main__":
    main()
