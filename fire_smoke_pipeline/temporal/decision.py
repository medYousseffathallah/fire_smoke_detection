from ..config.schema import TemporalConfig


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

