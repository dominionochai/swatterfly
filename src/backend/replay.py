import random
from datetime import datetime, timedelta, timezone

import numpy as np

from .config import Settings
from .schemas import ReplayFile, TelemetryFrame


class SyntheticReplay:
    """Produces made-up telemetry for UI development. It is not detection or flight math."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def build(self) -> ReplayFile:
        seed = self.settings.replay_seed
        random.seed(seed)
        rng = np.random.default_rng(seed)
        replay_id = f"demo-{seed}"
        start = datetime.now(timezone.utc).replace(microsecond=0)
        frames: list[TelemetryFrame] = []
        for index in range(self.settings.replay_frames):
            phase = index / max(1, self.settings.replay_frames - 1)
            theta = round(2.5 + 39.0 * phase + float(rng.uniform(-0.35, 0.35)), 2)
            tau = round(max(0.6, 12.0 - 10.8 * phase), 2)
            eta = round(min(1.0, 0.08 + phase * 0.46 + (0.48 if index % 9 in (6, 7) else 0.0)), 2)
            locked = index % 12 in range(4, 10)
            state = "LOCKED" if locked else ("COASTING" if index % 12 in (10, 11) else "SEARCHING")
            confidence = round(min(0.99, 0.42 + phase * 0.5 + float(rng.uniform(-0.03, 0.03))), 2)
            frames.append(TelemetryFrame(
                frame_id=f"{replay_id}-{index:04d}", replay_id=replay_id,
                theta=theta, eta=eta, tau=tau, target_lock_state=state,
                confidence=confidence, timestamp=start + timedelta(seconds=index * self.settings.replay_interval_seconds),
                closing_speed_mps=round(3.2 + phase * 8.4, 2), battery_percent=round(96.0 - phase * 3.1, 1),
                gps={"lat": 51.5007 + phase * 0.0003, "lon": -0.1246 + phase * 0.0004},
                threat_log=[f"{(start + timedelta(seconds=index * self.settings.replay_interval_seconds)).strftime('%H:%M:%S')}Z synthetic frame"],
                line_of_sight_deg=round(-6.0 + phase * 12.0, 2), bearing_rate_dps=round(0.8 + phase * 1.4, 2),
                predicted_intercept_point={"x": round(0.38 + phase * 0.24, 3), "y": round(0.5 - phase * 0.08, 3)},
                acceleration_vector={"x": round(0.18 + phase * 0.3, 2), "y": round(-0.1 + phase * 0.16, 2)},
                lgmd_spike=round(min(1.0, eta * (0.7 + (0.3 if index % 5 == 0 else 0))), 2),
            ))
        return ReplayFile(replay_id=replay_id, seed=seed, generated_at=start, frames=frames)
