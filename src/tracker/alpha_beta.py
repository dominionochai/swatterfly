"""Alpha-beta target tracker for event-camera and radar state estimation.

Provides:
- ``AlphaBetaState`` / ``AlphaBetaTracker``: Preserved 1D scalar implementation.
- ``AlphaBetaState2D`` / ``AlphaBetaTracker2D``: 2D position + velocity tracker
  (x, y, vx, vy) with exposed innovation-based confidence and uncertainty metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AlphaBetaState:
    """One-dimensional target state (position and velocity)."""

    position: float = 0.0
    velocity: float = 0.0


class AlphaBetaTracker:
    """One-dimensional scalar alpha-beta filter."""

    def __init__(
        self,
        state: AlphaBetaState | None = None,
        alpha: float = 0.85,
        beta: float = 0.05,
    ) -> None:
        self.state = state or AlphaBetaState()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def update(self, measurement: float, dt: float) -> AlphaBetaState:
        """Predict and correct one scalar measurement."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        prediction = self.state.position + self.state.velocity * dt
        residual = measurement - prediction
        self.state.position = prediction + self.alpha * residual
        self.state.velocity += (self.beta / dt) * residual
        return self.state


@dataclass
class AlphaBetaState2D:
    """Two-dimensional target state with exposed uncertainty and confidence."""

    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 1.0
    uncertainty: float = 0.0
    covariance: None = None  # Alpha-beta does not compute covariance; explicitly None
    innovation_norm: float = 0.0
    smoothed_residual: float = 0.0
    missed_updates: int = 0
    total_updates: int = 0


class AlphaBetaTracker2D:
    """Two-dimensional alpha-beta filter tracking (x, y, vx, vy).

    Exposes empirical innovation/residual-based confidence without faking a
    Kalman covariance matrix. Confidence degrades when observations are missed
    or when innovation residuals spike relative to baseline noise scale.
    """

    def __init__(
        self,
        initial_state: Optional[AlphaBetaState2D] = None,
        *,
        alpha: float = 0.85,
        beta: float = 0.10,
        nominal_noise_scale: float = 1.0,
        missed_decay: float = 0.85,
        residual_smoothing: float = 0.20,
    ) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        if not (0.0 <= beta <= 1.0):
            raise ValueError("beta must be in [0, 1]")
        if nominal_noise_scale <= 0.0:
            raise ValueError("nominal_noise_scale must be positive")

        self.state = initial_state or AlphaBetaState2D()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.nominal_noise_scale = float(nominal_noise_scale)
        self.missed_decay = float(missed_decay)
        self.residual_smoothing = float(residual_smoothing)
        self._initialized = initial_state is not None

    def predict(self, dt: float) -> AlphaBetaState2D:
        """Advance state forward by dt without an observation (missed update)."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        # Constant velocity prediction
        self.state.x += self.state.vx * dt
        self.state.y += self.state.vy * dt
        self.state.missed_updates += 1

        # Uncertainty grows with missed updates and elapsed time
        drift_growth = self.nominal_noise_scale * math.sqrt(self.state.missed_updates) * (1.0 + dt)
        self.state.uncertainty = self.state.smoothed_residual + drift_growth

        # Confidence decays exponentially with consecutive missed observations
        decay_factor = self.missed_decay ** self.state.missed_updates
        residual_factor = 1.0 / (1.0 + (self.state.uncertainty / self.nominal_noise_scale) ** 2)
        self.state.confidence = float(max(0.0, min(1.0, residual_factor * decay_factor)))
        self.state.innovation_norm = 0.0

        return self.state

    def update(
        self,
        measurement: Optional[Tuple[float, float]],
        dt: float,
    ) -> AlphaBetaState2D:
        """Process observation (z_x, z_y) or handle missed observation if None."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        if measurement is None:
            return self.predict(dt)

        z_x, z_y = float(measurement[0]), float(measurement[1])

        if not self._initialized:
            self.state.x = z_x
            self.state.y = z_y
            self.state.vx = 0.0
            self.state.vy = 0.0
            self.state.confidence = 0.5
            self.state.uncertainty = self.nominal_noise_scale
            self.state.smoothed_residual = 0.0
            self.state.missed_updates = 0
            self.state.total_updates = 1
            self._initialized = True
            return self.state

        # Prediction step
        pred_x = self.state.x + self.state.vx * dt
        pred_y = self.state.y + self.state.vy * dt

        # Innovation residual
        res_x = z_x - pred_x
        res_y = z_y - pred_y
        res_norm = math.hypot(res_x, res_y)
        self.state.innovation_norm = res_norm

        # Correction step
        self.state.x = pred_x + self.alpha * res_x
        self.state.y = pred_y + self.alpha * res_y
        self.state.vx += (self.beta / dt) * res_x
        self.state.vy += (self.beta / dt) * res_y

        # Reset missed update counter
        self.state.missed_updates = 0
        self.state.total_updates += 1

        # Smooth residual over time to avoid single-frame overreaction
        if self.state.total_updates <= 2:
            self.state.smoothed_residual = res_norm
        else:
            self.state.smoothed_residual = (
                (1.0 - self.residual_smoothing) * self.state.smoothed_residual
                + self.residual_smoothing * res_norm
            )

        self.state.uncertainty = self.state.smoothed_residual

        # Confidence: high when smoothed residual is in line with nominal noise
        res_ratio = self.state.uncertainty / self.nominal_noise_scale
        self.state.confidence = float(max(0.0, min(1.0, 1.0 / (1.0 + res_ratio ** 2))))

        return self.state


__all__ = [
    "AlphaBetaState",
    "AlphaBetaTracker",
    "AlphaBetaState2D",
    "AlphaBetaTracker2D",
]
