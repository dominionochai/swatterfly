"""Alpha-beta tracker placeholder for event-camera target states."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlphaBetaState:
    position: float = 0.0
    velocity: float = 0.0


class AlphaBetaTracker:
    """One-dimensional teaching implementation; extend to image/world vectors."""

    def __init__(self, state: AlphaBetaState | None = None, alpha: float = 0.85, beta: float = 0.05) -> None:
        self.state = state or AlphaBetaState()
        self.alpha = alpha
        self.beta = beta

    def update(self, measurement: float, dt: float) -> AlphaBetaState:
        """Predict and correct one scalar measurement.

        TODO: make dimensions explicit, validate timestamp monotonicity, and
        compare against a Kalman filter under event-camera noise and latency.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        prediction = self.state.position + self.state.velocity * dt
        residual = measurement - prediction
        self.state.position = prediction + self.alpha * residual
        self.state.velocity += (self.beta / dt) * residual
        return self.state
