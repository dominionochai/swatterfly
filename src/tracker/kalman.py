"""Linear 2D Constant-Velocity (CV) Kalman filter for target tracking.

Provides:
- ``KalmanState2D``: 2D state estimate with full 4x4 estimation covariance,
  positional uncertainty, innovation residual, and normalized confidence.
- ``KalmanTracker2D``: Standard CV linear Kalman filter sharing the same
  calling interface as ``AlphaBetaTracker2D``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class KalmanState2D:
    """Two-dimensional target state with rigorous covariance and exposed uncertainty."""

    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 1.0
    uncertainty: float = 0.0
    covariance: Optional[np.ndarray] = None
    innovation_norm: float = 0.0
    missed_updates: int = 0
    total_updates: int = 0


class KalmanTracker2D:
    """Constant-Velocity (CV) linear Kalman filter tracking [x, y, vx, vy]^T.

    State transition:
        F(dt) = [[1, 0, dt, 0],
                 [0, 1, 0, dt],
                 [0, 0, 1,  0],
                 [0, 0, 0,  1]]

    Measurement model:
        H = [[1, 0, 0, 0],
             [0, 1, 0, 0]]

    Process noise: Continuous White Noise Acceleration (CWNA) parameterized by q.
    Measurement noise: Diagonal matrix with variance sigma_m^2.
    """

    def __init__(
        self,
        initial_state: Optional[KalmanState2D] = None,
        *,
        process_noise_q: float = 1.0,
        measurement_noise_sigma: float = 1.0,
        nominal_pos_std: float = 1.0,
        missed_decay: float = 0.85,
    ) -> None:
        if process_noise_q <= 0.0:
            raise ValueError("process_noise_q must be positive")
        if measurement_noise_sigma <= 0.0:
            raise ValueError("measurement_noise_sigma must be positive")
        if nominal_pos_std <= 0.0:
            raise ValueError("nominal_pos_std must be positive")

        self.q = float(process_noise_q)
        self.r_sigma = float(measurement_noise_sigma)
        self.R = np.eye(2, dtype=float) * (self.r_sigma ** 2)
        self.nominal_pos_std = float(nominal_pos_std)
        self.missed_decay = float(missed_decay)

        self.state = initial_state or KalmanState2D()
        self._x = np.array([self.state.x, self.state.y, self.state.vx, self.state.vy], dtype=float)

        if self.state.covariance is not None:
            self._P = np.array(self.state.covariance, dtype=float, copy=True)
        else:
            # Default prior covariance
            self._P = np.diag([self.r_sigma ** 2, self.r_sigma ** 2, 10.0, 10.0]).astype(float)

        self.state.covariance = self._P
        self._H = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0]], dtype=float)
        self._I = np.eye(4, dtype=float)
        self._initialized = initial_state is not None

    def _compute_Q(self, dt: float) -> np.ndarray:
        """Discrete process noise covariance for continuous white noise acceleration."""
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        q = self.q

        Q = np.zeros((4, 4), dtype=float)
        Q[0, 0] = q * dt3 / 3.0
        Q[0, 2] = q * dt2 / 2.0
        Q[1, 1] = q * dt3 / 3.0
        Q[1, 3] = q * dt2 / 2.0
        Q[2, 0] = q * dt2 / 2.0
        Q[2, 2] = q * dt
        Q[3, 1] = q * dt2 / 2.0
        Q[3, 3] = q * dt
        return Q

    def _sync_state(self) -> None:
        """Sync internal numpy vectors with public dataclass attributes."""
        self.state.x = float(self._x[0])
        self.state.y = float(self._x[1])
        self.state.vx = float(self._x[2])
        self.state.vy = float(self._x[3])
        self.state.covariance = self._P

        # Positional uncertainty: standard deviation of position estimation error
        pos_var = float(0.5 * (self._P[0, 0] + self._P[1, 1]))
        pos_std = math.sqrt(max(0.0, pos_var))
        self.state.uncertainty = pos_std

        # Normalized confidence: based on ratio of positional std to nominal scale
        ratio = pos_std / self.nominal_pos_std
        cov_confidence = 1.0 / (1.0 + ratio ** 2)
        decay_factor = self.missed_decay ** self.state.missed_updates
        self.state.confidence = float(max(0.0, min(1.0, cov_confidence * decay_factor)))

    def predict(self, dt: float) -> KalmanState2D:
        """Propagate state and covariance forward by dt without measurement."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

        Q = self._compute_Q(dt)

        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        self.state.missed_updates += 1
        self.state.innovation_norm = 0.0

        self._sync_state()
        return self.state

    def update(
        self,
        measurement: Optional[Tuple[float, float]],
        dt: float,
    ) -> KalmanState2D:
        """Perform predict step by dt, then correct with observation (z_x, z_y)."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        if measurement is None:
            return self.predict(dt)

        z = np.array([float(measurement[0]), float(measurement[1])], dtype=float)

        if not self._initialized:
            self._x[0] = z[0]
            self._x[1] = z[1]
            self._x[2] = 0.0
            self._x[3] = 0.0
            self._P = np.diag([self.r_sigma ** 2, self.r_sigma ** 2, 10.0, 10.0]).astype(float)
            self.state.missed_updates = 0
            self.state.total_updates = 1
            self.state.innovation_norm = 0.0
            self._initialized = True
            self._sync_state()
            return self.state

        # Prediction step
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        Q = self._compute_Q(dt)

        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + Q

        # Innovation
        y = z - (self._H @ x_pred)
        self.state.innovation_norm = float(np.linalg.norm(y))

        # Innovation covariance
        S = self._H @ P_pred @ self._H.T + self.R

        # Kalman gain
        K = P_pred @ self._H.T @ np.linalg.inv(S)

        # State update
        self._x = x_pred + (K @ y)

        # Joseph form covariance update for numerical stability: P = (I - KH) P (I - KH)^T + K R K^T
        IKH = self._I - (K @ self._H)
        self._P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T

        self.state.missed_updates = 0
        self.state.total_updates += 1

        self._sync_state()
        return self.state


__all__ = [
    "KalmanState2D",
    "KalmanTracker2D",
]
