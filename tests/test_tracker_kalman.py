"""Tests for the 2D linear Kalman filter (tracker.kalman)."""

from __future__ import annotations

import pytest
import numpy as np

from tracker.kalman import KalmanState2D, KalmanTracker2D


def test_2d_kalman_filter_convergence_under_noise() -> None:
    """Kalman filter converges to ground truth state and reduces covariance."""
    rng = np.random.default_rng(2026)
    tracker = KalmanTracker2D(process_noise_q=0.5, measurement_noise_sigma=0.3, nominal_pos_std=0.5)

    x0, y0 = 5.0, 15.0
    vx_true, vy_true = 1.0, -2.0
    dt = 0.05

    cov_traces = []
    for step in range(50):
        t = step * dt
        true_x = x0 + vx_true * t
        true_y = y0 + vy_true * t
        meas_x = true_x + float(rng.normal(0.0, 0.3))
        meas_y = true_y + float(rng.normal(0.0, 0.3))

        state = tracker.update((meas_x, meas_y), dt)
        cov_traces.append(np.trace(state.covariance[:2, :2]))

    assert state.x == pytest.approx(true_x, abs=0.25)
    assert state.y == pytest.approx(true_y, abs=0.25)
    assert state.vx == pytest.approx(vx_true, abs=0.35)
    assert state.vy == pytest.approx(vy_true, abs=0.35)

    # Covariance trace must be lower after convergence than initial prior
    assert cov_traces[-1] < cov_traces[0]
    assert state.confidence > 0.6
    assert state.missed_updates == 0
    assert state.covariance.shape == (4, 4)


def test_2d_kalman_missed_observation_covariance_growth() -> None:
    """Missed observations increase covariance uncertainty and degrade confidence."""
    tracker = KalmanTracker2D(process_noise_q=1.0, measurement_noise_sigma=0.5, missed_decay=0.8)

    # Initial measurement
    state = tracker.update((10.0, 10.0), dt=0.05)
    initial_conf = state.confidence
    initial_cov_trace = np.trace(state.covariance)

    # Predict-only (missed observations)
    confidences = []
    traces = []
    for _ in range(5):
        state = tracker.predict(dt=0.05)
        confidences.append(state.confidence)
        traces.append(np.trace(state.covariance))

    assert state.missed_updates == 5
    # Covariance must monotonically increase during predict-only steps
    for prev_t, next_t in zip([initial_cov_trace] + traces[:-1], traces):
        assert next_t > prev_t
    # Confidence must monotonically decrease
    for prev_c, next_c in zip([initial_conf] + confidences[:-1], confidences):
        assert next_c < prev_c

    assert state.confidence < 0.4
    assert state.uncertainty > 0.5


def test_2d_kalman_parameter_validation() -> None:
    """Kalman tracker validates noise parameters and positive time step."""
    with pytest.raises(ValueError):
        KalmanTracker2D(process_noise_q=-1.0)
    with pytest.raises(ValueError):
        KalmanTracker2D(measurement_noise_sigma=0.0)
    with pytest.raises(ValueError):
        KalmanTracker2D(nominal_pos_std=-0.5)

    tracker = KalmanTracker2D()
    with pytest.raises(ValueError, match="dt must be positive"):
        tracker.update((0.0, 0.0), dt=0.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        tracker.predict(dt=-0.05)
