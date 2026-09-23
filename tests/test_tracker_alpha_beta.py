"""Tests for the 1D and 2D alpha-beta trackers (tracker.alpha_beta)."""

from __future__ import annotations

import pytest
import numpy as np

from tracker.alpha_beta import (
    AlphaBetaState,
    AlphaBetaState2D,
    AlphaBetaTracker,
    AlphaBetaTracker2D,
)


def test_legacy_1d_alpha_beta_tracker() -> None:
    """Preserved 1D scalar implementation tracks motion and validates dt."""
    tracker = AlphaBetaTracker(AlphaBetaState(position=10.0, velocity=-1.0), alpha=0.85, beta=0.05)
    with pytest.raises(ValueError, match="dt must be positive"):
        tracker.update(10.0, dt=0.0)

    # 1D approach from 10.0 at -1.0 m/s
    dt = 0.1
    for step in range(30):
        true_pos = 10.0 - 1.0 * (step * dt)
        state = tracker.update(true_pos, dt)

    assert state.position == pytest.approx(true_pos, abs=0.2)
    assert state.velocity == pytest.approx(-1.0, abs=0.2)


def test_2d_alpha_beta_tracker_convergence_under_noise() -> None:
    """2D tracker converges to ground-truth trajectory under measurement noise."""
    rng = np.random.default_rng(2026)
    tracker = AlphaBetaTracker2D(alpha=0.60, beta=0.02, nominal_noise_scale=0.5)

    x0, y0 = 10.0, -5.0
    vx_true, vy_true = -2.0, 1.5
    dt = 0.05

    for step in range(80):
        t = step * dt
        true_x = x0 + vx_true * t
        true_y = y0 + vy_true * t
        meas_x = true_x + float(rng.normal(0.0, 0.15))
        meas_y = true_y + float(rng.normal(0.0, 0.15))

        state = tracker.update((meas_x, meas_y), dt)

    # After 80 steps, filter should have converged close to ground truth
    assert state.x == pytest.approx(true_x, abs=0.3)
    assert state.y == pytest.approx(true_y, abs=0.3)
    assert state.vx == pytest.approx(vx_true, abs=0.4)
    assert state.vy == pytest.approx(vy_true, abs=0.4)
    assert state.confidence > 0.5
    assert state.missed_updates == 0
    assert state.covariance is None  # Covariance is not faked


def test_2d_alpha_beta_missed_observation_degradation() -> None:
    """Consecutive missed observations degrade confidence and increase uncertainty."""
    tracker = AlphaBetaTracker2D(alpha=0.8, beta=0.1, nominal_noise_scale=1.0, missed_decay=0.8)

    # Initialize with clean observation
    state = tracker.update((10.0, 20.0), dt=0.05)
    initial_conf = state.confidence

    # 5 consecutive missed observations
    confidences = []
    for _ in range(5):
        state = tracker.update(None, dt=0.05)
        confidences.append(state.confidence)

    assert state.missed_updates == 5
    # Confidences must monotonically decrease
    for prev_c, next_c in zip([initial_conf] + confidences[:-1], confidences):
        assert next_c < prev_c
    assert state.confidence < 0.35
    assert state.uncertainty > 1.0


def test_2d_alpha_beta_parameter_validation() -> None:
    """Tracker validates alpha, beta, noise scale, and dt inputs."""
    with pytest.raises(ValueError):
        AlphaBetaTracker2D(alpha=0.0)
    with pytest.raises(ValueError):
        AlphaBetaTracker2D(alpha=1.5)
    with pytest.raises(ValueError):
        AlphaBetaTracker2D(beta=-0.1)
    with pytest.raises(ValueError):
        AlphaBetaTracker2D(nominal_noise_scale=-1.0)

    tracker = AlphaBetaTracker2D()
    with pytest.raises(ValueError, match="dt must be positive"):
        tracker.update((0.0, 0.0), dt=-0.01)
    with pytest.raises(ValueError, match="dt must be positive"):
        tracker.predict(dt=0.0)
