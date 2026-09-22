"""Tests for the event-driven looming detector (lgmd.events_theta).

Covers:
1. Hand-verifiable synthetic expansion with known FOE and known expansion strength r(t).
2. Interface compatibility of estimate_tau with sim.events.estimate_tau and ApproachScenario.
3. Validation and edge-case guards (insufficient events, etc.).
4. Consistency of the angular-rate proxy relation theta_dot_proxy = r * theta_proxy.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from sim.events import ApproachScenario, Event, generate_synthetic_events, generate_trajectory
from lgmd.events_theta import (
    ExpansionBin,
    estimate_expansion_bins,
    estimate_tau,
    estimate_tau_series,
    fit_radial_expansion,
)


def test_hand_verified_synthetic_expansion() -> None:
    """Hand-verify least-squares fit recovers known radial expansion rate r.

    Consider an FOE at (cx, cy) = (640.0, 360.0).
    Given known expansion strength r_true = 2.5 (1/s) and bias = (0.0, 0.0),
    a circle of radius R = 20 px at angles theta_k produces radial velocities:
        vx = r_true * (x - cx) = r_true * R * cos(theta_k)
        vy = r_true * (y - cy) = r_true * R * sin(theta_k)
    The least-squares fit must recover r_true within 1e-6 relative tolerance.
    """
    cx, cy = 640.0, 360.0
    r_true = 2.5  # 1/s  =>  tau = 1/r = 0.4 s
    bias_x_true = 0.15
    bias_y_true = -0.10
    radius = 25.0

    n_points = 16
    angles = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    pos_X = radius * np.cos(angles)
    pos_Y = radius * np.sin(angles)

    flows_vx = r_true * pos_X + bias_x_true
    flows_vy = r_true * pos_Y + bias_y_true

    r_fit, bx_fit, by_fit, residual_rms = fit_radial_expansion(pos_X, pos_Y, flows_vx, flows_vy)

    assert r_fit == pytest.approx(r_true, rel=1e-5)
    assert bx_fit == pytest.approx(bias_x_true, abs=1e-5)
    assert by_fit == pytest.approx(bias_y_true, abs=1e-5)
    assert residual_rms == pytest.approx(0.0, abs=1e-5)


def test_estimate_tau_drop_in_compatible_with_approach_scenario() -> None:
    """estimate_tau accepts Sequence[Event] and ApproachScenario and returns valid tau."""
    scenario = ApproachScenario(
        object_size_m=0.16,
        initial_distance_m=6.0,
        approach_speed_mps=2.0,
        duration_s=1.5,
        dt_s=0.02,
        lighting=1.0,
        sensor_noise_px=0.0,
    )
    trajectory = generate_trajectory(scenario)
    stream = generate_synthetic_events(trajectory, scenario, seed=42)

    tau_true = trajectory[0].ground_truth.tau_s  # 3.0 s
    tau_hat = estimate_tau(stream.events, scenario)

    assert isinstance(tau_hat, float)
    assert math.isfinite(tau_hat)
    assert tau_hat > 0.0
    # Estimate should be within 25% tolerance of ground truth
    error_pct = 100.0 * abs(tau_hat - tau_true) / tau_true
    assert error_pct < 25.0


def test_estimate_tau_validates_event_count() -> None:
    """estimate_tau raises ValueError on empty or insufficient event sequences."""
    scenario = ApproachScenario(
        object_size_m=0.10,
        initial_distance_m=5.0,
        approach_speed_mps=1.0,
        duration_s=1.0,
    )
    with pytest.raises(ValueError, match="at least three events"):
        estimate_tau([], scenario)

    two_events = [
        Event(x_px=640, y_px=360, timestamp_s=0.01, polarity=1),
        Event(x_px=641, y_px=360, timestamp_s=0.02, polarity=1),
    ]
    with pytest.raises(ValueError, match="at least three events"):
        estimate_tau(two_events, scenario)


def test_angular_size_rate_proxy_relation() -> None:
    """theta_dot_proxy matches r * theta_proxy per the pinhole camera relation."""
    scenario = ApproachScenario(
        object_size_m=0.16,
        initial_distance_m=6.0,
        approach_speed_mps=4.0,
        duration_s=0.8,
        dt_s=0.02,
    )
    trajectory = generate_trajectory(scenario)
    stream = generate_synthetic_events(trajectory, scenario, seed=123)

    bins = estimate_tau_series(stream.events, scenario)
    valid_bins = [b for b in bins if b.valid]
    assert len(valid_bins) > 0

    for b in valid_bins:
        assert b.tau_hat_s == pytest.approx(1.0 / b.r_1ps, rel=1e-5)
        assert b.theta_dot_proxy_pxps == pytest.approx(b.r_1ps * b.theta_proxy_px, rel=1e-5)
