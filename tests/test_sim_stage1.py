"""Tests for the typed Stage-1 simulation baseline."""

from __future__ import annotations

import math

import pytest

from sim.events import ApproachScenario, estimate_tau, generate_synthetic_events, generate_trajectory
from sim.point_mass import EgoMotionScenario, PointMassState, WindScenario, step_point_mass


def test_point_mass_wind_and_ego_motion_are_explicit() -> None:
    state = step_point_mass(
        PointMassState(0.0, 0.0),
        0.0,
        0.0,
        1.0,
        wind=WindScenario(vx=2.0),
        ego_motion=EgoMotionScenario(vx=0.5),
    )
    assert state.x == pytest.approx(1.5)
    assert state.y == pytest.approx(0.0)
    assert state.vx == pytest.approx(0.0)


def test_trajectory_preserves_exact_l_x_u_ground_truth() -> None:
    scenario = ApproachScenario(0.08, 6.0, 2.0, 1.0, dt_s=0.02)
    trajectory = generate_trajectory(scenario)
    assert trajectory[0].ground_truth.l_m == pytest.approx(0.08)
    assert trajectory[0].ground_truth.x_m == pytest.approx(6.0)
    assert trajectory[0].ground_truth.u_mps == pytest.approx(2.0)
    assert trajectory[0].ground_truth.tau_s == pytest.approx(3.0)
    assert trajectory[-1].ground_truth.x_m == pytest.approx(4.0)


def test_deterministic_events_and_tau_estimate() -> None:
    scenario = ApproachScenario(0.08, 6.0, 2.0, 1.0, dt_s=0.02)
    trajectory = generate_trajectory(scenario)
    first = generate_synthetic_events(trajectory, scenario, seed=7)
    second = generate_synthetic_events(trajectory, scenario, seed=7)
    assert first.events == second.events
    estimate = estimate_tau(first.events, scenario)
    assert math.isfinite(estimate)
    assert estimate == pytest.approx(3.0, abs=0.15)
