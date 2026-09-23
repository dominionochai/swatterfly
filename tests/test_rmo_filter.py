"""Tests for the nonlinear Radial Motion Opponency (RMO) filter (lgmd.rmo_filter).

Verifies the implementation of Schubert et al. 2025 (DOI: 10.1088/2634-4386/add0da):
1. Symmetric looming flow produces positive, balanced opponent activations and passes.
2. Asymmetric / single-region noise produces zero opponent activation and is suppressed.
3. Uniform translation (ego-motion) produces one-sided activation and is suppressed.
4. Gated bins have expansion rates and angular-rate proxies suppressed to zero.
5. Four-quadrant mode opponency requires simultaneous 4-way expansion.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from lgmd.events_theta import ExpansionBin
from lgmd.rmo_filter import (
    RMOActivation,
    RMOFilter,
    compute_rmo_score,
    filter_expansion_bins_rmo,
)


def test_symmetric_looming_flow_passes_rmo() -> None:
    """True symmetric looming flow produces balanced L/R activations and opens gate."""
    # 32 points on circle of radius 20 px centered at (0, 0) expanding at 2.0 1/s
    angles = np.linspace(0, 2 * math.pi, 32, endpoint=False)
    pos_X = 20.0 * np.cos(angles)
    pos_Y = 20.0 * np.sin(angles)
    flows_vx = 2.0 * pos_X  # outward radial flow
    flows_vy = 2.0 * pos_Y

    l_act, r_act, top_act, bottom_act, rmo_score = compute_rmo_score(
        pos_X, pos_Y, flows_vx, flows_vy, mode="two_region"
    )

    assert l_act > 10.0
    assert r_act > 10.0
    assert l_act == pytest.approx(r_act, rel=1e-3)
    assert rmo_score == pytest.approx(l_act, rel=1e-3)
    assert rmo_score >= 0.5


def test_asymmetric_single_region_noise_suppressed() -> None:
    """Single-region noise cluster produces zero opponent activation and is suppressed."""
    # 15 events confined to left quadrant (X in [-25, -10]) with random velocities
    rng = np.random.default_rng(2026)
    pos_X = rng.uniform(-25.0, -10.0, 15)
    pos_Y = rng.uniform(-10.0, 10.0, 15)
    flows_vx = rng.uniform(-30.0, 30.0, 15)
    flows_vy = rng.uniform(-30.0, 30.0, 15)

    l_act, r_act, top_act, bottom_act, rmo_score = compute_rmo_score(
        pos_X, pos_Y, flows_vx, flows_vy, mode="two_region"
    )

    # Left may have spurious outward components, but Right has 0 events
    assert r_act == 0.0
    assert rmo_score == 0.0  # Nonlinear product sqrt(L * R) is exactly zero


def test_uniform_translation_suppressed() -> None:
    """Uniform translation (e.g. camera panning or lateral ego-motion) is rejected."""
    # Points on circle moving uniformly right at +40 px/s
    angles = np.linspace(0, 2 * math.pi, 32, endpoint=False)
    pos_X = 20.0 * np.cos(angles)
    pos_Y = 20.0 * np.sin(angles)
    flows_vx = np.full_like(pos_X, 40.0)
    flows_vy = np.zeros_like(pos_Y)

    l_act, r_act, top_act, bottom_act, rmo_score = compute_rmo_score(
        pos_X, pos_Y, flows_vx, flows_vy, mode="two_region"
    )

    # On left side, outward is negative x, but flow is +40 (inward), so l_act == 0
    assert l_act == 0.0
    assert r_act == pytest.approx(40.0)
    assert rmo_score == 0.0


def test_filter_expansion_bins_rmo_gating() -> None:
    """ExpansionBins are correctly suppressed when RMO condition is not met."""
    raw_bin = ExpansionBin(
        timestamp_s=0.5,
        r_1ps=2.0,
        tau_hat_s=0.5,
        theta_proxy_px=15.0,
        theta_dot_proxy_pxps=30.0,
        bias_x_pxps=0.0,
        bias_y_pxps=0.0,
        n_events=10,
        flow_count=8,
        fit_residual_pxps=0.1,
        valid=True,
    )

    # Synthesize events that form a single-sided translation (should be gated out)
    events = [
        {"x_px": 600.0 + i, "y_px": 360.0, "timestamp_s": 0.48 + 0.002 * i, "polarity": 1}
        for i in range(10)
    ]

    filtered_bins, activations = filter_expansion_bins_rmo(
        [raw_bin], events, rmo_threshold=1.0, bin_dt_s=0.05
    )

    assert len(filtered_bins) == 1
    assert len(activations) == 1

    f_bin = filtered_bins[0]
    act = activations[0]

    assert not act.gate_open
    assert not f_bin.valid
    assert f_bin.r_1ps == 0.0
    assert f_bin.theta_dot_proxy_pxps == 0.0
    assert math.isinf(f_bin.tau_hat_s)


def test_four_quadrant_mode_requires_vertical_and_horizontal_symmetry() -> None:
    """Four-quadrant mode requires all four subfields to exhibit outward flow."""
    angles = np.linspace(0, 2 * math.pi, 32, endpoint=False)
    pos_X = 20.0 * np.cos(angles)
    pos_Y = 20.0 * np.sin(angles)

    # Pure horizontal expansion, zero vertical expansion
    flows_vx = 2.0 * pos_X
    flows_vy = np.zeros_like(pos_Y)

    _, _, top_act, bottom_act, rmo_4q = compute_rmo_score(
        pos_X, pos_Y, flows_vx, flows_vy, mode="four_quadrant"
    )

    assert top_act == 0.0
    assert bottom_act == 0.0
    assert rmo_4q == 0.0
