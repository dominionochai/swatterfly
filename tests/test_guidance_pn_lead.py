"""Unit tests for guidance laws (pure pursuit, constant bearing, PN, blend)."""

from __future__ import annotations

import math
import pytest

from guidance.pn_lead import (
    GuidanceCommand,
    Vector2,
    blend_guidance,
    constant_bearing,
    proportional_navigation,
    pure_pursuit,
    wrap_angle,
)


def test_wrap_angle() -> None:
    """Angle wrapping correctly maps to [-pi, pi)."""
    assert wrap_angle(0.0) == pytest.approx(0.0)
    assert wrap_angle(3.0 * math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(-3.0 * math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(math.pi / 2.0) == pytest.approx(math.pi / 2.0)


def test_analytic_intercept_guidance_commands() -> None:
    """Assert guidance laws match exact hand-computed values on known geometry."""
    # Geometry:
    # Pursuer at (0, 0), speed V = 10 m/s, heading psi = 0 (along +x)
    # Target at (10, 10), rel_pos r = (10, 10)
    # Target rel_vel v_rel = (-10, 0)
    # R^2 = 200, R = 10 * sqrt(2)
    # Closing speed Vc = -(10 * -10 + 10 * 0) / (10 * sqrt(2)) = 10 / sqrt(2) = 5 * sqrt(2) ~= 7.071068 m/s
    # LOS rate lambda_dot = (10 * 0 - 10 * -10) / 200 = 100 / 200 = 0.5 rad/s
    rel_pos = Vector2(10.0, 10.0)
    rel_vel = Vector2(-10.0, 0.0)
    pursuer_speed = 10.0
    pursuer_heading = 0.0

    # 1. Proportional Navigation: a_raw = N * Vc * lambda_dot
    # For N = 3.0: 3.0 * (10 / sqrt(2)) * 0.5 = 15 / sqrt(2) ~= 10.60660 m/s^2
    pn_cmd = proportional_navigation(
        rel_pos,
        rel_vel,
        pursuer_speed=pursuer_speed,
        navigation_constant=3.0,
        max_accel_mps2=50.0,
    )
    expected_pn = 3.0 * (10.0 / math.sqrt(2.0)) * 0.5
    assert pn_cmd.raw_command == pytest.approx(expected_pn, rel=1e-5)
    assert pn_cmd.saturated_command == pytest.approx(expected_pn, rel=1e-5)
    assert not pn_cmd.is_saturated
    assert not pn_cmd.is_holding

    # 2. Constant Bearing: a_raw = K_cb * V * lambda_dot
    # For K_cb = 3.0: 3.0 * 10.0 * 0.5 = 15.0 m/s^2
    cb_cmd = constant_bearing(
        rel_pos,
        rel_vel,
        pursuer_speed=pursuer_speed,
        bearing_gain=3.0,
        max_accel_mps2=50.0,
    )
    assert cb_cmd.raw_command == pytest.approx(15.0, rel=1e-5)
    assert cb_cmd.saturated_command == pytest.approx(15.0, rel=1e-5)
    assert not cb_cmd.is_saturated

    # 3. Pure Pursuit: lambda = atan2(10, 10) = pi / 4
    # eta = pi / 4 - 0 = pi / 4
    # a_raw = (V / tau) * sin(eta) = (10 / 0.5) * sin(pi/4) = 20 * (sqrt(2)/2) = 10 * sqrt(2) ~= 14.14213 m/s^2
    pp_cmd = pure_pursuit(
        rel_pos,
        rel_vel,
        pursuer_speed=pursuer_speed,
        pursuer_heading=pursuer_heading,
        tau_pursuit=0.5,
        max_accel_mps2=50.0,
    )
    expected_pp = 20.0 * math.sin(math.pi / 4.0)
    assert pp_cmd.raw_command == pytest.approx(expected_pp, rel=1e-5)
    assert pp_cmd.saturated_command == pytest.approx(expected_pp, rel=1e-5)
    assert not pp_cmd.is_saturated

    # 4. Blend Guidance: weighted mix of PN and lead
    blend_cmd = blend_guidance(
        rel_pos,
        rel_vel,
        pursuer_speed=pursuer_speed,
        pursuer_heading=pursuer_heading,
        navigation_constant=3.0,
        lead_horizon_s=0.5,
        lead_weight=0.0,  # 0% lead -> pure PN
        max_accel_mps2=50.0,
    )
    assert isinstance(blend_cmd, GuidanceCommand)
    assert blend_cmd.raw_command == pytest.approx(expected_pn, rel=1e-4)


def test_degenerate_cases_handled_safely() -> None:
    """Degenerate zero range and opening target produce safe hold without crash."""
    # Zero range
    zero_pos = Vector2(0.0, 0.0)
    zero_vel = Vector2(0.0, 0.0)

    cmd_zero_pn = proportional_navigation(zero_pos, zero_vel, pursuer_speed=10.0)
    assert cmd_zero_pn.is_holding is True
    assert cmd_zero_pn.saturated_command == 0.0
    assert not math.isnan(cmd_zero_pn.raw_command)

    cmd_zero_pp = pure_pursuit(zero_pos, zero_vel, pursuer_speed=10.0)
    assert cmd_zero_pp.is_holding is True
    assert cmd_zero_pp.saturated_command == 0.0

    cmd_zero_cb = constant_bearing(zero_pos, zero_vel, pursuer_speed=10.0)
    assert cmd_zero_cb.is_holding is True
    assert cmd_zero_cb.saturated_command == 0.0

    # Opening target (closing speed <= 0)
    # Target is in front at (10, 0), but moving away at +20 m/s while pursuer speed is 10 m/s
    opening_pos = Vector2(10.0, 0.0)
    opening_vel = Vector2(20.0, 0.0)  # v_rel is positive x -> opening
    cmd_opening = proportional_navigation(opening_pos, opening_vel, pursuer_speed=10.0)
    assert cmd_opening.is_holding is True
    assert cmd_opening.saturated_command == 0.0


def test_saturation_clamping_and_observability() -> None:
    """Acceleration saturation clamps output and raw != saturated is observable."""
    rel_pos = Vector2(10.0, 10.0)
    rel_vel = Vector2(-10.0, 0.0)

    # Hand-computed raw command is 15.0 m/s^2. Set clamp to 4.0 m/s^2.
    cmd = constant_bearing(
        rel_pos,
        rel_vel,
        pursuer_speed=10.0,
        bearing_gain=3.0,
        max_accel_mps2=4.0,
        max_turn_rate_radps=2.0,
    )
    assert cmd.is_saturated is True
    assert cmd.saturated_command == pytest.approx(4.0)
    assert cmd.raw_command == pytest.approx(15.0)
    assert cmd.raw_command != cmd.saturated_command
    assert cmd.turn_rate_radps == pytest.approx(0.40)  # 4.0 / 10.0


def test_confidence_gating_produces_hold() -> None:
    """Low tracker confidence gates guidance into safe hold rather than commanding motion."""
    rel_pos = Vector2(10.0, 10.0)
    rel_vel = Vector2(-10.0, 0.0)

    # Sub-threshold confidence: 0.25 < min_confidence (0.40)
    cmd = proportional_navigation(
        rel_pos,
        rel_vel,
        pursuer_speed=10.0,
        confidence=0.25,
        min_confidence=0.40,
    )
    assert cmd.is_holding is True
    assert cmd.saturated_command == 0.0
    assert cmd.turn_rate_radps == 0.0

    # Above-threshold confidence: 0.85 >= min_confidence
    cmd_good = proportional_navigation(
        rel_pos,
        rel_vel,
        pursuer_speed=10.0,
        confidence=0.85,
        min_confidence=0.40,
    )
    assert cmd_good.is_holding is False
    assert cmd_good.saturated_command > 0.0
