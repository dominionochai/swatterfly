"""Planar guidance laws for 2D pursuit, interception, and safety gating.

Provides:
- ``Vector2``: 2D vector primitive with Euclidean norms and arithmetic.
- ``GuidanceCommand``: Structured lateral acceleration command exposing both
  pre-clamp ``raw_command`` and post-clamp ``saturated_command``, turn rate,
  saturation flag, holding status, and confidence.
- ``pure_pursuit``: Steers toward current target position (baseline / fallback).
- ``constant_bearing``: Drives line-of-sight rate to zero scaled by vehicle speed.
- ``proportional_navigation``: Intercept law scaling LOS rate by closing speed (N * Vc * lambda_dot).
- ``blend_guidance``: Combines proportional navigation with predictive lead pursuit.
- ``lead_pursuit``: Geometric future-aim point calculation.

Degenerate handling policy:
If relative range R < 1e-5 m, pursuer speed <= 0, or closing speed Vc <= 1e-5 m/s
(opening target / near-zero closing speed), guidance functions do not divide by
zero or emit NaN/Inf; they return a documented safe-fallback command with
``is_holding=True``, ``raw_command=0.0``, and ``saturated_command=0.0``.

Confidence gating policy:
If tracker confidence is below ``min_confidence`` (default 0.40), guidance
returns a safe-holding state (``is_holding=True, saturated_command=0.0``)
preventing erratic actuation on low-confidence or ghost tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


@dataclass(frozen=True)
class Vector2:
    """Two-dimensional vector with basic vector operations."""

    x: float
    y: float

    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2":
        return Vector2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector2":
        return Vector2(self.x * scalar, self.y * scalar)

    def dot(self, other: "Vector2") -> float:
        return self.x * other.x + self.y * other.y

    def norm_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    def norm(self) -> float:
        return math.sqrt(self.norm_sq())


@dataclass(frozen=True)
class GuidanceCommand:
    """Lateral acceleration command output with saturation and confidence flags.

    Attributes:
        raw_command: Unclamped lateral acceleration requested by guidance law [m/s^2].
        saturated_command: Clamped lateral acceleration respecting max_accel and max_turn_rate [m/s^2].
        turn_rate_radps: Saturated yaw rate command (saturated_command / pursuer_speed) [rad/s].
        is_saturated: True if acceleration or turn-rate clamping was active.
        is_holding: True if guidance held zero command due to low confidence or degenerate geometry.
        confidence: Tracker confidence input value.
    """

    raw_command: float
    saturated_command: float
    turn_rate_radps: float = 0.0
    is_saturated: bool = False
    is_holding: bool = False
    confidence: float = 1.0

    def __float__(self) -> float:
        """Allow implicit or explicit float conversion to the saturated command."""
        return float(self.saturated_command)


def wrap_angle(angle_rad: float) -> float:
    """Wrap angle in radians to [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _apply_saturation_and_confidence(
    raw_command: float,
    pursuer_speed: float,
    max_accel_mps2: float,
    max_turn_rate_radps: float,
    confidence: float,
    min_confidence: float,
) -> GuidanceCommand:
    """Apply confidence thresholding, acceleration clamping, and turn-rate clamping."""
    if confidence < min_confidence:
        return GuidanceCommand(
            raw_command=raw_command,
            saturated_command=0.0,
            turn_rate_radps=0.0,
            is_saturated=False,
            is_holding=True,
            confidence=confidence,
        )

    # Lateral acceleration limit based on turn rate: a_lat = V * omega
    turn_accel_limit = pursuer_speed * max_turn_rate_radps if pursuer_speed > 0 else 0.0
    effective_accel_limit = (
        min(max_accel_mps2, turn_accel_limit) if turn_accel_limit > 0.0 else max_accel_mps2
    )

    saturated = max(-effective_accel_limit, min(effective_accel_limit, raw_command))
    turn_rate = (saturated / pursuer_speed) if pursuer_speed > 0 else 0.0
    is_sat = abs(raw_command - saturated) > 1e-6

    return GuidanceCommand(
        raw_command=raw_command,
        saturated_command=saturated,
        turn_rate_radps=turn_rate,
        is_saturated=is_sat,
        is_holding=False,
        confidence=confidence,
    )


def pure_pursuit(
    relative_position: Vector2,
    relative_velocity: Vector2,
    pursuer_speed: float,
    pursuer_heading: Optional[float] = None,
    *,
    tau_pursuit: float = 0.5,
    max_accel_mps2: float = 20.0,
    max_turn_rate_radps: float = 2.0,
    confidence: float = 1.0,
    min_confidence: float = 0.40,
    covariance: Optional[Any] = None,
) -> GuidanceCommand:
    """Pure pursuit: steer directly toward current line of sight to target.

    Mathematical formulation:
        lambda = atan2(r_y, r_x)
        eta = wrap_angle(lambda - psi)
        a_cmd = (V / tau_pursuit) * sin(eta)

    Degenerate handling:
        Returns safe holding command if range R < 1e-5 m or pursuer_speed <= 0.
    """
    range_sq = relative_position.norm_sq()
    if range_sq <= 1e-10 or pursuer_speed <= 0.0:
        return GuidanceCommand(
            raw_command=0.0,
            saturated_command=0.0,
            turn_rate_radps=0.0,
            is_saturated=False,
            is_holding=True,
            confidence=confidence,
        )

    los_angle = math.atan2(relative_position.y, relative_position.x)
    if pursuer_heading is None:
        # Infer pursuer heading from target velocity - relative velocity if available
        heading = los_angle
    else:
        heading = pursuer_heading

    heading_error = wrap_angle(los_angle - heading)
    tau = max(1e-3, tau_pursuit)
    raw_command = (pursuer_speed / tau) * math.sin(heading_error)

    return _apply_saturation_and_confidence(
        raw_command,
        pursuer_speed,
        max_accel_mps2,
        max_turn_rate_radps,
        confidence,
        min_confidence,
    )


def constant_bearing(
    relative_position: Vector2,
    relative_velocity: Vector2,
    pursuer_speed: float,
    *,
    bearing_gain: float = 3.0,
    max_accel_mps2: float = 20.0,
    max_turn_rate_radps: float = 2.0,
    confidence: float = 1.0,
    min_confidence: float = 0.40,
    covariance: Optional[Any] = None,
) -> GuidanceCommand:
    """Constant bearing guidance: drive line-of-sight rate to zero scaled by vehicle speed.

    Mathematical formulation:
        lambda_dot = (r_x * v_rel_y - r_y * v_rel_x) / R^2
        a_cmd = K_cb * V * lambda_dot

    Unlike Proportional Navigation, Constant Bearing scales by vehicle speed V
    rather than closing velocity Vc, providing steady kinematic rate feedback.
    """
    range_sq = relative_position.norm_sq()
    if range_sq <= 1e-10 or pursuer_speed <= 0.0:
        return GuidanceCommand(
            raw_command=0.0,
            saturated_command=0.0,
            turn_rate_radps=0.0,
            is_saturated=False,
            is_holding=True,
            confidence=confidence,
        )

    lambda_dot = (
        relative_position.x * relative_velocity.y - relative_position.y * relative_velocity.x
    ) / range_sq
    raw_command = bearing_gain * pursuer_speed * lambda_dot

    return _apply_saturation_and_confidence(
        raw_command,
        pursuer_speed,
        max_accel_mps2,
        max_turn_rate_radps,
        confidence,
        min_confidence,
    )


def proportional_navigation(
    relative_position: Vector2,
    relative_velocity: Vector2,
    pursuer_speed: float,
    navigation_constant: float = 3.0,
    *,
    max_accel_mps2: float = 20.0,
    max_turn_rate_radps: float = 2.0,
    confidence: float = 1.0,
    min_confidence: float = 0.40,
    covariance: Optional[Any] = None,
) -> GuidanceCommand:
    """True Proportional Navigation (PN): lateral acceleration proportional to closing speed and LOS rate.

    Mathematical formulation:
        R = |r|
        lambda_dot = (r_x * v_rel_y - r_y * v_rel_x) / R^2
        Vc = - (r dot v_rel) / R
        a_cmd = N * Vc * lambda_dot

    Degenerate handling:
        Returns safe holding command if range R < 1e-5 m, pursuer_speed <= 0,
        or closing speed Vc <= 1e-5 m/s (opening or parallel trajectory).
    """
    range_sq = relative_position.norm_sq()
    if range_sq <= 1e-10 or pursuer_speed <= 0.0:
        return GuidanceCommand(
            raw_command=0.0,
            saturated_command=0.0,
            turn_rate_radps=0.0,
            is_saturated=False,
            is_holding=True,
            confidence=confidence,
        )

    range_m = math.sqrt(range_sq)
    dot_prod = relative_position.x * relative_velocity.x + relative_position.y * relative_velocity.y
    closing_speed = -dot_prod / range_m

    # Degenerate: target is opening or zero closing speed
    if closing_speed <= 1e-5:
        return GuidanceCommand(
            raw_command=0.0,
            saturated_command=0.0,
            turn_rate_radps=0.0,
            is_saturated=False,
            is_holding=True,
            confidence=confidence,
        )

    lambda_dot = (
        relative_position.x * relative_velocity.y - relative_position.y * relative_velocity.x
    ) / range_sq
    raw_command = navigation_constant * closing_speed * lambda_dot

    return _apply_saturation_and_confidence(
        raw_command,
        pursuer_speed,
        max_accel_mps2,
        max_turn_rate_radps,
        confidence,
        min_confidence,
    )


def lead_pursuit(
    relative_position: Vector2,
    target_velocity: Vector2,
    pursuer_speed: float,
    horizon: float,
) -> Vector2:
    """Calculate geometric aim-point vector for predictive intercept."""
    if pursuer_speed <= 0:
        raise ValueError("pursuer_speed must be positive")
    return relative_position + target_velocity * max(0.0, horizon)


def blend_guidance(
    relative_position: Union[Vector2, float],
    relative_velocity: Union[Vector2, Vector2],
    pursuer_speed: Union[float, float] = 10.0,
    pursuer_heading: Optional[float] = None,
    target_velocity: Optional[Vector2] = None,
    *,
    navigation_constant: float = 3.0,
    lead_horizon_s: float = 0.5,
    lead_weight: float = 0.3,
    max_accel_mps2: float = 20.0,
    max_turn_rate_radps: float = 2.0,
    confidence: float = 1.0,
    min_confidence: float = 0.40,
    covariance: Optional[Any] = None,
) -> Union[GuidanceCommand, Tuple[float, Vector2]]:
    """Blended guidance: PN dynamic acceleration + Lead pursuit predictive aim point.

    Mathematical formulation (docs/02-math-dragonfly.md Eq. 49-50):
        a_PN = N * Vc * lambda_dot
        p_aim = r + v_target * T_L
        lambda_L = atan2(p_aim.y, p_aim.x)
        a_lead = (V / T_L) * wrap_angle(lambda_L - psi)
        a_cmd = (1 - w_L) * a_PN + w_L * a_lead

    Also supports backward-compatible 3-argument signature:
        blend_guidance(pn_command: float, lead_vector: Vector2, weight: float) -> (float, Vector2)
    """
    # Backward compatibility with legacy scalar-and-vector signature:
    if isinstance(relative_position, (int, float)) and isinstance(relative_velocity, Vector2):
        pn_cmd = float(relative_position)
        lead_vec = relative_velocity
        w = min(1.0, max(0.0, float(pursuer_speed)))
        return w * pn_cmd, lead_vec * (1.0 - w)

    # Full GuidanceCommand mode:
    pos: Vector2 = relative_position  # type: ignore
    vel: Vector2 = relative_velocity  # type: ignore

    range_sq = pos.norm_sq()
    if range_sq <= 1e-10 or pursuer_speed <= 0.0:
        return GuidanceCommand(
            raw_command=0.0,
            saturated_command=0.0,
            turn_rate_radps=0.0,
            is_saturated=False,
            is_holding=True,
            confidence=confidence,
        )

    # 1. Proportional Navigation component
    pn_res = proportional_navigation(
        pos,
        vel,
        pursuer_speed,
        navigation_constant=navigation_constant,
        max_accel_mps2=1e6,  # Unclamped for blending
        max_turn_rate_radps=1e6,
        confidence=1.0,
        min_confidence=0.0,
    )
    a_pn = pn_res.raw_command

    # 2. Lead pursuit component
    heading = (
        pursuer_heading
        if pursuer_heading is not None
        else math.atan2(pos.y, pos.x)
    )
    t_horizon = max(0.05, lead_horizon_s)

    if target_velocity is not None:
        vt = target_velocity
    else:
        # Approximate target velocity from relative velocity and pursuer heading
        vp = Vector2(pursuer_speed * math.cos(heading), pursuer_speed * math.sin(heading))
        vt = vel + vp

    lead_point = lead_pursuit(pos, vt, pursuer_speed, t_horizon)
    lead_angle = math.atan2(lead_point.y, lead_point.x)
    lead_error = wrap_angle(lead_angle - heading)
    a_lead = (pursuer_speed / t_horizon) * math.sin(lead_error)

    # 3. Weighted dimensionally consistent blend [m/s^2]
    w = min(1.0, max(0.0, lead_weight))
    raw_command = (1.0 - w) * a_pn + w * a_lead

    return _apply_saturation_and_confidence(
        raw_command,
        pursuer_speed,
        max_accel_mps2,
        max_turn_rate_radps,
        confidence,
        min_confidence,
    )
