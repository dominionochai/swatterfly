"""Small 2-D guidance helpers for simulation-only experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Vector2:
    x: float
    y: float

    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2":
        return Vector2(self.x * scalar, self.y * scalar)


def proportional_navigation(relative_position: Vector2, relative_velocity: Vector2, pursuer_speed: float, navigation_constant: float = 3.0) -> float:
    """Return a planar PN command scalar for a simulation.

    TODO: add robust degenerate-vector handling, acceleration limits, and a
    vehicle-specific mapping. This is not suitable for real-world control.
    """
    radius_sq = relative_position.x**2 + relative_position.y**2
    if radius_sq <= 0 or pursuer_speed <= 0:
        return 0.0
    lambda_dot = (relative_position.x * relative_velocity.y - relative_position.y * relative_velocity.x) / radius_sq
    closing_speed = -(relative_position.x * relative_velocity.x + relative_position.y * relative_velocity.y) / math.sqrt(radius_sq)
    return navigation_constant * closing_speed * lambda_dot


def lead_pursuit(relative_position: Vector2, target_velocity: Vector2, pursuer_speed: float, horizon: float) -> Vector2:
    """Aim at a simple future point; solve a full intercept outside this stub."""
    if pursuer_speed <= 0:
        raise ValueError("pursuer_speed must be positive")
    return relative_position + target_velocity * max(0.0, horizon)


def blend_guidance(pn_command: float, lead_vector: Vector2, weight: float) -> tuple[float, Vector2]:
    """Return a weighted pair for a later vehicle-frame guidance adapter."""
    w = min(1.0, max(0.0, weight))
    return w * pn_command, lead_vector * (1.0 - w)
