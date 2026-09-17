"""Minimal point-mass chase scaffold.

This is not a flight dynamics model. It exists to make the first simulation
phase executable while keeping the future physics interface obvious.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PointMassState:
    """Position and velocity in a local Cartesian frame."""

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


def step_point_mass(state: PointMassState, ax: float, ay: float, dt: float) -> PointMassState:
    """Advance a constant-acceleration point mass by ``dt`` seconds.

    TODO: add gravity, thrust limits, motor lag, sensor latency, wind, and
    quaternion/attitude state in the physics-backed simulator.
    """
    if dt < 0:
        raise ValueError("dt must be non-negative")
    return PointMassState(
        x=state.x + state.vx * dt + 0.5 * ax * dt * dt,
        y=state.y + state.vy * dt + 0.5 * ay * dt * dt,
        vx=state.vx + ax * dt,
        vy=state.vy + ay * dt,
    )
