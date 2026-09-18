"""Small, typed point-mass primitives for the Stage-1 simulation.

This is deliberately a kinematic baseline, not a flight-dynamics model.  Wind
and ego motion are explicit constant-velocity scenarios; there are no thrust
curves, motor lag, or hidden physics tuning in this module.
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


# Preserve the original public name used by the Stage-1 scaffold.
PointMassState = PointMassState


@dataclass(frozen=True)
class WindScenario:
    """Constant wind velocity added to the moving object's world velocity."""

    vx: float = 0.0
    vy: float = 0.0

    @classmethod
    def calm(cls) -> "WindScenario":
        return cls()


@dataclass(frozen=True)
class EgoMotionScenario:
    """Constant ego velocity subtracted when producing relative motion."""

    vx: float = 0.0
    vy: float = 0.0

    @classmethod
    def stationary(cls) -> "EgoMotionScenario":
        return cls()


@dataclass(frozen=True)
class PointMassScenario:
    """Reproducible constant-acceleration scenario.

    ``acceleration`` is an explicit input.  It is not a thrust command and is
    intentionally not passed through a motor model or a thrust curve.
    """

    initial: PointMassState
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0
    duration_s: float = 1.0
    dt_s: float = 0.01
    wind: WindScenario = WindScenario.calm()
    ego_motion: EgoMotionScenario = EgoMotionScenario.stationary()


def step_point_mass(
    state: PointMassState,
    ax: float,
    ay: float,
    dt: float,
    wind: WindScenario | None = None,
    ego_motion: EgoMotionScenario | None = None,
) -> PointMassState:
    """Advance a point mass with constant acceleration for ``dt`` seconds.

    Wind and ego motion affect the relative position increment, while the
    returned ``vx``/``vy`` remain the object's own velocity.  Omitting both
    scenarios is backwards compatible with the original scaffold function.
    """

    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    wind = wind or WindScenario.calm()
    ego_motion = ego_motion or EgoMotionScenario.stationary()
    vx = state.vx + ax * dt
    vy = state.vy + ay * dt
    relative_vx = state.vx + 0.5 * ax * dt + wind.vx - ego_motion.vx
    relative_vy = state.vy + 0.5 * ay * dt + wind.vy - ego_motion.vy
    return PointMassState(
        x=state.x + relative_vx * dt,
        y=state.y + relative_vy * dt,
        vx=vx,
        vy=vy,
    )


def step_point_mass_legacy(state: PointMassState, ax: float, ay: float, dt: float) -> PointMassState:
    """Compatibility alias for callers that want the original three-input API."""

    return step_point_mass(state, ax, ay, dt)


def simulate_point_mass(scenario: PointMassScenario) -> tuple[PointMassState, ...]:
    """Return the initial state and uniformly sampled states for a scenario."""

    if scenario.duration_s < 0.0:
        raise ValueError("duration_s must be non-negative")
    if scenario.dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    steps = int(round(scenario.duration_s / scenario.dt_s))
    if abs(steps * scenario.dt_s - scenario.duration_s) > 1e-9:
        raise ValueError("duration_s must be an integer multiple of dt_s")
    states = [scenario.initial]
    for _ in range(steps):
        states.append(
            step_point_mass(
                states[-1],
                scenario.acceleration_x,
                scenario.acceleration_y,
                scenario.dt_s,
                wind=scenario.wind,
                ego_motion=scenario.ego_motion,
            )
        )
    return tuple(states)


# The original function name remains available and typed.
def step_point_mass_original(state: PointMassState, ax: float, ay: float, dt: float) -> PointMassState:
    return step_point_mass_legacy(state, ax, ay, dt)
