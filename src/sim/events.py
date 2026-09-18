"""Deterministic event-camera data generation for the Stage-1 baseline.

The renderer uses the event-camera invariant that the apparent radius of an
object is proportional to ``l / x``.  Each event contains only sensor-like
fields (x, y, timestamp, polarity); the exact ground truth is carried beside
the stream, never embedded in an event.

The optional v2e extra is detected when requested.  The deterministic renderer
is the canonical Windows-friendly path because v2e's upstream command-line
stack is Linux/CUDA-oriented and its Python API is not stable across releases.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Literal, Sequence

import numpy as np

EventBackend = Literal["auto", "synthetic", "v2e"]


@dataclass(frozen=True)
class GroundTruth:
    """Exact controllable quantities: l (size), x (distance), and u (speed)."""

    l_m: float
    x_m: float
    u_mps: float

    @property
    def tau_s(self) -> float:
        return self.x_m / self.u_mps


@dataclass(frozen=True)
class ApproachScenario:
    """Constant-speed approach used by the sweep and reusable in Phase 3."""

    object_size_m: float
    initial_distance_m: float
    approach_speed_mps: float
    duration_s: float
    dt_s: float = 0.01
    lighting: float = 1.0
    sensor_noise_px: float = 0.0
    injected_latency_s: float = 0.0
    focal_length_px: float = 640.0
    sensor_width_px: int = 1280
    sensor_height_px: int = 720

    def validate(self) -> None:
        if self.object_size_m <= 0 or self.initial_distance_m <= 0 or self.approach_speed_mps <= 0:
            raise ValueError("object size, distance, and speed must be positive")
        if self.duration_s <= 0 or self.dt_s <= 0:
            raise ValueError("duration_s and dt_s must be positive")
        if self.duration_s >= self.initial_distance_m / self.approach_speed_mps:
            raise ValueError("duration must end before contact")
        if self.lighting <= 0 or self.sensor_noise_px < 0 or self.injected_latency_s < 0:
            raise ValueError("lighting must be positive; noise and latency non-negative")


@dataclass(frozen=True)
class TrajectorySample:
    timestamp_s: float
    ground_truth: GroundTruth
    projected_radius_px: float


@dataclass(frozen=True)
class Event:
    x_px: int
    y_px: int
    timestamp_s: float
    polarity: int


@dataclass(frozen=True)
class EventStream:
    events: tuple[Event, ...]
    ground_truth_tau_s: float
    backend: str
    v2e_available: bool


def v2e_is_available() -> bool:
    """Return whether the optional upstream v2e package can be imported."""

    try:
        import v2e  # type: ignore[import-not-found,unused]
    except ImportError:
        return False
    return True


def generate_trajectory(scenario: ApproachScenario) -> tuple[TrajectorySample, ...]:
    """Generate a uniformly sampled approach with exact l/x/u ground truth."""

    scenario.validate()
    steps = int(round(scenario.duration_s / scenario.dt_s))
    times = np.arange(steps + 1, dtype=float) * scenario.dt_s
    samples: list[TrajectorySample] = []
    for timestamp in times:
        distance = scenario.initial_distance_m - scenario.approach_speed_mps * float(timestamp)
        radius = scenario.focal_length_px * scenario.object_size_m / (2.0 * distance)
        samples.append(
            TrajectorySample(
                timestamp_s=float(timestamp),
                ground_truth=GroundTruth(
                    l_m=scenario.object_size_m,
                    x_m=distance,
                    u_mps=scenario.approach_speed_mps,
                ),
                projected_radius_px=radius,
            )
        )
    return tuple(samples)


def _render_trajectory_events(
    trajectory: Sequence[TrajectorySample],
    scenario: ApproachScenario,
    seed: int,
) -> tuple[Event, ...]:
    """Render edge events deterministically from radius growth."""

    rng = np.random.default_rng(seed)
    events: list[Event] = []
    cx = (scenario.sensor_width_px - 1) / 2.0
    cy = (scenario.sensor_height_px - 1) / 2.0
    for index, (left, right) in enumerate(zip(trajectory, trajectory[1:])):
        radius_delta = right.projected_radius_px - left.projected_radius_px
        count = max(1, int(round(abs(radius_delta) * 18.0 * scenario.lighting)))
        for event_index in range(count):
            fraction = (event_index + 0.5) / count
            radius = left.projected_radius_px + fraction * radius_delta
            angle = (2.0 * pi * ((event_index * 0.61803398875 + index * 0.17320508) % 1.0))
            x = cx + radius * cos(angle)
            y = cy + radius * sin(angle)
            if scenario.sensor_noise_px:
                x += float(rng.normal(0.0, scenario.sensor_noise_px))
                y += float(rng.normal(0.0, scenario.sensor_noise_px))
            timestamp = left.timestamp_s + fraction * (right.timestamp_s - left.timestamp_s)
            events.append(
                Event(
                    x_px=int(round(x)),
                    y_px=int(round(y)),
                    timestamp_s=float(timestamp + scenario.injected_latency_s),
                    polarity=1 if radius_delta >= 0.0 else -1,
                )
            )
    return tuple(events)


def generate_synthetic_events(
    trajectory: Sequence[TrajectorySample],
    scenario: ApproachScenario,
    *,
    seed: int = 0,
    backend: EventBackend = "auto",
) -> EventStream:
    """Generate events from a trajectory using the reproducible baseline renderer.

    ``backend='v2e'`` verifies that the optional v2e dependency is installed,
    then uses this trajectory-native renderer so the same run is reproducible
    on Windows and Linux.  ``auto`` records v2e availability in metadata but
    never makes it a hard requirement for the Stage-1 sweep.
    """

    if not trajectory:
        raise ValueError("trajectory must contain at least one sample")
    if backend == "v2e" and not v2e_is_available():
        raise RuntimeError("backend='v2e' requested but v2e is not installed; use pip install -e .[event-camera]")
    if backend not in ("auto", "synthetic", "v2e"):
        raise ValueError(f"unsupported event backend: {backend}")
    events = _render_trajectory_events(trajectory, scenario, seed)
    return EventStream(
        events=events,
        ground_truth_tau_s=trajectory[0].ground_truth.tau_s,
        backend="v2e-compatible-synthetic" if backend == "v2e" else "synthetic",
        v2e_available=v2e_is_available(),
    )


def estimate_tau(events: Sequence[Event], scenario: ApproachScenario) -> float:
    """Estimate tau from event radius growth using a linear zero crossing.

    Since ``1/r(t)`` is affine in time for constant-speed approach, the
    intercept of a least-squares fit estimates time-to-contact.  The estimate
    intentionally sees delayed/noisy sensor timestamps, not the ground truth.
    """

    if len(events) < 3:
        raise ValueError("at least three events are required")
    cx = (scenario.sensor_width_px - 1) / 2.0
    cy = (scenario.sensor_height_px - 1) / 2.0
    times = np.asarray([event.timestamp_s for event in events], dtype=float)
    radii = np.asarray(
        [((event.x_px - cx) ** 2 + (event.y_px - cy) ** 2) ** 0.5 for event in events],
        dtype=float,
    )
    valid = radii > 0.5
    if int(valid.sum()) < 3:
        raise ValueError("event geometry is too small to estimate tau")
    slope, intercept = np.polyfit(times[valid], 1.0 / radii[valid], 1)
    if slope >= 0.0:
        raise ValueError("event radius did not grow monotonically")
    estimate = -intercept / slope
    if estimate <= 0.0:
        raise ValueError("non-positive tau estimate")
    return float(estimate)
