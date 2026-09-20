"""Gabbiani canonical firing model for an insect looming neuron.

This module implements the firing expression from Gabbiani, Krapp & Laurent,
"Computation of object approach by a wide-field, motion-sensitive neuron"
(J. Neurosci. 19, 1999), for an insect (locust) looming neuron:

    firing ∝ ψ(t − δ) · exp(−α · θ(t − δ))

where ψ is the angular size expansion rate, θ is the angular size, δ is a fixed
delay, and

    α = 1 / tan(θ_thres / 2)

with the threshold θ_thres in the range 15–40° and δ in the range 15–35 ms, as
configured below. This is the **Gabbiani canonical firing model** and must not
be confused with the **engineering scalar approximation** in ``lgmd.scalar_eta``
(``eta = theta_dot / theta``, ``tau_hat = 1 / eta``). The two detectors are
deliberately kept distinct: this module owns the canonical firing nonlinearity
and an explicit trigger/release hysteresis; the scalar module owns the simple
ratio and no hysteresis.

The network topology here (a feed-forward looming path that pools expansion
motion into a single wide-field firing unit, with a delayed, inhibited drive)
is an *original* Python implementation. It uses the open-source
``fuqinbing/LGMD2-open-source`` repository (github.com/fuqinbing/LGMD2-open-source)
as a *structural reference only* — the number and role of its stages — and does
not translate its code. That repository's specific cited paper is not in
``docs/04-papers.md`` and has not been verified; no claim is made that this
module reproduces it.

Units note: this baseline consumes the Phase 2 dataset's scalar theta proxy
``projected_radius_px`` as the detector's angular-size input (the same quantity
the engineering scalar approximation consumes). With α calibrated per the
formula above, the canonical firing score is therefore
``psi * exp(-alpha * theta)`` in px/s and is small for large theta. The
trigger thresholds ``eta_on``/``eta_off`` are calibrated on that scale and are
not comparable to the scalar detector's ``eta = theta_dot / theta`` values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .scalar_eta import LoomingSample

DEFAULT_THETA_THRESHOLD_DEG = 40.0
DEFAULT_DELAY_MS = 20.0
DEFAULT_ETA_ON = 2.0e-4
DEFAULT_ETA_OFF = 1.6e-4


class NetworkLgmdDetector:
    """Gabbiani canonical firing model with internal hysteresis.

    ``update`` mirrors the interface of ``lgmd.scalar_eta.ScalarEtaDetector``:
    it accepts one :class:`LoomingSample` and returns a two-tuple
    ``(score, tau_hat)``. Here ``score`` is the canonical firing response
    ``psi(t - delta) * exp(-alpha * theta(t - delta))`` and ``tau_hat`` is the
    geometric time-to-contact ``theta / psi`` reported alongside the firing
    score (the firing score drives detection; ``tau_hat`` is the reported
    estimate used for ground-truth comparison).

    Trigger/release hysteresis is internal: the detector latches ``active``
    when the firing score first reaches ``eta_on`` and clears it only when the
    score falls back to ``eta_off < eta_on``. This is the hysteresis the
    engineering scalar approximation does not have; it is implemented here and
    is not retrofitted into ``lgmd.scalar_eta``.
    """

    def __init__(
        self,
        *,
        theta_threshold_deg: float = DEFAULT_THETA_THRESHOLD_DEG,
        delay_ms: float = DEFAULT_DELAY_MS,
        eta_on: float = DEFAULT_ETA_ON,
        eta_off: float = DEFAULT_ETA_OFF,
        theta_floor: float = 1e-6,
    ) -> None:
        if not 15.0 <= theta_threshold_deg <= 40.0:
            raise ValueError("theta_threshold_deg must be in the Gabbiani range 15..40 deg")
        if not 15.0 <= delay_ms <= 35.0:
            raise ValueError("delay_ms must be in the Gabbiani range 15..35 ms")
        if eta_on <= 0.0 or eta_off <= 0.0 or eta_off >= eta_on:
            raise ValueError("require 0 < eta_off < eta_on")
        if theta_floor <= 0.0:
            raise ValueError("theta_floor must be positive")

        self.name = "network"
        self.theta_threshold_deg = theta_threshold_deg
        self.delay_s = delay_ms / 1000.0
        self.eta_on = float(eta_on)
        self.eta_off = float(eta_off)
        self.theta_floor = theta_floor
        # alpha = 1 / tan(theta_thres / 2), theta_thres in degrees, per the
        # Gabbiani, Krapp & Laurent 1999 firing expression.
        self.alpha = 1.0 / math.tan(math.radians(theta_threshold_deg / 2.0))

        self._history: list[tuple[float, float, float]] = []
        self._active = False
        self._triggered = False
        self._released = False
        self._trigger_timestamp: float | None = None
        self._peak_score = -math.inf
        self._peak_timestamp = math.nan

    @property
    def active(self) -> bool:
        """Whether the detector is currently latched above ``eta_off``."""
        return self._active

    @property
    def triggered(self) -> bool:
        """True only on the sample whose firing score crossed ``eta_on``."""
        return self._triggered

    @property
    def released(self) -> bool:
        """True only on the sample whose firing score fell back to ``eta_off``."""
        return self._released

    @property
    def trigger_timestamp(self) -> float | None:
        return self._trigger_timestamp

    @property
    def peak_score(self) -> float:
        return self._peak_score

    @property
    def peak_timestamp(self) -> float:
        return self._peak_timestamp

    def _delayed(self, timestamp: float) -> tuple[float, float]:
        """Return ``(psi, theta)`` from delay ``delta`` seconds earlier.

        Uses the most recent historical sample at or before ``timestamp -
        delay_s``; before the delay window fills, the earliest buffered sample
        is used.
        """
        target = timestamp - self.delay_s
        delayed = self._history[0]
        for entry in self._history:
            if entry[0] <= target:
                delayed = entry
            else:
                break
        return delayed[2], delayed[1]

    def update(self, sample: LoomingSample) -> tuple[float, float]:
        """Advance the canonical firing model by one observation.

        Returns ``(firing_score, tau_hat)`` where ``firing_score`` is the
        Gabbiani canonical firing response and ``tau_hat`` is the geometric
        time-to-contact estimate ``theta / psi``.
        """
        theta = max(float(sample.theta), 0.0)
        psi = max(float(sample.theta_dot), 0.0)
        timestamp = float(sample.timestamp)

        self._history.append((timestamp, theta, psi))
        while len(self._history) > 1 and self._history[0][0] < timestamp - self.delay_s - 1e-9:
            self._history.pop(0)

        psi_delayed, theta_delayed = self._delayed(timestamp)
        firing = psi_delayed * math.exp(-self.alpha * max(theta_delayed, self.theta_floor))
        tau_hat = theta / max(psi, self.theta_floor)

        self._triggered = False
        self._released = False
        if not self._active and firing >= self.eta_on:
            self._active = True
            self._triggered = True
            self._trigger_timestamp = timestamp
            self._peak_score = -math.inf
        elif self._active and firing <= self.eta_off:
            self._active = False
            self._released = True
        if firing > self._peak_score:
            self._peak_score = firing
            self._peak_timestamp = timestamp

        return firing, tau_hat


def build_detector(**kwargs: Any) -> NetworkLgmdDetector:
    """Factory with the same call pattern used by ``scripts/compare_lgmd.py``."""
    return NetworkLgmdDetector(**kwargs)


__all__ = ["LoomingSample", "NetworkLgmdDetector", "build_detector"]