"""Scalar eta looming detector placeholder.

The scalar detector is the first measurable baseline before a full temporal
LGMD2 implementation. It consumes angular size and angular expansion rate;
it intentionally does not claim canonical neuron-level fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LoomingSample:
    theta: float
    theta_dot: float
    timestamp: float


class ScalarEtaDetector:
    """Compute a guarded eta score and time-to-contact estimate."""

    def __init__(self, *, theta_floor: float = 1e-6) -> None:
        self.theta_floor = theta_floor

    def update(self, sample: LoomingSample) -> tuple[float, float]:
        """Return ``(eta, tau_hat)`` for one observation.

        TODO: replace the scalar score with delayed excitation/inhibition,
        spatial pooling, feed-forward inhibition, and calibrated hysteresis.
        """
        if sample.theta < 0 or sample.theta_dot < 0:
            raise ValueError("theta and theta_dot must be non-negative")
        eta = sample.theta_dot / max(sample.theta, self.theta_floor)
        tau_hat = math.inf if eta == 0 else 1.0 / eta
        return eta, tau_hat
