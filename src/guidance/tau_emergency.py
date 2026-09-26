"""Safety override and time-to-contact (tau) emergency boundary checks.

This is a stub interface for Phase 9's safety override system. It performs no
actuation. Phase 9 must wire this into the actual command path without
modifying this function's signature unless Phase 9's brief explicitly says so.

Math specification (docs/01-math-lgmd.md & docs/02-math-dragonfly.md):
- Imminent collision threshold: tau_hat <= tau_emergency_threshold
- Hysteresis requirement: tau_release_threshold > tau_emergency_threshold (eta_off < eta_on, tau_off > tau_on)
- Trigger and release thresholds must never be identical to prevent limit-cycle chattering.
"""

from __future__ import annotations


def tau_emergency_check(
    tau_hat: float,
    tau_emergency_threshold: float,
    confidence: float,
    min_confidence: float = 0.40,
) -> bool:
    """Return True if an immediate evasive override should be triggered.

    Parameters:
        tau_hat: Estimated time-to-contact [seconds].
        tau_emergency_threshold: Critical time-to-contact threshold [seconds] (e.g. 0.5s).
        confidence: Tracker or looming detector confidence in [0, 1].
        min_confidence: Minimum confidence required to trust the tau estimate.

    Returns:
        bool: True if an emergency condition is detected with sufficient confidence.
    """
    if tau_emergency_threshold <= 0.0:
        raise ValueError("tau_emergency_threshold must be strictly positive")
    if min_confidence < 0.0 or min_confidence > 1.0:
        raise ValueError("min_confidence must be between 0.0 and 1.0")

    # Only trigger if confidence meets minimum threshold
    if confidence < min_confidence:
        return False

    # Imminent collision: positive closing time within emergency threshold
    return 0.0 < tau_hat <= tau_emergency_threshold


def tau_emergency_release_check(
    tau_hat: float,
    tau_release_threshold: float,
    confidence: float,
    min_confidence: float = 0.40,
) -> bool:
    """Return True if the emergency condition has cleared and override can be released.

    Hysteresis requirement:
        tau_release_threshold > tau_emergency_threshold.

    Parameters:
        tau_hat: Estimated time-to-contact [seconds].
        tau_release_threshold: Threshold above which the threat has cleared [seconds].
        confidence: Tracker or looming detector confidence in [0, 1].
        min_confidence: Minimum confidence required for release evaluation.

    Returns:
        bool: True if the emergency state can safely be released.
    """
    if tau_release_threshold <= 0.0:
        raise ValueError("tau_release_threshold must be strictly positive")

    # Threat has cleared if time-to-contact has expanded past the release threshold,
    # or if target is receding / non-looming (tau_hat <= 0.0).
    if tau_hat <= 0.0 or tau_hat >= tau_release_threshold:
        return True

    return False
