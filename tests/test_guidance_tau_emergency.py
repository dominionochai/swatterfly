"""Unit tests for tau emergency safety check and release hysteresis."""

from __future__ import annotations

import pytest

from guidance.tau_emergency import (
    tau_emergency_check,
    tau_emergency_release_check,
)


def test_tau_emergency_trigger_and_confidence_gating() -> None:
    """Check triggers on imminent collision only when confidence exceeds threshold."""
    tau_thresh = 0.50  # 500 ms

    # High confidence, imminent collision (0.3s <= 0.5s) -> Trigger
    assert tau_emergency_check(tau_hat=0.30, tau_emergency_threshold=tau_thresh, confidence=0.85)

    # High confidence, safe distance (1.2s > 0.5s) -> No trigger
    assert not tau_emergency_check(tau_hat=1.20, tau_emergency_threshold=tau_thresh, confidence=0.85)

    # Low confidence (< min_confidence=0.40), even with small tau_hat -> Gate holds (No trigger)
    assert not tau_emergency_check(tau_hat=0.20, tau_emergency_threshold=tau_thresh, confidence=0.25)

    # Receding/opening or negative tau_hat -> No trigger
    assert not tau_emergency_check(tau_hat=-0.10, tau_emergency_threshold=tau_thresh, confidence=0.90)
    assert not tau_emergency_check(tau_hat=0.00, tau_emergency_threshold=tau_thresh, confidence=0.90)


def test_tau_emergency_hysteresis_separation() -> None:
    """Assert trigger and release thresholds are separate; state does not chatter at same tau."""
    tau_trigger = 0.50   # Trigger when tau <= 0.50s
    tau_release = 1.20   # Release only when tau >= 1.20s

    assert tau_release > tau_trigger

    # Condition 1: Imminent approach at tau_hat = 0.40s
    # Should trigger
    triggered = tau_emergency_check(0.40, tau_trigger, confidence=0.90)
    assert triggered is True
    # At the same tau_hat = 0.40s, release check MUST NOT release
    release_at_trigger = tau_emergency_release_check(0.40, tau_release, confidence=0.90)
    assert release_at_trigger is False

    # Condition 2: Vehicle evades, tau_hat expands to 0.80s (between trigger and release)
    # Trigger check would not re-trigger, but release check still does not release
    re_trigger = tau_emergency_check(0.80, tau_trigger, confidence=0.90)
    assert re_trigger is False
    release_in_band = tau_emergency_release_check(0.80, tau_release, confidence=0.90)
    assert release_in_band is False

    # Condition 3: Clear distance restored, tau_hat expands to 1.30s (>= release threshold)
    release_cleared = tau_emergency_release_check(1.30, tau_release, confidence=0.90)
    assert release_cleared is True


def test_tau_emergency_parameter_validation() -> None:
    """Validate threshold inputs and bounds."""
    with pytest.raises(ValueError, match="tau_emergency_threshold must be strictly positive"):
        tau_emergency_check(0.3, tau_emergency_threshold=-0.1, confidence=0.8)

    with pytest.raises(ValueError, match="tau_emergency_threshold must be strictly positive"):
        tau_emergency_check(0.3, tau_emergency_threshold=0.0, confidence=0.8)

    with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
        tau_emergency_check(0.3, tau_emergency_threshold=0.5, confidence=0.8, min_confidence=1.5)

    with pytest.raises(ValueError, match="tau_release_threshold must be strictly positive"):
        tau_emergency_release_check(0.3, tau_release_threshold=-0.5, confidence=0.8)
