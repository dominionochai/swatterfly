"""Tests for the Phase 3 Gabbiani canonical firing model (lgmd.network).

These tests verify the network detector's analytical trigger timing, its
internal hysteresis, and its interface compatibility with the engineering
scalar approximation detector so that ``scripts/compare_lgmd.py`` can call both
interchangeably.
"""

from __future__ import annotations

import inspect
import math

import pytest

from lgmd.network import NetworkLgmdDetector
from lgmd.scalar_eta import LoomingSample, ScalarEtaDetector


def _alpha(theta_threshold_deg: float) -> float:
    """alpha = 1 / tan(theta_thres / 2) from the Gabbiani firing expression."""
    return 1.0 / math.tan(math.radians(theta_threshold_deg / 2.0))


def _run(detector: NetworkLgmdDetector, samples: list[LoomingSample]) -> list[tuple[float, float]]:
    """Step a detector through a sample sequence; return (score, tau_hat)."""
    return [detector.update(sample) for sample in samples]


def _looming_signal(
    times: list[float],
    psi_break: float,
    psi_before: float,
    psi_after: float,
    theta0: float,
) -> list[LoomingSample]:
    """Piecewise-constant angular expansion psi with continuous theta.

    ``psi(t) = psi_before`` for ``t < psi_break`` and ``psi_after`` for
    ``t >= psi_break``; ``theta`` integrates psi so it stays continuous.
    """
    samples: list[LoomingSample] = []
    theta = theta0
    previous = 0.0
    for index, timestamp in enumerate(times):
        dt = timestamp - previous
        psi = psi_before if timestamp < psi_break else psi_after
        theta += psi * dt
        samples.append(LoomingSample(theta=theta, theta_dot=psi, timestamp=timestamp))
        previous = timestamp
    return samples


def test_hand_computed_firing_threshold_crossing() -> None:
    """Trigger time is analytically computable for a delayed step in psi.

    With theta constant until ``T`` and a step in angular expansion to
    ``psi_after`` at ``T``, the canonical firing score is zero until the delay
    window passes and then jumps to ``psi_after * exp(-alpha * theta)`` at
    ``T + delta``. The detector must cross ``eta_on`` on exactly that sample.
    """
    theta_threshold_deg = 40.0
    delta_s = 0.020  # 20 ms, within the required 15-35 ms range
    dt_s = 0.005
    break_t = 0.100
    theta0 = 1.0
    psi_after = 6.0
    times = [i * dt_s for i in range(60)]  # 0 .. 0.295 s
    samples = _looming_signal(times, break_t, 0.0, psi_after, theta0)

    alpha = _alpha(theta_threshold_deg)
    expected_crossing = break_t + delta_s  # 0.120 s
    # The profile's theta integration adds psi_after * dt on the break sample,
    # so the delayed theta at the crossing is theta0 + psi_after * dt_s.
    theta_delayed = theta0 + psi_after * dt_s
    expected_firing = psi_after * math.exp(-alpha * theta_delayed)

    eta_on = 0.3
    detector = NetworkLgmdDetector(
        theta_threshold_deg=theta_threshold_deg,
        delay_ms=delta_s * 1000.0,
        eta_on=eta_on,
        eta_off=0.24,
    )
    trigger_score = math.nan
    for sample in samples:
        score, _ = detector.update(sample)
        if detector.triggered:
            trigger_score = score

    assert detector.trigger_timestamp == pytest.approx(expected_crossing, abs=dt_s)
    assert trigger_score == pytest.approx(expected_firing, rel=1e-6)
    assert detector.peak_score == pytest.approx(expected_firing, rel=1e-6)


def test_hysteresis_triggers_and_releases_at_different_scores() -> None:
    """Trigger and release must occur at different firing scores and times.

    A slow expansion phase keeps the canonical firing below ``eta_on``; a
    fast expansion phase pushes it above ``eta_on`` (trigger), and the growing
    angular size suppresses the firing back down to ``eta_off`` (release).
    """
    theta_threshold_deg = 40.0
    delta_s = 0.020
    dt_s = 0.005
    break_t = 0.300
    theta0 = 0.2
    psi_before = 0.5
    psi_after = 8.0
    times = [i * dt_s for i in range(160)]  # 0 .. 0.795 s
    samples = _looming_signal(times, break_t, psi_before, psi_after, theta0)

    alpha = _alpha(theta_threshold_deg)
    eta_on = 0.6
    eta_off = 0.48

    detector = NetworkLgmdDetector(
        theta_threshold_deg=theta_threshold_deg,
        delay_ms=delta_s * 1000.0,
        eta_on=eta_on,
        eta_off=eta_off,
    )

    trigger_time: float | None = None
    trigger_score = math.nan
    release_time: float | None = None
    release_score = math.nan
    scores: list[float] = []
    for sample in samples:
        score, _ = detector.update(sample)
        scores.append(score)
        if detector.triggered:
            trigger_time = sample.timestamp
            trigger_score = score
        if detector.released:
            release_time = sample.timestamp
            release_score = score

    assert trigger_time is not None, "detector never triggered"
    assert release_time is not None, "detector never released"
    assert trigger_time < release_time, "trigger must precede release"
    assert trigger_score >= eta_on
    assert release_score <= eta_off
    assert eta_on != eta_off
    assert trigger_score != pytest.approx(release_score, rel=1e-3)

    # Analytic release: firing = psi_after * exp(-alpha * theta(t - delta))
    # falls to eta_off when theta(t - delta) = ln(psi_after / eta_off) / alpha.
    theta_at_release = math.log(psi_after / eta_off) / alpha
    expected_release = break_t + delta_s + (theta_at_release - theta0 - psi_before * break_t) / psi_after
    assert release_time == pytest.approx(expected_release, abs=2 * dt_s)


def test_network_update_interface_matches_scalar_detector() -> None:
    """Both detectors accept the same LoomingSample call pattern.

    ``scripts/compare_lgmd.py`` calls ``detector.update(sample)`` and reads the
    two-tuple ``(score, tau_hat)``; both detectors must honor that exact
    interface with identical method signatures.
    """
    scalar = ScalarEtaDetector()
    network = NetworkLgmdDetector()
    sample = LoomingSample(theta=3.0, theta_dot=0.4, timestamp=0.0)

    scalar_params = list(inspect.signature(ScalarEtaDetector.update).parameters)
    network_params = list(inspect.signature(NetworkLgmdDetector.update).parameters)
    assert network_params == scalar_params == ["self", "sample"]

    scalar_result = scalar.update(sample)
    network_result = network.update(sample)
    for result in (scalar_result, network_result):
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(value, float) for value in result)

    network_tau_hat = network_result[1]
    assert math.isfinite(network_tau_hat)
    assert network_tau_hat == pytest.approx(3.0 / 0.4)


def test_network_validates_gabbiani_parameter_ranges() -> None:
    """Threshold and delay must stay inside the cited Gabbiani ranges."""
    with pytest.raises(ValueError):
        NetworkLgmdDetector(theta_threshold_deg=10.0)
    with pytest.raises(ValueError):
        NetworkLgmdDetector(theta_threshold_deg=45.0)
    with pytest.raises(ValueError):
        NetworkLgmdDetector(delay_ms=10.0)
    with pytest.raises(ValueError):
        NetworkLgmdDetector(delay_ms=40.0)
    with pytest.raises(ValueError):
        NetworkLgmdDetector(eta_on=0.2, eta_off=0.2)
    with pytest.raises(ValueError):
        NetworkLgmdDetector(eta_on=0.2, eta_off=0.5)