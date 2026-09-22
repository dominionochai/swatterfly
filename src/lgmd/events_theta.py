"""Event-driven time-to-contact estimation from raw event streams.

This module implements the generative radial optical-flow expansion model from
the event-based time-to-contact (TTC) literature:
    Schubert et al. 2025, IOP, "Bio-inspired event-based looming object detection
    for automotive collision avoidance"
and the conceptual family of event-based TTC mapping (e.g. ICCV 2023).

Method:
--------
1. Discretize the raw event stream into fixed-width time bins (e.g. 10 ms - 50 ms).
2. For each event, estimate local optical flow via nearest-neighbor displacement:
   search prior events of identical polarity within a spatio-temporal window,
   preferring events in the same angular sector (to measure radial motion rather
   than tangential sampling jumps around circle contours).
3. Fit the linear radial expansion model:
       flow(x, y) ≈ r(t) * (x - cx, y - cy) + bias
   via least squares with focus-of-expansion (FOE) fixed at image center (cx, cy).
4. Compute instantaneous time-to-contact:
       tau_hat(t) = 1.0 / r(t)
5. Convert r(t) to an angular-size-rate proxy:
       theta_dot_proxy(t) = r(t) * theta_proxy(t)
   using the pinhole camera relation implicit in trajectories.csv's
   projected_radius_px (theta_dot / theta = 1 / tau = r).

The module exposes:
- ``estimate_tau(events, scenario) -> float``: Drop-in replacement for
  ``sim.events.estimate_tau``, returning the initial TTC at stream start (t = 0).
- ``estimate_tau_series(events, scenario) -> list[ExpansionBin]``: Time-resolved
  per-bin expansion rates, TTC estimates, and angular rate proxies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_BIN_DT_S: float = 0.050
DEFAULT_FLOW_T_WINDOW_S: float = 1.000
DEFAULT_FLOW_S_WINDOW_PX: float = 3.5
DEFAULT_SECTOR_DEG: float = 45.0
MIN_EVENTS_PER_BIN: int = 3
EPS: float = 1e-6


@dataclass(frozen=True)
class EventData:
    """Sensor event with coordinates, timestamp, and polarity."""

    x_px: float
    y_px: float
    timestamp_s: float
    polarity: int


@dataclass(frozen=True)
class ExpansionBin:
    """Per-time-bin radial expansion and time-to-contact estimate."""

    timestamp_s: float
    r_1ps: float
    tau_hat_s: float
    theta_proxy_px: float
    theta_dot_proxy_pxps: float
    bias_x_pxps: float
    bias_y_pxps: float
    n_events: int
    flow_count: int
    fit_residual_pxps: float
    valid: bool


def _parse_events(events: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract float numpy arrays (x, y, t, p) from Event dataclasses or dicts/tuples."""
    n = len(events)
    if n == 0:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=int),
        )
    first = events[0]
    if hasattr(first, "x_px"):
        xs = np.fromiter((float(e.x_px) for e in events), dtype=float, count=n)
        ys = np.fromiter((float(e.y_px) for e in events), dtype=float, count=n)
        ts = np.fromiter((float(e.timestamp_s) for e in events), dtype=float, count=n)
        ps = np.fromiter((int(e.polarity) for e in events), dtype=int, count=n)
    elif isinstance(first, dict):
        xs = np.fromiter((float(e["x_px"]) for e in events), dtype=float, count=n)
        ys = np.fromiter((float(e["y_px"]) for e in events), dtype=float, count=n)
        ts = np.fromiter((float(e["timestamp_s"]) for e in events), dtype=float, count=n)
        ps = np.fromiter((int(e["polarity"]) for e in events), dtype=int, count=n)
    elif isinstance(first, (tuple, list)):
        xs = np.fromiter((float(e[1] if len(e) >= 4 and isinstance(e[0], float) and e[0] < 100 else e[0]) for e in events), dtype=float, count=n)
        ys = np.fromiter((float(e[2] if len(e) >= 4 and isinstance(e[0], float) and e[0] < 100 else e[1]) for e in events), dtype=float, count=n)
        ts = np.fromiter((float(e[0] if len(e) >= 4 and isinstance(e[0], float) and e[0] < 100 else e[2]) for e in events), dtype=float, count=n)
        ps = np.fromiter((int(e[3]) for e in events), dtype=int, count=n)
    else:
        raise TypeError(f"Unsupported event type: {type(first)}")
    return xs, ys, ts, ps


def _resolve_foe(scenario: Optional[Any]) -> Tuple[float, float]:
    """Resolve focus-of-expansion center (cx, cy) from scenario or default 1280x720."""
    if scenario is not None and hasattr(scenario, "sensor_width_px") and hasattr(scenario, "sensor_height_px"):
        cx = (float(scenario.sensor_width_px) - 1.0) / 2.0
        cy = (float(scenario.sensor_height_px) - 1.0) / 2.0
        return cx, cy
    return 639.5, 359.5


def estimate_local_optical_flows(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    ps: np.ndarray,
    cx: float,
    cy: float,
    *,
    temporal_window_s: float = DEFAULT_FLOW_T_WINDOW_S,
    spatial_window_px: float = DEFAULT_FLOW_S_WINDOW_PX,
    sector_deg: float = DEFAULT_SECTOR_DEG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-event optical flow vectors via nearest prior neighbor displacement.

    Filters prior candidate events to matching polarity and angular sector around
    FOE (cx, cy), selecting the nearest spatial neighbor within temporal_window_s.
    Returns:
        (valid_indices, flow_vx, flow_vy, pos_X, pos_Y)
    """
    n = len(xs)
    if n < 2:
        return (
            np.empty(0, dtype=int),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )

    angles = np.arctan2(ys - cy, xs - cx)
    max_d_angle = math.radians(sector_deg)

    valid_indices: List[int] = []
    flows_vx: List[float] = []
    flows_vy: List[float] = []
    pos_X: List[float] = []
    pos_Y: List[float] = []

    for i in range(1, n):
        t_i = ts[i]
        dts = t_i - ts[:i]
        t_mask = (dts > 1e-5) & (dts <= temporal_window_s)
        if not np.any(t_mask):
            continue

        p_mask = ps[:i] == ps[i]
        candidate_mask = t_mask & p_mask
        if not np.any(candidate_mask):
            continue

        # Angular difference relative to radial line from FOE
        cand_idx = np.where(candidate_mask)[0]
        angle_diffs = np.abs(np.arctan2(np.sin(angles[i] - angles[cand_idx]), np.cos(angles[i] - angles[cand_idx])))
        ang_mask = angle_diffs <= max_d_angle
        if not np.any(ang_mask):
            continue

        valid_cands = cand_idx[ang_mask]
        dxs = xs[i] - xs[valid_cands]
        dys = ys[i] - ys[valid_cands]
        dists = np.hypot(dxs, dys)

        dist_mask = (dists > 0.0) & (dists <= spatial_window_px)
        if not np.any(dist_mask):
            continue

        best_k = np.argmin(dists[dist_mask])
        chosen_cand = valid_cands[dist_mask][best_k]
        dt_k = t_i - ts[chosen_cand]

        vx = (xs[i] - xs[chosen_cand]) / dt_k
        vy = (ys[i] - ys[chosen_cand]) / dt_k

        valid_indices.append(i)
        flows_vx.append(vx)
        flows_vy.append(vy)
        pos_X.append(xs[i] - cx)
        pos_Y.append(ys[i] - cy)

    return (
        np.array(valid_indices, dtype=int),
        np.array(flows_vx, dtype=float),
        np.array(flows_vy, dtype=float),
        np.array(pos_X, dtype=float),
        np.array(pos_Y, dtype=float),
    )


def fit_radial_expansion(
    pos_X: np.ndarray,
    pos_Y: np.ndarray,
    flows_vx: np.ndarray,
    flows_vy: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Fit radial expansion flow(x, y) = r * (X, Y) + (bx, by) via least squares.

    Returns (r_1ps, bias_x, bias_y, residual_rms).
    """
    m = len(pos_X)
    if m < 2:
        return 0.0, 0.0, 0.0, math.nan

    design_A = np.zeros((2 * m, 3), dtype=float)
    design_A[:m, 0] = pos_X
    design_A[:m, 1] = 1.0
    design_A[m:, 0] = pos_Y
    design_A[m:, 2] = 1.0

    target_b = np.concatenate([flows_vx, flows_vy])

    sol, _, _, _ = np.linalg.lstsq(design_A, target_b, rcond=None)
    r_val = float(sol[0])
    bx_val = float(sol[1])
    by_val = float(sol[2])

    pred = design_A @ sol
    res_rms = float(np.sqrt(np.mean((pred - target_b) ** 2)))
    return r_val, bx_val, by_val, res_rms


def estimate_expansion_bins(
    events: Sequence[Any],
    *,
    scenario: Optional[Any] = None,
    bin_dt_s: float = DEFAULT_BIN_DT_S,
    temporal_window_s: float = DEFAULT_FLOW_T_WINDOW_S,
    spatial_window_px: float = DEFAULT_FLOW_S_WINDOW_PX,
    sector_deg: float = DEFAULT_SECTOR_DEG,
) -> List[ExpansionBin]:
    """Compute per-time-bin radial expansion rates and TTC estimates.

    Parameters
    ----------
    events : Sequence of sensor events
    scenario : Optional ApproachScenario for sensor dimensions
    bin_dt_s : Bin temporal resolution in seconds (default 50 ms)
    temporal_window_s : History window for optical flow pairing (default 1.0 s)
    spatial_window_px : Search radius for neighbor pairing (default 3.5 px)
    sector_deg : Angular sector half-angle for radial motion (default 45 deg)
    """
    xs, ys, ts, ps = _parse_events(events)
    if len(xs) < MIN_EVENTS_PER_BIN:
        return []

    # Sort events chronologically
    sort_idx = np.argsort(ts)
    xs = xs[sort_idx]
    ys = ys[sort_idx]
    ts = ts[sort_idx]
    ps = ps[sort_idx]

    cx, cy = _resolve_foe(scenario)

    val_idx, flows_vx, flows_vy, pos_X, pos_Y = estimate_local_optical_flows(
        xs,
        ys,
        ts,
        ps,
        cx,
        cy,
        temporal_window_s=temporal_window_s,
        spatial_window_px=spatial_window_px,
        sector_deg=sector_deg,
    )

    t_min = float(ts[0])
    t_max = float(ts[-1])
    bin_edges = np.arange(t_min, t_max + bin_dt_s + 1e-9, bin_dt_s)
    if len(bin_edges) < 2:
        bin_edges = np.array([t_min, t_max + bin_dt_s])

    flow_ts = ts[val_idx] if len(val_idx) > 0 else np.empty(0, dtype=float)

    result_bins: List[ExpansionBin] = []
    for b_idx in range(len(bin_edges) - 1):
        b_start = bin_edges[b_idx]
        b_end = bin_edges[b_idx + 1]
        t_center = 0.5 * (b_start + b_end)

        ev_mask = (ts >= b_start) & (ts < b_end)
        n_ev = int(np.sum(ev_mask))

        fl_mask = (flow_ts >= b_start) & (flow_ts < b_end)
        n_fl = int(np.sum(fl_mask))

        if n_ev > 0:
            radii_in_bin = np.hypot(xs[ev_mask] - cx, ys[ev_mask] - cy)
            theta_proxy = float(np.median(radii_in_bin))
        else:
            theta_proxy = 0.0

        if n_fl < MIN_EVENTS_PER_BIN:
            result_bins.append(
                ExpansionBin(
                    timestamp_s=t_center,
                    r_1ps=0.0,
                    tau_hat_s=math.inf,
                    theta_proxy_px=theta_proxy,
                    theta_dot_proxy_pxps=0.0,
                    bias_x_pxps=0.0,
                    bias_y_pxps=0.0,
                    n_events=n_ev,
                    flow_count=n_fl,
                    fit_residual_pxps=math.nan,
                    valid=False,
                )
            )
            continue

        b_X = pos_X[fl_mask]
        b_Y = pos_Y[fl_mask]
        b_Vx = flows_vx[fl_mask]
        b_Vy = flows_vy[fl_mask]

        r_val, bx, by, res_rms = fit_radial_expansion(b_X, b_Y, b_Vx, b_Vy)

        valid = r_val > EPS
        tau_hat = 1.0 / r_val if valid else math.inf
        theta_dot_proxy = r_val * theta_proxy if valid else 0.0

        result_bins.append(
            ExpansionBin(
                timestamp_s=t_center,
                r_1ps=r_val,
                tau_hat_s=tau_hat,
                theta_proxy_px=theta_proxy,
                theta_dot_proxy_pxps=theta_dot_proxy,
                bias_x_pxps=bx,
                bias_y_pxps=by,
                n_events=n_ev,
                flow_count=n_fl,
                fit_residual_pxps=res_rms,
                valid=valid,
            )
        )

    return result_bins


def estimate_tau_series(
    events: Sequence[Any],
    scenario: Optional[Any] = None,
    **kwargs: Any,
) -> List[ExpansionBin]:
    """Expose time-resolved expansion bins (alias for estimate_expansion_bins)."""
    return estimate_expansion_bins(events, scenario=scenario, **kwargs)


def estimate_tau(
    events: Sequence[Any],
    scenario: Optional[Any] = None,
    **kwargs: Any,
) -> float:
    """Estimate time-to-contact at stream start (t = 0) from raw event data.

    Drop-in alternative to ``sim.events.estimate_tau``. Fits optical-flow
    expansion rate r(t) in time bins, converts each valid bin to an initial
    time-to-contact projection tau_0 = tau_hat(t) + t, and aggregates via the
    median over valid bins.

    Parameters
    ----------
    events : Sequence of event objects (dataclass, dict, or tuple)
    scenario : Optional ApproachScenario providing sensor dimensions
    **kwargs : Optional tuning overrides (bin_dt_s, temporal_window_s, etc.)

    Returns
    -------
    float : Estimated initial time-to-contact in seconds.

    Raises
    ------
    ValueError : If fewer than 3 events or insufficient valid bins to form an estimate.
    """
    if len(events) < 3:
        raise ValueError("at least three events are required")

    bins = estimate_expansion_bins(events, scenario=scenario, **kwargs)
    valid_bins = [b for b in bins if b.valid and math.isfinite(b.tau_hat_s)]

    if len(valid_bins) < 2:
        raise ValueError("insufficient valid expansion bins to estimate tau")

    # For constant-speed approach: tau(t) = tau_0 - t  =>  tau_0 = tau_hat(t) + t
    tau_0_candidates = [b.tau_hat_s + b.timestamp_s for b in valid_bins]
    tau_0_est = float(np.median(tau_0_candidates))

    if tau_0_est <= 0.0 or not math.isfinite(tau_0_est):
        raise ValueError("non-positive tau estimate")

    return tau_0_est


__all__ = [
    "EventData",
    "ExpansionBin",
    "estimate_expansion_bins",
    "estimate_tau_series",
    "estimate_tau",
    "estimate_local_optical_flows",
    "fit_radial_expansion",
]
