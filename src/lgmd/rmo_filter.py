"""Nonlinear Radial Motion Opponency (RMO) filtering for event-driven looming.

This module implements the nonlinear RMO filtering stage from:
    Schubert, Knight, Philippides, Nowotny 2025,
    Neuromorph. Comput. Eng. 5, 024016,
    "Bio-inspired event-based looming object detection for automotive collision avoidance"
    Section 3.1, "Nonlinear RMO-selectivity reduces false responses"
    DOI: 10.1088/2634-4386/add0da (Open Access, CC BY 4.0).

Theoretical Background & Equations:
-----------------------------------
In raw event streams, isolated noise events or non-looming translations
(e.g. ego-motion or lateral target motion) produce high instantaneous flow
vectors in single localized regions. When fed directly into an expansion
threshold gate, this causes premature false-trigger rates (>90%).

The paper resolves this through nonlinear radial motion opponency:
1. Directional pooling (Paper Eq. 7):
   Local optical flow estimates are pooled into opposing spatial subfields
   around the focus-of-expansion (FOE):
       H_L(t) = mean_{i in Left} max(0, -v_{x, i})   (leftward expansion)
       H_R(t) = mean_{i in Right} max(0, +v_{x, i})  (rightward expansion)
   Optional vertical opponency (four-quadrant mode):
       H_T(t) = mean_{i in Top} max(0, -v_{y, i})    (upward expansion)
       H_B(t) = mean_{i in Bottom} max(0, +v_{y, i}) (downward expansion)

2. Nonlinear temporal alignment & opponency (Paper Eq. 12):
   True looming requires opposing subfields to be simultaneously positive and
   temporally aligned:
       S_RMO(t) = sqrt( max(0, H_L(t)) * max(0, H_R(t)) )
   For four-quadrant opponency:
       S_RMO,4(t) = ( H_L(t) * H_R(t) * H_T(t) * H_B(t) )^(1/4)
   Uniform translation (e.g. v_x > 0 everywhere) yields H_L(t) = 0, so
   S_RMO(t) = 0. Single-region noise clusters similarly produce zero.

3. Gated looming output (Paper Eq. 13):
   The expansion score passes to downstream threshold detectors only when
   opposing subfields jointly exceed threshold:
       G(t) = r(t) * Theta( S_RMO(t) - theta_RMO )
   where Theta(.) is the binary step gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .events_theta import (
    ExpansionBin,
    _parse_events,
    _resolve_foe,
    estimate_local_optical_flows,
)

DEFAULT_RMO_THRESHOLD: float = 0.5
DEFAULT_MARGIN_PX: float = 0.2
DEFAULT_BIN_DT_S: float = 0.050

RMOMode = Literal["two_region", "four_quadrant"]


@dataclass(frozen=True)
class RMOActivation:
    """Activation strengths for opposing subfields in a time bin."""

    timestamp_s: float
    l_act: float
    r_act: float
    top_act: float
    bottom_act: float
    rmo_score: float
    gate_open: bool
    gated_r_1ps: float
    gated_theta_dot_proxy_pxps: float


def compute_rmo_score(
    pos_X: np.ndarray,
    pos_Y: np.ndarray,
    flows_vx: np.ndarray,
    flows_vy: np.ndarray,
    *,
    mode: RMOMode = "two_region",
    margin_px: float = DEFAULT_MARGIN_PX,
) -> Tuple[float, float, float, float, float]:
    """Compute directional opponent activations and nonlinear RMO score.

    Parameters
    ----------
    pos_X, pos_Y : Relative coordinates (x - cx, y - cy) from FOE.
    flows_vx, flows_vy : Flow vectors in px/s.
    mode : "two_region" (Left vs Right) or "four_quadrant" (L, R, Top, Bottom).
    margin_px : Deadband around optical axis to prevent near-center ambiguity.

    Returns
    -------
    (l_act, r_act, top_act, bottom_act, rmo_score)
    """
    if len(pos_X) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    left_mask = pos_X < -margin_px
    right_mask = pos_X > margin_px

    l_act = float(np.mean(np.maximum(0.0, -flows_vx[left_mask]))) if np.any(left_mask) else 0.0
    r_act = float(np.mean(np.maximum(0.0, flows_vx[right_mask]))) if np.any(right_mask) else 0.0

    top_mask = pos_Y < -margin_px
    bottom_mask = pos_Y > margin_px

    top_act = float(np.mean(np.maximum(0.0, -flows_vy[top_mask]))) if np.any(top_mask) else 0.0
    bottom_act = float(np.mean(np.maximum(0.0, flows_vy[bottom_mask]))) if np.any(bottom_mask) else 0.0

    if mode == "four_quadrant":
        prod = max(0.0, l_act) * max(0.0, r_act) * max(0.0, top_act) * max(0.0, bottom_act)
        rmo_score = float(prod ** 0.25)
    else:  # two_region (Schubert et al. 2025 Eq. 12)
        prod = max(0.0, l_act) * max(0.0, r_act)
        rmo_score = float(math.sqrt(prod))

    return l_act, r_act, top_act, bottom_act, rmo_score


def filter_expansion_bins_rmo(
    bins: Sequence[ExpansionBin],
    events: Sequence[Any],
    scenario: Optional[Any] = None,
    *,
    rmo_threshold: float = DEFAULT_RMO_THRESHOLD,
    mode: RMOMode = "two_region",
    bin_dt_s: float = DEFAULT_BIN_DT_S,
    margin_px: float = DEFAULT_MARGIN_PX,
) -> Tuple[List[ExpansionBin], List[RMOActivation]]:
    """Apply nonlinear RMO gating to a sequence of ExpansionBins.

    Sits between the raw expansion fitting in `events_theta.py` and downstream
    threshold gates. For bins where opposing subfield activation fails the joint
    RMO threshold (Paper Eq. 13), expansion and angular-rate proxy are gated to 0.

    Parameters
    ----------
    bins : Raw expansion bins from `events_theta.estimate_expansion_bins`.
    events : Raw event stream.
    scenario : Optional ApproachScenario for sensor center.
    rmo_threshold : Minimum joint RMO score to open the looming gate (theta_RMO).
    mode : "two_region" or "four_quadrant".
    bin_dt_s : Bin resolution in seconds.
    margin_px : Deadband around FOE.

    Returns
    -------
    (filtered_bins, activations)
    """
    if not bins or len(events) == 0:
        return list(bins), []

    xs, ys, ts, ps = _parse_events(events)
    sort_idx = np.argsort(ts)
    xs, ys, ts, ps = xs[sort_idx], ys[sort_idx], ts[sort_idx], ps[sort_idx]
    cx, cy = _resolve_foe(scenario)

    val_idx, flows_vx, flows_vy, pos_X, pos_Y = estimate_local_optical_flows(
        xs, ys, ts, ps, cx, cy
    )
    flow_ts = ts[val_idx] if len(val_idx) > 0 else np.empty(0, dtype=float)

    filtered_bins: List[ExpansionBin] = []
    activations: List[RMOActivation] = []

    for b in bins:
        t_center = b.timestamp_s
        fl_mask = (flow_ts >= t_center - 0.5 * bin_dt_s) & (flow_ts < t_center + 0.5 * bin_dt_s)

        if np.sum(fl_mask) < 2:
            l_act, r_act, top_act, bottom_act, rmo_score = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            b_X = pos_X[fl_mask]
            b_Y = pos_Y[fl_mask]
            b_Vx = flows_vx[fl_mask]
            b_Vy = flows_vy[fl_mask]
            l_act, r_act, top_act, bottom_act, rmo_score = compute_rmo_score(
                b_X, b_Y, b_Vx, b_Vy, mode=mode, margin_px=margin_px
            )

        gate_open = b.valid and (rmo_score >= rmo_threshold)

        gated_r = b.r_1ps if gate_open else 0.0
        gated_theta_dot = b.theta_dot_proxy_pxps if gate_open else 0.0
        gated_tau_hat = b.tau_hat_s if gate_open else math.inf

        filtered_bins.append(
            ExpansionBin(
                timestamp_s=b.timestamp_s,
                r_1ps=gated_r,
                tau_hat_s=gated_tau_hat,
                theta_proxy_px=b.theta_proxy_px,
                theta_dot_proxy_pxps=gated_theta_dot,
                bias_x_pxps=b.bias_x_pxps,
                bias_y_pxps=b.bias_y_pxps,
                n_events=b.n_events,
                flow_count=b.flow_count,
                fit_residual_pxps=b.fit_residual_pxps,
                valid=gate_open,
            )
        )

        activations.append(
            RMOActivation(
                timestamp_s=t_center,
                l_act=l_act,
                r_act=r_act,
                top_act=top_act,
                bottom_act=bottom_act,
                rmo_score=rmo_score,
                gate_open=gate_open,
                gated_r_1ps=gated_r,
                gated_theta_dot_proxy_pxps=gated_theta_dot,
            )
        )

    return filtered_bins, activations


class RMOFilter:
    """Stateful or batch nonlinear Radial Motion Opponency filter."""

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_RMO_THRESHOLD,
        mode: RMOMode = "two_region",
        bin_dt_s: float = DEFAULT_BIN_DT_S,
        margin_px: float = DEFAULT_MARGIN_PX,
    ) -> None:
        self.threshold = float(threshold)
        self.mode = mode
        self.bin_dt_s = float(bin_dt_s)
        self.margin_px = float(margin_px)

    def filter_bins(
        self,
        bins: Sequence[ExpansionBin],
        events: Sequence[Any],
        scenario: Optional[Any] = None,
    ) -> Tuple[List[ExpansionBin], List[RMOActivation]]:
        """Filter raw expansion bins via the RMO gate."""
        return filter_expansion_bins_rmo(
            bins,
            events,
            scenario=scenario,
            rmo_threshold=self.threshold,
            mode=self.mode,
            bin_dt_s=self.bin_dt_s,
            margin_px=self.margin_px,
        )


__all__ = [
    "RMOActivation",
    "RMOMode",
    "compute_rmo_score",
    "filter_expansion_bins_rmo",
    "RMOFilter",
]
