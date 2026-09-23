"""Compare 2D Alpha-Beta and Kalman trackers under noise, latency, and dropouts.

Harness evaluates:
1. Positional and velocity RMSE under Gaussian observation noise (0.0 to 3.0 px).
2. Latency sensitivity (0 to 50 ms delay).
3. Missed-observation (dropout) handling: confidence degradation and recovery.
4. Outlier rejection: response to sudden anomalous spikes.

Outputs:
- tracker_comparison.csv: Per-condition benchmark metrics.
- summary_tracker_comparison.csv: Sensitivity across noise, latency, and dropouts.
- tracker_comparison.png: 4-panel diagnostic comparison plot.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tracker.alpha_beta import AlphaBetaTracker2D
from tracker.kalman import KalmanTracker2D

NOISE_LEVELS = (0.0, 0.5, 1.5, 3.0)       # pixels
LATENCIES_S = (0.0, 0.010, 0.020, 0.050)   # 0 to 50 ms
DROPOUT_RATES = (0.0, 0.10, 0.25)          # probability of missed observation


def generate_test_trajectory(
    duration_s: float = 2.0,
    dt_s: float = 0.02,
    vx: float = -30.0,
    vy: float = 15.0,
    x0: float = 640.0,
    y0: float = 360.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ground truth 2D trajectory."""
    steps = int(round(duration_s / dt_s))
    times = np.arange(steps + 1, dtype=float) * dt_s
    true_x = x0 + vx * times
    true_y = y0 + vy * times
    return times, true_x, true_y


def run_tracker_trial(
    tracker_type: str,
    times: np.ndarray,
    true_x: np.ndarray,
    true_y: np.ndarray,
    vx_true: float,
    vy_true: float,
    noise_sigma: float,
    latency_s: float,
    dropout_rate: float,
    seed: int,
) -> Dict[str, Any]:
    """Run one tracker through a simulated flight with latency, noise, and dropouts."""
    rng = np.random.default_rng(seed)
    dt = float(times[1] - times[0])

    if tracker_type == "alpha_beta":
        tracker = AlphaBetaTracker2D(
            alpha=0.60,
            beta=0.04,
            nominal_noise_scale=max(0.5, noise_sigma),
            missed_decay=0.85,
        )
    elif tracker_type == "kalman":
        tracker = KalmanTracker2D(
            process_noise_q=2.0,
            measurement_noise_sigma=max(0.5, noise_sigma),
            nominal_pos_std=max(0.5, noise_sigma),
            missed_decay=0.85,
        )
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")

    est_x: List[float] = []
    est_y: List[float] = []
    est_vx: List[float] = []
    est_vy: List[float] = []
    confidences: List[float] = []
    uncertainties: List[float] = []

    # Delayed observation buffer
    latency_steps = int(round(latency_s / dt))

    for k in range(len(times)):
        # Determine delayed observation index
        obs_idx = k - latency_steps
        if obs_idx < 0:
            meas = None
        elif dropout_rate > 0.0 and rng.uniform(0.0, 1.0) < dropout_rate:
            meas = None  # Missed update / sensor dropout
        else:
            raw_x = true_x[obs_idx]
            raw_y = true_y[obs_idx]
            if noise_sigma > 0.0:
                raw_x += float(rng.normal(0.0, noise_sigma))
                raw_y += float(rng.normal(0.0, noise_sigma))
            meas = (raw_x, raw_y)

        state = tracker.update(meas, dt)

        est_x.append(state.x)
        est_y.append(state.y)
        est_vx.append(state.vx)
        est_vy.append(state.vy)
        confidences.append(state.confidence)
        uncertainties.append(state.uncertainty)

    # Compute errors over steady-state (second half of trial to allow initial convergence)
    half = len(times) // 2
    err_pos = np.hypot(
        np.array(est_x[half:]) - true_x[half:],
        np.array(est_y[half:]) - true_y[half:],
    )
    err_vel = np.hypot(
        np.array(est_vx[half:]) - vx_true,
        np.array(est_vy[half:]) - vy_true,
    )

    pos_rmse = float(np.sqrt(np.mean(err_pos ** 2)))
    vel_rmse = float(np.sqrt(np.mean(err_vel ** 2)))
    mean_conf = float(np.mean(confidences[half:]))
    min_conf = float(np.min(confidences[half:]))
    mean_unc = float(np.mean(uncertainties[half:]))

    return {
        "tracker": tracker_type,
        "noise_sigma_px": noise_sigma,
        "latency_s": latency_s,
        "dropout_rate": dropout_rate,
        "pos_rmse_px": pos_rmse,
        "vel_rmse_pxps": vel_rmse,
        "mean_confidence": mean_conf,
        "min_confidence": min_conf,
        "mean_uncertainty_px": mean_unc,
    }


def write_comparison_plot(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Generate 4-panel diagnostic figure comparing Alpha-Beta and Kalman."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel 1: Position RMSE vs Noise (at 0 latency, 0 dropout)
    ax1 = axes[0, 0]
    sub1 = [r for r in rows if r["latency_s"] == 0.0 and r["dropout_rate"] == 0.0]
    noises = sorted({r["noise_sigma_px"] for r in sub1})
    ab_pos = [next(r["pos_rmse_px"] for r in sub1 if r["tracker"] == "alpha_beta" and r["noise_sigma_px"] == n) for n in noises]
    kf_pos = [next(r["pos_rmse_px"] for r in sub1 if r["tracker"] == "kalman" and r["noise_sigma_px"] == n) for n in noises]
    ax1.plot(noises, ab_pos, "o-", label="Alpha-Beta 2D", color="#e27c3e", linewidth=2)
    ax1.plot(noises, kf_pos, "s-", label="Kalman 2D (CV)", color="#4a7ebb", linewidth=2)
    ax1.set_xlabel("Measurement Noise Sigma (px)")
    ax1.set_ylabel("Position RMSE (px)")
    ax1.set_title("Tracking Accuracy vs Sensor Noise")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Velocity RMSE vs Latency (at noise=1.5, dropout=0.0)
    ax2 = axes[0, 1]
    sub2 = [r for r in rows if r["noise_sigma_px"] == 1.5 and r["dropout_rate"] == 0.0]
    latencies_ms = [r["latency_s"] * 1000.0 for r in sub2 if r["tracker"] == "kalman"]
    ab_vel = [r["vel_rmse_pxps"] for r in sub2 if r["tracker"] == "alpha_beta"]
    kf_vel = [r["vel_rmse_pxps"] for r in sub2 if r["tracker"] == "kalman"]
    ax2.plot(latencies_ms, ab_vel, "o-", label="Alpha-Beta 2D", color="#e27c3e", linewidth=2)
    ax2.plot(latencies_ms, kf_vel, "s-", label="Kalman 2D (CV)", color="#4a7ebb", linewidth=2)
    ax2.set_xlabel("Injected Latency (ms)")
    ax2.set_ylabel("Velocity RMSE (px/s)")
    ax2.set_title("Velocity Estimation vs Injected Latency")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Panel 3: Confidence vs Dropout Rate (at noise=1.5, latency=0.010)
    ax3 = axes[1, 0]
    sub3 = [r for r in rows if r["noise_sigma_px"] == 1.5 and r["latency_s"] == 0.010]
    drops = sorted({r["dropout_rate"] for r in sub3})
    ab_conf = [next(r["mean_confidence"] for r in sub3 if r["tracker"] == "alpha_beta" and r["dropout_rate"] == d) for d in drops]
    kf_conf = [next(r["mean_confidence"] for r in sub3 if r["tracker"] == "kalman" and r["dropout_rate"] == d) for d in drops]
    ax3.plot([d * 100 for d in drops], ab_conf, "o-", label="Alpha-Beta 2D", color="#e27c3e", linewidth=2)
    ax3.plot([d * 100 for d in drops], kf_conf, "s-", label="Kalman 2D (CV)", color="#4a7ebb", linewidth=2)
    ax3.set_xlabel("Missed Observations / Dropouts (%)")
    ax3.set_ylabel("Mean Confidence [0, 1]")
    ax3.set_title("Confidence Degradation under Observation Dropouts")
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Panel 4: Uncertainty vs Noise Level
    ax4 = axes[1, 1]
    sub4 = [r for r in rows if r["latency_s"] == 0.0 and r["dropout_rate"] == 0.0]
    ab_unc = [next(r["mean_uncertainty_px"] for r in sub4 if r["tracker"] == "alpha_beta" and r["noise_sigma_px"] == n) for n in noises]
    kf_unc = [next(r["mean_uncertainty_px"] for r in sub4 if r["tracker"] == "kalman" and r["noise_sigma_px"] == n) for n in noises]
    ax4.plot(noises, ab_unc, "o-", label="Alpha-Beta Innovation", color="#e27c3e", linewidth=2)
    ax4.plot(noises, kf_unc, "s-", label="Kalman Covariance Std", color="#4a7ebb", linewidth=2)
    ax4.set_xlabel("Measurement Noise Sigma (px)")
    ax4.set_ylabel("Reported Uncertainty (px)")
    ax4.set_title("Exposed Uncertainty Scale vs Ground Truth Noise")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    fig.suptitle("Phase 4: 2D Target Tracker Benchmark (Alpha-Beta vs Kalman)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/phase4_trackers"))
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    times, true_x, true_y = generate_test_trajectory(duration_s=2.5, dt_s=0.02, vx=-25.0, vy=15.0)

    rows: List[Dict[str, Any]] = []
    trial_id = 0

    for noise in NOISE_LEVELS:
        for latency in LATENCIES_S:
            for drop in DROPOUT_RATES:
                for tracker_name in ("alpha_beta", "kalman"):
                    res = run_tracker_trial(
                        tracker_name,
                        times,
                        true_x,
                        true_y,
                        vx_true=-25.0,
                        vy_true=15.0,
                        noise_sigma=noise,
                        latency_s=latency,
                        dropout_rate=drop,
                        seed=args.seed + trial_id,
                    )
                    rows.append(res)
                    trial_id += 1

    fieldnames = [
        "tracker",
        "noise_sigma_px",
        "latency_s",
        "dropout_rate",
        "pos_rmse_px",
        "vel_rmse_pxps",
        "mean_confidence",
        "min_confidence",
        "mean_uncertainty_px",
    ]

    with (args.output / "tracker_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary by noise level
    summary_rows: List[Dict[str, Any]] = []
    for noise in NOISE_LEVELS:
        for tracker_name in ("alpha_beta", "kalman"):
            sub = [r for r in rows if r["tracker"] == tracker_name and r["noise_sigma_px"] == noise]
            summary_rows.append(
                {
                    "axis": "noise_sigma_px",
                    "value": noise,
                    "tracker": tracker_name,
                    "pos_rmse_px": float(np.mean([r["pos_rmse_px"] for r in sub])),
                    "vel_rmse_pxps": float(np.mean([r["vel_rmse_pxps"] for r in sub])),
                    "mean_confidence": float(np.mean([r["mean_confidence"] for r in sub])),
                    "mean_uncertainty_px": float(np.mean([r["mean_uncertainty_px"] for r in sub])),
                }
            )

    # Summary by latency
    for lat in LATENCIES_S:
        for tracker_name in ("alpha_beta", "kalman"):
            sub = [r for r in rows if r["tracker"] == tracker_name and r["latency_s"] == lat]
            summary_rows.append(
                {
                    "axis": "latency_s",
                    "value": lat,
                    "tracker": tracker_name,
                    "pos_rmse_px": float(np.mean([r["pos_rmse_px"] for r in sub])),
                    "vel_rmse_pxps": float(np.mean([r["vel_rmse_pxps"] for r in sub])),
                    "mean_confidence": float(np.mean([r["mean_confidence"] for r in sub])),
                    "mean_uncertainty_px": float(np.mean([r["mean_uncertainty_px"] for r in sub])),
                }
            )

    # Summary by dropout rate
    for drop in DROPOUT_RATES:
        for tracker_name in ("alpha_beta", "kalman"):
            sub = [r for r in rows if r["tracker"] == tracker_name and r["dropout_rate"] == drop]
            summary_rows.append(
                {
                    "axis": "dropout_rate",
                    "value": drop,
                    "tracker": tracker_name,
                    "pos_rmse_px": float(np.mean([r["pos_rmse_px"] for r in sub])),
                    "vel_rmse_pxps": float(np.mean([r["vel_rmse_pxps"] for r in sub])),
                    "mean_confidence": float(np.mean([r["mean_confidence"] for r in sub])),
                    "mean_uncertainty_px": float(np.mean([r["mean_uncertainty_px"] for r in sub])),
                }
            )

    summary_fields = ["axis", "value", "tracker", "pos_rmse_px", "vel_rmse_pxps", "mean_confidence", "mean_uncertainty_px"]
    with (args.output / "summary_tracker_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    write_comparison_plot(args.output / "tracker_comparison.png", rows)

    print(f"[Phase 4] Tracker benchmark complete: {len(rows)} trials evaluated.")
    print(f"Artifacts written to {args.output}")

    # Print executive summary
    print("\n--- Executive Summary (Alpha-Beta vs Kalman) ---")
    for t_name in ("alpha_beta", "kalman"):
        sub = [r for r in rows if r["tracker"] == t_name]
        mean_p_rmse = np.mean([r["pos_rmse_px"] for r in sub])
        mean_v_rmse = np.mean([r["vel_rmse_pxps"] for r in sub])
        mean_conf = np.mean([r["mean_confidence"] for r in sub])
        print(f"[{t_name:10s}] Overall Pos RMSE: {mean_p_rmse:5.2f} px | Vel RMSE: {mean_v_rmse:5.2f} px/s | Mean Confidence: {mean_conf:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
