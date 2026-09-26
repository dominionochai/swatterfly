"""Compare Guidance Laws (Pure Pursuit, Constant Bearing, PN, Blend) across Trackers.

Simulates closed-loop intercept engagements against a jinking target:
- Target executes an evasive maneuver (lateral acceleration steps) using
  ``src/sim/point_mass.py``.
- Observations are corrupted with measurement noise and processed through
  ``AlphaBetaTracker2D`` and ``KalmanTracker2D``.
- Guidance laws consume tracker state (position, velocity, confidence, covariance).
- Evaluates miss distance, control effort, saturation events, and confidence holds.

Outputs:
- guidance_comparison.csv: Quantitative metrics for all 8 combinations.
- guidance_comparison.png: 2D intercept trajectories and acceleration command traces.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from guidance.pn_lead import (
    GuidanceCommand,
    Vector2,
    blend_guidance,
    constant_bearing,
    proportional_navigation,
    pure_pursuit,
)
from sim.point_mass import PointMassState, step_point_mass
from tracker.alpha_beta import AlphaBetaTracker2D
from tracker.kalman import KalmanTracker2D


def generate_jinking_target_trajectory(
    duration_s: float = 3.0,
    dt_s: float = 0.02,
) -> List[PointMassState]:
    """Generate ground truth jinking target trajectory using step_point_mass.

    Profile:
    - Starts at (25.0, 12.0) m, vx = -8.0 m/s, vy = 0.0 m/s
    - t in [0.0, 0.8)s: constant velocity approach
    - t in [0.8, 1.4)s: evasive jink left (ay = +6.0 m/s^2)
    - t in [1.4, 2.0)s: evasive jink right (ay = -6.0 m/s^2)
    - t in [2.0, 3.0)s: constant velocity recovery
    """
    steps = int(round(duration_s / dt_s))
    current = PointMassState(x=25.0, y=12.0, vx=-8.0, vy=0.0)
    states = [current]

    for step in range(steps):
        t = step * dt_s
        if 0.8 <= t < 1.4:
            ay = 6.0
        elif 1.4 <= t < 2.0:
            ay = -6.0
        else:
            ay = 0.0
        ax = 0.0

        current = step_point_mass(current, ax=ax, ay=ay, dt=dt_s)
        states.append(current)

    return states


def run_engagement(
    tracker_name: str,
    guidance_name: str,
    target_states: List[PointMassState],
    dt_s: float = 0.02,
    noise_sigma: float = 0.20,
    seed: int = 42,
    pursuer_speed: float = 14.0,
    max_accel_mps2: float = 25.0,
    max_turn_rate_radps: float = 3.0,
    min_confidence: float = 0.40,
) -> Dict[str, Any]:
    """Simulate a full closed-loop intercept engagement."""
    rng = np.random.default_rng(seed)

    # Initialize tracker
    if tracker_name == "alpha_beta":
        tracker = AlphaBetaTracker2D(
            alpha=0.60,
            beta=0.04,
            nominal_noise_scale=max(0.2, noise_sigma),
            missed_decay=0.85,
        )
    elif tracker_name == "kalman":
        tracker = KalmanTracker2D(
            process_noise_q=2.0,
            measurement_noise_sigma=max(0.2, noise_sigma),
            nominal_pos_std=max(0.2, noise_sigma),
            missed_decay=0.85,
        )
    else:
        raise ValueError(f"Unknown tracker: {tracker_name}")

    # Pursuer initial state: at origin (0, 0), pointed toward initial target position
    init_t = target_states[0]
    init_heading = math.atan2(init_t.y, init_t.x)
    px, py = 0.0, 0.0
    heading = init_heading

    pursuer_x: List[float] = [px]
    pursuer_y: List[float] = [py]
    target_x: List[float] = [init_t.x]
    target_y: List[float] = [init_t.y]

    commands_raw: List[float] = []
    commands_sat: List[float] = []
    separations: List[float] = [math.hypot(init_t.x - px, init_t.y - py)]
    saturations: List[bool] = []
    holds: List[bool] = []
    confidences: List[float] = []

    for k in range(len(target_states) - 1):
        true_target = target_states[k]

        # 1. Noisy sensor observation
        meas_x = true_target.x + float(rng.normal(0.0, noise_sigma))
        meas_y = true_target.y + float(rng.normal(0.0, noise_sigma))

        # 2. Update tracker
        tr_state = tracker.update((meas_x, meas_y), dt=dt_s)

        # 3. Form relative guidance vectors
        rel_pos = Vector2(tr_state.x - px, tr_state.y - py)
        # Pursuer velocity vector
        vp = Vector2(pursuer_speed * math.cos(heading), pursuer_speed * math.sin(heading))
        vt_est = Vector2(tr_state.vx, tr_state.vy)
        rel_vel = vt_est - vp

        # 4. Evaluate guidance law
        if guidance_name == "pure_pursuit":
            cmd = pure_pursuit(
                rel_pos,
                rel_vel,
                pursuer_speed=pursuer_speed,
                pursuer_heading=heading,
                tau_pursuit=0.5,
                max_accel_mps2=max_accel_mps2,
                max_turn_rate_radps=max_turn_rate_radps,
                confidence=tr_state.confidence,
                min_confidence=min_confidence,
                covariance=tr_state.covariance,
            )
        elif guidance_name == "constant_bearing":
            cmd = constant_bearing(
                rel_pos,
                rel_vel,
                pursuer_speed=pursuer_speed,
                bearing_gain=3.5,
                max_accel_mps2=max_accel_mps2,
                max_turn_rate_radps=max_turn_rate_radps,
                confidence=tr_state.confidence,
                min_confidence=min_confidence,
                covariance=tr_state.covariance,
            )
        elif guidance_name == "proportional_navigation":
            cmd = proportional_navigation(
                rel_pos,
                rel_vel,
                pursuer_speed=pursuer_speed,
                navigation_constant=3.5,
                max_accel_mps2=max_accel_mps2,
                max_turn_rate_radps=max_turn_rate_radps,
                confidence=tr_state.confidence,
                min_confidence=min_confidence,
                covariance=tr_state.covariance,
            )
        elif guidance_name == "blend_guidance":
            cmd = blend_guidance(
                rel_pos,
                rel_vel,
                pursuer_speed=pursuer_speed,
                pursuer_heading=heading,
                target_velocity=vt_est,
                navigation_constant=3.5,
                lead_horizon_s=0.4,
                lead_weight=0.35,
                max_accel_mps2=max_accel_mps2,
                max_turn_rate_radps=max_turn_rate_radps,
                confidence=tr_state.confidence,
                min_confidence=min_confidence,
                covariance=tr_state.covariance,
            )
        else:
            raise ValueError(f"Unknown guidance law: {guidance_name}")

        commands_raw.append(cmd.raw_command)
        commands_sat.append(cmd.saturated_command)
        saturations.append(cmd.is_saturated)
        holds.append(cmd.is_holding)
        confidences.append(tr_state.confidence)

        # 5. Kinematic integration of pursuer
        # a_cmd = saturated lateral acceleration
        turn_rate = cmd.turn_rate_radps
        heading += turn_rate * dt_s
        px += pursuer_speed * math.cos(heading) * dt_s
        py += pursuer_speed * math.sin(heading) * dt_s

        pursuer_x.append(px)
        pursuer_y.append(py)
        next_target = target_states[k + 1]
        target_x.append(next_target.x)
        target_y.append(next_target.y)
        sep = math.hypot(next_target.x - px, next_target.y - py)
        separations.append(sep)

    min_miss = float(np.min(separations))
    final_miss = float(separations[-1])
    control_effort = float(np.sum(np.abs(commands_sat)) * dt_s)
    sat_count = int(np.sum(saturations))
    hold_count = int(np.sum(holds))
    mean_conf = float(np.mean(confidences))

    return {
        "tracker": tracker_name,
        "guidance": guidance_name,
        "min_miss_distance_m": min_miss,
        "final_miss_distance_m": final_miss,
        "control_effort_mps": control_effort,
        "saturation_count": sat_count,
        "saturation_pct": 100.0 * sat_count / len(commands_sat),
        "hold_count": hold_count,
        "mean_confidence": mean_conf,
        # Traces for plotting
        "pursuer_x": pursuer_x,
        "pursuer_y": pursuer_y,
        "target_x": target_x,
        "target_y": target_y,
        "commands_sat": commands_sat,
    }


def write_comparison_figure(path: Path, results: List[Dict[str, Any]]) -> None:
    """Generate diagnostic figure of trajectories and command profiles."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "pure_pursuit": "#d95f02",
        "constant_bearing": "#7570b3",
        "proportional_navigation": "#1b9e77",
        "blend_guidance": "#e7298a",
    }
    linestyles = {"alpha_beta": "--", "kalman": "-"}

    # Panel 1: 2D Intercept Trajectories
    ax1 = axes[0]
    # Draw target trajectory once
    ax1.plot(results[0]["target_x"], results[0]["target_y"], "k-", linewidth=2.5, label="Target (Jinking)")
    for res in results:
        label = f"{res['guidance']} ({res['tracker']})"
        ls = linestyles[res["tracker"]]
        c = colors[res["guidance"]]
        ax1.plot(res["pursuer_x"], res["pursuer_y"], color=c, linestyle=ls, linewidth=1.5, label=label)

    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.set_title("2D Intercept Trajectories (Jinking Target)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8)

    # Panel 2: Acceleration Command Profiles for Kalman
    ax2 = axes[1]
    kalman_res = [r for r in results if r["tracker"] == "kalman"]
    time_steps = np.arange(len(kalman_res[0]["commands_sat"])) * 0.02
    for r in kalman_res:
        ax2.plot(time_steps, r["commands_sat"], color=colors[r["guidance"]], label=r["guidance"], linewidth=1.8)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Saturated Lateral Accel (m/s²)")
    ax2.set_title("Guidance Lateral Acceleration Traces (Kalman Tracker)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle("Phase 5: Guidance Law Benchmark against Jinking Target", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/phase5_guidance"))
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    target_states = generate_jinking_target_trajectory(duration_s=3.0, dt_s=0.02)

    trackers = ("alpha_beta", "kalman")
    guidance_laws = ("pure_pursuit", "constant_bearing", "proportional_navigation", "blend_guidance")

    results: List[Dict[str, Any]] = []

    for tr_name in trackers:
        for g_name in guidance_laws:
            res = run_engagement(
                tracker_name=tr_name,
                guidance_name=g_name,
                target_states=target_states,
                dt_s=0.02,
                noise_sigma=0.20,
                seed=args.seed,
            )
            results.append(res)

    # Save CSV
    fieldnames = [
        "tracker",
        "guidance",
        "min_miss_distance_m",
        "final_miss_distance_m",
        "control_effort_mps",
        "saturation_count",
        "saturation_pct",
        "hold_count",
        "mean_confidence",
    ]

    csv_path = args.output / "guidance_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            clean_row = {k: r[k] for k in fieldnames}
            writer.writerow(clean_row)

    write_comparison_figure(args.output / "guidance_comparison.png", results)

    # Print formatted markdown table
    print("\n| Tracker | Guidance Law | Min Miss (m) | Final Miss (m) | Control Effort (m/s) | Saturation (%) | Holds | Mean Conf |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(
            f"| {r['tracker']:10s} | {r['guidance']:23s} | {r['min_miss_distance_m']:6.3f} | {r['final_miss_distance_m']:6.3f} | "
            f"{r['control_effort_mps']:6.2f} | {r['saturation_pct']:5.1f}% | {r['hold_count']:3d} | {r['mean_confidence']:.3f} |"
        )

    print(f"\n[Phase 5] Guidance benchmark complete. Artifacts written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
