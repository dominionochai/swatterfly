"""Run the complete, deterministic Phase-2 Stage-1 parameter sweep.

PowerShell:
    python scripts/sweep_stage1.py --output data/phase3_baseline

The output directory contains trajectories.csv, events.csv, results.csv,
metadata.json, and error_by_latency.png.  The raw CSV files make the baseline
reusable without rerunning the renderer in Phase 3.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from inspect import signature
from itertools import product
from pathlib import Path
from typing import Iterable

# Make the script runnable from a clean repository checkout before pip install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sim.events import (  # noqa: E402
    ApproachScenario,
    EventStream,
    estimate_tau,
    generate_synthetic_events,
    generate_trajectory,
    v2e_is_available,
)

SPEEDS_MPS = (0.5, 1.0, 2.0, 4.0)
SIZES_M = (0.04, 0.08, 0.16)
LIGHTING = (0.5, 1.0, 2.0)
NOISE_PX = (0.0, 0.5, 1.5)
LATENCIES_S = (0.0, 0.005, 0.02)


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _verified_v2e_available() -> bool:
    """Probe optional v2e without letting a broken import abort the baseline."""
    try:
        return v2e_is_available()
    except Exception as exc:  # pragma: no cover - depends on optional package
        print(f"[sweep] v2e import failed ({exc}); using the deterministic local renderer.", file=sys.stderr)
        return False


def _select_backend(requested: str) -> str:
    if requested == "auto":
        requested = "v2e" if _verified_v2e_available() else "synthetic"
    if requested == "v2e" and not _verified_v2e_available():
        print("[sweep] v2e was requested but did not import; falling back to the deterministic local renderer.", file=sys.stderr)
        return "synthetic"
    if requested == "v2e":
        print("[sweep] v2e imported successfully; using the v2e-compatible backend.")
        return "v2e"
    if requested == "synthetic":
        print("[sweep] using the deterministic local renderer (v2e unavailable or not selected).")
        return "synthetic"
    raise ValueError(f"unsupported event backend: {requested}")


def run_sweep(output: Path, seed: int = 20260918, backend: str = "auto") -> dict[str, object]:
    """Run every speed x size x lighting x noise x latency combination."""

    output.mkdir(parents=True, exist_ok=True)
    selected_backend = _select_backend(backend)
    result_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    run_id = 0
    for speed, size, lighting, noise, latency in product(
        SPEEDS_MPS, SIZES_M, LIGHTING, NOISE_PX, LATENCIES_S
    ):
        # End at 60% of the initial time-to-contact, leaving a well-conditioned
        # growth interval while keeping the default 324-run sweep lightweight.
        scenario = ApproachScenario(
            object_size_m=size,
            initial_distance_m=6.0,
            approach_speed_mps=speed,
            duration_s=0.60 * 6.0 / speed,
            dt_s=0.02,
            lighting=lighting,
            sensor_noise_px=noise,
            injected_latency_s=latency,
        )
        trajectory = generate_trajectory(scenario)
        stream: EventStream = generate_synthetic_events(
            trajectory, scenario, seed=seed + run_id, backend=selected_backend
        )
        try:
            tau_hat = estimate_tau(stream.events, scenario)
            estimate_status = "ok"
        except ValueError:
            tau_hat = float("nan")
            estimate_status = "failed"
        tau_true = stream.ground_truth_tau_s
        error_abs = abs(tau_hat - tau_true) if tau_hat == tau_hat else float("nan")
        error_pct = 100.0 * error_abs / tau_true if error_abs == error_abs else float("nan")
        result_rows.append(
            {
                "run_id": run_id,
                "approach_speed_mps": speed,
                "object_size_m": size,
                "lighting": lighting,
                "sensor_noise_px": noise,
                "injected_latency_s": latency,
                "l_m": size,
                "x_m": 6.0,
                "u_mps": speed,
                "tau_true_s": tau_true,
                "tau_hat_s": tau_hat,
                "error_abs_s": error_abs,
                "error_pct": error_pct,
                "event_count": len(stream.events),
                "estimate_status": estimate_status,
                "event_backend": stream.backend,
            }
        )
        trajectory_rows.extend(
            {
                "run_id": run_id,
                "timestamp_s": sample.timestamp_s,
                "l_m": sample.ground_truth.l_m,
                "x_m": sample.ground_truth.x_m,
                "u_mps": sample.ground_truth.u_mps,
                "tau_s": sample.ground_truth.tau_s,
                "projected_radius_px": sample.projected_radius_px,
            }
            for sample in trajectory
        )
        event_rows.extend(
            {
                "run_id": run_id,
                "x_px": event.x_px,
                "y_px": event.y_px,
                "timestamp_s": event.timestamp_s,
                "polarity": event.polarity,
            }
            for event in stream.events
        )
        run_id += 1

    _write_csv(
        output / "results.csv",
        [
            "run_id", "approach_speed_mps", "object_size_m", "lighting",
            "sensor_noise_px", "injected_latency_s", "l_m", "x_m", "u_mps",
            "tau_true_s", "tau_hat_s", "error_abs_s", "error_pct",
            "event_count", "estimate_status", "event_backend",
        ],
        result_rows,
    )
    _write_csv(
        output / "trajectories.csv",
        ["run_id", "timestamp_s", "l_m", "x_m", "u_mps", "tau_s", "projected_radius_px"],
        trajectory_rows,
    )
    _write_csv(
        output / "events.csv",
        ["run_id", "x_px", "y_px", "timestamp_s", "polarity"],
        event_rows,
    )
    metadata = {
        "seed": seed,
        "run_count": run_id,
        "parameter_grid": {
            "approach_speed_mps": SPEEDS_MPS,
            "object_size_m": SIZES_M,
            "lighting": LIGHTING,
            "sensor_noise_px": NOISE_PX,
            "injected_latency_s": LATENCIES_S,
        },
        "ground_truth_definition": "l=object size, x=initial distance, u=approach speed, tau=x/u",
        "requested_event_backend": backend,
        "event_backend": selected_backend,
        "v2e_available_at_generation": _verified_v2e_available(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_plot(output / "error_by_latency.png", result_rows)
    return metadata


def _write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latencies = sorted({float(row["injected_latency_s"]) for row in rows})
    groups = [
        [float(row["error_pct"]) for row in rows if float(row["injected_latency_s"]) == latency and row["error_pct"] == row["error_pct"]]
        for latency in latencies
    ]
    figure, axis = plt.subplots(figsize=(7, 4))
    labels = [f"{latency:g}s" for latency in latencies]
    # Matplotlib 3.9 renamed ``labels`` to ``tick_labels``.  Inspect the
    # installed API so the baseline also works with older Matplotlib releases.
    label_parameter = "tick_labels" if "tick_labels" in signature(axis.boxplot).parameters else "labels"
    axis.boxplot(groups, **{label_parameter: labels}, showfliers=False)
    axis.set_xlabel("Injected event latency")
    axis.set_ylabel("Absolute tau error (%)")
    axis.set_title("Stage-1 tau estimate error across full parameter sweep")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/phase3_baseline"))
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument(
        "--backend",
        choices=("auto", "synthetic", "v2e"),
        default=os.environ.get("SWATTERFLY_EVENT_BACKEND", "auto"),
        help="event backend; v2e is selected only after a successful import probe",
    )
    args = parser.parse_args()
    metadata = run_sweep(args.output, seed=args.seed, backend=args.backend)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
