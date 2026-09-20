"""Compare looming detectors against the Phase 2 deterministic event-camera sweep.

Data source
-----------
Reads the canonical Phase 2 output produced by ``scripts/sweep_stage1.py`` as
configured in ``.github/workflows/sweep.yml`` (``data/phase3_baseline``). The
script only ever reads that dataset; it never regenerates sweep data.

Detectors
---------
- ``scalar``  -- the engineering scalar approximation ``eta = theta_dot / theta``
  and ``tau_hat = 1 / eta`` from ``lgmd.scalar_eta``, measured as-is. The scalar
  detector is stateless, so this harness applies the trigger/release hysteresis
  externally. For a constant-speed approach ``eta = 1 / tau``, so the default
  trigger threshold ``eta_on = 0.5`` 1/s is a time-to-contact gate of 2 s.
- ``network`` -- the Gabbiani canonical firing model from ``lgmd.network``:
  ``firing = psi(t - delta) * exp(-alpha * theta(t - delta))`` with
  ``alpha = 1 / tan(theta_thres / 2)`` and an internal hysteresis. Its own
  ``eta_on``/``eta_off`` thresholds are read off the constructed detector.

Metrics (per case, i.e. per sweep run)
--------------------------------------
- ``triggered``: the response crossed ``eta_on`` while rising during the run.
- ``tau_hat_s``: the detector's time-to-contact estimate at the trigger.
- ``tau_true_s``: the ground-truth tau at the trigger timestamp (trajectory
  sample, not the sweep's initial tau).
- ``pass``: triggered AND ``abs(tau_hat - tau_true)/tau_true <= tolerance``.
- ``false_trigger``: triggered but the tau estimate missed ground truth by more
  than the tolerance (a detection whose timing would have driven a wrong
  decision).
- ``latency_s``: seconds from threshold crossing to the peak response observed
  at or after the trigger within the same active period.
- ``false_trigger_rate``: false triggers divided by triggers.
- ``detection_rate``: triggered cases divided by all cases.

Sensitivity axes are the sweep grid axes (speed, size, lighting, noise,
latency). Note: trajectories.csv only encodes speed and object size (lighting,
noise, and latency affect the event stream, which the detectors in this
baseline do not consume), so the lighting/noise/latency axes are expected to be
flat; that is a property of the dataset and is reported as-is.

Examples
--------
    python scripts/compare_lgmd.py --detector scalar --output /tmp/phase3_scalar_baseline
    python scripts/compare_lgmd.py --detector both --output /tmp/phase3_comparison
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lgmd.scalar_eta import LoomingSample, ScalarEtaDetector

try:
    from lgmd.network import NetworkLgmdDetector
except ImportError:  # pragma: no cover - network detector is introduced in Phase 3
    NetworkLgmdDetector = None

DEFAULT_DATA_DIR = ROOT / "data" / "phase3_baseline"

SCALAR_ETA_ON = 0.5  # 1/s, i.e. a tau <= 2 s time-to-contact gate
SCALAR_ETA_OFF = 0.4  # 80% release hysteresis, consistent with repo convention
TAU_TOLERANCE_PCT = 25.0

GRID_HEADERS = (
    "run_id",
    "approach_speed_mps",
    "object_size_m",
    "lighting",
    "sensor_noise_px",
    "injected_latency_s",
)


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_results(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv_dicts(path)
    if not rows:
        raise SystemExit(f"{path} is empty or missing")
    return {row["run_id"]: row for row in rows}


def load_trajectories(path: Path) -> dict[str, dict[str, np.ndarray]]:
    rows = _read_csv_dicts(path)
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    for row in rows:
        grouped.setdefault(row["run_id"], []).append(
            (float(row["timestamp_s"]), float(row["projected_radius_px"]), float(row["tau_s"]))
        )
    out: dict[str, dict[str, np.ndarray]] = {}
    for run_id, samples in grouped.items():
        samples.sort(key=lambda item: item[0])
        out[run_id] = {
            "timestamp_s": np.asarray([item[0] for item in samples], dtype=float),
            "projected_radius_px": np.asarray([item[1] for item in samples], dtype=float),
            "tau_s": np.asarray([item[2] for item in samples], dtype=float),
        }
    return out


def load_event_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in _read_csv_dicts(path):
        counts[row["run_id"]] = counts.get(row["run_id"], 0) + 1
    return counts


def build_detector(name: str) -> tuple[Any, float, float]:
    """Return ``(factory, eta_on, eta_off)`` for the requested detector.

    A *factory* (not an instance) is returned because the detectors are stepped
    per case and the network detector is stateful (delay buffer plus
    hysteresis); each sweep run must see a fresh detector.
    """
    if name == "scalar":
        return (lambda: ScalarEtaDetector()), SCALAR_ETA_ON, SCALAR_ETA_OFF
    if name == "network":
        if NetworkLgmdDetector is None:  # pragma: no cover
            raise SystemExit("network detector not importable; implement src/lgmd/network.py first")
        prototype = NetworkLgmdDetector()
        return (lambda: NetworkLgmdDetector()), prototype.eta_on, prototype.eta_off
    raise SystemExit(f"unsupported detector: {name}")


def run_case(
    detector: Any,
    times: np.ndarray,
    theta: np.ndarray,
    tau_true: np.ndarray,
    *,
    eta_on: float,
    eta_off: float,
) -> dict[str, Any]:
    """Step one detector through one run and return its per-case metrics."""
    theta_dot = np.gradient(theta, times)
    active = False
    trigger_index: int | None = None
    peak_score = -math.inf
    peak_index = 0
    trigger_score = math.nan
    trigger_tau_hat = math.nan
    for index in range(len(times)):
        sample = LoomingSample(
            theta=float(theta[index]),
            theta_dot=float(theta_dot[index]),
            timestamp=float(times[index]),
        )
        score, tau_hat = detector.update(sample)
        if not active and score >= eta_on:
            active = True
            if trigger_index is None:
                trigger_index = index
                trigger_score = float(score)
                trigger_tau_hat = float(tau_hat)
        elif active and score <= eta_off:
            active = False
        if score > peak_score:
            peak_score = float(score)
            peak_index = index
    if trigger_index is None:
        return {
            "triggered": False,
            "trigger_timestamp_s": math.nan,
            "trigger_score": math.nan,
            "trigger_tau_hat_s": math.nan,
            "trigger_tau_true_s": math.nan,
            "tau_error_pct": math.nan,
            "peak_timestamp_s": float(times[peak_index]),
            "peak_score": peak_score,
            "latency_s": math.nan,
            "pass": False,
            "false_trigger": False,
        }
    trigger_time = float(times[trigger_index])
    trigger_tau_true = float(tau_true[trigger_index])
    error_pct = 100.0 * abs(trigger_tau_hat - trigger_tau_true) / trigger_tau_true if trigger_tau_true > 0 else math.inf
    peak_time = float(times[peak_index])
    false_trigger = error_pct > TAU_TOLERANCE_PCT
    return {
        "triggered": True,
        "trigger_timestamp_s": trigger_time,
        "trigger_score": trigger_score,
        "trigger_tau_hat_s": trigger_tau_hat,
        "trigger_tau_true_s": trigger_tau_true,
        "tau_error_pct": error_pct,
        "peak_timestamp_s": peak_time,
        "peak_score": peak_score,
        "latency_s": max(0.0, peak_time - trigger_time),
        "pass": error_pct <= TAU_TOLERANCE_PCT,
        "false_trigger": false_trigger,
    }


def analyse_detector(
    detector_factory: Any,
    eta_on: float,
    eta_off: float,
    trajectories: dict[str, dict[str, np.ndarray]],
    results: dict[str, dict[str, str]],
    event_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    for run_id in results:
        header = {header: results[run_id][header] for header in GRID_HEADERS}
        traj = trajectories[run_id]
        detector = detector_factory()
        metrics = run_case(
            detector,
            traj["timestamp_s"],
            traj["projected_radius_px"],
            traj["tau_s"],
            eta_on=eta_on,
            eta_off=eta_off,
        )
        case_rows.append(
            {
                **header,
                "detector": getattr(detector, "name", type(detector).__name__),
                "tau_true_initial_s": results[run_id]["tau_true_s"],
                "event_count": event_counts.get(run_id, 0),
                **metrics,
            }
        )
    summary = summarize(case_rows)
    by_axis = summarize_by_axis(case_rows)
    return case_rows, summary, by_axis


def _rate(rows: list[dict[str, Any]], predicate) -> float:
    if not rows:
        return math.nan
    return sum(1 for row in rows if predicate(row)) / len(rows)


def summarize(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    triggered = [row for row in case_rows if row["triggered"]]
    latencies = [row["latency_s"] for row in triggered]
    return {
        "detector": case_rows[0]["detector"] if case_rows else "",
        "cases": len(case_rows),
        "triggered": len(triggered),
        "detection_rate": len(triggered) / len(case_rows) if case_rows else math.nan,
        "false_triggers": sum(1 for row in triggered if row["false_trigger"]),
        "false_trigger_rate": _rate(triggered, lambda row: row["false_trigger"]),
        "missed": sum(1 for row in case_rows if not row["triggered"]),
        "mean_latency_s": float(np.mean(latencies)) if latencies else math.nan,
        "median_latency_s": float(np.median(latencies)) if latencies else math.nan,
        "mean_tau_error_pct": float(np.mean([row["tau_error_pct"] for row in triggered])) if triggered else math.nan,
    }


AXES = ("approach_speed_mps", "object_size_m", "lighting", "sensor_noise_px", "injected_latency_s")


def summarize_by_axis(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for axis in AXES:
        values = sorted({row[axis] for row in case_rows})
        for value in values:
            group = [row for row in case_rows if row[axis] == value]
            summary = summarize(group)
            rows.append(
                {
                    "axis": axis,
                    "value": value,
                    "cases": summary["cases"],
                    "triggered": summary["triggered"],
                    "detection_rate": summary["detection_rate"],
                    "false_trigger_rate": summary["false_trigger_rate"],
                    "mean_latency_s": summary["mean_latency_s"],
                    "median_latency_s": summary["median_latency_s"],
                }
            )
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


CASE_FIELDS = [
    *GRID_HEADERS,
    "detector",
    "tau_true_initial_s",
    "event_count",
    "triggered",
    "trigger_timestamp_s",
    "trigger_score",
    "trigger_tau_hat_s",
    "trigger_tau_true_s",
    "tau_error_pct",
    "peak_timestamp_s",
    "peak_score",
    "latency_s",
    "pass",
    "false_trigger",
]


def write_outputs(output_dir: Path, name: str, case_rows: list[dict[str, Any]], summary: dict[str, Any], by_axis: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / f"results_{name}.csv", CASE_FIELDS, case_rows)
    _write_csv(output_dir / f"summary_{name}.csv", list(summary), [summary])
    _write_csv(
        output_dir / f"summary_by_axis_{name}.csv",
        ["axis", "value", "cases", "triggered", "detection_rate", "false_trigger_rate", "mean_latency_s", "median_latency_s"],
        by_axis,
    )


def print_summary(name: str, summary: dict[str, Any]) -> None:
    print(f"[{name}] cases={summary['cases']} triggered={summary['triggered']} "
          f"detection_rate={summary['detection_rate']:.3f} false_trigger_rate={summary['false_trigger_rate']:.3f} "
          f"missed={summary['missed']} mean_latency_s={summary['mean_latency_s']:.4f} "
          f"median_latency_s={summary['median_latency_s']:.4f}")


def build_comparison(rows_by_detector: dict[str, dict[str, Any]], axes_by_detector: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    comparison: list[dict[str, Any]] = []
    for name, summary in rows_by_detector.items():
        row: dict[str, Any] = {
            "detector": name,
            "cases": summary["cases"],
            "triggered": summary["triggered"],
            "detection_rate": summary["detection_rate"],
            "false_trigger_rate": summary["false_trigger_rate"],
            "missed": summary["missed"],
            "mean_latency_s": summary["mean_latency_s"],
            "median_latency_s": summary["median_latency_s"],
            "mean_tau_error_pct": summary["mean_tau_error_pct"],
        }
        for axis_row in axes_by_detector[name]:
            key = f"{axis_row['axis']}={axis_row['value']}"
            row[f"{key}_detection_rate"] = axis_row["detection_rate"]
            row[f"{key}_false_trigger_rate"] = axis_row["false_trigger_rate"]
            row[f"{key}_mean_latency_s"] = axis_row["mean_latency_s"]
        comparison.append(row)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="canonical Phase 2 sweep output")
    parser.add_argument("--output", type=Path, default=Path("/tmp/phase3_comparison"))
    parser.add_argument("--detector", choices=("scalar", "network", "both"), default="both")
    parser.add_argument("--scalar-eta-on", type=float, default=SCALAR_ETA_ON)
    parser.add_argument("--scalar-eta-off", type=float, default=SCALAR_ETA_OFF)
    parser.add_argument("--tau-tolerance-pct", type=float, default=TAU_TOLERANCE_PCT)
    args = parser.parse_args()

    results_path = args.data_dir / "results.csv"
    trajectories_path = args.data_dir / "trajectories.csv"
    events_path = args.data_dir / "events.csv"
    for path in (results_path, trajectories_path, events_path):
        if not path.is_file():
            raise SystemExit(f"required dataset file missing: {path}")

    results = load_results(results_path)
    trajectories = load_trajectories(trajectories_path)
    event_counts = load_event_counts(events_path)
    missing_runs = sorted(set(results) - set(trajectories))
    if missing_runs:
        raise SystemExit(f"trajectories.csv is missing runs: {missing_runs[:5]}")

    names = ["scalar", "network"] if args.detector == "both" else [args.detector]
    case_rows_by_detector: dict[str, list[dict[str, Any]]] = {}
    summary_by_detector: dict[str, dict[str, Any]] = {}
    axes_by_detector: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        detector_factory, eta_on, eta_off = build_detector(name)
        if name == "scalar":
            eta_on, eta_off = args.scalar_eta_on, args.scalar_eta_off
        case_rows, summary, by_axis = analyse_detector(detector_factory, eta_on, eta_off, trajectories, results, event_counts)
        case_rows_by_detector[name] = case_rows
        summary_by_detector[name] = summary
        axes_by_detector[name] = by_axis
        write_outputs(args.output, name, case_rows, summary, by_axis)
        print_summary(name, summary)

    if args.detector == "both":
        comparison = build_comparison(summary_by_detector, axes_by_detector)
        _write_csv(args.output / "comparison.csv", list(comparison[0]), comparison)
        print("comparison.csv written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())