#!/usr/bin/env python3
"""Analyze Phase 3 trajectories with the scalar engineering eta detector.

Gabbiani's firing expression is ``Firing ∝ ψ(t−δ)·e^(−α·θ(t−δ))``.
This script instead computes engineering ``eta = theta_dot / theta`` and
``tau_hat = 1 / eta`` sample by sample. The scalar is NOT the Gabbiani firing
model.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from lgmd.scalar_eta import LoomingSample, ScalarEtaDetector
except ImportError as exc:  # pragma: no cover
    raise ImportError(f"cannot import detector from {src_path}: {exc}") from exc

ETA_THRESHOLD = 10.0
ETA_RELEASE_THRESHOLD = ETA_THRESHOLD * 0.8  # explicit 80% hysteresis assumption
AXES = ("speed", "size", "lighting", "noise")
REQUIRED = {"run_id", "timestamp_s", "l_m", "x_m", "u_mps", "tau_s", "projected_radius_px"}
ALIASES = {
    "speed": ("approach_speed_mps", "approach_speed", "speed_mps", "speed", "u_mps"),
    "size": ("object_size_m", "object_size", "size_m", "size", "l_m"),
    "lighting": ("lighting", "light", "illumination"),
    "noise": ("sensor_noise_px", "sensor_noise", "noise_px", "noise"),
}


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def key(value: Any) -> str:
    try:
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return "" if value is None else str(value).strip()


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def shown(value: Any) -> str:
    result = number(value)
    if result is not None:
        return f"{result:g}"
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "null"} else "unknown"


def choose(columns: list[str], exact: tuple[str, ...], contains: tuple[str, ...], exclude: set[str] | None = None) -> str | None:
    exclude = exclude or set()
    names = {column: norm(column) for column in columns}
    for wanted in exact:
        for column in columns:
            if column not in exclude and names[column] == norm(wanted):
                return column
    for token in contains:
        token = norm(token)
        for column in columns:
            if column not in exclude and token in names[column]:
                return column
    return None


def mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def lookup(data: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    values = {norm(k): v for k, v in data.items()}
    for alias in aliases:
        if norm(alias) in values:
            return values[norm(alias)]
    for field, value in values.items():
        if any(norm(alias) in field for alias in aliases):
            return value
    return None


def run_metadata(metadata: dict[str, Any], run_id: Any) -> dict[str, Any]:
    run_key = key(run_id)
    for field in ("runs", "run_metadata", "run_mappings", "by_run", "metadata"):
        table = metadata.get(field)
        if isinstance(table, dict):
            for candidate, values in table.items():
                if key(candidate) == run_key and isinstance(values, dict):
                    return values
    value = metadata.get(str(run_id))
    return value if isinstance(value, dict) else {}


def id_axis(run_id: Any, axis: str) -> Any:
    tokens = {
        "speed": ("approach_speed_mps", "approach_speed", "speed_mps", "speed", "u_mps"),
        "size": ("object_size_m", "object_size", "size_m", "size", "l_m"),
        "lighting": ("lighting", "light"),
        "noise": ("sensor_noise_px", "sensor_noise", "noise_px", "noise"),
    }[axis]
    text = norm(run_id)
    value = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
    for token in sorted(tokens, key=len, reverse=True):
        match = re.search(rf"(?:^|_){re.escape(norm(token))}(?:_|=){value}", text)
        if match:
            return match.group(1)
    return None


def axes_for(row: dict[str, Any], metadata: dict[str, Any], run_id: Any) -> dict[str, str]:
    nested = mapping(row.get("metadata"))
    combined = dict(metadata)
    combined.update(nested)
    result = {}
    for axis in AXES:
        value = lookup(row, ALIASES[axis])
        if value is None:
            value = lookup(combined, ALIASES[axis])
        if value is None:
            value = id_axis(run_id, axis)
        result[axis] = shown(value)
    return result


def read_events(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, str | None]]:
    """Resolve event headers generically and return real-event rows.

    Header candidates contain run/trajectory, event/loom/real/label, and
    start/end/time. Fallback: no run header means global time matching; no
    real/label header means every row is a real event; no time header means the
    first numeric non-run column is used. This avoids assumptions about the
    producer's exact CSV names.
    """
    columns = [str(c) for c in frame.columns]
    run_col = choose(columns, ("run_id", "trajectory_id"), ("run", "trajectory"))
    excluded = {x for x in (run_col,) if x}
    time_col = choose(columns, ("timestamp_s", "time_s", "event_time_s"), ("timestamp", "time", "start", "end", "event", "loom"), excluded)
    label_col = choose(columns, ("is_real", "real_event", "event_label", "label"), ("real", "event", "loom", "label"), {x for x in (run_col, time_col) if x})
    if time_col is None:
        for column in columns:
            if column in {run_col, label_col}:
                continue
            if pd.to_numeric(frame[column], errors="coerce").notna().any():
                time_col = column
                break
    if time_col is None:
        raise ValueError("events.csv has no resolvable time column")
    rows = []
    for raw in frame.to_dict(orient="records"):
        timestamp = number(raw.get(time_col))
        if timestamp is None:
            continue
        real = True
        if label_col:
            text = str(raw.get(label_col)).strip().lower()
            if text in {"false", "0", "no", "n", "synthetic", "predicted", "negative"}:
                real = False
            elif text not in {"true", "1", "yes", "y", "real", "loom", "event", "positive"}:
                value = number(raw.get(label_col))
                real = value != 0 if value is not None else True
        rows.append({"run": key(raw.get(run_col)) if run_col else "__all__", "time": timestamp, "real": real})
    if not rows:
        raise ValueError("events.csv has no usable event rows")
    return rows, {"run": run_col, "time": time_col, "label": label_col}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def analyse(trajectories_path: Path, events_path: Path, metadata_path: Path, output_dir: Path) -> dict[str, Any]:
    for path in (trajectories_path, events_path):
        if not path.is_file():
            raise FileNotFoundError(f"required input missing: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = pd.read_csv(trajectories_path)
    missing = sorted(REQUIRED - set(trajectories.columns))
    if missing:
        raise ValueError(f"trajectories.csv missing required columns: {', '.join(missing)}")
    if trajectories.empty:
        raise ValueError("trajectories.csv contains no samples")
    for column in ("timestamp_s", "l_m", "x_m", "u_mps", "tau_s", "projected_radius_px"):
        trajectories[column] = pd.to_numeric(trajectories[column], errors="raise")
    if trajectories["run_id"].isna().any():
        raise ValueError("trajectories.csv contains a missing run_id")
    metadata = {}
    if metadata_path.is_file():
        parsed = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = parsed if isinstance(parsed, dict) else {}
    events, event_columns = read_events(pd.read_csv(events_path))
    real_events: dict[str, list[float]] = {}
    for event in events:
        if event["real"]:
            real_events.setdefault(event["run"], []).append(float(event["time"]))
    for values in real_events.values():
        values.sort()

    samples: list[dict[str, Any]] = []
    triggers: list[dict[str, Any]] = []
    run_count = 0
    for run_id, group in trajectories.groupby("run_id", sort=False, dropna=False):
        run_count += 1
        group = group.sort_values("timestamp_s", kind="mergesort")
        times = group["timestamp_s"].to_numpy(float)
        theta = group["projected_radius_px"].to_numpy(float)
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError(f"run {key(run_id)} timestamps must be strictly increasing")
        if np.any(~np.isfinite(theta)) or np.any(theta <= 0):
            raise ValueError(f"run {key(run_id)} projected_radius_px must be finite and positive")
        theta_dot = np.gradient(theta, times) if len(times) > 1 else np.array([0.0])
        if np.any(~np.isfinite(theta_dot)) or np.any(theta_dot < 0):
            raise ValueError(f"run {key(run_id)} has invalid theta_dot; detector requires non-negative expansion")
        detector = ScalarEtaDetector()
        axes = axes_for(group.iloc[0].to_dict(), run_metadata(metadata, run_id), run_id)
        active = False
        peak_eta = -math.inf
        peak_time = float(times[0])
        run_start = len(samples)
        trigger_indices: list[int] = []
        for index, (_, row) in enumerate(group.iterrows()):
            sample = LoomingSample(float(theta[index]), float(theta_dot[index]), float(times[index]))
            eta, tau_hat = detector.update(sample)
            triggered = not active and eta >= ETA_THRESHOLD
            if triggered:
                active = True
                trigger_indices.append(index)
            elif active and eta <= ETA_RELEASE_THRESHOLD:
                active = False
            if eta > peak_eta:
                peak_eta, peak_time = float(eta), float(times[index])
            output = row.to_dict()
            output.update({"theta": theta[index], "theta_dot": theta_dot[index], "eta": eta, "tau_hat": tau_hat, "triggered": triggered, "hysteresis_active": active, **axes})
            samples.append(output)
        run_samples = samples[run_start:]
        for index in trigger_indices:
            triggers.append({"run_id": run_id, "trigger_timestamp_s": times[index], "trigger_eta": run_samples[index]["eta"], "peak_timestamp_s": peak_time, "peak_eta": peak_eta, "threshold_to_peak_latency_s": max(0.0, peak_time - times[index]), **axes})

    sample_frame = pd.DataFrame(samples)
    trigger_columns = ["run_id", "trigger_timestamp_s", "trigger_eta", "peak_timestamp_s", "peak_eta", "threshold_to_peak_latency_s", *AXES]
    trigger_frame = pd.DataFrame(triggers, columns=trigger_columns)
    atomic_csv(sample_frame, output_dir / "phase3_eta_results.csv")

    # Match one detector trigger to at most one real event within a documented tolerance.
    times_all = trajectories.sort_values(["run_id", "timestamp_s"])["timestamp_s"].to_numpy(float)
    dts = np.diff(times_all)
    dts = dts[dts > 0]
    tolerance = max(0.05, float(2 * np.median(dts))) if len(dts) else 0.05
    used: set[tuple[str, int]] = set()
    for trigger in triggers:
        run = key(trigger["run_id"])
        source = run if run in real_events else "__all__"
        candidates = real_events.get(source, [])
        ranked = sorted((abs(t - trigger["trigger_timestamp_s"]), index) for index, t in enumerate(candidates) if (source, index) not in used)
        if ranked and ranked[0][0] <= tolerance:
            used.add((source, ranked[0][1]))
            trigger["matched_real_event"] = True
        else:
            trigger["matched_real_event"] = False
    trigger_columns.append("matched_real_event")
    trigger_frame = pd.DataFrame(triggers, columns=trigger_columns)
    atomic_csv(trigger_frame, output_dir / "phase3_triggers.csv")

    count = len(triggers)
    matched = sum(bool(t["matched_real_event"]) for t in triggers)
    false_count = count - matched
    false_rate = false_count / count if count else math.nan
    latencies = [float(t["threshold_to_peak_latency_s"]) for t in triggers]

    summary: list[dict[str, Any]] = []
    for axis in AXES:
        for value, sample_group in sample_frame.groupby(axis, dropna=False):
            value = shown(value)
            group_triggers = [t for t in triggers if shown(t.get(axis)) == value]
            group_latencies = [float(t["threshold_to_peak_latency_s"]) for t in group_triggers]
            group_false = sum(not bool(t["matched_real_event"]) for t in group_triggers)
            summary.append({"axis": axis, "value": value, "runs": sample_group["run_id"].nunique(), "samples": len(sample_group), "triggers": len(group_triggers), "mean_eta": sample_group["eta"].mean(), "median_eta": sample_group["eta"].median(), "mean_tau_hat": sample_group["tau_hat"].replace([np.inf, -np.inf], np.nan).mean(), "false_trigger_rate": group_false / len(group_triggers) if group_triggers else math.nan, "mean_threshold_to_peak_latency_s": np.mean(group_latencies) if group_latencies else math.nan})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes_plot = plt.subplots(1, 2, figsize=(12, 4.5))
    for index, (run_id, group) in enumerate(sample_frame.groupby("run_id", sort=False)):
        if index >= 12:
            break
        axes_plot[0].plot(group["timestamp_s"], group["eta"], linewidth=0.8, label=str(run_id))
    axes_plot[0].axhline(ETA_THRESHOLD, color="tab:red", linestyle="--", linewidth=1, label="trigger 10")
    axes_plot[0].axhline(ETA_RELEASE_THRESHOLD, color="tab:orange", linestyle=":", linewidth=1, label="release 8")
    axes_plot[0].set(xlabel="time (s)", ylabel="eta = theta_dot / theta", title="Scalar eta (first 12 runs)")
    if run_count:
        axes_plot[0].legend(fontsize=6, ncol=2)
    if latencies:
        axes_plot[1].hist(latencies, bins=min(20, max(1, len(latencies))), color="tab:blue")
    else:
        axes_plot[1].text(0.5, 0.5, "No threshold triggers", ha="center", va="center")
    axes_plot[1].set(xlabel="seconds", ylabel="triggers", title="Threshold-to-peak latency")
    figure.tight_layout()
    figure.savefig(output_dir / "phase3_baseline_analysis.png", dpi=140)
    plt.close(figure)

    lines = [
        "# Phase 3 scalar-eta baseline analysis", "",
        f"- Samples: {len(sample_frame)} across {run_count} runs.",
        f"- ETA threshold: `{ETA_THRESHOLD:g}`; release: `{ETA_RELEASE_THRESHOLD:g}` (80% hysteresis assumption).",
        f"- Triggers: {count}; matched real events: {matched}; false triggers: {false_count}; false-trigger rate: " + (f"{false_rate:.4f}" if math.isfinite(false_rate) else "n/a (no triggers)"),
        "- Threshold-to-peak latency: " + (f"mean {np.mean(latencies):.6g} s; median {np.median(latencies):.6g} s." if latencies else "n/a (no triggers)."), "",
        "## Model boundary", "",
        "Gabbiani is `Firing ∝ ψ(t−δ)·e^(−α·θ(t−δ))`; this analysis uses engineering `eta=theta_dot/theta` and `tau_hat=1/eta`. The scalar is **not** the Gabbiani firing model.",
        "`projected_radius_px` is the scalar theta proxy because the verified trajectory schema has no separate theta column; theta_dot is the within-run time derivative.", "",
        "## Event resolver", "",
        f"Resolved headers: run=`{event_columns['run'] or 'none'}`, time=`{event_columns['time']}`, label=`{event_columns['label'] or 'none'}`.",
        "The resolver searches headers containing run/trajectory, event/loom/real/label/start/end/time. Without a run header it matches globally; without a real/label header every events.csv row is treated as real.",
        f"Trigger/event matching tolerance: {tolerance:g} s (max of 0.05 s and twice the median positive sample interval).", "",
        "## Axis summaries", "",
        "Speed, size, lighting, and noise values come from metadata-like columns, per-run metadata mappings, or explicit axis/value tokens in run_id. Missing values remain `unknown`; no row-order inference is used.",
    ]
    if summary:
        headers = list(summary[0])
        lines += ["", "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        lines += ["| " + " | ".join("" if pd.isna(row[h]) else str(row[h]) for h in headers) + " |" for row in summary]
    lines += ["", "## Outputs", "", "- `phase3_eta_results.csv`: one detector update and eta/tau_hat for every trajectory sample.", "- `phase3_triggers.csv`: hysteresis trigger records and event matching.", "- `phase3_baseline_analysis.png`: Agg-rendered eta and latency plots.", "- `phase3_baseline_analysis.md`: this report. The input CSVs and existing `results.csv` are never written."]
    atomic_text("\n".join(lines) + "\n", output_dir / "phase3_baseline_analysis.md")
    return {"samples": len(sample_frame), "runs": run_count, "triggers": count, "false_rate": false_rate}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, default=repo_root / "data/phase3_baseline/trajectories.csv")
    parser.add_argument("--events", type=Path, default=repo_root / "data/phase3_baseline/events.csv")
    parser.add_argument("--metadata", type=Path, default=repo_root / "data/phase3_baseline/metadata.json")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "data/phase3_baseline")
    args = parser.parse_args()
    try:
        result = analyse(args.trajectories, args.events, args.metadata, args.output_dir)
    except Exception as exc:
        print(f"[phase3] error: {exc}", file=sys.stderr)
        return 1
    rate = "n/a" if not math.isfinite(result["false_rate"]) else f"{result['false_rate']:.4f}"
    print(f"[phase3] {result['samples']} samples, {result['runs']} runs, {result['triggers']} triggers, false-trigger rate {rate}")
    print(f"[phase3] wrote {args.output_dir / 'phase3_baseline_analysis.md'} and {args.output_dir / 'phase3_baseline_analysis.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
