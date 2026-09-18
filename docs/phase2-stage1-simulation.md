# Phase 2 Stage-1 simulation baseline

This adds a narrow, reproducible kinematic and event-camera baseline without
changing `src/lgmd/`, tracker/guidance code, physics tuning, or the dashboard.

## Run on Windows PowerShell

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest tests src/backend/tests -q
python scripts\sweep_stage1.py --output data\phase3_baseline
```

The sweep is deterministic (default seed `20260918`) and covers all
4 speeds x 3 object sizes x 3 lighting levels x 3 sensor-noise levels x 3
latencies = 324 cases. It writes reusable `results.csv`, `trajectories.csv`,
`events.csv`, `metadata.json`, and `error_by_latency.png`.

Every trajectory keeps the exact controllable ground truth beside the sensor
stream: `l_m` is object size, `x_m` is initial distance, `u_mps` is approach
speed, and `tau_true_s = x_m / u_mps`. `tau_hat_s` is estimated only from
synthetic event geometry and delayed/noisy event timestamps.

## v2e note

The project declares the upstream v2e repository as the real optional
`event-camera` dependency:

```powershell
python -m pip install -e ".[event-camera]"
```

v2e is not a hard prerequisite for the baseline. Its upstream command-line
stack commonly expects Linux/CUDA tooling and may fail to build on Windows.
`generate_synthetic_events(..., backend="auto")` therefore uses the typed,
deterministic trajectory-native renderer and records whether v2e was importable.
`backend="v2e"` verifies the optional install but still uses the same
trajectory-native renderer so a Phase-3 dataset is reproducible across hosts.
The rpg_esim integration remains stretch-only and is deliberately absent from
this baseline.
