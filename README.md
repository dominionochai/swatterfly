# Swatterfly

Swatterfly is a fly/dragonfly-inspired autonomous interceptor drone project for the Nebius x NVIDIA Global AI Hackathon. The project explores how insect vision and pursuit behavior can become a safety-first, explainable flight stack.

## Quick Start

Start the backend from the repository root:

```bash
cd src/backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cd ../..
uvicorn src.backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open the backend API docs at http://127.0.0.1:8000/docs.

In another terminal, start the dashboard:

```bash
cd src/dashboard
npm install
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 npm run dev
```

Open the dashboard at http://localhost:3000. `NEBIUS_API_KEY` is optional for the fake telemetry/cockpit demo; only the Nebius agent ping requires it.

## The three-layer brain

- **1 REFLEXES = fly LGMD looming** — a fast, low-latency visual threat signal that can trigger an evasive or intercept response.
- **2 INSTINCTS = dragonfly predictive interception** — target tracking and body-alignment behaviors that estimate where a moving target will be and guide the pursuer toward that future path.
- **3 JUDGMENT = NVIDIA model** — a higher-level model that briefs the mission, selects modes, explains uncertainty, and respects hard safety gates.

We trained the drone's eyes with Nvidia Cosmos, its instincts with Isaac on Nebius, brief it with Tavily

## Stack roles

- **NVIDIA runs onboard:** Jetson-class edge hardware, Isaac ROS, TensorRT, and Holoscan are the intended path for low-latency perception and flight-side inference.
- **Nebius trains + hosts coding agents:** Isaac Lab experiments and burst GPU training run on Nebius; GLM-5.3 or DeepSeek V4-Pro-0813 agents on Token Factory help iterate on the system.
- **Tavily pre-mission intel:** preflight research can collect weather, NOTAM, and other mission context before the vehicle is armed. It is not a replacement for onboard safety logic.

## Repository map

- `docs/01-math-lgmd.md` — looming geometry and the LGMD/Rind–Bramwell detector.
- `docs/02-math-dragonfly.md` — predictive interception, proportional navigation, and pursuit comparisons.
- `docs/03-repos.md` — verified implementation and tooling references.
- `docs/04-papers.md` — verified papers, reviews, URLs, and caveats.
- `docs/05-build-plan.md` — the twelve-phase build and validation plan.
- `docs/06-cockpit.md` — the preserved Stitch cockpit design specification.
- `docs/07-credits-budget.md` — credits, pricing reports, and eligibility caveats.
- `docs/08-stack-models.md` — model, hardware, and runtime choices.
- `src/` — Python 3.11+ scaffold for simulation, looming detection, tracking, guidance, and dashboard integration.

This repository is a research and demonstration scaffold. Any physical flight requires a controlled test area, a qualified operator, independent failsafes, and validation in simulation before hardware operation.

## Hackathon

Swatterfly is being prepared for the Nebius x NVIDIA Global AI Hackathon: https://nebiusglobalaihackathon.devpost.com/

The stated submission deadline is **Oct 30 2026 10am PDT**.

## Status

The modules under `src/` deliberately expose small, testable interfaces while marking research implementations as TODOs. The documentation distinguishes measured or cited behavior from assumptions and stretch goals; placeholder code must not be treated as flight-ready.

## Phase 2 closeout and Phase 3 looming baseline (session notes)

Work committed in this repository as of the latest two commits:

- **Phase 2 closeout:** removed the dead legacy aliases from `src/sim/point_mass.py`
  (`PointMassState = PointMassState`, `step_point_mass_legacy`, `step_point_mass_original`).
  Verified after removal: editable install works, `pytest tests src/backend/tests` reports
  **6 passed**, and `scripts/sweep_stage1.py` runs unchanged (324 runs, all
  `estimate_status == "ok"`, `error_by_latency.png` regenerated).
- **Phase 3 looming baseline:** added `src/lgmd/network.py` (Gabbiani canonical firing
  model, `firing ∝ ψ(t−δ)·e^(−α·θ(t−δ))`, `α = 1/tan(θ_thres/2)`, θ_thres 15–40°, δ
  15–35 ms, internal trigger/release hysteresis) and `scripts/compare_lgmd.py` to measure
  it against the existing engineering scalar approximation (`eta = theta_dot / theta`)
  on the canonical Phase 2 dataset at `data/phase3_baseline/`. Full results and caveats
  are in `docs/phase3-looming-baseline.md`; `tests/test_lgmd_network.py` adds 4 tests
  (full suite: **10 passed**).

Key findings from the Phase 3 comparison (324 cases, both detectors):

- Scalar: 50% detection, 0 false triggers, mean trigger→peak latency 0.83 s. It is a
  time-to-contact gate (`eta = 1/tau`), so it only fires for runs reaching `tau <= 2 s`
  in the sweep window (speeds 2.0/4.0 m/s); slow runs are missed.
- Network: 33% detection (only the smallest-object runs), 0 false triggers, fires at the
  first frame (latency 0.000 s). With this dataset's `projected_radius_px` theta proxy,
  the exponential `exp(−α·θ)` suppresses the score for larger theta, so it behaves as a
  size gate. Not a preprocessing claim — a measured property of the model as configured.
- Lighting/noise/latency axes are flat for both detectors because those parameters only
  affect the event stream (`events.csv`), which these trajectory-driven detectors do not
  consume. Zero false-trigger rate reflects the noise-free analytic trajectories.

Known pre-existing issues (noted, not yet fixed):

- `scripts/phase3_baseline_analysis.py:28` imports `LookingSample` from `lgmd.scalar_eta`,
  but the class is `LoomingSample` (`src/lgmd/scalar_eta.py:15`) — the script crashes on
  import and is not covered by CI.
- `.github/workflows/ci.yml` installs only `requirements*.txt` and runs pytest with
  `PYTHONPATH=.`; root tests importing `sim` (e.g. `tests/test_sim_stage1.py`) will fail
  collection there. The sweep workflow avoids this via `pip install -e ".[dev]"`.
- v2e is unusable on this stack (no top-level import in the 1.5.1 distribution); the
  deterministic synthetic event backend is the canonical path.
