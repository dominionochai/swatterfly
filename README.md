# Swatterfly

Swatterfly is a fly/dragonfly-inspired autonomous interceptor drone project for the Nebius x NVIDIA Global AI Hackathon. The project explores how insect vision and pursuit behavior can become a safety-first, explainable flight stack.

## The three-layer brain

- **1 REFLEXES = fly LGMD looming** — a fast, low-latency visual threat signal that can trigger an evasive or intercept response.
- **2 INSTINCTS = dragonfly predictive interception** — target tracking and body-alignment behaviors that estimate where a moving target will be and guide the pursuer toward that future path.
- **3 JUDGMENT = NVIDIA model** — a higher-level model that briefs the mission, selects modes, explains uncertainty, and respects hard safety gates.

we trained the drone's eyes with Nvidia Cosmos, its instincts with Isaac on Nebius, brief it with Tavily

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
