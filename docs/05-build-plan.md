# Twelve-phase build plan

The phases are ordered to keep fast reflexes and safety gates ahead of model complexity. Every phase produces an artifact that can be tested without claiming flight readiness.

## 1. Scaffold and contracts

Create the FastAPI backend, preserve the Stitch frontend design, and define telemetry and replay schemas. Keep the TypeScript/Next.js dashboard as the only frontend language. Use Python 3.11+ as the primary language and Docker for repeatable services. Host the GLM-5.3 agent on Nebius Token Factory through its OpenAI-compatible interface. Establish configuration, logging, deterministic seeds, and a no-actuator local demo.

## 2. Stage-1 simulation

Build the Python point-mass chase simulator, then connect v2e and ESIM-style event generation. Verify the estimated time-to-contact `tau_hat` against the geometric `x/u` result across approach speeds, object sizes, lighting changes, sensor noise, and latency. Add wind and ego-motion scenarios before trusting detector plots.

## 3. Looming baseline

Implement and test the scalar `eta` detector first, including hysteresis and `tau_hat`. Then port LGMD2 in Python behind the same interface. Compare latency, false triggers, threshold-to-peak behavior, and sensitivity to size/speed/lighting. Keep canonical biology claims separate from engineering substitutions.

## 4. Event tracker

Convert event clusters into target observations and implement an alpha-beta tracker, followed by a Kalman comparison. Measure track confidence, missed observations, outliers, frame/event synchronization, and the effect of 10-50 ms latency. Expose uncertainty to guidance rather than hiding it.

## 5. Guidance ladder

Implement and compare pure pursuit, constant bearing, proportional navigation, and the PN/lead blend. Start with the simplest controller, log miss distance and effort, and add acceleration/turn-rate limits. Include a `tau_emergency` safety override and keep guidance as a desired command, never a raw motor command.

## 6. Isaac Lab reinforcement learning on Nebius

Use Isaac Lab on Nebius for controlled RL experiments, spending the Nebius $50 in GPU bursts. Prefer 2-3 hour sessions, checkpoint every run, and kill idle resources. Start with shaped, interpretable objectives and domain randomization; do not let an RL policy bypass deterministic safety and geofence logic.

## 7. Cosmos-Transfer2.5 augmentation

Use Cosmos-Transfer2.5 to augment sim-to-real appearance while preserving depth and segmentation structure. Keep source labels and augmentation metadata. Compare original, augmented, and held-out real-like validation sets; augmentation is not permission to invent safety-critical data.

For Nebius Token Factory, set `base_url` to `https://api.tokenfactory.nebius.com/v1/` and read the credential from the `NEBIUS_API_KEY` environment variable, not bare `OPENAI_API_KEY`. This configuration applies to GLM/DeepSeek and to OpenAI-SDK tools including IsaacLabEureka; those tools must be explicitly patched to use the Token Factory base URL.

## 8. Tavily pre-mission intelligence

Use Tavily for pre-mission intelligence such as NOTAMs, weather, and known threats. Store query time, source context, and confidence in the mission brief. Treat the result as advisory and staleable; it cannot authorize flight or replace local regulations, operator judgment, or onboard sensing.

## 9. Safety override and supervision

Implement hard gates for arming, geofence, lost track, low battery, excessive latency, uncertain state, `tau_emergency`, and operator abort. Test hysteresis, graceful release, return/land behavior, and command timeouts. A model may recommend a mode, but it must not remove these gates.

## 10. Hardware path

Move from simulation to Jetson plus PX4 Offboard or ArduPilot Guided through MAVSDK. Use PX4 SITL first, then bench tests with props removed, tethered tests, and only a controlled flight envelope. Profile sensor-to-command latency, thermal behavior, and CPU/GPU headroom before enabling any intercept behavior.

## 11. Validation sequence

Run the sequence: unit tests for geometry and filters; deterministic replay; event-camera simulation; Monte Carlo size/speed/lighting/noise/latency; SITL with injected target trajectories; hardware-in-the-loop; bench and tethered checks; supervised open-space tests; and a final demo rehearsal. Record false positives, missed detections, latency percentiles, miss distance, energy, and every safety intervention.

## 12. Pitch and evidence package

Package the three-layer brain story, a replayable demo, architecture diagram, measured plots, safety envelope, and an explicit limitations slide. Show reflex latency, instinctive predictive interception, and judgment as separate traces. Report what is simulated, what is hardware-tested, and what remains stretch work; never imply that a placeholder is flight-ready.

## Real-physics tuning checklist

Use gravity `9.81` m/s². Model thrust as `T = k_T*omega^2` and include first-order motor lag of `0.05-0.1s`. Target an inner loop of `250-500Hz` and guidance at `20-50Hz`. Inject sensor noise, `10-50ms` latency, and wind gusts. Match the real quad mass and thrust-to-weight ratio before transferring any controller or policy. Record all units, coordinate frames, saturations, and sign conventions.
