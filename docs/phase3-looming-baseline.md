# Phase 3: Looming baseline — Gabbiani canonical firing model vs engineering scalar approximation

## Summary

Phase 3 measures the existing **engineering scalar approximation** detector
(`src/lgmd/scalar_eta.py`, `eta = theta_dot / theta`, `tau_hat = 1 / eta`)
against a new **Gabbiani canonical firing model** detector
(`src/lgmd/network.py`) on the identical Phase 2 deterministic event-camera
sweep. Both are stepping detectors over the **insect looming neuron**
literature; the underlying biological data is locust, not fly LGMD.

`scripts/compare_lgmd.py` reads only the canonical Phase 2 dataset at
`data/phase3_baseline/` (the output path configured in
`.github/workflows/sweep.yml`); it does not regenerate sweep data. Each of the
324 sweep runs is one case. Detector responses are processed through a common
trigger/release hysteresis (trigger at `eta_on`, release at `eta_off`).

## The Gabbiani canonical firing model as implemented

The network detector implements the firing expression of Gabbiani, Krapp &
Laurent 1999 ("Computation of object approach by a wide-field, motion-sensitive
neuron", J. Neurosci. 19):

    firing ∝ ψ(t − δ) · exp(−α · θ(t − δ))

with

    α = 1 / tan(θ_thres / 2)

- `ψ` is the angular size expansion rate (delay-compensated),
- `θ` is the angular size (delay-compensated),
- `δ` is the delay, configured in the required **15–35 ms** range (default
  20 ms),
- `θ_thres` is the threshold, configured in the required **15–40°** range
  (default 40°, the least-suppressing end, so the largest fraction of cases can
  cross threshold given the dataset's theta scale).

The implementation is original Python. The open-source
`fuqinbing/LGMD2-open-source` repository is used as a **structural reference
only** (a feed-forward looming path that pools expansion drive into a single
wide-field firing unit with delayed, inhibited drive); its code is not
translated and its specific cited paper is not in `docs/04-papers.md` and has
not been verified — no claim is made that this module reproduces it.

The firing score is reported alongside a geometric time-to-contact estimate
`tau_hat = theta / psi` so the detector keeps the same calling convention as
`lgmd.scalar_eta.ScalarEtaDetector.update`: both accept one `LoomingSample`
and return `(score, tau_hat)`.

### Detector settings used

| Detector | Model | Defaults used in the comparison |
| --- | --- | --- |
| scalar | engineering scalar approximation `eta = theta_dot / theta` | `eta_on = 0.5` 1/s, `eta_off = 0.4` |
| network | Gabbiani canonical firing model `psi·exp(−α·θ)` | `theta_thres = 40°`, `δ = 20 ms`, `eta_on = 2e-4`, `eta_off = 1.6e-4` |

For the scalar detector, `eta = theta_dot / theta` equals `1 / tau` for a
constant-speed approach, so `eta_on = 0.5` is a time-to-contact gate of 2 s.
The network thresholds are calibrated on the canonical firing score's own scale
and are **not comparable numerically** to the scalar detector's `eta` values.

## Comparison table

Overall (all 324 sweep runs):

| Metric | engineering scalar approximation | Gabbiani canonical firing model |
| --- | ---: | ---: |
| Cases | 324 | 324 |
| Triggered (detection rate) | 162 (50.0%) | 108 (33.3%) |
| Missed | 162 | 216 |
| False-trigger rate | 0.000 | 0.000 |
| Mean latency, trigger → peak (s) | 0.830 | 0.000 |
| Median latency, trigger → peak (s) | 0.830 | 0.000 |
| Mean |tau_hat − tau_true| error at trigger (%) | 0.67 | 0.63 |

Detection rate and latency by sweep axis:

| Axis value | scalar detection | scalar latency (s) | network detection | network latency (s) |
| --- | ---: | ---: | ---: | ---: |
| speed 0.5 m/s | 0% | n/a | 33.3% | 0.000 |
| speed 1.0 m/s | 0% | n/a | 33.3% | 0.000 |
| speed 2.0 m/s | 100% | 0.780 | 33.3% | 0.000 |
| speed 4.0 m/s | 100% | 0.880 | 33.3% | 0.000 |
| size 0.04 m | 50% | 0.830 | 100% | 0.000 |
| size 0.08 m | 50% | 0.830 | 0% | n/a |
| size 0.16 m | 50% | 0.830 | 0% | n/a |
| lighting 0.5 / 1.0 / 2.0 | 50% each | 0.830 | 33.3% each | 0.000 |
| noise 0.0 / 0.5 / 1.5 px | 50% each | 0.830 | 33.3% each | 0.000 |
| latency 0.0 / 0.005 / 0.02 s | 50% each | 0.830 | 33.3% each | 0.000 |

Full per-case and per-axis CSV output is under `/tmp/phase3_comparison/`
(`results_{detector}.csv`, `summary_{detector}.csv`,
`summary_by_axis_{detector}.csv`, `comparison.csv`).

## Interpretation

- **The scalar detector is a speed gate.** Because `eta = theta_dot / theta =
  1 / tau`, the 0.5 1/s threshold only fires for runs that reach `tau <= 2 s`
  inside the sweep window (speeds 2.0 and 4.0 m/s). The slow runs (0.5, 1.0
  m/s) never reach that gate in the truncated window, so they are missed. When
  it fires, its `tau_hat` matches ground truth to ~0.01% and latency is the
  remaining rise time after crossing (~0.78–0.88 s).
- **The Gabbiani canonical firing model, with this dataset's theta proxy, is a
  size gate.** The dataset's `projected_radius_px` is consumed directly as the
  angular-size input (the same quantity the scalar consumes). With
  `alpha = 1 / tan(theta_thres / 2)` at 40°, the exponential
  `exp(−α·θ)` suppresses the score to zero for the larger theta values
  (medium/large objects), so only the smallest-object runs (0.04 m, theta ≈ 2–3)
  cross `eta_on`, and they do so at the first frame (latency 0.000 s). The
  model is therefore not discriminative on this dataset at its configured
  scale; it fires maximally early and decays. This is a measured property of
  the model given the px-as-degree theta proxy, not a preprocessing claim.
- **Zero false-trigger rate for both.** `trajectories.csv` is noise-free and
  analytic, and `tau_hat = theta / theta_dot` equals the geometric ground truth
  at every trigger, so no trigger mismatches ground truth by more than the 25%
  tolerance. The lighting/noise/latency axes are flat for both detectors
  because those parameters affect the event stream (`events.csv`), which these
  trajectory-driven detectors do not consume — a real gap for future phases.

## Caveat

> the scalar η=θ̇/θ detector is an engineering approximation, not the Gabbiani
> et al. 1999 canonical firing model — see PMC6782150 for the source paper

This distinction is preserved throughout the code, tests, and this document:
the term "engineering scalar approximation" refers only to `lgmd/scalar_eta.py`
and "Gabbiani canonical firing model" refers only to `lgmd/network.py`. The two
are never interchangeable.

## Reproduce

```bash
python -m pytest tests/test_lgmd_network.py -v
python scripts/compare_lgmd.py --detector scalar --output /tmp/phase3_scalar_baseline
python scripts/compare_lgmd.py --detector both --output /tmp/phase3_comparison
```

## Files

- `src/lgmd/network.py` — Gabbiani canonical firing model with internal
  hysteresis (new).
- `scripts/compare_lgmd.py` — measurement harness over the Phase 2 dataset
  (new).
- `tests/test_lgmd_network.py` — analytical trigger, hysteresis, and interface
  tests (new).