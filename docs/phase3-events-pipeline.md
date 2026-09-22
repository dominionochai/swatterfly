# Phase 3: Event-Driven Looming Detection and Time-to-Contact Pipeline

## Summary

This document reports the implementation and validation of the **event-driven radial optical-flow time-to-contact (TTC) pipeline** (`src/lgmd/events_theta.py`), measured against the existing **trajectory-derived baseline** from Phase 2 across the full 324-case parameter sweep dataset at `data/phase3_baseline/`.

The implementation follows the generative-model approach of the event-based looming literature:
- **Schubert et al. 2025**, IOP Publishing, *"Bio-inspired event-based looming object detection for automotive collision avoidance"*
- The conceptual family of event-based TTC mapping (e.g. ICCV 2023 event-based TTC estimation frameworks).

In accordance with project conventions, this is an original Python implementation that uses the cited literature as a **structural and methodological reference only**, with zero dependency on compiled SNN engines (GeNN, PyTorch, etc.).

---

## Method as Implemented

For each sweep case in `events.csv`, the pipeline executes the following steps:

1. **Temporal Discretization:** The event stream is discretized into fixed-width time bins ($\Delta t = 50\text{ ms}$).
2. **Local Displacement Optical Flow:** For each event $(x_i, y_i, t_i, p_i)$, prior events of matching polarity within history window $\Delta T \le 1.0\text{ s}$ and angular sector $\Delta \phi \le 45^\circ$ from the focus-of-expansion (FOE) are evaluated. The nearest spatial neighbor within radius $\Delta r \le 3.5\text{ px}$ yields local optical flow:
   $$\mathbf{v}_i = \frac{(x_i - x_j, y_i - y_j)}{t_i - t_j}$$
3. **Least-Squares Radial Expansion Fit:** Assuming FOE centered at image optical axis $(c_x, c_y) = (639.5, 359.5)$, the radial optical flow model:
   $$\mathbf{v}(x, y) \approx r(t) \cdot (x - c_x, y - c_y) + \mathbf{b}$$
   is solved via linear least squares for the expansion strength $r(t)$ ($1/\text{s}$) and bias $\mathbf{b}$.
4. **Time-to-Contact Estimation:** Instantaneous time-to-contact is derived per bin:
   $$\hat{\tau}(t) = \frac{1}{r(t)}$$
   For constant-speed approach where $\tau(t) = \tau_0 - t$, each valid bin projects the stream-start TTC:
   $$\hat{\tau}_0 = \hat{\tau}(t) + t$$
   The overall stream estimate $\hat{\tau}_{\text{events}}$ is taken as the median over valid bins, matching the calling contract of `sim.events.estimate_tau(events, scenario)`.
5. **Angular Rate Proxy:** Using the pinhole geometry relation implicit in `trajectories.csv` ($\theta = \text{projected\_radius\_px}$, $\dot{\theta}/\theta = 1/\tau = r$), the angular expansion rate proxy:
   $$\dot{\theta}_{\text{proxy}}(t) = r(t) \cdot \theta_{\text{proxy}}(t)$$
   is output per bin, allowing direct drop-in evaluation by `lgmd.scalar_eta.ScalarEtaDetector` and `lgmd.network.NetworkLgmdDetector`.

---

## Validation Results: Event-Derived vs Trajectory-Derived Accuracy

The complete 324-case parameter sweep (`data/phase3_baseline/`) was evaluated with `scripts/compare_lgmd.py --detector all`.

### Overall Comparison (All 324 Cases)

| Metric | Phase 2 Trajectory Baseline (`tau_hat_s`) | Phase 3 Event-Driven Model (`tau_hat_events_s`) | Ground Truth (`tau_true_s`) |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (%)** | **3.57%** | **15.24%** | 0.00% (reference) |
| **Median Absolute Error (%)** | **1.64%** | **9.54%** | 0.00% (reference) |
| **Valid Estimation Rate** | 100% (324 / 324) | 100% (324 / 324) | 100% (324 / 324) |

The event-driven pipeline recovers ground-truth time-to-contact with **9.54% median error** directly from asynchronous, noisy event coordinates without trajectory or radius annotations.

---

### Accuracy Breakdown Across Parameter Axes

#### 1. Sensor Noise Axis (`sensor_noise_px`)

| Noise Level | Trajectory Baseline Mean Error (%) | Trajectory Median Error (%) | Event-Driven Mean Error (%) | Event-Driven Median Error (%) |
| :--- | :---: | :---: | :---: | :---: |
| **0.0 px (noise-free)** | 1.70% | 0.79% | **7.54%** | **5.10%** |
| **0.5 px** | 2.27% | 1.08% | **12.26%** | **10.04%** |
| **1.5 px** | 6.74% | 3.60% | **25.93%** | **22.54%** |

#### 2. Lighting / Event Density Axis (`lighting`)

| Lighting Multiplier | Trajectory Baseline Mean Error (%) | Trajectory Median Error (%) | Event-Driven Mean Error (%) | Event-Driven Median Error (%) |
| :--- | :---: | :---: | :---: | :---: |
| **0.5 (sparse events)** | 3.86% | 1.90% | **18.17%** | **13.82%** |
| **1.0 (nominal)** | 4.17% | 2.12% | **15.45%** | **10.02%** |
| **2.0 (dense events)** | 2.68% | 0.94% | **12.11%** | **7.14%** |

#### 3. Object Size Axis (`object_size_m`)

| Target Size | Trajectory Baseline Mean Error (%) | Trajectory Median Error (%) | Event-Driven Mean Error (%) | Event-Driven Median Error (%) |
| :--- | :---: | :---: | :---: | :---: |
| **0.04 m** | 6.44% | 4.21% | **18.56%** | **16.33%** |
| **0.08 m** | 3.15% | 1.12% | **13.31%** | **7.65%** |
| **0.16 m** | 1.13% | 0.66% | **13.86%** | **6.96%** |

#### 4. Approach Speed Axis (`approach_speed_mps`)

| Speed | Ground Truth Initial $\tau_0$ | Trajectory Mean Error (%) | Event-Driven Mean Error (%) | Event-Driven Median Error (%) |
| :--- | :---: | :---: | :---: | :---: |
| **0.5 m/s** | 12.0 s | 2.97% | **18.11%** | **18.26%** |
| **1.0 m/s** | 6.0 s | 2.74% | **10.45%** | **6.96%** |
| **2.0 m/s** | 3.0 s | 4.39% | **15.01%** | **11.24%** |
| **4.0 m/s** | 1.5 s | 4.19% | **17.40%** | **8.29%** |

#### 5. Injected Sensor Latency Axis (`injected_latency_s`)

| Injected Latency | Trajectory Baseline Mean Error (%) | Event-Driven Mean Error (%) | Event-Driven Median Error (%) |
| :--- | :---: | :---: | :---: |
| **0.0 s** | 3.45% | **14.19%** | **8.35%** |
| **0.005 s** | 3.70% | **15.58%** | **10.37%** |
| **0.02 s** | 3.56% | **15.97%** | **10.04%** |

---

## Detailed Analysis: Where and Why Event-Driven Accuracy Diverges

1. **High Sensor Noise ($1.5\text{ px}$):**
   - *Observation:* Mean error increases from **7.54%** at 0 noise to **25.93%** at 1.5 px noise (median **22.54%**).
   - *Cause:* Event cameras report discrete integer pixel coordinates. When a target is small (e.g. radius $2\text{--}5\text{ px}$), a $1.5\text{ px}$ spatial perturbation represents $30\%\text{--}75\%$ of the object's radius. This distorts the spatial displacement vector between matched events, producing noisy instantaneous velocity candidates.
2. **Event Sparsity at Low Lighting ($0.5\times$):**
   - *Observation:* Error increases to **18.17%** mean (**13.82%** median) under low lighting, whereas high lighting ($2.0\times$) achieves **12.11%** mean (**7.14%** median).
   - *Cause:* Low lighting reduces event count ($N \approx 46\text{--}150$ per run). With fewer events per unit time, time gaps between nearest neighbors widen, making the constant-velocity assumption between successive event pairs less accurate.
3. **Small Object Size ($0.04\text{ m}$):**
   - *Observation:* Smallest targets exhibit **18.56%** mean error vs **13.86%** for large targets ($0.16\text{ m}$, median **6.96%**).
   - *Cause:* At $6\text{ m}$ initial distance with $f=640\text{ px}$, projected radius is only $2.13\text{ px}$. Integer quantization on a 2-pixel contour severely degrades spatial derivative estimation until the object looms closer.
4. **Trigger Threshold Behavior on Raw Events:**
   - When fed into the scalar detector hysteresis gate ($\eta_{\text{on}} = 0.5$, corresponding to $\tau \le 2\text{ s}$), the event pipeline achieves a **96.6% detection rate** (313 / 324 cases), compared to 50.0% for the trajectory scalar baseline.
   - However, false-trigger rate is **93.9%** under the fixed scalar gate because transient event noise spikes briefly cross $\eta_{\text{on}}$ at long range ($\tau_{\text{true}} > 10\text{ s}$).
   - *Biological implication:* This finding directly validates the architectural necessity of wide-field low-pass filtering and delayed lateral inhibition present in biological LGMD neurons (Gabbiani et al. 1999) — instantaneous optical flow must not be gated directly without temporal pooling.

---

## Artifacts Produced

Running `python scripts/compare_lgmd.py --detector all --output <output_dir>` generates:
- `results_events.csv`: Per-case metrics for all 324 runs.
- `summary_events.csv`: Overall aggregate metrics.
- `summary_by_axis_events.csv`: Per-axis detector sensitivity.
- `events_vs_trajectory.csv`: Side-by-side run-by-run comparison of trajectory $\hat{\tau}$ vs event $\hat{\tau}$ vs $\tau_{\text{true}}$.
- `summary_events_vs_trajectory.csv`: Mean and median error comparison table by axis.
- `event_vs_traj_error.png`: Five-panel bar comparison across parameter sweep axes.
- `comparison.csv`: Comparative summary across all 3 detectors (`scalar`, `network`, `events`).

---

## Verification and Reproduction

```bash
# Run unit and integration tests (14 passed)
python -m pytest tests/test_events_theta.py -v
python -m pytest -q

# Run event-driven comparison
python scripts/compare_lgmd.py --detector events --output /tmp/phase3_events

# Run three-way comparison (scalar, network, events)
python scripts/compare_lgmd.py --detector all --output /tmp/phase3_comparison_all
```
