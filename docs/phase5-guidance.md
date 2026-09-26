# Phase 5: Guidance Ladder, Confidence Gating, and Emergency Tau Interface

## Summary

Phase 5 implements and benchmarks a planar guidance ladder (`src/guidance/pn_lead.py`) connecting target tracking estimates from Phase 4 directly to intercept control commands. Four guidance laws are implemented with explicit acceleration/turn-rate saturation and tracker confidence gating:
1. **Pure Pursuit (`pure_pursuit`)**: Direct line-of-sight tracking (baseline/fallback).
2. **Constant Bearing (`constant_bearing`)**: Nulls line-of-sight angular rate scaled by vehicle speed.
3. **Proportional Navigation (`proportional_navigation`)**: True PN scaling LOS rate by closing velocity ($a_{\text{cmd}} = N V_c \dot{\lambda}$).
4. **PN + Lead Pursuit Blend (`blend_guidance`)**: Dimensionally consistent combination of dynamic PN acceleration and future geometric aim-point correction.

Additionally, `src/guidance/tau_emergency.py` establishes the non-actuating safety stub with hysteresis for Phase 9's upcoming safety supervisor.

---

## Guidance Laws and Kinematic Formulations

All guidance laws produce a structured `GuidanceCommand` object exposing both pre-clamp `raw_command` and post-clamp `saturated_command` [$\text{m/s}^2$], commanded yaw rate [$\text{rad/s}$], a saturation flag (`is_saturated`), a holding flag (`is_holding`), and the input tracker confidence.

### 1. Pure Pursuit
$$a_{\text{raw}} = \frac{V_p}{\tau_{\text{pursuit}}} \sin(\lambda - \psi)$$
Where $\lambda = \text{atan2}(r_y, r_x)$ is the line-of-sight (LOS) angle, $\psi$ is pursuer heading, $V_p$ is pursuer speed, and $\tau_{\text{pursuit}}$ is pursuit response time constant ($0.5\text{ s}$). Pure pursuit aims directly at the current target position, exhibiting the classic lag and "cutting the corner" path curvature against maneuvering targets.

### 2. Constant Bearing (CB)
$$a_{\text{raw}} = K_{\text{cb}} \cdot V_p \cdot \dot{\lambda}$$
Where $\dot{\lambda} = \frac{r_x v_{\text{rel},y} - r_y v_{\text{rel},x}}{R^2}$. Constant Bearing attempts to drive LOS rotation rate to zero. Unlike PN, it scales with constant vehicle speed $V_p$ rather than closing velocity $V_c$, providing steady angular rate damping.

### 3. Proportional Navigation (PN)
$$a_{\text{raw}} = N \cdot V_c \cdot \dot{\lambda}$$
Where $V_c = -\frac{\mathbf{r} \cdot \mathbf{v}_{\text{rel}}}{R}$ is the instantaneous closing speed, and $N$ is the dimensionless navigation constant ($3.0\text{--}3.5$). PN is the optimal guidance law for non-accelerating targets in the absence of lag, steering the pursuer onto a collision triangle.

### 4. PN + Lead Blend
$$a_{\text{raw}} = (1 - w_L) \cdot a_{\text{PN}} + w_L \cdot \frac{V_p}{T_L} \sin(\lambda_L - \psi)$$
Where $\mathbf{p}_{\text{lead}} = \mathbf{r} + \mathbf{v}_t T_L$, $\lambda_L = \text{atan2}(p_{\text{lead},y}, p_{\text{lead},x})$, and $w_L \in [0, 1]$ is the lead weight. Both components have consistent acceleration units ($\text{m/s}^2$), matching the formulation in `docs/02-math-dragonfly.md` (Eq. 49–50).

---

## Degenerate and Confidence-Gating Policies

### Degenerate Geometry Handling
- **Zero Range ($R \le 10^{-5}\text{ m}$) / Non-Positive Pursuer Speed ($V_p \le 0$):**
  Guidance returns a safe holding command (`is_holding=True`, `raw_command=0.0`, `saturated_command=0.0`) avoiding division by zero or singularity.
- **Zero or Negative Closing Speed ($V_c \le 10^{-5}\text{ m/s}$):**
  When a target is opening (moving away faster than pursuer approaches) or has passed post-intercept, PN ceases aggressive maneuvering and returns `is_holding=True, saturated_command=0.0`.

### Confidence Gating (`min_confidence = 0.40`)
- **Reasoning:** In Phase 4 benchmarks, tracker confidence $\in [0, 1]$ was shown to degrade during high-noise bursts or target acceleration transients. When tracker confidence drops below `min_confidence = 0.40`, guidance commands enter a safe "hold" state (`is_holding=True, saturated_command=0.0`), inhibiting wild erratic maneuvers driven by ghost tracks or high estimation uncertainty.
- In `scripts/compare_guidance.py`, this gate prevented the pursuer from chasing noisy transients during sharp maneuvers.

---

## Benchmark Results: Jinking Target Engagement

The test harness in `scripts/compare_guidance.py` simulated an engagement where the target executes an evasive double-jink maneuver (accelerations of $+6.0\text{ m/s}^2$ and $-6.0\text{ m/s}^2$) with measurement noise $\sigma = 0.20\text{ m}$. Guidance laws operated purely on tracker output.

### Quantitative Comparison (8 Combinations)

| Tracker | Guidance Law | Min Miss Distance (m) | Final Miss Distance (m) | Control Effort (m/s) | Saturation (%) | Holds (steps) | Mean Confidence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **alpha_beta** | pure_pursuit | 5.098 | 36.654 | 2.07 | 0.0% | 141 | 0.278 |
| **alpha_beta** | constant_bearing | 5.097 | 36.688 | 1.87 | 0.0% | 141 | 0.278 |
| **alpha_beta** | proportional_navigation | 5.097 | 37.082 | 0.00 | 0.0% | 149 | 0.278 |
| **alpha_beta** | blend_guidance | 5.098 | 36.917 | 0.86 | 0.0% | 141 | 0.278 |
| **kalman** | pure_pursuit | 3.344 | 6.427 | 46.40 | 38.7% | 0 | 0.833 |
| **kalman** | constant_bearing | 0.241 | 6.213 | 51.19 | 50.0% | 0 | 0.833 |
| **kalman** | **proportional_navigation** | **0.187** | 30.232 | **10.52** | **2.7%** | 81 | 0.833 |
| **kalman** | **blend_guidance** | **0.231** | 21.095 | 29.11 | **2.0%** | 0 | 0.833 |

---

## Tracker Guidance Performance Analysis

Tied directly to Phase 4's findings (`docs/phase4-tracker.md`), the Kalman filter decisively produced superior guidance outcomes compared to the $\alpha$-$\beta$ filter:

1. **Velocity Accuracy and Phase Lag:**
   - In Phase 4, Kalman demonstrated a velocity RMSE of $0.67\text{ px/s}$ compared to $\alpha$-$\beta$'s $3.90\text{ px/s}$ under noise.
   - In closed-loop guidance, velocity estimates feed directly into $\mathbf{v}_{\text{rel}}$ and LOS rate $\dot{\lambda}$. $\alpha$-$\beta$'s noisy velocity estimate under target jinking caused confidence to drop to $0.278$ (below the $0.40$ gate threshold), causing the confidence gate to hold command.
   - When configured with Kalman tracking, PN achieved a **$0.187\text{ m}$ intercept miss distance** with minimal control effort ($10.52\text{ m/s}$) and only $2.7\%$ saturation.

2. **Controller Efficiency:**
   - **Proportional Navigation (PN)** was by far the most efficient law: once the intercept point was reached ($t \approx 1.4\text{ s}$), closing speed reversed ($V_c \le 0$) and PN cleanly held rather than turning to chase an opening target.
   - **Pure Pursuit** suffered severe tail-chase lag, missing by $3.34\text{ m}$ with $38.7\%$ saturation.
   - **Constant Bearing** achieved close miss ($0.241\text{ m}$) but demanded $51.19\text{ m/s}$ control effort and $50\%$ saturation because it lacked closing-speed adaptive attenuation.

---

## Tau Emergency Interface (`src/guidance/tau_emergency.py`)

As specified in Phase 5, `tau_emergency.py` provides the non-actuating stub interface for Phase 9's safety override system:
- **`tau_emergency_check(tau_hat, tau_emergency_threshold, confidence, min_confidence=0.40) -> bool`**: Returns `True` if imminent collision is detected ($\hat{\tau} \le \tau_{\text{emergency}}$) with sufficient tracker confidence.
- **`tau_emergency_release_check(tau_hat, tau_release_threshold, confidence, min_confidence=0.40) -> bool`**: Evaluates hysteresis release ($\tau_{\text{release}} > \tau_{\text{emergency}}$), preventing chattering limit-cycles when evasive maneuvers widen the time-to-contact margin.
- **Note:** This module performs **no physical actuation**; Phase 9 will wire these outputs into the flight supervisor command path.

---

## Verification & Test Commands

```bash
# Run guidance unit tests (analytic intercept, saturation, degenerate handling, hysteresis)
python -m pytest tests/test_guidance_pn_lead.py tests/test_guidance_tau_emergency.py -v

# Run full guidance comparison benchmark
python scripts/compare_guidance.py --output /tmp/phase5_guidance
```
