# Dragonfly predictive interception math

Dragonfly pursuit is used here as an inspiration for predictive body alignment, not as evidence that an insect implements a literal missile guidance law. Mischiati et al. (2015) found evidence for a predictive internal model: the animal aligns its body to the future path of prey rather than merely pointing at the prey's current position.

## Relative geometry

Let `r = p_target - p_pursuer` and let `v_rel = v_target - v_pursuer`. With `r_x`, `r_y` as planar components and `R = |r|`, define line-of-sight angle and rate as

```text
lambda = atan2(r_y,r_x);
lambda_dot = (r_x*v_rel_y - r_y*v_rel_x)/R^2;
```

The closing speed is

```text
Vc = -(r·v_rel)/R;
```

A proportional-navigation-style command is

```text
a_cmd = N*Vc*lambda_dot with N=3-5;
heading_rate_cmd = N*lambda_dot;
```

The navigation constant is a starting range for simulation, not a universal tuning value. Acceleration, turn-rate, tilt, thrust, geofence, and target-class limits must be applied downstream.

## Lead pursuit

For a constant-velocity target, a simple lead point is

```text
p_aim = p_target + v_target*T
```

where `T` is selected so that

```text
|r+v_rel*T| = V_pursuer*T;
```

The positive, physically useful root must be selected and rejected when it is not well-conditioned. A blended command can be represented as

```text
a_cmd = w*a_PN + (1-w)*a_lead;
```

with `w` scheduled by range, track quality, closing speed, and mission mode.

## Guidance comparison

| Method | Strength | Weakness | Swatterfly role |
|---|---|---|---|
| Pure pursuit | Simple, intuitive, easy to visualize | Can cut inside a turn, create lag and larger miss distance | First baseline and fallback |
| Constant bearing | Interception intuition from maintaining line-of-sight direction | Sensitive to noisy derivatives and poor observability; does not alone specify acceleration | Diagnostic baseline |
| Proportional navigation | Uses line-of-sight rate and closing speed to shape an intercept | Needs reliable relative velocity, saturation handling, and safety envelopes | Main predictive guidance experiment |
| PN + lead blend | Combines a dynamic correction with an explicit future aim point | More parameters and possible mode-transition discontinuities | Candidate final research controller |

## Safety and assumptions

- Trigger a safety override when `tau < tau_emergency`; the override must reduce risk and cannot be disabled by the model layer.
- Relative velocity must be expressed in a consistent frame, with ego motion and camera rotation compensated.
- Guidance output is a desired acceleration or heading-rate request, not a direct motor command.
- Track confidence, latency, actuator limits, geofence, and a human abort are first-class inputs.
- Dragonflies do not literally run missile PN. PN is an engineering abstraction used to compare an interpretable controller against pursuit behavior.
