# Dragonfly predictive interception math

Dragonfly pursuit is used here as inspiration for predictive body alignment, not as evidence that an insect implements a literal missile-guidance law. Mischiati et al. (2015) found evidence for a predictive internal model: the animal aligns its body to the future path of prey rather than merely pointing at the prey's current position.

## Relative geometry

Let `r = p_target - p_pursuer` and `v_rel = v_target - v_pursuer`. With `r_x`, `r_y` as planar components and `R = |r|`, define line-of-sight angle and rate as

```text
lambda     = atan2(r_y, r_x)
lambda_dot = (r_x*v_rel_y - r_y*v_rel_x)/R^2
```

The closing speed is

```text
Vc = -(r dot v_rel)/R
```

A proportional-navigation-style command is

```text
a_PN = N*Vc*lambda_dot
```

where `N` is a dimensionless navigation constant. The PN law `a = N*Vc*lambda_dot` already has units of `m/s^2`: `Vc` is `m/s` and `lambda_dot` is `1/s`.

## Lead pursuit

For a constant-velocity target, a simple lead point is

```text
p_aim = p_target + v_target*T
```

where `T` is selected so that `|r + v_rel*T| = |v_pursuer*T|`; choose the positive, physically useful root and reject it when it is not well-conditioned.

A lead angle cannot be added directly to an acceleration command: an angle is dimensionless, whereas the command is `m/s^2`. Convert the angular error through a time scale and speed:

```text
a_lead = (V/T_L)*(lambda_L - psi)
```

`V` is pursuer speed in `m/s`, `T_L` is lead-response time in seconds, `lambda_L` is the lead line-of-sight angle, and `psi` is pursuer heading; therefore `a_lead` is `m/s^2`.

A dimensionally consistent blended command is

```text
a_cmd = w_PN*N*Vc*lambda_dot
      + w_L*(V/T_L)*(lambda_L - psi)
```

where `w_PN` and `w_L` are dimensionless weights. The weights may be scheduled by range, track quality, closing speed, and mission mode, but both terms remain acceleration commands. A heading-rate command, if desired, must be kept as a separate output with its own units and conversion.

## Guidance comparison

| Method | Strength | Weakness | Swatterfly role |
|---|---|---|---|
| Pure pursuit | Simple, intuitive, easy to visualize | Can cut inside a turn, create lag and larger miss distance | First baseline and fallback |
| Constant bearing | Interception intuition from maintaining line-of-sight direction | Sensitive to noisy derivatives and poor observability; does not alone specify acceleration | Diagnostic baseline |
| Proportional navigation | Uses line-of-sight rate and closing speed to shape an intercept | Needs reliable relative velocity, saturation handling, and safety envelopes | Main predictive guidance experiment |
| PN + lead blend | Combines dynamic correction with an explicit future aim point | More parameters and possible mode-transition discontinuities | Candidate final research controller |

The navigation constant is a starting range for simulation, not a universal tuning value. Acceleration, turn-rate, tilt, thrust, geofence, and target-class limits must be applied downstream.

## Safety and assumptions

- Trigger a safety override when `tau < tau_emergency`; the override must reduce risk and cannot be disabled by the model layer.
- Relative velocity must be expressed in a consistent frame, with ego motion and camera rotation compensated.
- Guidance output is a desired acceleration or heading-rate request, not a direct motor command.
- Track confidence, latency, actuator limits, geofence, and a human abort are first-class inputs.
- Dragonflies do not literally run missile PN. PN is an engineering abstraction used to compare an interpretable controller against pursuit behavior.

## Modeling note: event-camera projection

A simulated projected pixel radius is a small-angle pinhole-projection proxy, not angular size. Any comparison of an angular quantity such as `lambda`, `theta`, or an `eta`-style threat score with projected radius must state the projection approximation and camera geometry used to convert pixels to angles.

## References

- Gabbiani, Krapp & Laurent (1999), *J Neurosci* 19:1122. DOI: [10.1523/JNEUROSCI.19-03-01122.1999](https://doi.org/10.1523/JNEUROSCI.19-03-01122.1999).
- Gabbiani, Krapp, Laurent & Koch (2001), *J Neurosci* 21:314. DOI: [10.1523/JNEUROSCI.21-01-00314.2001](https://doi.org/10.1523/JNEUROSCI.21-01-00314.2001).
- Lee (1976), *Perception* 5:437. DOI: [10.1068/p050437](https://doi.org/10.1068/p050437).
- Rind & Simmons (1998), *J Comp Neurol* 395:405. DOI: [10.1002/(SICI)1096-9861(19980808)395:3<405::AID-CNE9>3.0.CO;2-6](https://doi.org/10.1002/%28SICI%291096-9861%2819980808%29395%3A3%3C405%3A%3AAID-CNE9%3E3.0.CO%3B2-6).
- Fossen & Breivik (2008), *Guidance Laws for Planar Motion Control*.
