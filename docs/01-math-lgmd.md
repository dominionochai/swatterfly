# LGMD looming math

This note defines a compact geometric and network-level model for a looming detector. It is a design and implementation guide, not a claim that the scaffold reproduces a biological neuron exactly.

## Geometry and expansion

Assume a circular target of half-size `l` approaches a camera along its optical axis. Let `x(t)` be range and let `u = -dx/dt > 0` be positive closing speed. Use `theta` for the angular half-size of the target:

```text
theta(t) = arctan(l/x(t))
full angular size = 2*theta(t)
theta_dot(t) = l*u/(x(t)^2 + l^2)
```

For a small angle, the dimensionless geometric expansion ratio is `alpha_geom = l/x` (the quantity sometimes written `alpha = l/x`): it is an angle in radians, hence dimensionless, and is never a length. Do not confuse this instantaneous geometric ratio with the peak-time threshold convention below, where `alpha` is reserved for `theta_th`.

The half-angle time-to-contact is exactly

```text
tau = theta/theta_dot
    = ((x^2 + l^2)/(l*u)) * arctan(l/x)
```

The shortcut `tau ≈ x/u` is valid only when `l << x` (small angle) and the closing speed is constant. The expression `x*(x^2 + l^2)/(l*u)` is not the exact tau form.

### Delayed peak

If the whole response is delayed, write

```text
y(t) = x(t - delta)
```

so a peak that occurs at `t_p` without the delay occurs at `t_p + delta` with the delay. Thus the delayed firing peak timing is **`t_p + delta`**, not `t_p - delta`. This statement applies when the whole response is delayed. If only inhibition is delayed, the peak must instead be derived from the delayed excitation-inhibition equation; it cannot be shifted by inspection.

### Peak-time convention

Use the explicit convention

```text
alpha = theta_th
```

where `theta_th` is the angular threshold and `alpha` is dimensionless (radians). With `x0` the initial range, `l` the target half-size, `u` the constant positive closing speed, and `delta` a whole-response delay, use

```text
t_peak = (x0 - l/alpha)/u + delta
```

The exact angular version is

```text
x_th = l/tan(theta_th/2)
t_peak = (x0 - l/tan(theta_th/2))/u + delta
```

The threshold convention is stated here to prevent the old, dimensionally inconsistent `alpha*(l/|v|) - delta` form from being used. The geometric small-angle ratio `l/x` remains dimensionless and should be written `alpha_geom` when it appears alongside the threshold `alpha`.

## Formal definitions and notation

The following block defines the symbols used by the discrete LGMD scaffold in one place. For each spatial element `i` and time `t`,

```text
P+_i(t) = [L_i(t) - L_i(t-dt)]+
[z]+     = max(0, z)                         (rectification)
E(t)     = sum_i w_i P+_i(t)
I(t)     = K_I * sum_i w_i P+_i(t-delta_I)
V(t+dt)  = rho*V(t) + g_E*E(t) - K_I*I(t)
fire     when V(t) >= V_th
```

Here `i` is a spatial sampling location, `t` is time, `dt` is the time step, `L_i(t)` is luminance or log-luminance, `P+_i(t)` is rectified positive temporal change, `[z]+` is the rectifier, `w_i` is the spatial weight, `E(t)` is excitation, `I(t)` is delayed inhibition, `K_I` is inhibitory gain/kernel (with `*` denoting the stated gain or convolution operation), `delta_I` is inhibitory delay, `V(t)` is the leaky membrane-like state, `rho` is leak/retention, `g_E` is excitatory gain, and `V_th` is the firing threshold. The symbol `fire` is the output event.

A more detailed implementation may add spatial kernels, low-pass filters, feed-forward inhibition, normalization, saturation, boundary handling, refractory behavior, and a mapping from pixels or events to `L_i`; these are engineering choices and must be calibrated rather than presented as biological constants.

## Trigger and hysteresis

Use a two-threshold state machine for noise immunity:

```text
fire/engage when V >= V_on
hold       while V > V_off
release    when V <= V_off
with       V_off < V_on
```

An implementation can additionally require a threat measure such as `eta > eta_on` and `tau < tau_on` to engage, and release when `eta < eta_off` or `tau > tau_off`, with `eta_off < eta_on` and `tau_off > tau_on`. Hysteresis is an engineering/noise-immunity layer. The Gabbiani biological model is a threshold-crossing model; the ON/OFF pair is not attributed to that biology.

## Rind-Bramwell P/E/I/S network

Let `L_i[k]` be the luminance or log-luminance sample at spatial location `i` and time index `k`. A compact discrete network is:

```text
P_i[k]       = L_i[k] - L_i[k-1]
P+_i[k]      = max(0, P_i[k])
E[k]         = LPF_E(P+)
I[k]         = LPF_I(sum_i K_I*P+_i plus delayed terms)
S[k]         = max(0, E[k] - I[k])
V[k]         = sum_i S_i[k] - w_FFI*FFI[k]
output[k]    = g(V[k] - V_th)
```

Here `P+` denotes rectified positive temporal change, `K_I` is an inhibitory spatial kernel, `LPF_E` and `LPF_I` are low-pass filters, `FFI` is feed-forward inhibition, `w_FFI` is its weight, and `g` is a threshold/nonlinearity. This P/E/I/S notation is an implementation scaffold; filter time constants, saturation, normalization, and pixel/event mapping must be specified for a concrete implementation.

## Assumptions

1. The target is approximated by a circle or equivalent angular extent; real silhouettes are not circular.
2. Optical-axis closing is used for the clean `x/u` result. Lateral motion, camera motion, rolling shutter, and ego-rotation require compensation.
3. `u` is positive for closing, `theta_dot` is measured or estimated after filtering, and `delta` includes sensor and processing latency.
4. `alpha`, thresholds, filter constants, and `w_FFI` are calibration parameters, not biological facts to copy blindly.
5. Event-camera polarity, refractory behavior, contrast thresholds, and lighting changes need explicit tests.

## Modeling note: event-camera projection

In a simulation, a projected pixel radius is a small-angle pinhole-projection proxy, not an angular size by itself. Any comparison between `theta` (or an `eta` derived from it) and a simulated projected radius must explicitly state that projection approximation and its camera geometry; do not treat pixel radius as an angle without that conversion.

## Caveats and verified facts

- Reported angular thresholds and threshold-to-peak latencies are biological/experimental ranges, not acceptance criteria for a flight controller.
- Looming responses are reported as robust across object size, speed, and lighting in relevant experimental settings; an engineering detector still needs its own robustness evaluation.
- The evidence base is from locusts rather than flies. In project language, say insect looming neuron when generalizing, and do not present every result as a fly-specific measurement.
- The scalar `eta` baseline is intentionally simpler than the full P/E/I/S network. A Python port of the network belongs after the scalar baseline and simulation checks.
- Units and sign conventions must be tested with synthetic approaches before any hardware-in-the-loop experiment.

## References

- Gabbiani, Krapp & Laurent (1999), *J Neurosci* 19:1122. DOI: [10.1523/JNEUROSCI.19-03-01122.1999](https://doi.org/10.1523/JNEUROSCI.19-03-01122.1999).
- Gabbiani, Krapp, Laurent & Koch (2001), *J Neurosci* 21:314. DOI: [10.1523/JNEUROSCI.21-01-00314.2001](https://doi.org/10.1523/JNEUROSCI.21-01-00314.2001).
- Lee (1976), *Perception* 5:437. DOI: [10.1068/p050437](https://doi.org/10.1068/p050437).
- Rind & Simmons (1998), *J Comp Neurol* 395:405. DOI: [10.1002/(SICI)1096-9861(19980808)395:3<405::AID-CNE9>3.0.CO;2-6](https://doi.org/10.1002/%28SICI%291096-9861%2819980808%29395%3A3%3C405%3A%3AAID-CNE9%3E3.0.CO%3B2-6).
- Fossen & Breivik (2008), *Guidance Laws for Planar Motion Control*.
