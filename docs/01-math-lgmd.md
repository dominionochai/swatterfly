# LGMD looming math

This note defines a compact geometric and network-level model for a looming detector. It is a design and implementation guide, not a claim that the scaffold reproduces a biological neuron exactly.

## Geometry and expansion

Assume a circular target of half-size `l` approaches a camera along its optical axis. Let `x(t)` be range and let `u = -dx/dt` be positive closing speed. The angular size is

```text
theta(t) = 2*atan(l/x(t));
```

For the half-angle expansion convention, define

```text
psi = (dtheta/dt)/2;
dtheta/dt = 2*l*u/(x^2+l^2);
```

The dimensionless expansion score can be formed from delayed angular expansion:

```text
firing ∝ psi(t-delta)*exp(-alpha*theta(t-delta));
```

For a common small-angle time-to-contact approximation,

```text
tau = theta/theta_dot ≈ x/u;
theta_threshold = 2*atan(1/alpha);
t_peak = alpha*(l/|v|)-delta;
```

The last two expressions are useful scaling relationships under the stated geometry and a particular alpha/delay parameterization. They are not universal constants.

## Rind–Bramwell P/E/I/S network

Let `L_i[k]` be the luminance or log-luminance sample at spatial location `i` and time index `k`. A compact discrete network is:

```text
P_i = L_i[k]-L_i[k-1],
E = LPF_E(Pplus),
I = LPF_I(sum K_I*Pplus delayed),
S = max(0,E-I),
V = sum(S) - w_FFI*FFI,
output = g(V - V_th);
```

Here `Pplus` denotes the rectified positive temporal change, `K_I` is an inhibitory spatial kernel, `LPF_E` and `LPF_I` are low-pass filters, `FFI` is feed-forward inhibition, and `g` is a threshold/nonlinearity. A practical implementation should specify filter time constants, boundary behavior, normalization, saturation, and the mapping from pixels or events to `L_i`.

## Trigger and hysteresis

Use a two-threshold policy rather than a single noisy crossing. Engage only when both threat measures satisfy

```text
eta > eta_on, tau < tau_on  -> engage
```

and release when either relaxed condition is met:

```text
eta < eta_off or tau > tau_off -> release
```

Choose `eta_off < eta_on` and `tau_off > tau_on` to prevent chatter. A safety supervisor must also be able to inhibit engagement and force a controlled release.

## Assumptions

1. The target is approximated by a circle or an equivalent angular extent; real silhouettes are not circular.
2. Optical-axis closing is used for the clean `x/u` result. Lateral motion, camera motion, rolling shutter, and ego-rotation require compensation.
3. `u` is positive for closing, `theta_dot` is measured or estimated after filtering, and delay `delta` includes sensor and processing latency.
4. `alpha`, thresholds, filter constants, and `w_FFI` are calibration parameters, not biological facts to copy blindly.
5. Event-camera polarity, refractory behavior, contrast thresholds, and lighting changes need explicit tests.

## Caveats and verified facts

- A useful reported angular threshold range is **15–40 degrees**, and reported threshold-to-peak latency is **15–35 ms**. These are ranges from biological/experimental evidence, not acceptance criteria for a flight controller.
- Looming responses are reported as robust across object size, speed, and lighting in relevant experimental settings; the engineering detector still needs its own robustness evaluation.
- The evidence base is from **locust**, not fly. In project language, say **insect looming neuron** when generalizing; do not present every result as a fly-specific measurement.
- The scalar `eta` baseline is intentionally simpler than the full P/E/I/S network. A Python port of LGMD2 belongs after the scalar baseline and simulation checks.
- Units and sign conventions must be tested with synthetic approaches before any hardware-in-the-loop experiment.
