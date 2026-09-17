# Stitch cockpit UI — preserved exactly

The Stitch cockpit UI is preserved exactly as the product requirement. The future TypeScript/Next.js dashboard should implement this surface without moving flight authority into the browser.

- **live event-cam feed with target lock**
- **LOOMING RING growing with theta**
- **tau countdown**
- **eta meter**
- **LGMD spike trace**
- **guidance overlay (LOS, lambda_dot, predicted intercept point, accel vector)**
- **telemetry (closing speed, battery, GPS, threat log)**
- **demo mode replay with slow-mo overlays**

## Interaction and safety notes

Target lock must show confidence and stale-data state. The looming ring should make the angular-size signal legible without suggesting that a ring alone is a certified threat decision. The `tau` countdown and `eta` meter must show units/threshold state. Guidance overlays are explanatory and read-only; they must not be a browser-side actuator path. Telemetry should distinguish measured, estimated, and advisory values. Demo replay must be deterministic, timestamped, and clearly separated from live mode.
