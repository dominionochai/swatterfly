# Source scaffold

Python 3.11+ is the primary language. Each package below is intentionally small and explicit about what is and is not implemented:

- `sim/` — stage-1 point-mass chase simulation; no aerodynamics or flight claims.
- `lgmd/` — scalar eta looming detector, with a seam for a later Python LGMD2 port.
- `tracker/` — alpha-beta tracking placeholder for event-derived target states.
- `guidance/` — proportional-navigation and lead-pursuit helpers, not an actuator controller.
- `dashboard/` — integration notes for the preserved Stitch design.

The modules are safe to import as a scaffold. TODOs identify the work needed before tests, hardware-in-the-loop, or flight use.
