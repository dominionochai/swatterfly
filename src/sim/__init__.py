"""Stage-1 simulation primitives for Swatterfly.

TODO: add deterministic scenario fixtures and property-based tests.
"""

from .point_mass import PointMassState, step_point_mass

__all__ = ["PointMassState", "step_point_mass"]
