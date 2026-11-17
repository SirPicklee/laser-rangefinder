"""lrfsim: physics-based laser rangefinder simulation."""

from .tof import (
    C0,
    N_AIR,
    max_unambiguous_range,
    measure,
    range_bin,
    range_from_tof,
    tof_from_range,
    two_target_separation,
)

__all__ = [
    "C0",
    "N_AIR",
    "max_unambiguous_range",
    "measure",
    "range_bin",
    "range_from_tof",
    "tof_from_range",
    "two_target_separation",
]
__version__ = "0.1.0"