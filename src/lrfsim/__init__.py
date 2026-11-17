"""lrfsim: physics-based laser rangefinder simulation."""

from .beam import (
    Transmitter,
    beam_radius,
    divergence_half_angle,
    fraction_on_square,
    mean_fraction,
    pointing_fractions,
    rayleigh_range,
)
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
    "Transmitter",
    "beam_radius",
    "divergence_half_angle",
    "fraction_on_square",
    "mean_fraction",
    "pointing_fractions",
    "rayleigh_range",
]
__version__ = "0.1.0"