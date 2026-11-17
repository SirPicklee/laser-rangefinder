"""Laser rangefinder link budget (range equation) for a Lambertian target.

P_received = P_tx * T_tx_optics * T_atm(round trip) * F_on_target * (rho / pi) * (A_rx / R^2) * T_rx_optics

F_on_target ~ 1 (beam inside the target)  -> received power falls as 1/R^2 (extended target)
F_on_target ~ 1/R^2 (target inside the beam) -> received power falls as 1/R^4 (point target)
Units: SI (m, s, J, W).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .atmosphere import transmission
from .beam import Transmitter
from .tof import C0

H_PLANCK = 6.62607015e-34  # [J s]


@dataclass(frozen=True)
class LRF:
    tx: Transmitter
    pulse_energy: float  # [J]
    pulse_width: float  # [s]
    rx_aperture: float  # receiver aperture diameter [m]
    tx_optics: float = 0.9  # transmitter optics transmission
    rx_optics: float = 0.8  # receiver optics transmission (lenses + filter)

    @property
    def wavelength(self) -> float:
        return self.tx.wavelength

    @property
    def peak_power(self) -> float:
        """Peak power of a rectangular pulse with the same energy and width."""
        return self.pulse_energy / self.pulse_width

    @property
    def rx_area(self) -> float:
        return math.pi * (self.rx_aperture / 2.0) ** 2


@dataclass(frozen=True)
class Target:
    side: float = 2.3  # square target side [m] (NATO standard target)
    reflectivity: float = 0.3  # Lambertian reflectivity


def link_terms(lrf: LRF, target: Target, distance, visibility_km: float, dx=0.0, dy=0.0) -> dict:
    """Every multiplicative factor of the range equation."""
    R = np.asarray(distance, dtype=float)
    return {
        "transmitter optics": lrf.tx_optics,
        "atmosphere (round trip)": transmission(R, visibility_km, lrf.wavelength),
        "beam on target": lrf.tx.fraction_on_target(R, target.side, dx, dy),
        "Lambertian target rho/pi": target.reflectivity / math.pi,
        "receiver solid angle A/R^2": lrf.rx_area / R**2,
        "receiver optics": lrf.rx_optics,
    }


def total_factor(lrf: LRF, target: Target, distance, visibility_km: float, dx=0.0, dy=0.0):
    out = 1.0
    for v in link_terms(lrf, target, distance, visibility_km, dx, dy).values():
        out = out * v
    return out


def received_power(lrf: LRF, target: Target, distance, visibility_km: float, dx=0.0, dy=0.0):
    """Peak optical power on the detector [W]."""
    return lrf.peak_power * total_factor(lrf, target, distance, visibility_km, dx, dy)


def received_energy(lrf: LRF, target: Target, distance, visibility_km: float, dx=0.0, dy=0.0):
    """Optical energy per pulse on the detector [J]."""
    return lrf.pulse_energy * total_factor(lrf, target, distance, visibility_km, dx, dy)


def photons(energy, wavelength: float):
    """Number of photons in the given energy."""
    return np.asarray(energy, dtype=float) * wavelength / (H_PLANCK * C0)


def link_budget_db(lrf: LRF, target: Target, distance: float, visibility_km: float) -> dict:
    """Every factor in dB (negative = loss) for one distance."""
    return {k: 10.0 * math.log10(float(v)) for k, v in link_terms(lrf, target, distance, visibility_km).items()}