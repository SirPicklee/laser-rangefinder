"""Gaussian laser beam: waist, divergence, spot size at range and power on a square target.

Beam radius w is the 1/e^2 intensity radius. Divergence angles are FULL angles unless the
name says half. Lengths in metres, angles in radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import erf


def rayleigh_range(w0: float, wavelength: float, m2: float = 1.0) -> float:
    """Distance over which the beam area doubles: z_R = pi w0^2 / (M^2 lambda)."""
    return math.pi * w0**2 / (m2 * wavelength)


def beam_radius(z, w0: float, wavelength: float, m2: float = 1.0):
    """1/e^2 radius at distance z from the waist."""
    z = np.asarray(z, dtype=float)
    return w0 * np.sqrt(1.0 + (z / rayleigh_range(w0, wavelength, m2)) ** 2)


def divergence_half_angle(w0: float, wavelength: float, m2: float = 1.0) -> float:
    """Far-field half-angle divergence theta = M^2 lambda / (pi w0)."""
    return m2 * wavelength / (math.pi * w0)


def fraction_on_square(side: float, w, dx=0.0, dy=0.0):
    """Fraction of a Gaussian beam (1/e^2 radius w) that falls on a square target of the given side,
    with the beam centre offset (dx, dy) from the target centre."""
    s = math.sqrt(2.0) / np.asarray(w, dtype=float)

    def axis(d):
        d = np.asarray(d, dtype=float)
        return 0.5 * (erf((side / 2 - d) * s) + erf((side / 2 + d) * s))

    return axis(dx) * axis(dy)


@dataclass(frozen=True)
class Transmitter:
    """Laser transmitter: wavelength, beam waist radius at the exit aperture and beam quality M^2."""

    wavelength: float
    w0: float
    m2: float = 1.0

    def __post_init__(self) -> None:
        if self.m2 < 1.0:
            raise ValueError("M^2 cannot be below 1 (the diffraction limit)")

    @classmethod
    def from_divergence(cls, wavelength: float, w0: float, full_divergence: float) -> "Transmitter":
        """Transmitter with a given full-angle divergence; M^2 follows from the waist size."""
        m2 = math.pi * w0 * (full_divergence / 2.0) / wavelength
        return cls(wavelength, w0, m2)

    @property
    def divergence_half(self) -> float:
        return divergence_half_angle(self.w0, self.wavelength, self.m2)

    @property
    def divergence_full(self) -> float:
        return 2.0 * self.divergence_half

    def radius_at(self, distance):
        return beam_radius(distance, self.w0, self.wavelength, self.m2)

    def fraction_on_target(self, distance, side: float, dx=0.0, dy=0.0):
        return fraction_on_square(side, self.radius_at(distance), dx, dy)


def pointing_fractions(tx: Transmitter, distance: float, side: float, sigma_pointing: float,
                       n: int, rng: np.random.Generator) -> np.ndarray:
    """Fraction on target for n shots with random aiming error (sigma_pointing [rad] per axis)."""
    dx = rng.normal(0.0, sigma_pointing * distance, n)
    dy = rng.normal(0.0, sigma_pointing * distance, n)
    return tx.fraction_on_target(distance, side, dx, dy)



def mean_fraction(tx: Transmitter, distance: float, side: float, sigma_pointing: float):
    """Exact expected fraction on target with Gaussian aiming error (sigma_pointing [rad] per axis).

    A Gaussian beam (sigma_beam = w/2) smeared by a Gaussian aiming error is again Gaussian with
    sigma_eff^2 = (w/2)^2 + (sigma_pointing * distance)^2, so the narrowest beam always wins on average.
    """
    w = np.asarray(tx.radius_at(distance), dtype=float)
    sigma_eff = np.sqrt((w / 2.0) ** 2 + (sigma_pointing * distance) ** 2)
    return erf(side / (2.0 * math.sqrt(2.0) * sigma_eff)) ** 2