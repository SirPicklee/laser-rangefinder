"""Atmospheric effects on a laser rangefinder beam.

Extinction: Beer-Lambert law with the aerosol extinction coefficient estimated from the
meteorological visibility (Koschmieder 3.912 / V) and the Kim wavelength model.
Turbulence: Rytov variance and the Andrews-Phillips plane-wave scintillation index
(weak to strong fluctuations), log-normal intensity statistics.
Ranges in metres, visibility in km, wavelengths in metres.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

KOSCHMIEDER = 3.912  # -ln(0.02): 2 % contrast threshold of the eye


def kim_q(visibility_km: float) -> float:
    """Wavelength exponent q of the Kim model."""
    v = float(visibility_km)
    if v > 50:
        return 1.6
    if v > 6:
        return 1.3
    if v > 1:
        return 0.16 * v + 0.34
    if v > 0.5:
        return v - 0.5
    return 0.0


def extinction(visibility_km: float, wavelength: float) -> float:
    """Aerosol extinction coefficient [1/m]."""
    sigma_550 = KOSCHMIEDER / (visibility_km * 1e3)
    return sigma_550 * (wavelength / 550e-9) ** (-kim_q(visibility_km))


def transmission(distance, visibility_km: float, wavelength: float, round_trip: bool = True):
    """Atmospheric transmission over the given distance (one way or there and back)."""
    k = 2.0 if round_trip else 1.0
    return np.exp(-k * extinction(visibility_km, wavelength) * np.asarray(distance, dtype=float))


def attenuation_db_per_km(visibility_km: float, wavelength: float) -> float:
    """One-way attenuation [dB/km]."""
    return 10.0 * math.log10(math.e) * extinction(visibility_km, wavelength) * 1e3


def rytov_variance(cn2: float, wavelength: float, distance):
    """Plane-wave Rytov variance sigma_R^2 = 1.23 Cn^2 k^(7/6) L^(11/6)."""
    k = 2.0 * math.pi / wavelength
    return 1.23 * cn2 * k ** (7.0 / 6.0) * np.asarray(distance, dtype=float) ** (11.0 / 6.0)


def scintillation_index(cn2: float, wavelength: float, distance):
    """Plane-wave scintillation index valid from weak to strong turbulence (Andrews & Phillips)."""
    s = rytov_variance(cn2, wavelength, distance)
    a = 0.49 * s / (1.0 + 1.11 * s**1.2) ** (7.0 / 6.0)
    b = 0.51 * s / (1.0 + 0.69 * s**1.2) ** (5.0 / 6.0)
    return np.exp(a + b) - 1.0


def lognormal_samples(scint_index: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Normalised intensity samples I/<I> with mean 1 and variance = scintillation index."""
    s2 = math.log1p(scint_index)
    return np.exp(rng.normal(-s2 / 2.0, math.sqrt(s2), n))


def fade_probability(scint_index: float, level: float) -> float:
    """Probability that the intensity drops below level * mean (log-normal)."""
    s2 = math.log1p(scint_index)
    return float(norm.cdf((math.log(level) + s2 / 2.0) / math.sqrt(s2)))