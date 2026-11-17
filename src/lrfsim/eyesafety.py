"""Laser eye safety, simplified IEC 60825-1 / ANSI Z136.1 style.

Educational model, NOT a certification tool. Two bands are modelled:
  700-1050 nm  : retinal thermal hazard (e.g. 905 nm diode)
  1500-1800 nm : cornea / lens hazard, the "eye-safe" band (e.g. 1550 nm Er:glass)
Exposure = radiant exposure H [J/m^2] averaged over the limiting aperture.
Repetitive pulses: (a) single pulse, (b) average over the exposure time,
(c) pulse train MPE * N^-0.25 (retina only, conservative older rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq

from .beam import Transmitter

EXPOSURE_TIME = 10.0  # [s] accidental exposure to an invisible beam


def _nm(wavelength: float) -> float:
    return wavelength * 1e9


def retinal_band(wavelength: float) -> bool:
    return 700.0 <= _nm(wavelength) < 1050.0


def cornea_band(wavelength: float) -> bool:
    return 1500.0 <= _nm(wavelength) <= 1800.0


def c4(wavelength: float) -> float:
    if not retinal_band(wavelength):
        raise ValueError("C4 is only modelled for 700-1050 nm")
    return 10 ** (0.002 * (_nm(wavelength) - 700.0))


def mpe_single(wavelength: float, duration: float) -> float:
    """MPE for one exposure of the given duration [J/m^2] (valid 1 ns .. 10 s)."""
    if not 1e-9 <= duration <= 10.0:
        raise ValueError("duration must be between 1 ns and 10 s")
    if retinal_band(wavelength):
        k = c4(wavelength)
        return 2e-3 * k if duration < 5e-6 else 18.0 * k * duration**0.75
    if cornea_band(wavelength):
        return 1e4
    raise ValueError("wavelength band not modelled")


def limiting_aperture(wavelength: float, duration: float) -> float:
    """Diameter [m] over which the exposure is averaged."""
    if retinal_band(wavelength):
        return 7e-3
    if cornea_band(wavelength):
        return 1e-3 if duration <= 0.35 else 1.5e-3 * duration**0.375
    raise ValueError("wavelength band not modelled")


@dataclass(frozen=True)
class OpticalAid:
    aperture: float = 50e-3  # entrance diameter [m]
    transmission: float = 0.7
    magnification: float = 7.0


CLASS1_TELESCOPE = OpticalAid(50e-3, 1.0, 7.0)


def aperture_fraction(w: float, diameter: float) -> float:
    """Fraction of a Gaussian beam (1/e^2 radius w) through a centred circular aperture."""
    return 1.0 - math.exp(-(diameter**2) / (2.0 * w**2))


def aperture_exposure(energy: float, w: float, diameter: float, aid: OptionalAid | None = None) -> float:
    """Radiant exposure averaged over the limiting aperture [J/m^2]."""
    if aid is None:
        return energy * aperture_fraction(w, diameter) / (math.pi * diameter**2 / 4)
    captured = energy * aperture_fraction(w, aid.aperture) * aid.transmission
    spot = max(aid.aperture / aid.magnification, diameter)  # exit pupil or limiting aperture
    return captured / (math.pi * spot**2 / 4)


def exposure_ratios(tx: Transmitter, energy: float, pulse_width: float, distance: float, prf: float = 0.0,
                    exposure_time: float = EXPOSURE_TIME, aid: OpticalAid | None = None) -> dict:
    """Exposure / MPE for each applicable rule (> 1 means hazardous)."""
    lam = tx.wavelength
    w = float(tx.radius_at(distance))
    d_a = limiting_aperture(lam, pulse_width)
    ratios = {"single pulse": aperture_exposure(energy, w, d_a, aid) / mpe_single(lam, pulse_width)}
    n = prf * exposure_time
    if n > 1:
        d_b = limiting_aperture(lam, exposure_time)
        ratios["average"] = n * aperture_exposure(energy, w, d_b, aid) / mpe_single(lam, exposure_time)
        if retinal_band(lam):
            ratios["pulse train"] = ratios["single pulse"] * n**0.25
    return ratios


def exposure_ratio(tx: Transmitter, energy: float, pulse_width: float, distance: float, prf: float = 0.0,
                   exposure_time: float = EXPOSURE_TIME, aid: OpticalAid | None = None) -> float:
    return max(exposure_ratios(tx, energy, pulse_width, distance, prf, exposure_time, aid).values())


def nohd(tx: Transmitter, energy: float, pulse_width: float, prf: float = 0.0,
         exposure_time: float = EXPOSURE_TIME, aid: OpticalAid | None = None, z_max: float = 1e6) -> float:
    """Nominal ocular hazard distance [m]: where exposure / MPE falls to 1."""

    def f(z):
        return math.log(exposure_ratio(tx, energy, pulse_width, z, prf, exposure_time, aid))

    if f(0.0) <= 0:
        return 0.0
    if f(z_max) > 0:
        return math.inf
    return brentq(f, 0.0, z_max, xtol=1e-3)


def max_class1_energy(tx: Transmitter, pulse_width: float, prf: float = 0.0,
                      exposure_time: float = EXPOSURE_TIME) -> float:
    """Largest pulse energy [J] that keeps both simplified Class 1 conditions below the MPE:
    naked eye at 100 mm and a 50 mm telescope at 2 m."""
    worst = max(exposure_ratio(tx, 1.0, pulse_width, 0.1, prf, exposure_time),
                exposure_ratio(tx, 1.0, pulse_width, 2.0, prf, exposure_time, CLASS1_TELESCOPE))
    return 1.0 / worst