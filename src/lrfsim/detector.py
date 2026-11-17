"""APD receiver: responsivity, excess noise, noise budget, SNR, optimum gain, solar background.

Current-mode APD followed by a transimpedance amplifier (TIA). Amplitude SNR = peak signal current /
RMS noise current in the electrical bandwidth B. All units SI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .linkbudget import H_PLANCK
from .tof import C0

Q_E = 1.602176634e-19  # electron charge [C]


@dataclass(frozen=True)
class APD:
    name: str
    quantum_efficiency: float
    k_ion: float  # ionisation ratio (Si ~0.02, InGaAs ~0.45)
    dark_bulk: float  # bulk (multiplied) dark current, primary level [A]
    dark_surface: float  # surface (unmultiplied) dark current [A]
    amp_noise: float  # TIA input-referred current noise density [A/sqrt(Hz)]
    max_gain: float = 200.0

    def responsivity(self, wavelength: float) -> float:
        """Unity-gain responsivity R0 = eta q lambda / (h c) [A/W]."""
        return self.quantum_efficiency * Q_E * wavelength / (H_PLANCK * C0)

    def excess_noise(self, gain):
        """McIntyre excess noise factor F(M) = k M + (1 - k)(2 - 1/M)."""
        M = np.asarray(gain, dtype=float)
        return self.k_ion * M + (1.0 - self.k_ion) * (2.0 - 1.0 / M)


def bandwidth_for_pulse(pulse_width: float) -> float:
    """Electrical bandwidth matched to the pulse (rule of thumb B = 0.5 / tau)."""
    return 0.5 / pulse_width


def noise_terms(apd: APD, wavelength: float, p_signal, p_background, gain, bandwidth: float) -> dict:
    """Noise current variances [A^2] of every source."""
    r0 = apd.responsivity(wavelength)
    mf = np.asarray(gain, dtype=float) ** 2 * apd.excess_noise(gain)
    return {
        "signal shot": 2 * Q_E * r0 * np.asarray(p_signal, dtype=float) * mf * bandwidth,
        "background shot": 2 * Q_E * r0 * np.asarray(p_background, dtype=float) * mf * bandwidth,
        "bulk dark": 2 * Q_E * apd.dark_bulk * mf * bandwidth,
        "surface dark": 2 * Q_E * apd.dark_surface * bandwidth + 0 * mf,
        "amplifier": apd.amp_noise**2 * bandwidth + 0 * mf,
    }


def snr(apd: APD, wavelength: float, p_signal, p_background, gain, bandwidth: float):
    """Amplitude SNR: peak signal current / RMS noise current."""
    signal = apd.responsivity(wavelength) * np.asarray(gain, dtype=float) * np.asarray(p_signal, dtype=float)
    var = sum(noise_terms(apd, wavelength, p_signal, p_background, gain, bandwidth).values())
    return signal / np.sqrt(var)


def optimum_gain(apd: APD, wavelength: float, p_signal: float, p_background: float,
                 bandwidth: float) -> tuple[float, float]:
    """Gain that maximises the SNR and the SNR there."""
    res = minimize_scalar(lambda M: -float(snr(apd, wavelength, p_signal, p_background, M, bandwidth)),
                          bounds=(1.0, apd.max_gain), method="bounded", options={"xatol": 1e-3})
    return float(res.x), float(-res.fun)


def solar_background(irradiance_per_nm: float, filter_nm: float, reflectivity: float, rx_area: float,
                     fov_full: float, rx_optics: float) -> float:
    """Sunlit Lambertian scene filling the receiver field of view -> background power on the detector [W]."""
    radiance = reflectivity * irradiance_per_nm * filter_nm / math.pi  # [W / m^2 / sr]
    omega = math.pi * (fov_full / 2.0) ** 2  # field-of-view solid angle [sr]
    return radiance * rx_area * omega * rx_optics


# Typical datasheet-like detectors (approximate)
SI_APD = APD("Si APD", quantum_efficiency=0.6, k_ion=0.02, dark_bulk=0.1e-9, dark_surface=1e-9,
             amp_noise=2e-12, max_gain=300)
INGAAS_APD = APD("InGaAs APD", quantum_efficiency=0.8, k_ion=0.45, dark_bulk=2e-9, dark_surface=50e-9,
                 amp_noise=2e-12, max_gain=40)