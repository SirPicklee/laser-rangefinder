"""End-to-end range performance: probability of a successful range reading vs distance.

One shot: aiming error (constant during a pulse burst) -> beam on target -> atmosphere ->
Lambertian target -> receiver -> log-normal scintillation (independent per pulse) ->
N pulses accumulated (SNR_N = sqrt(N) * mean single-pulse SNR) -> noise -> threshold (Pfa per window).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .atmosphere import lognormal_samples, scintillation_index
from .detection import detection_probability_fading, echo_peak_current, threshold_for_pfa
from .detector import solar_background
from .linkbudget import LRF, Target, received_energy
from .waveform import Receiver, noise_rms


@dataclass(frozen=True)
class System:
    name: str
    lrf: LRF
    rx: Receiver
    n_pulses: int = 1
    window: float = 110e-6  # listening window [s]
    pfa: float = 1e-3  # false alarm probability per measurement
    sun_per_nm: float = 0.25  # solar spectral irradiance at the wavelength [W/m^2/nm]
    filter_nm: float = 10.0
    fov: float = 0.75e-3  # receiver full field of view [rad]


@dataclass(frozen=True)
class Conditions:
    visibility_km: float = 23.0
    cn2: float = 0.0
    aiming_sigma: float = 0.0  # aiming error per axis [rad]
    daylight: bool = False
    target: Target = field(default_factory=Target)


def background_power(system: System, cond: Conditions) -> float:
    if not cond.daylight:
        return 0.0
    lrf = system.lrf
    return solar_background(system.sun_per_nm, system.filter_nm, cond.target.reflectivity, lrf.rx_area,
                            system.fov, lrf.rx_optics)


def threshold(system: System) -> float:
    return threshold_for_pfa(system.rx.bandwidth, system.window, system.pfa)


def snr_per_joule(system: System, cond: Conditions) -> float:
    """Peak SNR of a single echo per joule of received energy."""
    lrf, rx = system.lrf, system.rx
    peak = echo_peak_current(rx.apd.responsivity(lrf.wavelength), rx.gain, 1.0, lrf.pulse_width, rx.bandwidth)
    return peak / noise_rms(rx, lrf.wavelength, background_power(system, cond))


def mean_snr(system: System, cond: Conditions, distance: float, dx=0.0, dy=0.0):
    """Single-pulse peak SNR without fading."""
    e = received_energy(system.lrf, cond.target, distance, cond.visibility_km, dx, dy)
    return e * snr_per_joule(system, cond)


def pd_analytic(system: System, cond: Conditions, distance: float) -> float:
    """Pd without aiming error (fading averaged over N pulses: index / N, log-normal approximation)."""
    if cond.aiming_sigma > 0:
        raise ValueError("the analytic model has no aiming error; use pd_monte_carlo")
    n = system.n_pulses
    snr = float(mean_snr(system, cond, distance)) * math.sqrt(n)
    idx = float(scintillation_index(cond.cn2, system.lrf.wavelength, distance)) if cond.cn2 > 0 else 0.0
    return detection_probability_fading(snr, threshold(system), idx / n)


def pd_monte_carlo(system: System, cond: Conditions, distance: float, n_trials: int,
                   rng: np.random.Generator) -> float:
    """Fraction of successful range readings over n_trials simulated measurements."""
    n = system.n_pulses
    s = cond.aiming_sigma * distance
    dx = rng.normal(0.0, s, (n_trials, 1)) if s > 0 else np.zeros((n_trials, 1))
    dy = rng.normal(0.0, s, (n_trials, 1)) if s > 0 else np.zeros((n_trials, 1))
    snr = np.broadcast_to(mean_snr(system, cond, distance, dx, dy), (n_trials, n))
    if cond.cn2 > 0:
        idx = float(scintillation_index(cond.cn2, system.lrf.wavelength, distance))
        snr = snr * lognormal_samples(idx, n_trials * n, rng).reshape(n_trials, n)
    snr_acc = math.sqrt(n) * snr.mean(axis=1)
    detected = snr_acc + rng.standard_normal(n_trials) > threshold(system)
    return float(detected.mean())


def range_at_pd(distances, pd_values, level: float = 0.9) -> float:
    """Distance at which Pd first falls below the level, linearly interpolated (nan if never)."""
    d, p = np.asarray(distances, float), np.asarray(pd_values, float)
    below = np.nonzero(p < level)[0]
    if below.size == 0:
        return math.nan
    i = below[0]
    if i == 0:
        return float(d[0])
    return float(np.interp(level, [p[i], p[i - 1]], [d[i], d[i - 1]]))