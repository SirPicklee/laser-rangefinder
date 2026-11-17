"""Full-waveform simulation of the LRF receiver.

optical echo P(t) -> APD current R0 M P(t) + noise -> Gaussian amplifier filter -> TIA (A -> V) -> ADC samples

Noise is generated as white current noise with one-sided PSD
    S(t) = 2 q (R0 P(t) + R0 P_b + I_db) M^2 F + 2 q I_ds + i_n^2      [A^2/Hz]
(the signal shot noise only exists while the echo is there). A discrete white sequence with
one-sided PSD S has variance S / (2 dt) per sample. The receiver is AC coupled (DC removed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .detector import APD, Q_E
from .linkbudget import LRF, Target, received_energy
from .tof import tof_from_range


@dataclass(frozen=True)
class Receiver:
    apd: APD
    gain: float
    bandwidth: float  # -3 dB bandwidth of the amplifier chain [Hz]
    transimpedance: float = 20e3  # TIA gain [V/A]
    adc_rate: float = 1e9  # [samples/s]
    adc_bits: int = 12
    adc_full_scale: float = 0.5  # bipolar range: -FS/2 ... +FS/2 [V]
    dt: float = 0.1e-9  # simulation time step [s]


def gaussian_pulse_power(t, t0: float, energy: float, fwhm: float):
    """Optical power of a Gaussian pulse with the given energy and FWHM, centred at t0."""
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return energy / (sigma * math.sqrt(2.0 * math.pi)) * np.exp(-((np.asarray(t) - t0) ** 2) / (2.0 * sigma**2))


def gaussian_filter_taps(bandwidth: float, dt: float) -> np.ndarray:
    """FIR taps of a Gaussian low-pass filter with the given -3 dB bandwidth (unity DC gain)."""
    sigma_f = bandwidth / math.sqrt(math.log(2.0))
    sigma_t = 1.0 / (2.0 * math.pi * sigma_f)
    n = int(math.ceil(5.0 * sigma_t / dt))
    t = np.arange(-n, n + 1) * dt
    h = np.exp(-(t**2) / (2.0 * sigma_t**2))
    return h / h.sum()


def noise_equivalent_bandwidth(h: np.ndarray, dt: float) -> float:
    """B_N such that white noise of one-sided PSD S has variance S * B_N after the filter."""
    return float(np.sum(h**2) / (2.0 * dt))


def noise_psd(rx: Receiver, wavelength: float, p_background: float, p_signal=0.0):
    """One-sided current noise PSD [A^2/Hz] (array if p_signal is an array)."""
    a, M = rx.apd, rx.gain
    F = float(a.excess_noise(M))
    r0 = a.responsivity(wavelength)
    multiplied = 2.0 * Q_E * (r0 * np.asarray(p_signal, dtype=float) + r0 * p_background + a.dark_bulk) * M**2 * F
    return multiplied + 2.0 * Q_E * a.dark_surface + a.amp_noise**2


def noise_rms(rx: Receiver, wavelength: float, p_background: float, p_signal: float = 0.0) -> float:
    """Expected RMS noise current after the filter [A]."""
    bn = noise_equivalent_bandwidth(gaussian_filter_taps(rx.bandwidth, rx.dt), rx.dt)
    return float(np.sqrt(noise_psd(rx, wavelength, p_background, p_signal) * bn))


def simulate_current(rx: Receiver, wavelength: float, t: np.ndarray, p_signal: np.ndarray,
                     p_background: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Filtered APD current: (noiseless, noisy) [A] on the time grid t."""
    r0 = rx.apd.responsivity(wavelength)
    i_sig = r0 * rx.gain * p_signal
    psd = noise_psd(rx, wavelength, p_background, p_signal)
    noise = rng.standard_normal(t.size) * np.sqrt(psd / (2.0 * rx.dt))
    h = gaussian_filter_taps(rx.bandwidth, rx.dt)
    clean = np.convolve(i_sig, h, mode="same")
    noisy = np.convolve(i_sig + noise, h, mode="same")
    return clean, noisy


def digitize(rx: Receiver, t: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TIA + ADC: returns sample times, integer codes and the corresponding voltages."""
    step = max(1, int(round(1.0 / (rx.adc_rate * rx.dt))))
    v = current[::step] * rx.transimpedance
    lsb = rx.adc_full_scale / 2**rx.adc_bits
    codes = np.clip(np.round(v / lsb), -(2 ** (rx.adc_bits - 1)), 2 ** (rx.adc_bits - 1) - 1)
    return t[::step], codes.astype(int), codes * lsb


def echo_power(t: np.ndarray, lrf: LRF, target: Target, distance: float, visibility_km: float):
    """Optical power on the detector for one echo from a target at the given distance."""
    e = float(received_energy(lrf, target, distance, visibility_km))
    return gaussian_pulse_power(t, float(tof_from_range(distance)), e, lrf.pulse_width)