"""Threshold detection of a laser echo in Gaussian receiver noise.

Noise threshold crossings follow Rice's formula: upward crossings of level T*sigma per second
    nu(T) = nu0 exp(-T^2 / 2),   nu0 = sigma_f / sqrt(2) for Gaussian-filtered white noise,
    sigma_f = B / sqrt(ln 2)  (B = -3 dB bandwidth).
False alarm probability in a listening window W: 1 - exp(-nu W).
Detection probability of an echo with peak SNR s: Q(T - s); averaged over log-normal fading if needed.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def zero_crossing_rate(bandwidth: float) -> float:
    """nu0: upward zero crossings per second of Gaussian-filtered white noise."""
    sigma_f = bandwidth / math.sqrt(math.log(2.0))
    return sigma_f / math.sqrt(2.0)


def crossing_rate(bandwidth: float, threshold_sigma):
    """Upward crossings per second of the level threshold_sigma * sigma."""
    return zero_crossing_rate(bandwidth) * np.exp(-np.asarray(threshold_sigma, dtype=float) ** 2 / 2.0)


def false_alarm_probability(bandwidth: float, window: float, threshold_sigma):
    """Probability of at least one noise crossing in a listening window of length window [s]."""
    return 1.0 - np.exp(-crossing_rate(bandwidth, threshold_sigma) * window)


def threshold_for_pfa(bandwidth: float, window: float, pfa: float) -> float:
    """Threshold (in units of the noise RMS) that gives the requested false alarm probability."""
    return math.sqrt(2.0 * math.log(zero_crossing_rate(bandwidth) * window / -math.log1p(-pfa)))


def detection_probability(snr, threshold_sigma):
    """Probability that the echo peak exceeds the threshold: Q(T - SNR)."""
    return norm.sf(np.asarray(threshold_sigma, dtype=float) - np.asarray(snr, dtype=float))


def detection_probability_fading(snr_mean: float, threshold_sigma: float, scint_index: float,
                                 n_quad: int = 2001) -> float:
    """Pd averaged over log-normal fading of the echo power (mean 1, variance = scint_index)."""
    if scint_index <= 0:
        return float(detection_probability(snr_mean, threshold_sigma))
    s2 = math.log1p(scint_index)
    z = norm.ppf((np.arange(n_quad) + 0.5) / n_quad)
    amplitude = np.exp(-s2 / 2.0 + math.sqrt(s2) * z)
    return float(np.mean(detection_probability(snr_mean * amplitude, threshold_sigma)))


def count_upward_crossings(x, level: float) -> int:
    x = np.asarray(x)
    return int(np.sum((x[:-1] < level) & (x[1:] >= level)))


def echo_peak_current(responsivity: float, gain: float, energy: float, pulse_fwhm: float, bandwidth: float) -> float:
    """Peak current of a Gaussian echo after the Gaussian amplifier filter (both Gaussians add in quadrature)."""
    sigma_p = pulse_fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    sigma_t = 1.0 / (2.0 * math.pi * bandwidth / math.sqrt(math.log(2.0)))
    charge = responsivity * gain * energy
    return charge / (math.sqrt(2.0 * math.pi) * math.sqrt(sigma_p**2 + sigma_t**2))