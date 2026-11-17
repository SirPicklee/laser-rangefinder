"""Echo timing: leading-edge threshold, constant-fraction discrimination (CFD), matched filter,
analytic walk of the leading edge and the Cramér-Rao lower bound (CRLB) on timing jitter.

All functions take a sampled waveform (times t, values x) and return the estimated echo time
(np.nan if no estimate is possible).
"""

from __future__ import annotations

import math

import numpy as np


def _interp_crossing(t, x, k, level):
    """Linear interpolation of the crossing of level between samples k-1 and k."""
    return t[k - 1] + (level - x[k - 1]) * (t[k] - t[k - 1]) / (x[k] - x[k - 1])


def leading_edge_time(t, x, threshold: float) -> float:
    """First upward crossing of a fixed threshold."""
    above = np.nonzero((x[:-1] < threshold) & (x[1:] >= threshold))[0]
    if above.size == 0:
        return np.nan
    return float(_interp_crossing(t, x, above[0] + 1, threshold))


def cfd_time(t, x, fraction: float = 0.5) -> float:
    """Time at which the leading edge reaches a fixed fraction of the pulse's own peak."""
    k_peak = int(np.argmax(x))
    level = fraction * x[k_peak]
    j = k_peak
    while j > 0 and x[j - 1] >= level:
        j -= 1
    if j == 0:
        return np.nan
    return float(_interp_crossing(t, x, j, level))


def matched_filter_time(t, x, template) -> float:
    """Peak of the cross-correlation with the expected pulse shape, refined by a parabola.

    template must be sampled with the same step as t, centred on its middle sample (odd length).
    """
    c = np.correlate(x, template, mode="same")
    k = int(np.argmax(c))
    if k == 0 or k == len(c) - 1:
        return float(t[k])
    y0, y1, y2 = c[k - 1], c[k], c[k + 1]
    denom = y0 - 2 * y1 + y2
    delta = 0.0 if denom == 0 else 0.5 * (y0 - y2) / denom
    return float(t[k] + delta * (t[1] - t[0]))


def gaussian_template(sigma: float, dt: float) -> np.ndarray:
    """Odd-length Gaussian template on a grid of step dt, centred on its middle sample."""
    n = int(math.ceil(5 * sigma / dt))
    tt = np.arange(-n, n + 1) * dt
    return np.exp(-(tt**2) / (2 * sigma**2))


def leading_edge_walk(sigma: float, amplitude_over_threshold):
    """Analytic leading-edge time error of a Gaussian pulse: -sigma sqrt(2 ln(A / threshold))."""
    a = np.asarray(amplitude_over_threshold, dtype=float)
    return -sigma * np.sqrt(2.0 * np.log(a))


def crlb_timing(noise_psd_one_sided: float, charge: float, pulse_fwhm: float) -> float:
    """Cramér-Rao bound on the timing standard deviation [s] for a Gaussian current pulse of total
    charge Q in white noise: var >= (S/2) / integral(i'(t)^2 dt),  integral = Q^2 / (4 sqrt(pi) sigma^3)."""
    sigma_p = pulse_fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    integral = charge**2 / (4.0 * math.sqrt(math.pi) * sigma_p**3)
    return math.sqrt(noise_psd_one_sided / 2.0 / integral)