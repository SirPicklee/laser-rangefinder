"""Multi-pulse accumulation, binary (k-of-N) integration and multiple echoes.

Coherent accumulation (averaging aligned waveforms): signal adds, independent noise averages down
    -> SNR_N = sqrt(N) * SNR_1.
Binary integration: declare a target if at least k of N single-shot detections fire.
Multiple echoes: peaks above the threshold separated by at least one echo width;
first-target / last-target logic picks the nearest / farthest.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import binom


def accumulate(waveforms) -> np.ndarray:
    """Average of N time-aligned waveforms (rows)."""
    return np.mean(np.asarray(waveforms, dtype=float), axis=0)


def pulses_for_snr(snr_single: float, snr_required: float) -> int:
    """Number of accumulated pulses needed to lift the SNR from snr_single to snr_required."""
    return int(math.ceil((snr_required / snr_single) ** 2))


def binary_integration_pd(pd_single: float, n: int, k: int) -> float:
    """Probability of at least k detections in n independent shots."""
    return float(binom.sf(k - 1, n, pd_single))


def find_echoes(t, x, threshold: float, min_separation: float) -> np.ndarray:
    """Times of the peaks above threshold that are at least min_separation apart."""
    dt = t[1] - t[0]
    idx, _ = find_peaks(x, height=threshold, distance=max(1, int(round(min_separation / dt))))
    return np.asarray(t)[idx]


def first_and_last(echo_times) -> tuple[float, float]:
    """First-target and last-target times (nan if nothing was found)."""
    e = np.asarray(echo_times)
    if e.size == 0:
        return math.nan, math.nan
    return float(e.min()), float(e.max())