"""Time of flight (TOF) basics for a pulsed laser rangefinder.

R = c t / (2 n): the pulse travels to the target and back, n is the group index of air.
All times in seconds, distances in metres, frequencies in Hz.
"""

from __future__ import annotations

import numpy as np

C0 = 299_792_458.0  # speed of light in vacuum [m/s]
N_AIR = 1.00027  # group index of air near 1.5 µm at standard conditions (approximate)


def range_from_tof(t, n: float = N_AIR):
    """Distance to the target for a round-trip time t."""
    return C0 * np.asarray(t, dtype=float) / (2.0 * n)


def tof_from_range(r, n: float = N_AIR):
    """Round-trip time for a target at distance r."""
    return 2.0 * n * np.asarray(r, dtype=float) / C0


def range_bin(clock_hz: float, n: float = N_AIR) -> float:
    """Range step of a time counter running at clock_hz."""
    return C0 / (2.0 * n * clock_hz)


def max_unambiguous_range(prf_hz: float, n: float = N_AIR) -> float:
    """Farthest range whose echo arrives before the next pulse is fired."""
    return C0 / (2.0 * n * prf_hz)


def two_target_separation(pulse_width_s: float, n: float = N_AIR) -> float:
    """Closest spacing at which two targets give two separate echoes."""
    return C0 * pulse_width_s / (2.0 * n)


def measure(true_range: float, n_shots: int, clock_hz: float, jitter_s: float = 0.0,
            rng: np.random.Generator | None = None, n: float = N_AIR) -> np.ndarray:
    """Simulate counter-based TOF measurements.

    true round-trip time + Gaussian timing jitter -> counted in whole clock ticks
    -> reported at the centre of the tick (bin).
    """
    rng = np.random.default_rng() if rng is None else rng
    t = np.full(n_shots, tof_from_range(true_range, n))
    if jitter_s > 0:
        t = t + rng.normal(0.0, jitter_s, n_shots)
    ticks = np.floor(t * clock_hz)
    t_measured = (ticks + 0.5) / clock_hz
    return range_from_tof(t_measured, n)