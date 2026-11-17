import math

import numpy as np
import pytest

from lrfsim import accumulate, binary_integration_pd, find_echoes, first_and_last, pulses_for_snr

DT = 0.5e-9
t = np.arange(0, 2e-6, DT)


def gauss(t0, sigma=10e-9, a=1.0):
    return a * np.exp(-((t - t0) ** 2) / (2 * sigma**2))


def test_pulses_for_snr():
    assert pulses_for_snr(2.0, 8.0) == 16


def test_averaging_reduces_noise_by_sqrt_n():
    rng = np.random.default_rng(0)
    waves = rng.normal(0.0, 1.0, (64, t.size))
    assert accumulate(waves).std() == pytest.approx(1 / 8, rel=0.05)


def test_averaging_keeps_the_echo():
    waves = np.tile(gauss(1e-6), (10, 1))
    assert accumulate(waves).max() == pytest.approx(1.0)


def test_binary_integration_limits():
    p, n = 0.3, 8
    assert binary_integration_pd(p, n, 1) == pytest.approx(1 - (1 - p) ** n)
    assert binary_integration_pd(p, n, n) == pytest.approx(p**n)


def test_two_separated_echoes_are_found():
    x = gauss(0.8e-6) + gauss(1.2e-6, a=0.6)
    e = find_echoes(t, x, 0.3, 20e-9)
    assert e.size == 2
    assert e == pytest.approx([0.8e-6, 1.2e-6], abs=DT)


def test_echoes_closer_than_their_width_merge():
    x = gauss(1.0e-6) + gauss(1.0e-6 + 10e-9)  # 1 sigma apart: one bump
    assert find_echoes(t, x, 0.3, 20e-9).size == 1


def test_first_and_last():
    assert first_and_last(np.array([3.0, 1.0, 2.0])) == (1.0, 3.0)
    f, l = first_and_last(np.array([]))
    assert math.isnan(f) and math.isnan(l)