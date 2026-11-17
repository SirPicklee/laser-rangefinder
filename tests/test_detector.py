import math

import numpy as np
import pytest

from lrfsim import APD, Q_E, optimum_gain, snr, solar_background

LAM = 1550e-9
B = 100e6


def ideal(k=0.0, **kw):
    params = dict(quantum_efficiency=1.0, k_ion=k, dark_bulk=0.0, dark_surface=0.0, amp_noise=0.0, max_gain=1000)
    params.update(kw)
    return APD("test", **params)


def test_ideal_responsivity_at_1550_nm():
    assert ideal().responsivity(LAM) == pytest.approx(1.2502, rel=1e-4)


def test_excess_noise_limits():
    assert float(ideal(0.3).excess_noise(1.0)) == pytest.approx(1.0)
    assert float(ideal(1.0).excess_noise(50.0)) == pytest.approx(50.0)
    assert float(ideal(0.0).excess_noise(1e6)) == pytest.approx(2.0, rel=1e-5)


def test_shot_noise_limited_snr():
    d = ideal()
    p = 1e-8
    expected = math.sqrt(d.responsivity(LAM) * p / (2 * Q_E * B))
    assert float(snr(d, LAM, p, 0.0, 1.0, B)) == pytest.approx(expected)


def test_gain_helps_linearly_when_amplifier_noise_dominates():
    d = ideal(amp_noise=2e-12)
    p = 1e-12  # tiny signal: its shot noise is negligible next to the amplifier
    ratio = float(snr(d, LAM, p, 0.0, 10.0, B)) / float(snr(d, LAM, p, 0.0, 1.0, B))
    assert ratio == pytest.approx(10.0, rel=0.01)


def test_noisier_multiplication_means_lower_optimum_gain():
    low_k = ideal(0.02, amp_noise=2e-12)
    high_k = ideal(0.45, amp_noise=2e-12)
    m_low, _ = optimum_gain(low_k, LAM, 1e-8, 0.0, B)
    m_high, _ = optimum_gain(high_k, LAM, 1e-8, 0.0, B)
    assert m_high < m_low


def test_background_lowers_optimum_gain_and_snr():
    d = ideal(0.45, amp_noise=2e-12)
    m_night, s_night = optimum_gain(d, LAM, 1e-8, 0.0, B)
    m_day, s_day = optimum_gain(d, LAM, 1e-8, 1e-7, B)
    assert m_day < m_night
    assert s_day < s_night


def test_solar_background_formula():
    p = solar_background(0.25, 10.0, 0.3, 1e-3, 1e-3, 0.8)
    expected = (0.3 * 0.25 * 10 / math.pi) * 1e-3 * (math.pi * (0.5e-3) ** 2) * 0.8
    assert p == pytest.approx(expected)