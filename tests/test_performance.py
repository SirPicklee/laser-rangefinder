import numpy as np
import pytest

from lrfsim import (
    INGAAS_APD,
    LRF,
    SI_APD,
    Conditions,
    Receiver,
    System,
    Target,
    Transmitter,
    detection_probability,
    mean_snr,
    pd_analytic,
    pd_monte_carlo,
    range_at_pd,
    threshold,
)

ER = System("Er", LRF(Transmitter.from_divergence(1550e-9, 10e-3, 0.5e-3), 5e-3, 5e-9, 40e-3),
            Receiver(INGAAS_APD, gain=5.0, bandwidth=100e6, transimpedance=5e3))
DIODE = System("diode", LRF(Transmitter.from_divergence(905e-9, 5e-3, 1.0e-3), 2.25e-6, 30e-9, 25e-3),
               Receiver(SI_APD, gain=100.0, bandwidth=0.5 / 30e-9), n_pulses=16, window=20e-6,
               sun_per_nm=0.75, filter_nm=40.0, fov=1.5e-3)


def test_without_turbulence_analytic_is_the_plain_q_function():
    c = Conditions()
    s = float(mean_snr(ER, c, 12000.0))
    assert pd_analytic(ER, c, 12000.0) == pytest.approx(float(detection_probability(s, threshold(ER))))


def test_accumulation_multiplies_snr_by_sqrt_n():
    c = Conditions()
    s1 = float(mean_snr(DIODE, c, 1500.0))
    assert pd_analytic(DIODE, c, 1500.0) == pytest.approx(float(detection_probability(4 * s1, threshold(DIODE))))


def test_monte_carlo_matches_analytic_with_fading():
    rng = np.random.default_rng(0)
    c = Conditions(cn2=1e-14)
    for r in (11000.0, 13000.0):
        assert pd_monte_carlo(ER, c, r, 20000, rng) == pytest.approx(pd_analytic(ER, c, r), abs=0.02)


def test_daylight_costs_the_diode_snr():
    night, day = Conditions(), Conditions(daylight=True)
    assert float(mean_snr(DIODE, day, 1000.0)) < float(mean_snr(DIODE, night, 1000.0))


def test_aiming_jitter_lowers_pd():
    rng = np.random.default_rng(1)
    still = pd_monte_carlo(ER, Conditions(), 10000.0, 4000, rng)
    shaky = pd_monte_carlo(ER, Conditions(aiming_sigma=0.5e-3), 10000.0, 4000, rng)
    assert shaky < still


def test_analytic_refuses_aiming_error():
    with pytest.raises(ValueError):
        pd_analytic(ER, Conditions(aiming_sigma=0.1e-3), 5000.0)


def test_range_at_pd():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    assert range_at_pd(d, np.array([1.0, 0.95, 0.85, 0.5]), 0.9) == pytest.approx(2.5)
    assert np.isnan(range_at_pd(d, np.ones(4), 0.9))