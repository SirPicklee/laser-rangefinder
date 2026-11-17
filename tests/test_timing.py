import math

import numpy as np
import pytest

from lrfsim import (
    cfd_time,
    crlb_timing,
    gaussian_template,
    leading_edge_time,
    leading_edge_walk,
    matched_filter_time,
)

DT = 0.05e-9
SIGMA = 2.5e-9
T0 = 100e-9
t = np.arange(0, 200e-9, DT)


def pulse(amplitude):
    return amplitude * np.exp(-((t - T0) ** 2) / (2 * SIGMA**2))


def test_leading_edge_walk_matches_analytic_formula():
    for a in (2.0, 10.0, 100.0):
        err = leading_edge_time(t, pulse(a), 1.0) - T0
        assert err == pytest.approx(float(leading_edge_walk(SIGMA, a)), abs=0.01e-9)


def test_strong_pulses_cross_earlier():
    assert leading_edge_time(t, pulse(100.0), 1.0) < leading_edge_time(t, pulse(3.0), 1.0)


def test_cfd_does_not_depend_on_amplitude():
    times = [cfd_time(t, pulse(a), 0.5) for a in (2.0, 10.0, 100.0)]
    assert max(times) - min(times) < 0.01e-9


def test_cfd_at_half_maximum_is_one_hwhm_early():
    hwhm = SIGMA * math.sqrt(2 * math.log(2))
    assert cfd_time(t, pulse(5.0), 0.5) - T0 == pytest.approx(-hwhm, abs=0.01e-9)


def test_matched_filter_finds_the_centre():
    x = pulse(3.0)
    assert matched_filter_time(t, x, gaussian_template(SIGMA, DT)) == pytest.approx(T0, abs=0.01e-9)


def test_no_crossing_gives_nan():
    assert math.isnan(leading_edge_time(t, pulse(0.5), 1.0))


def test_crlb_formula_matches_numerical_integral():
    Q, fwhm, S = 1e-12, 5e-9, 1e-23
    sigma_p = fwhm / (2 * math.sqrt(2 * math.log(2)))
    i = Q / (sigma_p * math.sqrt(2 * math.pi)) * np.exp(-((t - T0) ** 2) / (2 * sigma_p**2))
    integral = np.sum(np.gradient(i, DT) ** 2) * DT
    assert crlb_timing(S, Q, fwhm) == pytest.approx(math.sqrt(S / 2 / integral), rel=1e-3)