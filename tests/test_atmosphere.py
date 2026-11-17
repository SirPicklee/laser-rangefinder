import math

import numpy as np
import pytest

from lrfsim import (
    KOSCHMIEDER,
    extinction,
    fade_probability,
    kim_q,
    lognormal_samples,
    rytov_variance,
    scintillation_index,
    transmission,
)


@pytest.mark.parametrize("v, q", [(100, 1.6), (20, 1.3), (3, 0.82), (0.7, 0.2), (0.3, 0.0)])
def test_kim_exponent(v, q):
    assert kim_q(v) == pytest.approx(q)


def test_extinction_at_550_nm_is_koschmieder():
    assert extinction(10.0, 550e-9) == pytest.approx(KOSCHMIEDER / 10e3)


def test_visibility_means_2_percent_contrast_left():
    # one way over the visibility distance at 550 nm
    assert float(transmission(5e3, 5.0, 550e-9, round_trip=False)) == pytest.approx(0.02, rel=1e-3)


def test_round_trip_is_one_way_squared():
    one = float(transmission(2e3, 5.0, 1550e-9, round_trip=False))
    assert float(transmission(2e3, 5.0, 1550e-9)) == pytest.approx(one**2)


def test_1550_beats_905_in_haze_but_not_in_thick_fog():
    assert extinction(5.0, 1550e-9) < extinction(5.0, 905e-9)
    assert extinction(0.3, 1550e-9) == pytest.approx(extinction(0.3, 905e-9))


def test_rytov_scales_with_distance_to_the_11_6():
    r1 = float(rytov_variance(1e-14, 1550e-9, 1000.0))
    r2 = float(rytov_variance(1e-14, 1550e-9, 2000.0))
    assert r2 / r1 == pytest.approx(2 ** (11 / 6))


def test_weak_turbulence_scintillation_equals_rytov():
    L = 1000.0
    s = float(rytov_variance(1e-17, 1550e-9, L))
    assert float(scintillation_index(1e-17, 1550e-9, L)) == pytest.approx(s, rel=0.01)


def test_strong_turbulence_saturates():
    assert float(scintillation_index(1e-12, 905e-9, 5000.0)) < 2.0


def test_lognormal_mean_one_and_variance_equal_to_index():
    rng = np.random.default_rng(0)
    x = lognormal_samples(0.5, 400000, rng)
    assert x.mean() == pytest.approx(1.0, rel=0.01)
    assert x.var() == pytest.approx(0.5, rel=0.03)


def test_fade_probability_matches_samples():
    rng = np.random.default_rng(1)
    x = lognormal_samples(0.4, 400000, rng)
    assert fade_probability(0.4, 0.5) == pytest.approx((x < 0.5).mean(), abs=0.003)