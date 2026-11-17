import math

import numpy as np
import pytest

from lrfsim import (
    Transmitter,
    beam_radius,
    divergence_half_angle,
    fraction_on_square,
    pointing_fractions,
    rayleigh_range,
)

LAM = 1550e-9
W0 = 10e-3


def test_radius_at_waist_is_w0():
    assert beam_radius(0.0, W0, LAM) == pytest.approx(W0)


def test_radius_at_rayleigh_range_is_w0_sqrt2():
    zr = rayleigh_range(W0, LAM)
    assert beam_radius(zr, W0, LAM) == pytest.approx(W0 * math.sqrt(2))


def test_far_field_grows_linearly_with_divergence():
    z = 100 * rayleigh_range(W0, LAM)
    assert beam_radius(z, W0, LAM) == pytest.approx(z * divergence_half_angle(W0, LAM), rel=1e-3)


def test_m2_scales_divergence():
    assert divergence_half_angle(W0, LAM, m2=3.0) == pytest.approx(3 * divergence_half_angle(W0, LAM))


def test_from_divergence_round_trip():
    tx = Transmitter.from_divergence(LAM, W0, 1e-3)
    assert tx.divergence_full == pytest.approx(1e-3)
    assert tx.m2 > 1.0


def test_below_diffraction_limit_is_rejected():
    with pytest.raises(ValueError):
        Transmitter.from_divergence(LAM, W0, 0.05e-3)


def test_huge_target_catches_everything():
    assert fraction_on_square(100.0, 0.5) == pytest.approx(1.0)


def test_known_fraction_value():
    w = 1.0
    side = math.sqrt(2) * w  # erf(1)^2
    assert fraction_on_square(side, w) == pytest.approx(math.erf(1.0) ** 2)


def test_offset_reduces_fraction_and_is_symmetric():
    f0 = fraction_on_square(2.3, 1.0)
    f_plus = fraction_on_square(2.3, 1.0, dx=0.5)
    f_minus = fraction_on_square(2.3, 1.0, dx=-0.5)
    assert f_plus < f0
    assert f_plus == pytest.approx(f_minus)


def test_pointing_without_jitter_equals_centred_beam():
    tx = Transmitter.from_divergence(LAM, W0, 1e-3)
    rng = np.random.default_rng(0)
    f = pointing_fractions(tx, 3000.0, 2.3, 0.0, 10, rng)
    assert f == pytest.approx(np.full(10, tx.fraction_on_target(3000.0, 2.3)))



def test_monte_carlo_mean_matches_exact_formula():
    from lrfsim import mean_fraction

    tx = Transmitter.from_divergence(LAM, W0, 0.5e-3)
    rng = np.random.default_rng(7)
    mc = pointing_fractions(tx, 3000.0, 2.3, 0.3e-3, 40000, rng).mean()
    assert mc == pytest.approx(float(mean_fraction(tx, 3000.0, 2.3, 0.3e-3)), rel=0.02)