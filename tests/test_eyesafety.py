import math

import pytest

from lrfsim import (
    OpticalAid,
    Transmitter,
    aperture_exposure,
    c4,
    exposure_ratio,
    exposure_ratios,
    max_class1_energy,
    mpe_single,
    nohd,
)

DIODE_TX = Transmitter.from_divergence(905e-9, 5e-3, 1.0e-3)
ER_TX = Transmitter.from_divergence(1550e-9, 10e-3, 0.5e-3)


def test_c4_band_edges():
    assert c4(700e-9) == pytest.approx(1.0)
    assert c4(1049.999e-9) == pytest.approx(10**0.7, rel=1e-4)


def test_retinal_mpe_value_and_continuity():
    assert mpe_single(905e-9, 30e-9) == pytest.approx(2e-3 * 10**0.41, rel=1e-6)
    assert mpe_single(905e-9, 5e-6) == pytest.approx(mpe_single(905e-9, 4.99e-6), rel=0.06)


def test_1550_is_a_million_times_more_tolerant():
    assert mpe_single(1550e-9, 5e-9) == 1e4
    assert mpe_single(1550e-9, 5e-9) / mpe_single(905e-9, 5e-9) > 1e6


def test_unmodelled_inputs_raise():
    with pytest.raises(ValueError):
        mpe_single(1300e-9, 1e-9)
    with pytest.raises(ValueError):
        mpe_single(905e-9, 100.0)


def test_aperture_exposure_limits():
    e, d = 1e-6, 7e-3
    assert aperture_exposure(e, 1.0, d) == pytest.approx(2 * e / math.pi, rel=1e-3)  # wide beam: peak exposure
    assert aperture_exposure(e, 1e-4, d) == pytest.approx(e / (math.pi * d**2 / 4), rel=1e-6)  # all energy inside


def test_pulse_train_rule_only_for_retina_and_many_pulses():
    single = exposure_ratios(DIODE_TX, 1e-6, 30e-9, 1.0)
    assert set(single) == {"single pulse"}
    train = exposure_ratios(DIODE_TX, 1e-6, 30e-9, 1.0, prf=1e4)
    assert train["pulse train"] == pytest.approx(single["single pulse"] * 1e5**0.25)
    assert "pulse train" not in exposure_ratios(ER_TX, 5e-3, 5e-9, 1.0, prf=10.0)


def test_nohd_is_where_ratio_crosses_one():
    z = nohd(DIODE_TX, 2.25e-6, 30e-9, prf=1e4)
    assert 0 < z < math.inf
    assert exposure_ratio(DIODE_TX, 2.25e-6, 30e-9, z, prf=1e4) == pytest.approx(1.0, rel=1e-3)
    assert nohd(ER_TX, 5e-3, 5e-9, prf=1.0) == 0.0


def test_binoculars_extend_nohd():
    aid = OpticalAid(50e-3, 0.7, 7.0)
    assert nohd(DIODE_TX, 2.25e-6, 30e-9, prf=1e4, aid=aid) > nohd(DIODE_TX, 2.25e-6, 30e-9, prf=1e4)


def test_max_class1_energy():
    assert max_class1_energy(DIODE_TX, 30e-9, 1e2) > max_class1_energy(DIODE_TX, 30e-9, 1e4)
    assert max_class1_energy(ER_TX, 5e-9, 1.0) > 5e-3