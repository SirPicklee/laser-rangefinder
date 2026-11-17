import math

import numpy as np
import pytest

from lrfsim import (
    LRF,
    Target,
    Transmitter,
    link_budget_db,
    photons,
    received_energy,
    received_power,
    total_factor,
)

CLEAR = 1e9  # visibility [km]: practically no extinction


@pytest.fixture
def lrf():
    tx = Transmitter.from_divergence(1550e-9, 10e-3, 0.5e-3)
    return LRF(tx, pulse_energy=5e-3, pulse_width=5e-9, rx_aperture=40e-3)


def test_photon_count_of_one_joule():
    assert float(photons(1.0, 1550e-9)) == pytest.approx(7.80e18, rel=1e-3)


def test_energy_is_power_times_pulse_width(lrf):
    t = Target()
    e = float(received_energy(lrf, t, 2000.0, 23.0))
    p = float(received_power(lrf, t, 2000.0, 23.0))
    assert e == pytest.approx(p * lrf.pulse_width)


def test_extended_target_falls_as_one_over_r_squared(lrf):
    t = Target(side=1000.0)
    ratio = float(total_factor(lrf, t, 4000.0, CLEAR)) / float(total_factor(lrf, t, 2000.0, CLEAR))
    assert ratio == pytest.approx(1 / 4, rel=1e-3)


def test_point_target_falls_as_one_over_r_to_the_fourth(lrf):
    t = Target(side=0.01)
    ratio = float(total_factor(lrf, t, 4000.0, CLEAR)) / float(total_factor(lrf, t, 2000.0, CLEAR))
    assert ratio == pytest.approx(1 / 16, rel=1e-3)


def test_signal_is_linear_in_reflectivity(lrf):
    a = float(total_factor(lrf, Target(reflectivity=0.1), 3000.0, 23.0))
    b = float(total_factor(lrf, Target(reflectivity=0.3), 3000.0, 23.0))
    assert b / a == pytest.approx(3.0)


def test_budget_in_db_adds_up_to_total(lrf):
    t = Target()
    db = link_budget_db(lrf, t, 3000.0, 23.0)
    total = 10 * math.log10(float(total_factor(lrf, t, 3000.0, 23.0)))
    assert sum(db.values()) == pytest.approx(total)