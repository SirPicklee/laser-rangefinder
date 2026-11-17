import numpy as np
import pytest

from lrfsim import (
    C0,
    max_unambiguous_range,
    measure,
    range_bin,
    range_from_tof,
    tof_from_range,
    two_target_separation,
)


def test_round_trip_conversion():
    r = np.array([10.0, 1234.5, 20000.0])
    assert range_from_tof(tof_from_range(r)) == pytest.approx(r)


def test_one_nanosecond_is_about_15_cm():
    assert range_from_tof(1e-9, n=1.0) == pytest.approx(0.1499, abs=1e-4)


def test_range_bin_500_mhz():
    assert range_bin(500e6, n=1.0) == pytest.approx(C0 / 1e9)


def test_unambiguous_range_1_khz():
    assert max_unambiguous_range(1e3, n=1.0) == pytest.approx(C0 / 2e3)


def test_two_target_separation_10_ns():
    assert two_target_separation(10e-9, n=1.0) == pytest.approx(1.499, abs=1e-3)


def test_without_jitter_every_shot_is_identical():
    m = measure(1000.0, 100, 500e6, jitter_s=0.0)
    assert np.ptp(m) == 0.0


def test_quantization_error_is_at_most_half_a_bin():
    rng = np.random.default_rng(1)
    for r in rng.uniform(100, 5000, 200):
        err = measure(r, 1, 500e6)[0] - r
        assert abs(err) <= range_bin(500e6) / 2 + 1e-9


def test_jitter_spreads_measurements_as_c_sigma_over_2():
    rng = np.random.default_rng(2)
    m = measure(1000.0, 20000, 10e9, jitter_s=1e-9, rng=rng)  # fine clock, 1 ns jitter
    assert np.std(m) == pytest.approx(range_from_tof(1e-9), rel=0.05)


def test_jitter_lets_averaging_beat_the_bin():
    rng = np.random.default_rng(3)
    clock = 500e6
    true = 1000.0 + 0.3 * range_bin(clock)  # deliberately between bin centres
    no_jitter = measure(true, 5000, clock, 0.0, rng).mean() - true
    with_jitter = measure(true, 5000, clock, 1.0 / clock, rng).mean() - true
    assert abs(with_jitter) < abs(no_jitter) / 5