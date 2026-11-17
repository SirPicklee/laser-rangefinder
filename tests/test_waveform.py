import numpy as np
import pytest

from lrfsim import (
    LRF,
    SI_APD,
    Receiver,
    Target,
    Transmitter,
    digitize,
    echo_power,
    gaussian_filter_taps,
    gaussian_pulse_power,
    noise_rms,
    received_energy,
    simulate_current,
    tof_from_range,
)

DT = 0.1e-9


@pytest.fixture
def rx():
    return Receiver(SI_APD, gain=100.0, bandwidth=17e6)


def test_pulse_energy_is_conserved():
    t = np.arange(0, 200e-9, DT)
    p = gaussian_pulse_power(t, 100e-9, 1e-12, 10e-9)
    assert p.sum() * DT == pytest.approx(1e-12, rel=1e-4)


def test_filter_minus_3db_at_bandwidth():
    B = 50e6
    h = gaussian_filter_taps(B, DT)
    n = 2**18
    H = np.abs(np.fft.rfft(h, n))
    f = np.fft.rfftfreq(n, DT)
    assert np.interp(B, f, H) == pytest.approx(1 / np.sqrt(2), rel=0.01)


def test_filtered_noise_has_the_predicted_rms(rx):
    rng = np.random.default_rng(0)
    t = np.arange(0, 100e-6, DT)
    _, noisy = simulate_current(rx, 905e-9, t, np.zeros_like(t), 1e-9, rng)
    assert noisy.std() == pytest.approx(noise_rms(rx, 905e-9, 1e-9), rel=0.03)


def test_filter_keeps_the_total_charge(rx):
    t = np.arange(0, 1e-6, DT)
    p = gaussian_pulse_power(t, 0.5e-6, 1e-15, 30e-9)
    clean, _ = simulate_current(rx, 905e-9, t, p, 0.0, np.random.default_rng(1))
    expected = SI_APD.responsivity(905e-9) * rx.gain * 1e-15
    assert clean.sum() * DT == pytest.approx(expected, rel=1e-3)


def test_echo_arrives_at_the_round_trip_time():
    lrf = LRF(Transmitter.from_divergence(905e-9, 5e-3, 1e-3), 2.25e-6, 30e-9, 25e-3)
    t = np.arange(0, 10e-6, DT)
    p = echo_power(t, lrf, Target(), 700.0, 23.0)
    assert t[np.argmax(p)] == pytest.approx(float(tof_from_range(700.0)), abs=DT)
    assert p.sum() * DT == pytest.approx(float(received_energy(lrf, Target(), 700.0, 23.0)), rel=1e-3)


def test_adc_clips_at_full_scale(rx):
    t = np.arange(0, 100e-9, DT)
    _, codes, volts = digitize(rx, t, np.full(t.size, 1.0))  # 1 A: far beyond full scale
    assert codes.max() == 2 ** (rx.adc_bits - 1) - 1
    assert volts.max() < rx.adc_full_scale / 2


def test_adc_output_is_quantised(rx):
    t = np.arange(0, 1e-6, DT)
    i = np.linspace(-5e-6, 5e-6, t.size)
    _, codes, volts = digitize(rx, t, i)
    lsb = rx.adc_full_scale / 2**rx.adc_bits
    assert np.allclose(volts / lsb, np.round(volts / lsb))