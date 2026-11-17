import numpy as np
import pytest

from lrfsim import (
    SI_APD,
    Receiver,
    count_upward_crossings,
    crossing_rate,
    detection_probability,
    detection_probability_fading,
    echo_peak_current,
    false_alarm_probability,
    gaussian_pulse_power,
    noise_rms,
    simulate_current,
    threshold_for_pfa,
)

RX = Receiver(SI_APD, gain=100.0, bandwidth=50e6, dt=0.2e-9)


def test_rice_crossing_rate_matches_simulated_noise():
    rng = np.random.default_rng(0)
    t = np.arange(0, 0.5e-3, RX.dt)
    _, noise = simulate_current(RX, 905e-9, t, np.zeros_like(t), 0.0, rng)
    sigma = noise_rms(RX, 905e-9, 0.0)
    n_sim = count_upward_crossings(noise, 2.5 * sigma)
    n_rice = float(crossing_rate(RX.bandwidth, 2.5)) * t[-1]
    assert n_sim == pytest.approx(n_rice, rel=0.1)


def test_threshold_for_pfa_inverts_false_alarm_probability():
    T = threshold_for_pfa(100e6, 110e-6, 1e-3)
    assert float(false_alarm_probability(100e6, 110e-6, T)) == pytest.approx(1e-3, rel=1e-6)


def test_longer_window_needs_higher_threshold():
    assert threshold_for_pfa(100e6, 200e-6, 1e-3) > threshold_for_pfa(100e6, 20e-6, 1e-3)


def test_echo_exactly_at_threshold_is_a_coin_flip():
    assert float(detection_probability(5.0, 5.0)) == pytest.approx(0.5)


def test_fading_hurts_strong_echoes_and_helps_weak_ones():
    T = 5.0
    assert detection_probability_fading(8.0, T, 0.5) < float(detection_probability(8.0, T))
    assert detection_probability_fading(3.0, T, 0.5) > float(detection_probability(3.0, T))


def test_echo_peak_current_matches_waveform_simulation():
    t = np.arange(0, 2e-6, RX.dt)
    p = gaussian_pulse_power(t, 1e-6, 1e-15, 30e-9)
    clean, _ = simulate_current(RX, 905e-9, t, p, 0.0, np.random.default_rng(1))
    peak = echo_peak_current(SI_APD.responsivity(905e-9), RX.gain, 1e-15, 30e-9, RX.bandwidth)
    assert clean.max() == pytest.approx(peak, rel=0.01)