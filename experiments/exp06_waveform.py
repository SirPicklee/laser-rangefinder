"""Exp06: what the receiver actually sees - full waveforms on an "oscilloscope".

1) Er:glass + InGaAs at 5, 10, 14 km and diode + Si at 0.3, 0.7, 1.0 km (night, clear air):
   photons, predicted SNR, simulated noise vs prediction.
2) Oscilloscope traces around the echo (ADC samples).
3) The whole listening window: can you even see the echo?
4) A close target: the echo saturates the ADC.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lrfsim import (
    INGAAS_APD,
    LRF,
    SI_APD,
    Receiver,
    Target,
    Transmitter,
    bandwidth_for_pulse,
    digitize,
    echo_power,
    noise_rms,
    photons,
    received_energy,
    simulate_current,
    tof_from_range,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
NATO = Target(2.3, 0.3)
VIS = 23.0

SYSTEMS = {
    "Er:glass + InGaAs": dict(
        lrf=LRF(Transmitter.from_divergence(1550e-9, 10e-3, 0.5e-3), 5e-3, 5e-9, 40e-3),
        rx=Receiver(INGAAS_APD, gain=5.0, bandwidth=bandwidth_for_pulse(5e-9), transimpedance=5e3),
        ranges=(5000.0, 10000.0, 14000.0), window=60e-9),
    "diode + Si": dict(
        lrf=LRF(Transmitter.from_divergence(905e-9, 5e-3, 1.0e-3), 2.25e-6, 30e-9, 25e-3),
        rx=Receiver(SI_APD, gain=100.0, bandwidth=bandwidth_for_pulse(30e-9), transimpedance=20e3),
        ranges=(300.0, 700.0, 1000.0), window=200e-9),
}


def scope(ax, ts, volts, clean_t, clean_v, title):
    ax.set_facecolor("black")
    ax.plot(ts * 1e9, volts * 1e3, color="lime", lw=0.8, drawstyle="steps-mid", label="ADC samples")
    ax.plot(clean_t * 1e9, clean_v * 1e3, color="orange", lw=1, ls="--", label="noiseless echo")
    ax.grid(color="0.35", lw=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time from expected echo [ns]")
    ax.set_ylabel("TIA output [mV]")


# ---------- 1) + 2) ----------
print("1) Echo, SNR and noise check (night, clear air, NATO target)")
print(f"   {'system':<20}{'range':>8}{'photons':>10}{'peak [mV]':>11}{'noise pred [mV]':>17}{'noise sim [mV]':>16}{'SNR':>7}")
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
for row, (name, s) in enumerate(SYSTEMS.items()):
    lrf, rx = s["lrf"], s["rx"]
    for col, R in enumerate(s["ranges"]):
        t0 = float(tof_from_range(R))
        t = np.arange(t0 - 2e-6, t0 + 2e-6, rx.dt)
        p = echo_power(t, lrf, NATO, R, VIS)
        clean, noisy = simulate_current(rx, lrf.wavelength, t, p, 0.0, rng)
        ts, _, volts = digitize(rx, t, noisy)
        quiet = (ts < t0 - 0.5e-6)
        peak_v = clean.max() * rx.transimpedance
        pred = noise_rms(rx, lrf.wavelength, 0.0) * rx.transimpedance
        sim = volts[quiet].std()
        n_ph = float(photons(received_energy(lrf, NATO, R, VIS), lrf.wavelength))
        print(f"   {name:<20}{R / 1e3:>6.1f}km{n_ph:>10.0f}{peak_v * 1e3:>11.2f}{pred * 1e3:>17.3f}{sim * 1e3:>16.3f}"
              f"{peak_v / pred:>7.1f}")
        w = s["window"]
        k = (ts > t0 - w) & (ts < t0 + w)
        kc = (t > t0 - w) & (t < t0 + w)
        scope(axes[row, col], ts[k] - t0, volts[k], t[kc] - t0, clean[kc] * rx.transimpedance,
              f"{name}, {R / 1e3:g} km: {n_ph:.0f} photons, SNR {peak_v / pred:.1f}")
axes[0, 0].legend(fontsize=7, loc="upper right")
fig.tight_layout()
out1 = RESULTS / "exp06_scope.png"
fig.savefig(out1, dpi=150)

# ---------- 3) + 4) ----------
s = SYSTEMS["Er:glass + InGaAs"]
lrf, rx = s["lrf"], s["rx"]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))

R_far = 14000.0
t = np.arange(0, 110e-6, rx.dt)  # listening window up to ~16.5 km
p = echo_power(t, lrf, NATO, R_far, VIS)
clean, noisy = simulate_current(rx, lrf.wavelength, t, p, 0.0, rng)
ts, _, volts = digitize(rx, t, noisy)
ax[0].set_facecolor("black")
ax[0].plot(ts * 1e6, volts * 1e3, color="lime", lw=0.3)
ax[0].axvline(float(tof_from_range(R_far)) * 1e6, color="orange", ls="--", lw=1, label=f"true echo ({R_far / 1e3:g} km)")
ax[0].grid(color="0.35", lw=0.5)
ax[0].set_xlabel("time after the laser fired [µs]")
ax[0].set_ylabel("TIA output [mV]")
ax[0].set_title("The whole listening window: where is the echo?")
ax[0].legend(fontsize=8)
k_max = int(np.argmax(volts))
print(f"\n3) Listening window 0-110 µs, echo at {R_far / 1e3:g} km ({float(tof_from_range(R_far)) * 1e6:.2f} µs)")
print(f"   largest sample at {ts[k_max] * 1e6:.2f} µs ({volts[k_max] * 1e3:.2f} mV) -> "
      f"{'the echo' if abs(ts[k_max] - float(tof_from_range(R_far))) < 50e-9 else 'a NOISE PEAK (false alarm)'}")

R_near = 1000.0
t0 = float(tof_from_range(R_near))
t = np.arange(t0 - 300e-9, t0 + 300e-9, rx.dt)
p = echo_power(t, lrf, NATO, R_near, VIS)
clean, noisy = simulate_current(rx, lrf.wavelength, t, p, 0.0, rng)
ts, codes, volts = digitize(rx, t, noisy)
scope(ax[1], ts - t0, volts, t - t0, clean * rx.transimpedance, f"Er:glass at {R_near / 1e3:g} km: the ADC saturates")
ax[1].axhline(rx.adc_full_scale / 2 * 1e3, color="red", lw=1, label="ADC full scale")
ax[1].set_ylim(-50, clean.max() * rx.transimpedance * 1.1e3)
ax[1].legend(fontsize=7)
clipped = int(np.sum(codes == 2 ** (rx.adc_bits - 1) - 1))
print(f"\n4) Er:glass at {R_near / 1e3:g} km: noiseless peak {clean.max() * rx.transimpedance:.2f} V vs ADC full scale "
      f"{rx.adc_full_scale / 2:.2f} V -> {clipped} samples clipped")
fig.tight_layout()
out2 = RESULTS / "exp06_listening.png"
fig.savefig(out2, dpi=150)
print(f"\nSaved {out1}\nSaved {out2}")