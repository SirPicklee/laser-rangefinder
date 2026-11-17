"""Exp08: many pulses and many echoes.

1) Laser diode 905 nm at 1 km: averaging N waveforms, SNR vs sqrt(N).
2) How many pulses the diode needs at each range, how long that takes, and how far a moving
   target travels meanwhile.
3) Coherent accumulation vs binary k-of-N integration.
4) Bush in front of a vehicle: first / last target, and when the two echoes merge.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from lrfsim import (
    C0,
    INGAAS_APD,
    LRF,
    N_AIR,
    SI_APD,
    Receiver,
    Target,
    Transmitter,
    accumulate,
    binary_integration_pd,
    detection_probability,
    digitize,
    echo_peak_current,
    find_echoes,
    first_and_last,
    gaussian_pulse_power,
    noise_rms,
    pulses_for_snr,
    received_energy,
    simulate_current,
    threshold_for_pfa,
    tof_from_range,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
NATO = Target(2.3, 0.3)
VIS = 23.0
TO_M = C0 / (2 * N_AIR)

DIODE = LRF(Transmitter.from_divergence(905e-9, 5e-3, 1.0e-3), 2.25e-6, 30e-9, 25e-3)
RX_D = Receiver(SI_APD, gain=100.0, bandwidth=0.5 / 30e-9, transimpedance=20e3)
ER = LRF(Transmitter.from_divergence(1550e-9, 10e-3, 0.5e-3), 5e-3, 5e-9, 40e-3)
RX_E = Receiver(INGAAS_APD, gain=5.0, bandwidth=100e6, transimpedance=5e3)

WINDOW_D = 20e-6  # diode listening window (3 km)
PFA = 1e-3
T_D = threshold_for_pfa(RX_D.bandwidth, WINDOW_D, PFA)
SNR_REQ = T_D + norm.ppf(0.99)  # Pd = 0.99 without fading
sig_d = noise_rms(RX_D, DIODE.wavelength, 0.0)
r0_d = SI_APD.responsivity(DIODE.wavelength)


def single_snr(r):
    e = float(received_energy(DIODE, NATO, r, VIS))
    return echo_peak_current(r0_d, RX_D.gain, e, DIODE.pulse_width, RX_D.bandwidth) / sig_d


# ---------- 1) Accumulation ----------
R1 = 1000.0
t0 = float(tof_from_range(R1))
t = np.arange(t0 - 1e-6, t0 + 1e-6, RX_D.dt)
p = gaussian_pulse_power(t, t0, float(received_energy(DIODE, NATO, R1, VIS)), DIODE.pulse_width)
shots = []
for _ in range(256):
    clean, noisy = simulate_current(RX_D, DIODE.wavelength, t, p, 0.0, rng)
    shots.append(noisy)
shots = np.array(shots)
quiet = t < t0 - 0.3e-6
print(f"1) Diode at {R1 / 1e3:g} km: single-shot SNR {single_snr(R1):.1f}; "
      f"Pd 0.99 at Pfa {PFA:g} needs SNR {SNR_REQ:.2f} (threshold {T_D:.2f} σ)")
print(f"   {'N':>5}{'SNR theory':>12}{'SNR simulated':>15}")
acc = {}
for N in (1, 4, 16, 64, 256):
    a = accumulate(shots[:N])
    acc[N] = a
    s_sim = clean.max() / a[quiet].std()
    print(f"   {N:>5}{single_snr(R1) * np.sqrt(N):>12.1f}{s_sim:>15.1f}")

# ---------- 2) Pulses needed ----------
PRF = 10e3
print(f"\n2) Pulses the diode needs for Pd 0.99 (PRF {PRF / 1e3:.0f} kHz, NATO target, clear night)")
print(f"   {'range':>7}{'SNR 1 shot':>12}{'pulses':>9}{'time [ms]':>11}{'vehicle at 20 m/s moves':>25}")
for r in (500, 700, 1000, 1500, 2000, 3000):
    s1 = single_snr(r)
    n = max(1, pulses_for_snr(s1, SNR_REQ))
    ms = n / PRF * 1e3
    print(f"   {r:>5.0f} m{s1:>12.2f}{n:>9d}{ms:>11.1f}{20 * ms / 1e3:>22.2f} m")
print(f"   range resolution of the 30 ns pulse: {DIODE.pulse_width * TO_M:.1f} m")

# ---------- 3) Coherent vs binary ----------
N3 = 16
s_axis = np.linspace(0.5, 6, 200)
pd_coh = detection_probability(s_axis * np.sqrt(N3), T_D)
pd_bin = {k: [binary_integration_pd(float(detection_probability(s, T_D)), N3, k) for s in s_axis] for k in (2, 4, 8)}
print(f"\n3) {N3} pulses: single-shot SNR needed for Pd = 0.99")
print(f"   coherent accumulation: {s_axis[np.argmax(pd_coh >= 0.99)]:.2f}")
for k, c in pd_bin.items():
    c = np.array(c)
    print(f"   binary {k}-of-{N3}:       {s_axis[np.argmax(c >= 0.99)] if np.any(c >= 0.99) else np.nan:.2f}")
print("   (binary needs per-shot false alarms in the SAME range bin, so its own threshold could be lower;"
      " here both use the same threshold for a like-for-like comparison)")

# ---------- 4) Bush and vehicle ----------
R_BUSH, COVER = 800.0, 0.4
BUSH = Target(side=20.0, reflectivity=0.4)
print(f"\n4) Bush at {R_BUSH:g} m covering {COVER:.0%} of the beam, vehicle behind it")
print(f"   {'system':<10}{'gap [m]':>8}{'echoes':>8}{'first [m]':>11}{'last [m]':>10}")
scenes = {}
for name, lrf, rx, n_acc in (("diode", DIODE, RX_D, 64), ("Er:glass", ER, RX_E, 1)):
    sig = noise_rms(rx, lrf.wavelength, 0.0)
    th = threshold_for_pfa(rx.bandwidth, WINDOW_D, PFA) * sig / np.sqrt(n_acc)
    width = lrf.pulse_width
    for gap in (1.0, 3.0, 6.0, 15.0):
        r_veh = R_BUSH + gap
        tb, tv = float(tof_from_range(R_BUSH)), float(tof_from_range(r_veh))
        tt = np.arange(tb - 0.5e-6, tv + 0.5e-6, rx.dt)
        pw = (COVER * gaussian_pulse_power(tt, tb, float(received_energy(lrf, BUSH, R_BUSH, VIS)), width)
              + (1 - COVER) * gaussian_pulse_power(tt, tv, float(received_energy(lrf, NATO, r_veh, VIS)), width))
        waves = [simulate_current(rx, lrf.wavelength, tt, pw, 0.0, rng)[1] for _ in range(n_acc)]
        a = accumulate(waves)
        echoes = find_echoes(tt, a, th, width)
        f, l = first_and_last(echoes)
        scenes[(name, gap)] = (tt - tb, a / sig * np.sqrt(n_acc), th / sig * np.sqrt(n_acc))
        print(f"   {name:<10}{gap:>8.0f}{echoes.size:>8d}{f * TO_M:>11.1f}{l * TO_M:>10.1f}")

# ---------- Figure ----------
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for N, c in ((1, "lime"), (64, "orange")):
    ax[0, 0].plot((t - t0) * 1e9, acc[N] / sig_d * np.sqrt(N), color=c, lw=0.8, label=f"N = {N}")
ax[0, 0].set_facecolor("black")
ax[0, 0].axhline(T_D, color="red", lw=1, label="threshold")
ax[0, 0].grid(color="0.35", lw=0.5)
ax[0, 0].set_xlim(-400, 400)
ax[0, 0].set_xlabel("time from echo [ns]")
ax[0, 0].set_ylabel("signal / noise RMS of the average")
ax[0, 0].set_title(f"Diode at {R1 / 1e3:g} km: 1 shot vs 64 averaged")
ax[0, 0].legend(fontsize=8)

ax[0, 1].plot(s_axis, pd_coh, "k-", lw=2, label=f"coherent, N = {N3}")
for k, c in pd_bin.items():
    ax[0, 1].plot(s_axis, c, label=f"binary {k}-of-{N3}")
ax[0, 1].plot(s_axis, detection_probability(s_axis, T_D), "k:", label="single shot")
ax[0, 1].set_xlabel("single-shot SNR")
ax[0, 1].set_ylabel("detection probability")
ax[0, 1].set_title("Accumulating waveforms vs counting detections")
ax[0, 1].legend(fontsize=8)
ax[0, 1].grid(alpha=0.3)

for a_, name in ((ax[1, 0], "diode"), (ax[1, 1], "Er:glass")):
    a_.set_facecolor("black")
    for gap, c in ((1.0, "cyan"), (6.0, "yellow"), (15.0, "magenta")):
        tt, y, th = scenes[(name, gap)]
        a_.plot(tt * TO_M, y, color=c, lw=0.8, label=f"vehicle {gap:g} m behind")
    a_.axhline(th, color="red", lw=1)
    a_.grid(color="0.35", lw=0.5)
    a_.set_xlim(-10, 25)
    a_.set_xlabel("range behind the bush [m]")
    a_.set_ylabel("signal / noise RMS")
    a_.set_title(f"{name}: bush + vehicle")
    a_.legend(fontsize=7)
fig.tight_layout()
out = RESULTS / "exp08_multipulse.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")