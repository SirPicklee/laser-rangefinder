"""Exp07a: detecting the echo - thresholds, false alarms and detection probability.

Er:glass + InGaAs receiver from Exp06 (B = 100 MHz), listening window 110 µs (16.5 km).
1) Noise threshold crossings: Rice formula vs simulated receiver noise.
2) Threshold needed for a given false alarm probability per shot.
3) Pd vs SNR at Pfa = 1e-3, with and without turbulent fading; Monte Carlo check with full waveforms.
4) ROC curves.
5) Maximum range for Pd >= 0.99 (replaces the SNR = 10 placeholder of Exp05).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lrfsim import (
    INGAAS_APD,
    LRF,
    Receiver,
    Target,
    Transmitter,
    count_upward_crossings,
    crossing_rate,
    detection_probability,
    detection_probability_fading,
    echo_peak_current,
    false_alarm_probability,
    gaussian_pulse_power,
    noise_rms,
    received_energy,
    scintillation_index,
    simulate_current,
    threshold_for_pfa,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

LRF_ER = LRF(Transmitter.from_divergence(1550e-9, 10e-3, 0.5e-3), 5e-3, 5e-9, 40e-3)
RX = Receiver(INGAAS_APD, gain=5.0, bandwidth=100e6, transimpedance=5e3)
WL = LRF_ER.wavelength
WINDOW = 110e-6
PFA = 1e-3
sigma = noise_rms(RX, WL, 0.0)
r0 = INGAAS_APD.responsivity(WL)

# ---------- 1) Rice vs simulation ----------
t = np.arange(0, 1e-3, RX.dt)
_, noise = simulate_current(RX, WL, t, np.zeros_like(t), 0.0, rng)
levels = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
sim_rate = np.array([count_upward_crossings(noise, T * sigma) / t[-1] for T in levels])
print("1) Noise crossings per second (1 ms of simulated receiver noise)")
print(f"   {'threshold':>10}{'Rice':>12}{'simulated':>12}")
for T, s in zip(levels, sim_rate):
    print(f"   {T:>7.1f} σ{float(crossing_rate(RX.bandwidth, T)):>12.3g}{s:>12.3g}")

# ---------- 2) Threshold for a given Pfa ----------
print(f"\n2) Threshold for a false alarm probability per shot (window {WINDOW * 1e6:.0f} µs)")
print(f"   {'Pfa':>8}{'B = 100 MHz':>13}{'B = 17 MHz':>12}")
for pfa in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
    print(f"   {pfa:>8.0e}{threshold_for_pfa(100e6, WINDOW, pfa):>11.2f} σ{threshold_for_pfa(17e6, WINDOW, pfa):>10.2f} σ")
T_req = threshold_for_pfa(RX.bandwidth, WINDOW, PFA)

# ---------- 3) Pd vs SNR ----------
snrs = np.linspace(2, 12, 101)
fading = {"no fading": 0.0, "fading index 0.3": 0.3, "fading index 1.0": 1.0}
pd_curves = {name: np.array([detection_probability_fading(s, T_req, idx) for s in snrs]) for name, idx in fading.items()}

print(f"\n3) Pd at threshold {T_req:.2f} σ (Pfa = {PFA:g}); Monte Carlo with full waveforms, no fading")
print(f"   {'SNR':>5}{'Pd formula':>12}{'Pd simulated':>14}")
mc_snr, mc_pd = [4.0, 5.0, 6.0, 7.0], []
t = np.arange(0, 400e-9, RX.dt)
sigma_p = LRF_ER.pulse_width / (2 * np.sqrt(2 * np.log(2)))
for s in mc_snr:
    # pulse energy that gives exactly this peak SNR after the filter
    e = s * sigma / echo_peak_current(r0, RX.gain, 1.0, LRF_ER.pulse_width, RX.bandwidth)
    p = gaussian_pulse_power(t, 200e-9, e, LRF_ER.pulse_width)
    gate = (t > 190e-9) & (t < 210e-9)
    hits = 0
    for _ in range(1000):
        _, noisy = simulate_current(RX, WL, t, p, 0.0, rng)
        hits += noisy[gate].max() > T_req * sigma
    mc_pd.append(hits / 1000)
    print(f"   {s:>5.1f}{float(detection_probability(s, T_req)):>12.3f}{mc_pd[-1]:>14.3f}")

print("\n   SNR needed for Pd = 0.99 / 0.90")
for name in fading:
    c = pd_curves[name]
    s99 = snrs[np.argmax(c >= 0.99)] if np.any(c >= 0.99) else np.nan
    s90 = snrs[np.argmax(c >= 0.90)] if np.any(c >= 0.90) else np.nan
    print(f"   {name:<18}{s99:>6.2f}{s90:>8.2f}")

# ---------- 4) ROC ----------
T_scan = np.linspace(2.0, 8.0, 300)
pfa_scan = false_alarm_probability(RX.bandwidth, WINDOW, T_scan)

# ---------- 5) Range ----------
NATO = Target(2.3, 0.3)
R = np.linspace(3000, 20000, 300)
snr_R = np.array([echo_peak_current(r0, RX.gain, float(received_energy(LRF_ER, NATO, r, 23.0)),
                                    LRF_ER.pulse_width, RX.bandwidth) for r in R]) / sigma
turb = {"no turbulence": 0.0, "moderate Cn² 1e-14": 1e-14, "strong Cn² 1e-13": 1e-13}
pd_R = {}
print(f"\n5) Er:glass, NATO target, clear night, threshold {T_req:.2f} σ (Pfa {PFA:g} per shot)")
print(f"   {'turbulence':<22}{'Pd >= 0.99 up to':>18}{'Pd >= 0.90 up to':>18}")
for name, cn2 in turb.items():
    pd_R[name] = np.array([detection_probability_fading(s, T_req, float(scintillation_index(cn2, WL, r)) if cn2 else 0.0)
                           for s, r in zip(snr_R, R)])
    r99 = R[np.argmax(pd_R[name] < 0.99)] if np.any(pd_R[name] < 0.99) else np.nan
    r90 = R[np.argmax(pd_R[name] < 0.90)] if np.any(pd_R[name] < 0.90) else np.nan
    print(f"   {name:<22}{r99 / 1e3:>15.2f} km{r90 / 1e3:>15.2f} km")

# ---------- Figure ----------
fig, ax = plt.subplots(2, 2, figsize=(12, 9.5))
Tl = np.linspace(0, 5, 200)
ax[0, 0].semilogy(Tl, crossing_rate(RX.bandwidth, Tl), label="Rice formula")
ax[0, 0].semilogy(levels, sim_rate, "ko", label="simulated noise (1 ms)")
ax[0, 0].set_xlabel("threshold [noise σ]")
ax[0, 0].set_ylabel("noise crossings per second")
ax[0, 0].set_title("False alarms fall exponentially with the threshold")
ax[0, 0].legend(fontsize=8)
ax[0, 0].grid(alpha=0.3, which="both")

for s in (4, 5, 6, 7):
    ax[0, 1].semilogx(pfa_scan, detection_probability(s, T_scan), label=f"SNR {s}")
ax[0, 1].axvline(PFA, color="k", ls=":", lw=1)
ax[0, 1].set_xlim(1e-8, 1)
ax[0, 1].set_xlabel("false alarm probability per shot")
ax[0, 1].set_ylabel("detection probability")
ax[0, 1].set_title("ROC curves")
ax[0, 1].legend(fontsize=8)
ax[0, 1].grid(alpha=0.3, which="both")

for name, c in pd_curves.items():
    ax[1, 0].plot(snrs, c, label=name)
ax[1, 0].plot(mc_snr, mc_pd, "ko", label="full-waveform Monte Carlo")
ax[1, 0].axvline(T_req, color="k", ls=":", lw=1)
ax[1, 0].set_xlabel("mean peak SNR")
ax[1, 0].set_ylabel("detection probability")
ax[1, 0].set_title(f"Pd at Pfa = {PFA:g} (threshold {T_req:.2f} σ)")
ax[1, 0].legend(fontsize=8)
ax[1, 0].grid(alpha=0.3)

for name, c in pd_R.items():
    ax[1, 1].plot(R / 1e3, c, label=name)
ax[1, 1].axhline(0.99, color="k", ls=":", lw=1)
ax[1, 1].set_xlabel("range [km]")
ax[1, 1].set_ylabel("detection probability per shot")
ax[1, 1].set_title("Er:glass, NATO target, clear night")
ax[1, 1].legend(fontsize=8)
ax[1, 1].grid(alpha=0.3)
fig.tight_layout()
out = RESULTS / "exp07a_detection.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")