"""Exp07b: timing the echo - leading edge vs CFD vs matched filter, and the Cramér-Rao bound.

Er:glass + InGaAs receiver (B = 100 MHz), ADC 1 GS/s, threshold for Pfa = 1e-3 per 110 µs window.
1) Walk: noiseless echoes of different strength.
2) Noise: 500 shots per SNR -> bias and jitter of each method vs the CRLB.
3) A saturated echo (close target): which method survives?
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lrfsim import (
    C0,
    INGAAS_APD,
    N_AIR,
    LRF,
    Receiver,
    Transmitter,
    cfd_time,
    crlb_timing,
    digitize,
    echo_peak_current,
    gaussian_pulse_power,
    gaussian_template,
    leading_edge_time,
    leading_edge_walk,
    matched_filter_time,
    noise_psd,
    noise_rms,
    simulate_current,
    threshold_for_pfa,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

LRF_ER = LRF(Transmitter.from_divergence(1550e-9, 10e-3, 0.5e-3), 5e-3, 5e-9, 40e-3)
RX = Receiver(INGAAS_APD, gain=5.0, bandwidth=100e6, transimpedance=5e3)
WL = LRF_ER.wavelength
r0 = INGAAS_APD.responsivity(WL)
sigma_noise = noise_rms(RX, WL, 0.0)  # [A]
T_REQ = threshold_for_pfa(RX.bandwidth, 110e-6, 1e-3)
TH_V = T_REQ * sigma_noise * RX.transimpedance  # threshold in volts at the ADC

sigma_p = LRF_ER.pulse_width / (2 * math.sqrt(2 * math.log(2)))
sigma_f = RX.bandwidth / math.sqrt(math.log(2))
sigma_tot = math.sqrt(sigma_p**2 + (1 / (2 * math.pi * sigma_f)) ** 2)  # filtered echo width
dts = 1.0 / RX.adc_rate
template = gaussian_template(sigma_tot, dts)
TO_M = C0 / (2 * N_AIR)  # seconds -> metres of range
T0 = 1e-6  # echo time inside the simulated record
t = np.arange(0, 2e-6, RX.dt)
hwhm = sigma_tot * math.sqrt(2 * math.log(2))


def energy_for_snr(snr):
    return snr * sigma_noise / echo_peak_current(r0, RX.gain, 1.0, LRF_ER.pulse_width, RX.bandwidth)


def time_echo(snr, noisy=True):
    """One shot: returns (leading edge, CFD, matched filter) time errors [s]."""
    p = gaussian_pulse_power(t, T0, energy_for_snr(snr), LRF_ER.pulse_width)
    clean, noisy_i = simulate_current(RX, WL, t, p, 0.0, rng)
    ts, _, volts = digitize(RX, t, noisy_i if noisy else clean)
    gate = (ts > T0 - 50e-9) & (ts < T0 + 50e-9)
    tg, xg = ts[gate], volts[gate]
    return (leading_edge_time(tg, xg, TH_V) - T0,
            cfd_time(tg, xg, 0.5) - (T0 - hwhm),  # CFD at 50 % fires one half-width early: a fixed, known offset
            matched_filter_time(tg, xg, template) - T0)


# ---------- 1) Walk ----------
snr_walk = np.array([6, 8, 10, 15, 20, 30, 50, 100, 200])
walk = np.array([time_echo(s, noisy=False) for s in snr_walk])
print(f"1) Systematic error (no noise), threshold {T_REQ:.2f} σ, filtered echo width σ = {sigma_tot * 1e9:.2f} ns")
print(f"   {'SNR':>5}{'leading edge [cm]':>19}{'analytic [cm]':>15}{'CFD [cm]':>10}{'matched [cm]':>14}")
for s, w in zip(snr_walk, walk):
    a = float(leading_edge_walk(sigma_tot, s / T_REQ)) * TO_M * 100
    print(f"   {s:>5.0f}{w[0] * TO_M * 100:>19.1f}{a:>15.1f}{w[1] * TO_M * 100:>10.1f}{w[2] * TO_M * 100:>14.1f}")
print(f"   leading-edge walk between SNR {snr_walk[0]} and {snr_walk[-1]}: "
      f"{(walk[0, 0] - walk[-1, 0]) * TO_M * 100:.0f} cm")

# ---------- 2) Jitter ----------
snr_noise = [6, 8, 12, 20, 50, 100]
N = 500
S = float(noise_psd(RX, WL, 0.0))
stats = {}
print(f"\n2) Random error, {N} shots per SNR [cm of range]")
print(f"   {'SNR':>5}{'LE bias':>9}{'LE std':>8}{'CFD std':>9}{'MF std':>8}{'CRLB':>7}{'MF / CRLB':>11}{'LE misses':>11}")
for s in snr_noise:
    res = np.array([time_echo(s) for _ in range(N)])
    q = r0 * RX.gain * energy_for_snr(s)
    crlb = crlb_timing(S, q, LRF_ER.pulse_width)
    le, cfd, mf = (res[:, k] * TO_M * 100 for k in range(3))
    stats[s] = dict(le=np.nanstd(le), cfd=np.nanstd(cfd), mf=np.nanstd(mf), crlb=crlb * TO_M * 100)
    print(f"   {s:>5}{np.nanmean(le):>9.1f}{np.nanstd(le):>8.1f}{np.nanstd(cfd):>9.1f}{np.nanstd(mf):>8.1f}"
          f"{stats[s]['crlb']:>7.1f}{np.nanstd(mf) / stats[s]['crlb']:>11.2f}{np.isnan(le).mean() * 100:>10.1f}%")

# ---------- 3) Saturation ----------
snr_sat = 20000.0
res = time_echo(snr_sat)
peak_v = snr_sat * sigma_noise * RX.transimpedance
print(f"\n3) Saturated echo: noiseless peak {peak_v:.2f} V vs ADC limit {RX.adc_full_scale / 2:.2f} V")
for name, e in zip(("leading edge", "CFD", "matched filter"), res):
    print(f"   {name:<15} error {e * TO_M * 100:>8.1f} cm")
print(f"   (leading-edge walk predicted without clipping: "
      f"{float(leading_edge_walk(sigma_tot, snr_sat / T_REQ)) * TO_M * 100:.1f} cm)")

# ---------- Figure ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
s_fine = np.linspace(snr_walk[0], snr_walk[-1], 300)
ax[0].semilogx(s_fine, leading_edge_walk(sigma_tot, s_fine / T_REQ) * TO_M * 100, "k-", lw=1, label="analytic walk")
ax[0].semilogx(snr_walk, walk[:, 0] * TO_M * 100, "o", label="leading edge")
ax[0].semilogx(snr_walk, walk[:, 1] * TO_M * 100, "s", label="CFD 50 %")
ax[0].semilogx(snr_walk, walk[:, 2] * TO_M * 100, "^", label="matched filter")
ax[0].set_xlabel("echo SNR")
ax[0].set_ylabel("systematic range error [cm]")
ax[0].set_title("Walk: strong echoes cross a fixed threshold earlier")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3, which="both")

sn = np.array(snr_noise)
for key, lab, m in (("le", "leading edge", "o-"), ("cfd", "CFD 50 %", "s-"), ("mf", "matched filter", "^-")):
    ax[1].loglog(sn, [stats[s][key] for s in snr_noise], m, label=lab)
ax[1].loglog(sn, [stats[s]["crlb"] for s in snr_noise], "k--", label="Cramér-Rao bound")
ax[1].set_xlabel("echo SNR")
ax[1].set_ylabel("range jitter (1σ) [cm]")
ax[1].set_title(f"Random error ({N} shots per point)")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, which="both")

p = gaussian_pulse_power(t, T0, energy_for_snr(snr_sat), LRF_ER.pulse_width)
clean, noisy_i = simulate_current(RX, WL, t, p, 0.0, rng)
ts, _, volts = digitize(RX, t, noisy_i)
g = (ts > T0 - 30e-9) & (ts < T0 + 30e-9)
gc = (t > T0 - 30e-9) & (t < T0 + 30e-9)
ax[2].set_facecolor("black")
ax[2].plot((t[gc] - T0) * 1e9, clean[gc] * RX.transimpedance, color="orange", ls="--", lw=1, label="true echo")
ax[2].plot((ts[g] - T0) * 1e9, volts[g], color="lime", drawstyle="steps-mid", label="ADC (clipped)")
for e, c, lab in zip(res, ("cyan", "magenta", "yellow"), ("leading edge", "CFD", "matched filter")):
    ax[2].axvline(e * 1e9, color=c, lw=1, label=f"{lab}: {e * TO_M * 100:.0f} cm")
ax[2].set_ylim(-0.05, 0.4)
ax[2].grid(color="0.35", lw=0.5)
ax[2].set_xlabel("time from true echo [ns]")
ax[2].set_ylabel("TIA output [V]")
ax[2].set_title("Saturated echo")
ax[2].legend(fontsize=7, loc="upper right")
fig.tight_layout()
out = RESULTS / "exp07b_timing.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")