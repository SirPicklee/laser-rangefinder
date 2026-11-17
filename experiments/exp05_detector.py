"""Exp05: APD receiver noise and SNR.

1) The two detectors: responsivity and excess noise.
2) Noise budget at 3 km (Er:glass) and 1 km (laser diode), night and sunny day, at the optimum gain.
3) SNR vs APD gain: why there is an optimum.
4) SNR vs range at the optimum gain; maximum range for SNR >= 10 (placeholder threshold, Step 7).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lrfsim import (
    INGAAS_APD,
    LRF,
    SI_APD,
    Target,
    Transmitter,
    bandwidth_for_pulse,
    noise_terms,
    optimum_gain,
    received_power,
    snr,
    solar_background,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

NATO = Target(2.3, 0.3)
SNR_REQ = 10.0
SYSTEMS = {
    "Er:glass 1550 nm + InGaAs": dict(
        lrf=LRF(Transmitter.from_divergence(1550e-9, 10e-3, 0.5e-3), 5e-3, 5e-9, 40e-3),
        apd=INGAAS_APD, sun=0.25, filter_nm=10.0, fov=0.75e-3, r_test=3000.0),
    "diode 905 nm + Si": dict(
        lrf=LRF(Transmitter.from_divergence(905e-9, 5e-3, 1.0e-3), 2.25e-6, 30e-9, 25e-3),
        apd=SI_APD, sun=0.75, filter_nm=40.0, fov=1.5e-3, r_test=1000.0),
}


def background(s, day):
    if not day:
        return 0.0
    lrf = s["lrf"]
    return solar_background(s["sun"], s["filter_nm"], NATO.reflectivity, lrf.rx_area, s["fov"], lrf.rx_optics)


# ---------- 1) Detectors ----------
print("1) Detectors")
print(f"   {'detector':<12}{'R0 [A/W]':>10}{'k':>6}{'F(10)':>8}{'F(50)':>8}{'F(100)':>8}")
for name, s in SYSTEMS.items():
    a, wl = s["apd"], s["lrf"].wavelength
    print(f"   {a.name:<12}{a.responsivity(wl):>10.3f}{a.k_ion:>6.2f}"
          + "".join(f"{float(a.excess_noise(m)):>8.2f}" for m in (10, 50, 100)))

# ---------- 2) Noise budget ----------
print("\n2) Noise budget at the optimum gain (share of the total noise variance)")
budgets = {}
for name, s in SYSTEMS.items():
    lrf, apd, R = s["lrf"], s["apd"], s["r_test"]
    B = bandwidth_for_pulse(lrf.pulse_width)
    ps = float(received_power(lrf, NATO, R, 23.0))
    for day in (False, True):
        pb = background(s, day)
        M, S = optimum_gain(apd, lrf.wavelength, ps, pb, B)
        terms = {k: float(v) for k, v in noise_terms(apd, lrf.wavelength, ps, pb, M, B).items()}
        tot = sum(terms.values())
        label = f"{name}, {R / 1e3:.0f} km, {'day' if day else 'night'}"
        budgets[label] = {k: v / tot for k, v in terms.items()}
        print(f"   {label}")
        print(f"      signal {ps * 1e9:.2f} nW, background {pb * 1e9:.3f} nW, B = {B / 1e6:.0f} MHz -> "
              f"optimum M = {M:.1f}, SNR = {S:.1f}")
        print("      " + ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in terms.items()))

# ---------- 3) SNR vs gain ----------
gains = np.linspace(1, 300, 600)
gain_curves = {}
for name, s in SYSTEMS.items():
    lrf, apd, R = s["lrf"], s["apd"], s["r_test"]
    B = bandwidth_for_pulse(lrf.pulse_width)
    ps = float(received_power(lrf, NATO, R, 23.0))
    g = gains[gains <= apd.max_gain]
    for day in (False, True):
        gain_curves[(name, day)] = (g, snr(apd, lrf.wavelength, ps, background(s, day), g, B))

# ---------- 4) SNR vs range ----------
R = np.linspace(200, 15000, 150)
range_curves = {}
print(f"\n4) Maximum range with SNR >= {SNR_REQ:.0f} (NATO target, optimum gain at every range)")
print(f"   {'system':<28}{'clear night':>12}{'clear day':>11}{'haze night':>12}{'haze day':>10}")
for name, s in SYSTEMS.items():
    lrf, apd = s["lrf"], s["apd"]
    B = bandwidth_for_pulse(lrf.pulse_width)
    row = ""
    for vis in (23.0, 5.0):
        for day in (False, True):
            pb = background(s, day)
            snrs = np.array([optimum_gain(apd, lrf.wavelength, float(received_power(lrf, NATO, r, vis)), pb, B)[1]
                             for r in R])
            range_curves[(name, vis, day)] = snrs
            r_max = R[np.argmax(snrs < SNR_REQ)] if np.any(snrs < SNR_REQ) else np.nan
            row += f"{r_max / 1e3:>9.2f} km"
    print(f"   {name:<28}{row}")

# ---------- Figure ----------
fig, ax = plt.subplots(2, 2, figsize=(12, 9.5))
for a, name in zip((ax[0, 0], ax[0, 1]), SYSTEMS):
    for day, ls in ((False, "-"), (True, "--")):
        g, sn = gain_curves[(name, day)]
        a.plot(g, sn, ls=ls, label="sunny day" if day else "night")
        k = int(np.argmax(sn))
        a.plot(g[k], sn[k], "ko", ms=4)
    a.set_xlabel("APD gain M")
    a.set_ylabel("SNR")
    a.set_title(f"{name} at {SYSTEMS[name]['r_test'] / 1e3:.0f} km (dots = optimum)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

styles = {(23.0, False): "-", (23.0, True): "--", (5.0, False): "-.", (5.0, True): ":"}
for (name, vis, day), snrs in range_curves.items():
    color = "tab:red" if "Er" in name else "tab:blue"
    ax[1, 0].semilogy(R / 1e3, snrs, ls=styles[(vis, day)], color=color,
                      label=f"{name.split(' +')[0]}, V {vis:g} km, {'day' if day else 'night'}")
ax[1, 0].axhline(SNR_REQ, color="k", lw=1)
ax[1, 0].set_ylim(0.1, 1e4)
ax[1, 0].set_xlabel("range [km]")
ax[1, 0].set_ylabel("SNR at optimum gain")
ax[1, 0].set_title(f"SNR vs range (black line: SNR = {SNR_REQ:.0f})")
ax[1, 0].legend(fontsize=6)
ax[1, 0].grid(alpha=0.3, which="both")

labels = list(budgets)
sources = list(next(iter(budgets.values())).keys())
bottom = np.zeros(len(labels))
for src in sources:
    vals = np.array([budgets[l][src] for l in labels])
    ax[1, 1].bar(range(len(labels)), vals, bottom=bottom, label=src)
    bottom += vals
ax[1, 1].set_xticks(range(len(labels)), [l.replace(", ", "\n") for l in labels], fontsize=6)
ax[1, 1].set_ylabel("share of noise variance")
ax[1, 1].set_title("Which noise dominates at the optimum gain?")
ax[1, 1].legend(fontsize=7)
fig.tight_layout()
out = RESULTS / "exp05_detector.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")