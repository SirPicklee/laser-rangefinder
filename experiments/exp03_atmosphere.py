"""Exp03: the atmosphere between the LRF and the target.

1) Extinction vs visibility at 905 nm and 1550 nm (Kim model), one-way dB/km and round-trip
   transmission at 2 km.
2) Range at which the round trip costs 20 dB (1 %).
3) Turbulence: Rytov variance, scintillation index (with saturation) and the probability that a
   single shot gets less than half the mean light.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lrfsim import (
    attenuation_db_per_km,
    extinction,
    fade_probability,
    lognormal_samples,
    rytov_variance,
    scintillation_index,
    transmission,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
WL = {"905 nm": 905e-9, "1550 nm": 1550e-9}
WEATHER = {
    "very clear": 50.0,
    "clear": 23.0,
    "light haze": 10.0,
    "haze": 5.0,
    "mist": 2.0,
    "light fog": 1.0,
    "moderate fog": 0.5,
    "thick fog": 0.2,
}

# ---------- 1) Extinction ----------
print("1) Extinction from visibility (Kim model)")
print(f"   {'weather':<14}{'V [km]':>7}{'905 dB/km':>11}{'1550 dB/km':>12}{'T 2 km 905':>12}{'T 2 km 1550':>13}")
for name, v in WEATHER.items():
    print(f"   {name:<14}{v:>7.1f}{attenuation_db_per_km(v, 905e-9):>11.3f}{attenuation_db_per_km(v, 1550e-9):>12.3f}"
          f"{float(transmission(2e3, v, 905e-9)):>12.2e}{float(transmission(2e3, v, 1550e-9)):>13.2e}")

# ---------- 2) 20 dB range ----------
print("\n2) Range where the round trip costs 20 dB (1 % left)")
print(f"   {'weather':<14}{'905 nm':>10}{'1550 nm':>10}")
for name, v in WEATHER.items():
    r = [math.log(100) / (2 * extinction(v, wl)) for wl in WL.values()]
    print(f"   {name:<14}{r[0] / 1e3:>7.2f} km{r[1] / 1e3:>7.2f} km")

# ---------- 3) Turbulence ----------
CN2 = {"weak 1e-16": 1e-16, "moderate 1e-14": 1e-14, "strong 1e-13": 1e-13}
print("\n3) Turbulence (plane wave, one way)")
print(f"   {'Cn^2':<16}{'wl':>8}" + "".join(f"{'Rytov ' + str(d) + 'km':>13}{'index':>8}" for d in (1, 3, 5))
      + f"{'P(fade<0.5) 3 km':>18}")
for cname, cn2 in CN2.items():
    for wname, wl in WL.items():
        row = ""
        for d in (1, 3, 5):
            row += f"{float(rytov_variance(cn2, wl, d * 1e3)):>13.3f}{float(scintillation_index(cn2, wl, d * 1e3)):>8.3f}"
        idx3 = float(scintillation_index(cn2, wl, 3e3))
        print(f"   {cname:<16}{wname:>8}{row}{fade_probability(idx3, 0.5):>18.3f}")

# ---------- Figure ----------
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
R = np.linspace(0, 10e3, 400)
for v, ls in ((23.0, "-"), (5.0, "--"), (1.0, ":")):
    for wname, wl in WL.items():
        ax[0, 0].semilogy(R / 1e3, transmission(R, v, wl), ls=ls, color="tab:blue" if wname == "905 nm" else "tab:red",
                          label=f"{wname}, V = {v:g} km")
ax[0, 0].set_ylim(1e-6, 1.2)
ax[0, 0].set_xlabel("range [km]")
ax[0, 0].set_ylabel("round-trip transmission")
ax[0, 0].set_title("Beer-Lambert extinction")
ax[0, 0].legend(fontsize=7)
ax[0, 0].grid(alpha=0.3, which="both")

V = np.logspace(np.log10(0.1), np.log10(80), 200)
for wname, wl in WL.items():
    ax[0, 1].loglog(V, [attenuation_db_per_km(v, wl) for v in V], label=wname)
ax[0, 1].set_xlabel("visibility [km]")
ax[0, 1].set_ylabel("one-way attenuation [dB/km]")
ax[0, 1].set_title("Haze favours 1550 nm, thick fog does not")
ax[0, 1].legend(fontsize=8)
ax[0, 1].grid(alpha=0.3, which="both")

R2 = np.linspace(100, 10e3, 300)
for cname, cn2 in CN2.items():
    line, = ax[1, 0].loglog(R2 / 1e3, scintillation_index(cn2, 1550e-9, R2), label=f"index, Cn² {cname}")
    ax[1, 0].loglog(R2 / 1e3, rytov_variance(cn2, 1550e-9, R2), ls="--", color=line.get_color(), lw=1)
ax[1, 0].plot([], [], "k--", lw=1, label="Rytov variance (weak-turbulence theory)")
ax[1, 0].set_ylim(1e-4, 1e2)
ax[1, 0].set_xlabel("range [km]")
ax[1, 0].set_ylabel("scintillation index (1550 nm)")
ax[1, 0].set_title("Scintillation grows, then saturates")
ax[1, 0].legend(fontsize=7)
ax[1, 0].grid(alpha=0.3, which="both")

x = np.linspace(0.01, 4, 400)
for cname in ("moderate 1e-14", "strong 1e-13"):
    idx = float(scintillation_index(CN2[cname], 1550e-9, 3e3))
    samples = lognormal_samples(idx, 50000, rng)
    s2 = math.log1p(idx)
    pdf = np.exp(-(np.log(x) + s2 / 2) ** 2 / (2 * s2)) / (x * math.sqrt(2 * math.pi * s2))
    line, = ax[1, 1].plot(x, pdf, label=f"{cname}: index {idx:.2f}")
    ax[1, 1].hist(samples, bins=120, range=(0, 4), density=True, alpha=0.3, color=line.get_color())
ax[1, 1].axvline(0.5, color="k", ls="--", lw=1, label="half the mean")
ax[1, 1].set_xlabel("intensity / mean")
ax[1, 1].set_ylabel("probability density")
ax[1, 1].set_title("Single-shot intensity at 3 km (log-normal)")
ax[1, 1].legend(fontsize=7)
ax[1, 1].grid(alpha=0.3)
fig.tight_layout()
out = RESULTS / "exp03_atmosphere.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")