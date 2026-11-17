"""Exp09: end-to-end range performance, Er:glass 1550 nm vs laser diode 905 nm.

1) Validation: Monte Carlo vs analytic Pd (no aiming error, with turbulence).
2) Range table: distance at which Pd falls below 0.9 under realistic conditions.
3) Pd vs range curves.
4) Er:glass detection map: Pd over range and visibility.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lrfsim import (
    INGAAS_APD,
    LRF,
    SI_APD,
    Conditions,
    Receiver,
    System,
    Target,
    Transmitter,
    pd_analytic,
    pd_monte_carlo,
    range_at_pd,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

SYSTEMS = {
    "Er:glass 1550 nm, 1 pulse": System(
        "Er:glass", LRF(Transmitter.from_divergence(1550e-9, 10e-3, 0.5e-3), 5e-3, 5e-9, 40e-3),
        Receiver(INGAAS_APD, gain=5.0, bandwidth=100e6, transimpedance=5e3),
        n_pulses=1, window=135e-6, sun_per_nm=0.25, filter_nm=10.0, fov=0.75e-3),
    "diode 905 nm, 256 pulses": System(
        "diode", LRF(Transmitter.from_divergence(905e-9, 5e-3, 1.0e-3), 2.25e-6, 30e-9, 25e-3),
        Receiver(SI_APD, gain=100.0, bandwidth=0.5 / 30e-9, transimpedance=20e3),
        n_pulses=256, window=20e-6, sun_per_nm=0.75, filter_nm=40.0, fov=1.5e-3),
}
NATO = Target(2.3, 0.3)
CONDITIONS = {
    "clear night, tripod": Conditions(23.0, 0.0, 0.0, False, NATO),
    "clear day, tripod": Conditions(23.0, 0.0, 0.0, True, NATO),
    "clear day, turbulence, tripod": Conditions(23.0, 1e-14, 0.0, True, NATO),
    "clear day, turbulence, hand-held": Conditions(23.0, 1e-14, 0.5e-3, True, NATO),
    "haze day, turbulence, hand-held": Conditions(5.0, 1e-14, 0.5e-3, True, NATO),
    "haze day, dark target, hand-held": Conditions(5.0, 1e-14, 0.5e-3, True, Target(2.3, 0.1)),
}
RANGES = {"Er:glass 1550 nm, 1 pulse": np.linspace(500, 20000, 40), "diode 905 nm, 256 pulses": np.linspace(100, 3000, 30)}
TRIALS = {"Er:glass 1550 nm, 1 pulse": 3000, "diode 905 nm, 256 pulses": 600}

# ---------- 1) Validation ----------
print("1) Monte Carlo vs analytic Pd (Er:glass, clear day, moderate turbulence, no aiming error)")
s = SYSTEMS["Er:glass 1550 nm, 1 pulse"]
c = CONDITIONS["clear day, turbulence, tripod"]
print(f"   {'range':>8}{'analytic':>10}{'Monte Carlo':>13}")
for r in (8000.0, 10000.0, 12000.0, 14000.0):
    print(f"   {r / 1e3:>5.0f} km{pd_analytic(s, c, r):>10.3f}{pd_monte_carlo(s, c, r, 20000, rng):>13.3f}")

# ---------- 2) + 3) Range table and curves ----------
curves = {}
print("\n2) Range for Pd >= 0.9 per measurement (Pfa 1e-3)")
print(f"   {'conditions':<36}" + "".join(f"{n:>28}" for n in SYSTEMS))
for cname, cond in CONDITIONS.items():
    row = ""
    for sname, sys_ in SYSTEMS.items():
        R = RANGES[sname]
        pd = np.array([pd_monte_carlo(sys_, cond, r, TRIALS[sname], rng) for r in R])
        curves[(sname, cname)] = (R, pd)
        row += f"{range_at_pd(R, pd, 0.9) / 1e3:>25.2f} km"
    print(f"   {cname:<36}{row}")

# ---------- 4) Detection map ----------
s = SYSTEMS["Er:glass 1550 nm, 1 pulse"]
vis = np.logspace(np.log10(1.0), np.log10(50.0), 40)
Rm = np.linspace(500, 20000, 60)
pd_map = np.array([[pd_analytic(s, Conditions(v, 1e-14, 0.0, True, NATO), r) for r in Rm] for v in vis])

# ---------- Figure ----------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for a_, sname in zip(ax[:2], SYSTEMS):
    for cname in CONDITIONS:
        R, pd = curves[(sname, cname)]
        a_.plot(R / 1e3, pd, label=cname)
    a_.axhline(0.9, color="k", ls=":", lw=1)
    a_.set_xlabel("range [km]")
    a_.set_ylabel("probability of a range reading")
    a_.set_title(sname)
    a_.legend(fontsize=7)
    a_.grid(alpha=0.3)

im = ax[2].pcolormesh(Rm / 1e3, vis, pd_map, cmap="viridis", vmin=0, vmax=1, shading="auto")
cs = ax[2].contour(Rm / 1e3, vis, pd_map, levels=[0.5, 0.9, 0.99], colors="w", linewidths=1)
ax[2].clabel(cs, fmt="%.2f", fontsize=8)
ax[2].set_yscale("log")
ax[2].set_xlabel("range [km]")
ax[2].set_ylabel("visibility [km]")
ax[2].set_title("Er:glass: Pd over range and weather (day, turbulence, tripod)")
fig.colorbar(im, ax=ax[2], label="Pd")
fig.tight_layout()
out = RESULTS / "exp09_performance.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")