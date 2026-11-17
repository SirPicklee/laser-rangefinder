"""Exp04: link budget of two laser rangefinders.

Er:glass 1550 nm (military, long range) vs pulsed laser diode 905 nm (commercial).
1) The two designs.
2) Link budget in dB at 3 km, clear air: where does the energy go?
3) Photons back per pulse vs range, clear air and haze.
4) The geometry: 1/R^2 (extended target) turning into 1/R^4 (point target).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lrfsim import LRF, Target, Transmitter, link_budget_db, photons, received_energy, received_power, total_factor

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

DESIGNS = {
    "Er:glass 1550 nm": LRF(Transmitter.from_divergence(1550e-9, 10e-3, 0.5e-3), 5e-3, 5e-9, 40e-3),
    "laser diode 905 nm": LRF(Transmitter.from_divergence(905e-9, 5e-3, 1.0e-3), 2.25e-6, 30e-9, 25e-3),
}
NATO = Target(side=2.3, reflectivity=0.3)
CLEAR, HAZE, NO_ATM = 23.0, 5.0, 1e9  # visibility [km]

# ---------- 1) Designs ----------
print("1) The two designs")
print(f"   {'design':<20}{'energy':>10}{'width':>8}{'peak power':>12}{'divergence':>12}{'rx lens':>9}{'photons sent':>14}")
for name, d in DESIGNS.items():
    print(f"   {name:<20}{d.pulse_energy * 1e3:>7.4f} mJ{d.pulse_width * 1e9:>5.0f} ns{d.peak_power / 1e3:>9.2f} kW"
          f"{d.tx.divergence_full * 1e3:>8.2f} mrad{d.rx_aperture * 1e3:>6.0f} mm{float(photons(d.pulse_energy, d.wavelength)):>14.2e}")

# ---------- 2) Link budget at 3 km ----------
R0 = 3000.0
d = DESIGNS["Er:glass 1550 nm"]
budget = link_budget_db(d, NATO, R0, CLEAR)
print(f"\n2) Link budget, Er:glass, NATO target (rho = {NATO.reflectivity}), {R0 / 1e3:.0f} km, visibility {CLEAR:.0f} km")
for k, v in budget.items():
    print(f"   {k:<30}{v:>8.1f} dB")
print(f"   {'TOTAL':<30}{sum(budget.values()):>8.1f} dB")
e = float(received_energy(d, NATO, R0, CLEAR))
print(f"   received: {e * 1e15:.1f} fJ per pulse = {float(photons(e, d.wavelength)):.0f} photons, "
      f"peak power {float(received_power(d, NATO, R0, CLEAR)) * 1e9:.1f} nW")

# ---------- 3) Photons vs range ----------
ranges = [500, 1000, 2000, 3000, 5000, 10000]
print("\n3) Photons back per pulse (NATO target)")
print(f"   {'design':<20}{'weather':>8}" + "".join(f"{r / 1e3:>9.1f} km" for r in ranges))
for name, d in DESIGNS.items():
    for wname, v in (("clear", CLEAR), ("haze", HAZE)):
        row = "".join(f"{float(photons(received_energy(d, NATO, r, v), d.wavelength)):>12.3g}" for r in ranges)
        print(f"   {name:<20}{wname:>8}{row}")

# ---------- 4) Geometry slope ----------
R = np.logspace(2, np.log10(30e3), 300)
print("\n4) Local slope d log(P) / d log(R), geometry only (no atmosphere)")
print(f"   {'design':<20}" + "".join(f"{r:>9.1f} km" for r in (0.5, 2, 5, 10, 20)))
slopes = {}
for name, d in DESIGNS.items():
    P = total_factor(d, NATO, R, NO_ATM)
    s = np.gradient(np.log(P), np.log(R))
    slopes[name] = s
    print(f"   {name:<20}" + "".join(f"{np.interp(r, R, s):>12.2f}" for r in (500, 2000, 5000, 10000, 20000)))

# ---------- Figure ----------
fig, ax = plt.subplots(2, 2, figsize=(12, 9.5))

names, vals = list(budget.keys()), list(budget.values())
cum = np.concatenate(([0.0], np.cumsum(vals)))
for i, (n, v) in enumerate(zip(names, vals)):
    ax[0, 0].bar(i, v, bottom=cum[i], color="tab:red" if v < -3 else "tab:orange")
    ax[0, 0].text(i, cum[i + 1] - 2, f"{v:.1f}", ha="center", va="top", fontsize=8)
ax[0, 0].bar(len(vals), cum[-1], color="k")
ax[0, 0].text(len(vals), cum[-1] - 2, f"{cum[-1]:.1f}", ha="center", va="top", fontsize=8)
ax[0, 0].set_xticks(range(len(vals) + 1), names + ["total"], rotation=30, ha="right", fontsize=8)
ax[0, 0].set_ylabel("dB")
ax[0, 0].set_title(f"Where the energy goes (Er:glass, {R0 / 1e3:.0f} km, clear air)")
ax[0, 0].grid(alpha=0.3, axis="y")

Rl = np.linspace(200, 15000, 400)
for name, d in DESIGNS.items():
    for v, ls in ((CLEAR, "-"), (HAZE, "--")):
        ax[0, 1].semilogy(Rl / 1e3, photons(received_energy(d, NATO, Rl, v), d.wavelength), ls=ls,
                          label=f"{name}, V = {v:g} km")
ax[0, 1].set_xlabel("range [km]")
ax[0, 1].set_ylabel("photons back per pulse")
ax[0, 1].set_title("Received photons (NATO target, rho = 0.3)")
ax[0, 1].legend(fontsize=7)
ax[0, 1].grid(alpha=0.3, which="both")

for name, s in slopes.items():
    ax[1, 0].semilogx(R / 1e3, s, label=name)
ax[1, 0].axhline(-2, color="k", ls=":", lw=1)
ax[1, 0].axhline(-4, color="k", ls=":", lw=1)
ax[1, 0].text(0.12, -1.85, "extended target: 1/R²", fontsize=8)
ax[1, 0].text(0.12, -3.85, "point target: 1/R⁴", fontsize=8)
ax[1, 0].set_ylim(-4.5, -1.5)
ax[1, 0].set_xlabel("range [km]")
ax[1, 0].set_ylabel("d log P / d log R")
ax[1, 0].set_title("Geometry only: the slope steepens when the target no longer fills the beam")
ax[1, 0].legend(fontsize=8)
ax[1, 0].grid(alpha=0.3, which="both")

d = DESIGNS["Er:glass 1550 nm"]
for rho in (0.1, 0.3, 0.6):
    ax[1, 1].semilogy(Rl / 1e3, photons(received_energy(d, Target(2.3, rho), Rl, CLEAR), d.wavelength),
                      label=f"rho = {rho}")
ax[1, 1].set_xlabel("range [km]")
ax[1, 1].set_ylabel("photons back per pulse")
ax[1, 1].set_title("Er:glass, clear air: dark vs bright target")
ax[1, 1].legend(fontsize=8)
ax[1, 1].grid(alpha=0.3, which="both")
fig.tight_layout()
out = RESULTS / "exp04_link_budget.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")