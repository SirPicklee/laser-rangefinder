"""Exp10: laser eye safety, 905 nm diode vs 1550 nm Er:glass (simplified IEC 60825-1).

1) MPE vs pulse duration for both wavelengths.
2) Our two rangefinders: exposure / MPE per rule, NOHD with and without 7x50 binoculars, Class 1 limit.
3) Design trade: largest Class 1 pulse energy of the diode vs PRF and exit beam size.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lrfsim import (
    OpticalAid,
    Transmitter,
    exposure_ratio,
    exposure_ratios,
    max_class1_energy,
    mpe_single,
    nohd,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

BINOCULARS = OpticalAid(50e-3, 0.7, 7.0)
SYSTEMS = {
    "diode 905 nm": dict(tx=Transmitter.from_divergence(905e-9, 5e-3, 1.0e-3), energy=2.25e-6, width=30e-9, prf=10e3),
    "Er:glass 1550 nm": dict(tx=Transmitter.from_divergence(1550e-9, 10e-3, 0.5e-3), energy=5e-3, width=5e-9, prf=1.0),
}

# ---------- 1) MPE table ----------
print("1) Single-exposure MPE at the cornea [J/m^2]")
print(f"   {'duration':>10}{'905 nm':>12}{'1550 nm':>12}{'1550/905':>12}")
for t in (1e-9, 5e-9, 30e-9, 1e-6, 1e-3, 1.0):
    a, b = mpe_single(905e-9, t), mpe_single(1550e-9, t)
    print(f"   {t:>10.0e}{a:>12.3e}{b:>12.3e}{b / a:>12.0f}")

# ---------- 2) Our rangefinders ----------
print("\n2) Our rangefinders (continuous ranging, 10 s accidental exposure)")
for name, s in SYSTEMS.items():
    tx, e, tp, prf = s["tx"], s["energy"], s["width"], s["prf"]
    print(f"   {name}: {e * 1e6:.2f} uJ, {tp * 1e9:.0f} ns, {prf:.0f} Hz")
    for rule, r in exposure_ratios(tx, e, tp, 0.1, prf).items():
        print(f"      exposure/MPE at 10 cm, {rule:<13}: {r:10.3g}")
    print(f"      NOHD naked eye             : {nohd(tx, e, tp, prf):10.1f} m")
    print(f"      NOHD with 7x50 binoculars  : {nohd(tx, e, tp, prf, aid=BINOCULARS):10.1f} m")
    emax = max_class1_energy(tx, tp, prf)
    verdict = "Class 1" if e <= emax else "NOT Class 1"
    print(f"      largest Class 1 energy     : {emax * 1e6:10.3g} uJ  -> {verdict}")

# ---------- 3) Diode design trade ----------
prfs = np.logspace(1, np.log10(5e4), 40)
beams = (5e-3, 10e-3, 20e-3)
curves = {w: np.array([max_class1_energy(Transmitter.from_divergence(905e-9, w, 1.0e-3), 30e-9, p) for p in prfs])
          for w in beams}
print("\n3) Diode: largest Class 1 pulse energy [uJ] vs PRF and exit beam radius")
print(f"   {'PRF':>9}" + "".join(f"{'w0 = ' + str(int(w * 1e3)) + ' mm':>14}" for w in beams))
for p in (100.0, 1e3, 1e4):
    row = "".join(f"{max_class1_energy(Transmitter.from_divergence(905e-9, w, 1.0e-3), 30e-9, p) * 1e6:>14.3g}"
                  for w in beams)
    print(f"   {p:>6.0f} Hz{row}")

# ---------- Figure ----------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

t = np.logspace(-9, 1, 200)
ax[0].loglog(t, [mpe_single(905e-9, x) for x in t], label="905 nm (retina)")
ax[0].loglog(t, [mpe_single(1550e-9, x) for x in t], label="1550 nm (cornea)")
ax[0].set_xlabel("exposure duration [s]")
ax[0].set_ylabel("MPE [J/m$^2$]")
ax[0].set_title("Maximum permissible exposure")
ax[0].legend()
ax[0].grid(alpha=0.3, which="both")

z = np.logspace(-1, 4, 200)
for (name, s), color in zip(SYSTEMS.items(), ("tab:red", "tab:blue")):
    for aid, ls, label in ((None, "-", "naked eye"), (BINOCULARS, "--", "7x50 binoculars")):
        r = [exposure_ratio(s["tx"], s["energy"], s["width"], zz, s["prf"], aid=aid) for zz in z]
        ax[1].loglog(z, r, ls, color=color, label=f"{name}, {label}")
ax[1].axhline(1.0, color="k", ls=":", lw=1)
ax[1].set_xlabel("distance from the rangefinder [m]")
ax[1].set_ylabel("exposure / MPE (worst rule)")
ax[1].set_title("Hazard vs distance (above 1 = hazardous)")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, which="both")

for w, em in curves.items():
    ax[2].loglog(prfs, em * 1e6, label=f"exit beam radius {w * 1e3:.0f} mm")
ax[2].plot([SYSTEMS["diode 905 nm"]["prf"]], [SYSTEMS["diode 905 nm"]["energy"] * 1e6], "r*", ms=12,
           label="our diode design")
ax[2].set_xlabel("pulse repetition frequency [Hz]")
ax[2].set_ylabel("largest Class 1 pulse energy [uJ]")
ax[2].set_title("905 nm diode: Class 1 design space")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.3, which="both")

fig.tight_layout()
out = RESULTS / "exp10_eye_safety.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")