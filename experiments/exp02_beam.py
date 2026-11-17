"""Exp02: the laser beam on its way to the target.

1) Diffraction limit: the narrowest beam a given exit aperture allows, 905 nm vs 1550 nm.
2) Spot size at range for typical LRF divergences.
3) How much of the beam hits a 2.3 m x 2.3 m NATO target, and where the target stops filling the beam.
4) Aiming jitter: the average fraction always favours the narrowest beam, but what matters for a
   range reading is the probability that ENOUGH light lands on the target in a given shot.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lrfsim import Transmitter, mean_fraction, pointing_fractions

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

LAM = 1550e-9
W0 = 10e-3  # 1/e^2 beam radius at the exit aperture (20 mm beam)
TARGET = 2.3  # NATO standard target side [m]
DIVS = [0.3e-3, 0.5e-3, 1.0e-3, 2.0e-3]  # full-angle divergences

# ---------- 1) Diffraction limit ----------
print("1) Narrowest possible beam (M^2 = 1), full-angle divergence [mrad]")
print(f"   {'exit radius':>12}{'905 nm':>10}{'1550 nm':>10}")
for w0 in (5e-3, 10e-3, 20e-3):
    d905 = 2 * 905e-9 / (math.pi * w0) * 1e3
    d1550 = 2 * 1550e-9 / (math.pi * w0) * 1e3
    print(f"   {w0 * 1e3:>9.0f} mm{d905:>10.3f}{d1550:>10.3f}")

# ---------- 2) Spot size ----------
txs = {d: Transmitter.from_divergence(LAM, W0, d) for d in DIVS}
ranges = [500, 1000, 2000, 5000, 10000]
print(f"\n2) Spot diameter 2w [m] at range (1550 nm, 20 mm exit beam)")
print(f"   {'divergence':>12}{'M^2':>6}" + "".join(f"{r / 1e3:>8.1f} km" for r in ranges))
for d, tx in txs.items():
    print(f"   {d * 1e3:>8.1f} mrad{tx.m2:>6.1f}" + "".join(f"{2 * float(tx.radius_at(r)):>11.2f}" for r in ranges))

# ---------- 3) Fraction on the target ----------
print(f"\n3) Fraction of the beam on a {TARGET} m x {TARGET} m target")
print(f"   {'divergence':>12}" + "".join(f"{r / 1e3:>8.1f} km" for r in ranges) + "   f < 0.9 beyond   f < 0.5 beyond")
R_grid = np.linspace(100, 20000, 4000)
for d, tx in txs.items():
    f = tx.fraction_on_target(R_grid, TARGET)
    r90 = R_grid[np.argmax(f < 0.9)] if np.any(f < 0.9) else np.nan
    r50 = R_grid[np.argmax(f < 0.5)] if np.any(f < 0.5) else np.nan
    row = "".join(f"{float(tx.fraction_on_target(r, TARGET)):>11.3f}" for r in ranges)
    print(f"   {d * 1e3:>8.1f} mrad{row}{r90 / 1e3:>12.2f} km{r50 / 1e3:>14.2f} km")

# ---------- 4) Aiming jitter ----------
R_AIM = 3000.0
F_REQ = 0.25  # fraction needed for a range reading (placeholder; Step 4 gives the real number)
sigmas = {"tripod (0.05 mrad)": 0.05e-3, "braced (0.2 mrad)": 0.2e-3, "hand-held (0.5 mrad)": 0.5e-3}
div_scan = np.linspace(0.12e-3, 4.0e-3, 60)
print(f"\n4) Target at {R_AIM / 1e3:.0f} km with random aiming error (per axis), 4000 shots per point")
print(f"   {'aiming':<22}{'mean f @0.12':>13}{'mean f @1.0':>12}{'best div for hit':>17}{'P(f >= ' + str(F_REQ) + ')':>14}")
mean_curves, hit_curves = {}, {}
for name, s in sigmas.items():
    means = np.array([float(mean_fraction(Transmitter.from_divergence(LAM, W0, d), R_AIM, TARGET, s)) for d in div_scan])
    hits = np.array([
        (pointing_fractions(Transmitter.from_divergence(LAM, W0, d), R_AIM, TARGET, s, 4000, rng) >= F_REQ).mean()
        for d in div_scan
    ])
    mean_curves[name], hit_curves[name] = means, hits
    k = int(np.argmax(hits))
    m1 = float(mean_fraction(Transmitter.from_divergence(LAM, W0, 1e-3), R_AIM, TARGET, s))
    print(f"   {name:<22}{means[0]:>13.3f}{m1:>12.3f}{div_scan[k] * 1e3:>12.2f} mrad{hits[k]:>14.3f}")

# ---------- Figure ----------
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
R_plot = np.linspace(0, 10000, 500)
for d, tx in txs.items():
    ax[0, 0].plot(R_plot / 1e3, 2 * tx.radius_at(R_plot), label=f"{d * 1e3:.1f} mrad")
ax[0, 0].axhline(TARGET, color="k", ls="--", lw=1, label=f"target {TARGET} m")
ax[0, 0].set_xlabel("range [km]")
ax[0, 0].set_ylabel("spot diameter 2w [m]")
ax[0, 0].set_title("Beam size at the target")
ax[0, 0].legend(fontsize=8)
ax[0, 0].grid(alpha=0.3)

for d, tx in txs.items():
    ax[0, 1].semilogy(R_grid / 1e3, tx.fraction_on_target(R_grid, TARGET), label=f"{d * 1e3:.1f} mrad")
ax[0, 1].set_xlabel("range [km]")
ax[0, 1].set_ylabel("fraction of beam on target")
ax[0, 1].set_title("Extended target -> point target")
ax[0, 1].legend(fontsize=8)
ax[0, 1].grid(alpha=0.3, which="both")

colors = dict(zip(sigmas, ("tab:blue", "tab:orange", "tab:red")))
for name in sigmas:
    ax[1, 0].plot(div_scan * 1e3, hit_curves[name], color=colors[name], label=f"{name}: P(f >= {F_REQ})")
    ax[1, 0].plot(div_scan * 1e3, mean_curves[name], color=colors[name], ls="--", lw=1)
ax[1, 0].plot([], [], "k--", lw=1, label="mean fraction (exact)")
ax[1, 0].set_xlabel("full divergence [mrad]")
ax[1, 0].set_ylabel(f"at {R_AIM / 1e3:.0f} km")
ax[1, 0].set_title("Mean fraction vs probability of enough light")
ax[1, 0].legend(fontsize=7)
ax[1, 0].grid(alpha=0.3)

tx = txs[0.5e-3]
w = float(tx.radius_at(R_AIM))
g = np.linspace(-3 * w, 3 * w, 300)
X, Y = np.meshgrid(g, g)
ax[1, 1].imshow(np.exp(-2 * (X**2 + Y**2) / w**2), extent=[g[0], g[-1], g[0], g[-1]], origin="lower", cmap="inferno")
ax[1, 1].add_patch(plt.Rectangle((-TARGET / 2, -TARGET / 2), TARGET, TARGET, fill=False, color="c", lw=2, label="target"))
ax[1, 1].add_patch(plt.Circle((0, 0), w, fill=False, color="w", ls="--", label="1/e² radius"))
ax[1, 1].set_xlabel("x [m]")
ax[1, 1].set_ylabel("y [m]")
ax[1, 1].set_title(f"0.5 mrad beam at {R_AIM / 1e3:.0f} km")
ax[1, 1].legend(fontsize=8, loc="upper right")
fig.tight_layout()
out = RESULTS / "exp02_beam.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")