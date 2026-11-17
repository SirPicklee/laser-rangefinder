"""Exp01: time-of-flight basics.

1) Numbers every LRF engineer should know: time per metre, counter bins, unambiguous range,
   two-target separation, error from ignoring the refractive index of air.
2) Quantization and jitter: measured range vs true range inside one counter bin.
3) Averaging N shots: does it help? (dithering)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lrfsim import (
    N_AIR,
    max_unambiguous_range,
    measure,
    range_bin,
    range_from_tof,
    two_target_separation,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

# ---------- 1) Basic numbers ----------
print("1) Basic numbers")
print(f"   1 ns round trip = {range_from_tof(1e-9) * 100:.2f} cm of range")
print(f"   {'counter clock':>14}{'range bin [m]':>15}")
for f in (100e6, 500e6, 1e9, 4e9):
    print(f"   {f / 1e6:>10.0f} MHz{range_bin(f):>15.3f}")
print(f"   {'PRF':>14}{'max range [km]':>16}")
for prf in (1, 10, 1e3, 10e3, 100e3):
    print(f"   {prf:>11.0f} Hz{max_unambiguous_range(prf) / 1e3:>16.1f}")
print(f"   {'pulse width':>14}{'2-target sep. [m]':>19}")
for tau in (2e-9, 5e-9, 10e-9, 20e-9):
    print(f"   {tau * 1e9:>11.0f} ns{two_target_separation(tau):>19.2f}")
print(f"   ignoring the air index (n = {N_AIR}):")
for r in (1e3, 3e3, 5e3, 10e3):
    err = r * (N_AIR - 1)
    print(f"   {r / 1e3:>8.0f} km -> error {err:.2f} m")

# ---------- 2) Quantization and jitter ----------
clock = 500e6
dR = range_bin(clock)
dt = 1.0 / clock
base = 1000.0
offsets = np.linspace(0.0, dR, 61)
jitters = {"no jitter": 0.0, "jitter = 0.3 bin": 0.3 * dt, "jitter = 1 bin": 1.0 * dt}
N_AVG = 200

print(f"\n2) Clock {clock / 1e6:.0f} MHz, bin = {dR:.3f} m, quantization limit bin/sqrt(12) = {dR / np.sqrt(12):.3f} m")
curves = {}
for name, j in jitters.items():
    errs = np.array([measure(base + o, N_AVG, clock, j, rng).mean() - (base + o) for o in offsets])
    curves[name] = errs
    print(f"   {name:<18} average of {N_AVG} shots: RMS error over the bin = {np.sqrt(np.mean(errs**2)):.3f} m")

# ---------- 3) Averaging ----------
Ns = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
test_offsets = np.linspace(0.0, dR, 31)
rms = {}
for name, j in jitters.items():
    rms[name] = []
    for N in Ns:
        errs = [measure(base + o, N, clock, j, rng).mean() - (base + o) for o in test_offsets]
        rms[name].append(np.sqrt(np.mean(np.square(errs))))
    rms[name] = np.array(rms[name])

print("\n3) RMS range error [m] after averaging N shots")
print(f"   {'N':>6}" + "".join(f"{name:>20}" for name in jitters))
for k, N in enumerate(Ns):
    if N in (1, 10, 100, 1000):
        print(f"   {N:>6}" + "".join(f"{rms[name][k]:>20.4f}" for name in jitters))

# ---------- Figure ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name, errs in curves.items():
    ax[0].plot(offsets / dR, errs * 100, label=name)
ax[0].set_xlabel("true position inside the counter bin [bins]")
ax[0].set_ylabel(f"error of the {N_AVG}-shot average [cm]")
ax[0].set_title("Quantization: the error depends on where the target sits")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

for name in jitters:
    ax[1].loglog(Ns, rms[name] * 100, "o-", ms=4, label=name)
ax[1].loglog(Ns, rms["jitter = 1 bin"][0] * 100 / np.sqrt(Ns), "k--", lw=1, label="1/sqrt(N)")
ax[1].set_xlabel("shots averaged N")
ax[1].set_ylabel("RMS range error [cm]")
ax[1].set_title("Averaging only helps when there is noise (dithering)")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, which="both")
fig.tight_layout()
out = RESULTS / "exp01_tof.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")