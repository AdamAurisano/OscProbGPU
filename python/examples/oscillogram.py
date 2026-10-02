"""Atmospheric oscillogram P(numu -> numu) and bin-averaged version.

Usage: python oscillogram.py [cpu]
Writes oscillogram.npz (and oscillogram.png if matplotlib is available).
"""
import sys
import time

import numpy as np

import oscprobgpu as opg

devices = [] if "cpu" in sys.argv else opg.default_devices()
p = opg.Fast(devices=devices)
p.set_angle(2, 3, np.arcsin(np.sqrt(0.451)))
p.set_delta(1, 3, np.radians(232))

E = np.geomspace(1, 100, 1000)
C = np.linspace(-1, 0, 1000)
p.set_grid(E, C)

t0 = time.perf_counter()
p.calculate()
P = p.probs()
t1 = time.perf_counter()
print(f"{'GPU' if p.on_gpu else 'CPU'}: 2 x {E.size} x {C.size} points in "
      f"{t1 - t0:.3f} s")

# Bin averages: 40 log-E bins x 20 cosZ bins, 8 GL nodes per direction
Ee = np.geomspace(1, 100, 41)
Ce = np.linspace(-1, 0, 21)
p.set_bins(Ee, Ce, n_gl_energy=8, n_gl_cosine=8, measure="log")
p.calculate_binned()
A = p.binned()

np.savez("oscillogram.npz", E=E, cosZ=C, Pmumu=P[0, 1, 1], Ee=Ee, Ce=Ce,
         Amumu=A[0, 1, 1])
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].pcolormesh(E, C, P[0, 1, 1], shading="auto", vmin=0, vmax=1)
    ax[1].pcolormesh(Ee, Ce, A[0, 1, 1], vmin=0, vmax=1)
    for a, t in zip(ax, ("P(numu->numu)", "bin averaged")):
        a.set_xscale("log")
        a.set_xlabel("E [GeV]")
        a.set_ylabel("cos zenith")
        a.set_title(t)
    fig.savefig("oscillogram.png", dpi=120)
    print("wrote oscillogram.png")
except ImportError:
    print("matplotlib not available; wrote oscillogram.npz")
