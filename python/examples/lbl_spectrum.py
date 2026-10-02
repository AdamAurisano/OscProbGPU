"""DUNE-like long-baseline appearance/disappearance with standard and NSI
physics, as point probabilities and 1D Gauss-Legendre bin averages.
"""
import sys

import numpy as np

import oscprobgpu as opg

devices = [] if "cpu" in sys.argv else opg.default_devices()
path = np.array([[1285.0, 2.84, 0.5]])          # L [km], rho [g/cm3], Z/A
edges = np.linspace(0.5, 8.0, 31)
E = 0.5 * (edges[1:] + edges[:-1])

std = opg.Fast(devices=devices)
nsi = opg.NSI(devices=devices)
nsi.set_eps(0, 1, 0.1, 0.5 * np.pi)
nsi.set_eps(0, 2, 0.2, 0.0)

for name, m in (("standard", std), ("NSI", nsi)):
    for nb, label in ((False, "nu"), (True, "nubar")):
        P = m.prob_path(E, path, nubar=nb)            # [a][b][iE]
        A = m.avg_path(edges, 8, path, nubar=nb)      # [a][b][iEbin]
        print(f"{name:8s} {label:5s}  P(mu->e) at bin centres: "
              + " ".join(f"{x:.3f}" for x in P[1, 0, ::5]))
        print(f"{'':8s} {'':5s}  bin-averaged          : "
              + " ".join(f"{x:.3f}" for x in A[1, 0, ::5]))
