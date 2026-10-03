"""Fit oscillation parameters to a binned atmospheric toy analysis with exact
gradients.

Expected numu-CC events per (cos zenith, E) bin, for nu and nubar,

    N_b = T_b * [ f_e A_b(nue -> numu) + f_mu A_b(numu -> numu) ],

with bin-averaged probabilities A_b (2D Gauss-Legendre), a toy flux x
cross-section x exposure normalisation T_b and flavour fractions f_a. The
Poisson likelihood -2 ln L = 2 sum_b [N_b - D_b + D_b ln(D_b / N_b)] is
minimised with Levenberg-Marquardt / Fisher-scoring steps, whose Jacobian
dN_b/dp comes from binned_grad(): one gradient evaluation per iteration, no
finite differences. The gradient of -2 ln L is also obtained directly with
weighted_gradient_binned() (only one number per parameter leaves the device),
as a fit with a quasi-Newton minimiser (e.g. scipy's L-BFGS-B) would use it.

Usage: python gradient_fit.py [cpu] [core]
  cpu  : use the CPU backend
  core : also fit the outer-core Z/A (Earth parameter zoa_1)
"""
import sys
import time

import numpy as np

import oscprobgpu as opg

devices = [] if "cpu" in sys.argv else opg.default_devices()
fit_core = "core" in sys.argv

# --- parameters -------------------------------------------------------------
setters = {
    "th23": lambda m, x: m.set_angle(2, 3, x),
    "dm31": lambda m, x: m.set_dm(3, x),
    "d13": lambda m, x: m.set_delta(1, 3, x),
}
truth = {"th23": np.arcsin(np.sqrt(0.451)), "dm31": 2.507e-3, "d13": np.radians(232)}
# start in the true octant: from sin^2(th23) > 0.5 the fit finds the mirror
# minimum (octant degeneracy), a genuine local minimum of -2lnL
start = {"th23": np.arcsin(np.sqrt(0.40)), "dm31": 2.35e-3, "d13": np.radians(150)}
scale = {"th23": 0.1, "dm31": 1e-4, "d13": 1.0}   # typical step sizes
names = list(truth)
if fit_core:
    truth["zoa_1"] = opg.PremModel().get_layer_zoa(1)
    start["zoa_1"] = truth["zoa_1"] + 0.03
    scale["zoa_1"] = 0.01
    names.append("zoa_1")


def make_model():
    m = opg.Fast(devices=devices)
    m.set_angle(1, 2, np.arcsin(np.sqrt(0.303)))
    m.set_angle(1, 3, np.arcsin(np.sqrt(0.02225)))
    m.set_dm(2, 7.41e-5)
    return m


def set_params(m, x):
    for n, v in zip(names, x):
        if n == "zoa_1":
            earth = opg.PremModel()
            earth.set_layer_zoa(1, v)
            m.set_earth(earth)
        else:
            setters[n](m, v)


# --- binning and toy event normalisation ------------------------------------
Ee = np.geomspace(1.0, 50.0, 31)          # GeV
Ce = np.linspace(-1.0, 0.0, 21)           # cos zenith (up-going)
Ec = np.sqrt(Ee[1:] * Ee[:-1])
# flux (E^-2.7) x cross-section (E) x exposure, nubar ~ 1/2; nue/numu = 1/2
T = 2e3 * (Ec / Ee[0]) ** -1.7 * np.diff(np.log(Ee))
T = np.array([T, 0.5 * T])[:, None, :] * np.ones((2, Ce.size - 1, 1))
fa = np.array([0.5, 1.0])                 # f_e, f_mu


def expected(m, with_grad):
    """N[nubar, iC, iE] and, optionally, J[p, nubar, iC, iE] = dN/dp."""
    if with_grad:
        m.calculate_binned_gradient()
    else:
        m.calculate_binned()
    A = m.binned()                        # [nubar, a, b, iC, iE]
    N = T * (fa[0] * A[:, 0, 1] + fa[1] * A[:, 1, 1])
    if not with_grad:
        return N, None
    G = m.binned_grad()                   # [nubar, p, a, b, iC, iE]
    J = T[None] * (fa[0] * G[:, :, 0, 1] + fa[1] * G[:, :, 1, 1]).transpose(1, 0, 2, 3)
    return N, J


def m2lnl(N, D):
    return 2 * np.sum(N - D + np.where(D > 0, D * np.log(np.maximum(D, 1e-300) / N), 0))


m = make_model()
m.set_bins(Ee, Ce, 4, 4, measure="log")
m.set_gradient_params(names)

set_params(m, [truth[n] for n in names])
D, _ = expected(m, False)                 # Asimov data set
print(f"{'GPU' if m.on_gpu else 'CPU'} backend, {D.size} bins, "
      f"{D.sum():.0f} events; fitting {', '.join(names)}")

# --- Levenberg-Marquardt with Fisher information ----------------------------
x = np.array([start[n] for n in names])
s = np.array([scale[n] for n in names])
lam, t0 = 1e-3, time.perf_counter()
set_params(m, x)
N, J = expected(m, True)
f = m2lnl(N, D)
print(f"{'it':>3} {'-2lnL':>12} " + " ".join(f"{n:>11}" for n in names))
for it in range(30):
    print(f"{it:3d} {f:12.5f} " + " ".join(f"{v:11.6g}" for v in x))
    Jf = J.reshape(len(names), -1) * s[:, None]         # scaled parameters
    r = (1 - D / N).ravel()
    g = 2 * Jf @ r                                      # d(-2lnL)/d(x/s)
    H = 2 * (Jf / N.ravel()) @ Jf.T                     # Fisher information
    while True:
        step = -np.linalg.solve(H + lam * np.diag(np.diag(H)), g) * s
        set_params(m, x + step)
        N1, J1 = expected(m, True)
        f1 = m2lnl(N1, D)
        if f1 <= f:
            x, N, J, lam = x + step, N1, J1, lam / 10
            break
        lam *= 10
    converged = f - f1 < 1e-8
    f = f1
    if converged:
        break
t1 = time.perf_counter()
print(f"converged after {it + 1} iterations in {t1 - t0:.2f} s, -2lnL = {f:.3g}")

# parabolic errors: covariance = inverse Fisher information J^T diag(1/N) J
Jf = J.reshape(len(names), -1)
cov = np.linalg.inv((Jf / N.ravel()) @ Jf.T)
for i, n in enumerate(names):
    print(f"  {n:6s} fit {x[i]:.6g} +- {np.sqrt(cov[i, i]):.2g}   (truth {truth[n]:.6g})")

# --- the same gradient from the weighted mode ---------------------------------
# d(-2lnL)/dp = sum_b 2 (1 - D_b/N_b) dN_b/dp; weights in the layout of binned()
x_test = x + 0.5 * s
set_params(m, x_test)
N, J = expected(m, True)
w = np.zeros(m.binned().shape)
w[:, 0, 1] = 2 * (1 - D / N) * T * fa[0]
w[:, 1, 1] = 2 * (1 - D / N) * T * fa[1]
g_w = m.weighted_gradient_binned(w)
g_j = 2 * J.reshape(len(names), -1) @ (1 - D / N).ravel()
print("gradient of -2lnL, weighted mode vs Jacobian: max rel. diff "
      f"{np.max(np.abs(g_w - g_j) / np.abs(g_j)):.1e}")
