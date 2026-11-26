"""Production-height averaging through the Python bindings."""

import numpy as np
import pytest

import oscprobgpu as opg
from conftest import set_nominal

EDGES = [np.array([0.0, 10, 20, 40]), np.array([15.0, 15.0]), np.array([0.0, 60.0])]
PROBS = [np.array([0.3, 0.5, 0.2]), np.array([1.0]), np.array([1.0])]


def events():
    E = np.repeat(np.geomspace(0.5, 30, 5), 6)
    C = np.tile(np.array([-0.9, -0.3, -0.02, 0.0, 0.2, 1.0]), 5)
    nb = (np.arange(E.size) % 2).astype(np.uint8)
    d = (np.arange(E.size) % 3).astype(np.int64)
    return E, C, nb, d


def test_brute_force_and_fixed(devices):
    p = set_nominal(opg.Fast(devices=devices))
    assert opg.Fast.has_prod_height and not opg.Decay.has_prod_height
    p.set_prod_heights(EDGES, PROBS)
    assert p.n_prod_heights == 3
    assert p.prod_height_path_length(1.0, 20.0) == pytest.approx(20.0, rel=1e-14)
    E, C, nb, d = events()
    P = p.prob_points_prod_height(E, C, nb, d)
    assert P.shape == (3, 3, E.size)
    assert np.abs(P.sum(axis=1) - 1).max() < 1e-13
    # brute force on distribution 0: 24-point Gauss-Legendre per bin over
    # fixed heights (the C++ tests check fixed heights against explicit paths)
    x, w = np.polynomial.legendre.leggauss(24)
    e = EDGES[0]
    hs, ws = [], []
    for b in range(len(e) - 1):
        hs += list(0.5 * (e[b] + e[b + 1]) + 0.5 * (e[b + 1] - e[b]) * x)
        ws += list(PROBS[0][b] / PROBS[0].sum() * w / 2)
    q = set_nominal(opg.Fast(devices=devices))
    q.set_prod_heights([np.array([h, h]) for h in hs], [np.array([1.0])] * len(hs))
    ref = sum(wq * q.prob_points_prod_height(E, C, nb, np.full(E.size, k))
              for k, wq in enumerate(ws))
    sel = d == 0
    assert np.abs(P[:, :, sel] - ref[:, :, sel]).max() < 1e-12
    # fixed height of 15 km (the default atmosphere) and -1: the plain list
    Pp = p.prob_points(E, C, nb)
    P1 = p.prob_points_prod_height(E, C, nb, np.ones(E.size, np.int32))
    assert np.abs(P1 - Pp).max() < 1e-12
    Pm = p.prob_points_prod_height(E, C, nb, -np.ones(E.size, np.int64))
    if not devices:
        assert np.array_equal(Pm, Pp)
    else:
        assert np.abs(Pm - Pp).max() < 1e-12


def test_grid_and_gradients(devices):
    p = set_nominal(opg.Fast(devices=devices))
    p.set_prod_heights(EDGES, PROBS)
    Eg = np.array([0.6, 3.0, 20.0])
    Cg = np.array([-0.8, -0.1, 0.0, 0.5])
    p.set_grid(Eg, Cg)
    with pytest.raises(RuntimeError):
        p.calculate_prod_height()           # distributions of the grid not set
    dg = (np.arange(12).reshape(4, 3) % 4 - 1).astype(np.int64)
    p.set_grid_prod_heights(dg)
    p.calculate_prod_height()
    G = p.probs()
    for nb in (0, 1):
        Ev = np.tile(Eg, 4)
        Cv = np.repeat(Cg, 3)
        Pe = p.prob_points_prod_height(Ev, Cv, np.full(12, nb, np.uint8), dg.ravel())
        assert np.abs(Pe.reshape(3, 3, 4, 3) - G[nb]).max() < 1e-14
    p.set_grid_prod_heights(np.array([0, 1, 2, -1], np.int32))   # per cosine
    p.calculate_prod_height()

    if not opg.Fast.has_gradients:
        return
    p.set_gradient_params(["th23", "dm31", "zoa_6", "rho_6"])
    p.calculate_prod_height(gradient=True)
    assert p.grad().shape == (2, 4, 3, 3, 4, 3)
    E, C, nb, d = events()
    P, dP = p.prob_points_grad_prod_height(E, C, nb, d)
    assert dP.shape == (4, 3, 3, E.size)
    rng = np.random.default_rng(1)
    w = rng.normal(size=(3, 3, E.size))
    g_ref = np.einsum("pabi,abi->p", dP, w)
    for adj in (False, True):
        p.adjoint_gradients = adj
        g = p.weighted_gradient_points_prod_height(E, C, nb, d, w)
        assert np.abs(g - g_ref).max() < 1e-10 * max(1.0, np.abs(g_ref).max())
    wg = rng.normal(size=(2, 3, 3, 4, 3))
    g_ref = np.einsum("npabce,nabce->p", p.grad(), wg)
    for adj in (False, True):
        p.adjoint_gradients = adj
        g = p.weighted_gradient_prod_height(wg)
        assert np.abs(g - g_ref).max() < 1e-10 * max(1.0, np.abs(g_ref).max())
    # dm31 against a central difference
    h = 1e-8
    q = set_nominal(opg.Fast(devices=devices))
    q.set_prod_heights(EDGES, PROBS)
    q.set_dm(3, 2.507e-3 + h)
    Pp = q.prob_points_prod_height(E, C, nb, d)
    q.set_dm(3, 2.507e-3 - h)
    Pm = q.prob_points_prod_height(E, C, nb, d)
    fd = (Pp - Pm) / (2 * h)
    assert np.abs(dP[1] - fd).max() / np.abs(fd).max() < 1e-5


def test_errors(devices):
    p = set_nominal(opg.Fast(devices=devices))
    E, C, nb, d = events()
    with pytest.raises(RuntimeError):
        p.prob_points_prod_height(E, C, nb, d)  # no distributions
    p.set_prod_heights(EDGES, PROBS)
    with pytest.raises(IndexError):
        p.prob_points_prod_height(E, C, nb, d + 3)
    with pytest.raises(ValueError):
        p.prob_points_prod_height(E, C, nb, d.astype(float))
    with pytest.raises(ValueError):
        p.set_prod_heights([np.array([0.0, 10.0])], [np.array([1.0, 2.0])])
    with pytest.raises(RuntimeError):
        opg.Decay(devices=devices).set_prod_heights(EDGES, PROBS)
