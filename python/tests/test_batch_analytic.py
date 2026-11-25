"""Python bindings of the batch and analytic APIs, DeviceArray validity and
GIL release (CPU and GPU backends)."""

import threading
import time

import numpy as np
import pytest

import oscprobgpu as opg
from conftest import set_nominal

FD = np.array([[810, 2.84, 0.5]], float)                       # one segment
TWO = np.array([[400, 2.6, 0.49], [410, 3.1, 0.5]], float)    # two segments


def _set(p, name, v):
    """Set a mixing parameter by gradient name through the OscProb setters."""
    if name.startswith("th"):
        p.set_angle(int(name[2]), int(name[3]), v)
    elif name.startswith("dm"):
        p.set_dm(int(name[2]), v)
    elif name.startswith("d"):
        p.set_delta(int(name[1]), int(name[2]), v)
    else:
        raise KeyError(name)


def _need_gradients():
    if not opg.Fast.has_gradients:
        pytest.skip("gradients disabled at build time")


def _sterile(devices):
    p = set_nominal(opg.Sterile(devices=devices))
    p.set_angle(2, 4, 0.1)
    p.set_angle(1, 4, 0.05)
    p.set_dm(4, 1.0)
    return p


# ---------------------------------------------------------------------------
# binned_batch / calculate_binned_batch

@pytest.mark.parametrize("rule", ["layer", "plain"])
def test_binned_batch_equals_single_points(devices, rule):
    _need_gradients()
    p = set_nominal(opg.NSI(devices=devices))
    p.set_eps(0, 1, 0.05, 0.3)
    p.set_bins(np.geomspace(1, 50, 7), np.linspace(-1, 0, 6), 2, 3, "log", rule)
    names = ["th23", "dm31"]
    earth_par = [n for n in p.earth_parameter_names if n.startswith("zoa_")][:1]
    p.set_gradient_params(names + earth_par)
    p.calculate_binned_gradient()
    ref0, gref0 = p.binned(), p.binned_grad()

    vals = np.array([[0.70, 2.4e-3], [0.80, 2.5e-3], [0.75, 2.6e-3]])
    earths = []
    for k in range(len(vals)):
        e = opg.PremModel()
        e.set_layer_zoa(1, 0.47 + 0.005 * k)
        earths.append(e)
    A, dA = p.binned_batch(names, vals, earths=earths, gradient=True)
    nq = len(p.gradient_params)
    assert A.shape == (3,) + ref0.shape
    assert dA.shape == (3, 2, nq) + ref0.shape[1:]
    A0 = p.binned_batch(names, vals, earths=earths)   # values only
    assert np.array_equal(A0, A)

    # the propagator's parameters and Earth are unchanged
    p.calculate_binned_gradient()
    assert np.array_equal(p.binned(), ref0)
    assert np.array_equal(p.binned_grad(), gref0)

    for k, row in enumerate(vals):
        q = set_nominal(opg.NSI(earth=earths[k], devices=devices))
        q.set_eps(0, 1, 0.05, 0.3)
        q.set_bins(np.geomspace(1, 50, 7), np.linspace(-1, 0, 6), 2, 3, "log", rule)
        q.set_gradient_params(names + earth_par)
        for n, v in zip(names, row):
            _set(q, n, v)
        q.calculate_binned_gradient()
        assert np.array_equal(A[k], q.binned())
        assert np.array_equal(dA[k], q.binned_grad())

    if not devices:
        with pytest.raises((TypeError, RuntimeError)):   # CUDA output only
            p.calculate_binned_batch(names, vals, np.zeros(1))
        return
    out = opg.cuda_empty(A.shape, devices[0])
    dout = opg.cuda_empty(dA.shape, devices[0])
    p.calculate_binned_batch(names, vals, out, dout, earths=earths)
    p.wait()
    assert np.array_equal(out.get(), A)
    assert np.array_equal(dout.get(), dA)
    with pytest.raises(ValueError):           # wrong size
        p.calculate_binned_batch(names, vals[:2], out)


def test_batch_point_arguments(devices):
    p = set_nominal(opg.Fast(devices=devices))
    p.set_bins(np.geomspace(1, 50, 4), np.linspace(-1, 0, 3), 2, 2)
    with pytest.raises(ValueError):
        p.binned_batch(["nope"], np.zeros((1, 1)))
    with pytest.raises(ValueError):
        p.binned_batch(["zoa_1"], np.zeros((1, 1)))  # Earth parameters go through earths
    with pytest.raises(ValueError):
        p.binned_batch(["th23"], np.zeros((2, 2)))
    with pytest.raises(ValueError):
        p.binned_batch([], np.zeros((2, 0)), earths=[opg.PremModel()])
    # no names: every point is the model's current parameters
    A = p.binned_batch([], np.zeros((2, 0)))
    p.calculate_binned()
    assert np.array_equal(A[0], p.binned()) and np.array_equal(A[1], p.binned())


# ---------------------------------------------------------------------------
# analytic averages

def test_analytic_models():
    for cls in (opg.Fast, opg.NSI, opg.NUNM, opg.Sterile):
        assert hasattr(cls, "avg_path_analytic") and hasattr(cls, "analytic_batch")
    for cls in (opg.Decay, opg.LIV, opg.SNSI, opg.Deco, opg.SiderealLIV, opg.OQS):
        assert not hasattr(cls, "avg_path_analytic")


@pytest.mark.parametrize("path", [FD, TWO], ids=["one_segment", "two_segments"])
def test_analytic_single_calls(devices, path):
    _need_gradients()
    p = _sterile(devices)
    p.set_gradient_params(["th24", "dm41"] + (["zoa_0", "rho_0"] if path is TWO else []))
    E = np.geomspace(0.5, 10, 21)
    A = p.avg_path_analytic(E, path, measure="invE")
    A2, dA = p.avg_path_analytic_grad(E, path, measure="invE")
    assert A.shape == (4, 4, 20) and dA.shape == (len(p.gradient_params), 4, 4, 20)
    assert np.array_equal(A, A2)
    # probabilities: rows sum to 1 (no absorption, unitary model)
    assert np.allclose(A.sum(axis=1), 1, atol=1e-12)
    # explicit sub-bins reproduce the edge-based call bit for bit
    sub = opg.Sterile.analytic_subbins(E, path, "invE")
    assert sub.shape[1] == 5
    assert np.array_equal(p.avg_path_analytic_subbins(sub, 20, path), A)
    S, dS = p.avg_path_analytic_subbins_grad(sub, 20, path)
    assert np.array_equal(S, A) and np.array_equal(dS, dA)
    # against converged Gauss-Legendre (uniform in 1/E; up to ~50 periods per bin)
    G = p.avg_path(E, 400, path, False, "invE")
    assert np.abs(A - G).max() < 2e-5
    # L/E edges give the same bins
    L = path[:, 0].sum()
    B = p.avg_path_analytic(L / E[::-1], path, measure="invE", var="LoE")
    assert np.allclose(B[..., ::-1], A, atol=1e-12)
    # options: a smaller tol changes the sub-bins, the result stays within tol
    o = opg.AnalyticAvgOptions()
    o.tol = 1e-8
    assert np.abs(p.avg_path_analytic(E, path, measure="invE", options=o) - A).max() < 2e-6
    # derivative check: central differences in th24
    # (the sub-bins depend on the path only, not on th24)
    h = 1e-6
    _set(p, "th24", 0.1 + h)
    up = p.avg_path_analytic(E, path, measure="invE")
    _set(p, "th24", 0.1 - h)
    dn = p.avg_path_analytic(E, path, measure="invE")
    _set(p, "th24", 0.1)
    assert np.abs((up - dn) / (2 * h) - dA[0]).max() < 1e-6


@pytest.mark.parametrize("path", [FD, TWO], ids=["one_segment", "two_segments"])
def test_analytic_batch(devices, path):
    _need_gradients()
    p = _sterile(devices)
    names = ["th24", "dm41"]
    p.set_gradient_params(names)
    E = np.geomspace(0.5, 10, 21)
    vals = np.array([[0.10, 1.0], [0.15, 0.5], [0.05, 2.0], [0.12, 1.3]])
    h = p.analytic_batch(E, path, measure="invE")
    assert h.n_bins == 20 and h.nubar is False
    assert h.on_device == bool(devices) and h.device == (0 if devices else -1)
    A, dA = p.avg_path_analytic_batch(h, names, vals, gradient=True)
    assert A.shape == (4, 4, 4, 20) and dA.shape == (4, 2, 4, 4, 20)
    assert np.array_equal(p.avg_path_analytic_batch(h, names, vals), A)
    # any batch size and order: bit-identical per point
    B, dB = p.avg_path_analytic_batch(h, names, vals[::-1].copy(), gradient=True)
    assert np.array_equal(B[::-1], A) and np.array_equal(dB[::-1], dA)
    # each point against the single-call API (GPU: round-off of the reduction)
    tol = 1e-12 if devices else 0
    for k, row in enumerate(vals):
        for n, v in zip(names, row):
            _set(p, n, v)
        S, dS = p.avg_path_analytic_grad(E, path, measure="invE")
        assert np.abs(S - A[k]).max() <= tol
        assert np.abs(dS - dA[k]).max() <= tol * max(1, np.abs(dS).max())
    # weighted gradients: the contraction of dA
    w = np.random.default_rng(3).normal(size=A.shape)
    g = p.avg_path_analytic_batch_weighted(h, names, vals, w)
    ref = np.einsum("pabk,pqabk->pq", w, dA)
    sc = np.einsum("pabk,pqabk->pq", np.abs(w), np.abs(dA))
    assert g.shape == (4, 2)
    assert np.all(np.abs(g - ref) <= 1e-12 * sc)
    with pytest.raises(ValueError):
        p.avg_path_analytic_batch_weighted(h, names, vals, w[:2])
    # explicit sub-bins
    sub = opg.Sterile.analytic_subbins(E, path, "invE")
    hs = p.analytic_batch_subbins(sub, 20, path)
    assert np.array_equal(p.avg_path_analytic_batch(hs, names, vals), A)

    if not devices:
        with pytest.raises((TypeError, RuntimeError)):
            p.avg_path_analytic_batch_device(h, names, vals, np.zeros(1))
        return
    out = opg.cuda_empty(A.shape, devices[0])
    dout = opg.cuda_empty(dA.shape, devices[0])
    p.avg_path_analytic_batch_device(h, names, vals, out, dout)
    assert np.array_equal(out.get(), A) and np.array_equal(dout.get(), dA)
    wd, gd = opg.cuda_empty(w.shape, devices[0]), opg.cuda_empty((4, 2), devices[0])
    # weights to the device through a binned batch is not possible; use the
    # host weights path for the reference and zero CUDA weights here
    p.avg_path_analytic_batch_weighted(h, names, vals, wd, gd)
    assert np.array_equal(gd.get(), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        p.avg_path_analytic_batch_weighted(h, names, vals, wd)   # CUDA weights need g


def test_analytic_batch_device_weights(devices):
    """CUDA weights give the host-weights result (device copy of w via a
    batch output: A itself as weights)."""
    _need_gradients()
    if not devices:
        pytest.skip("GPU only")
    p = _sterile(devices)
    names = ["th24", "dm41"]
    p.set_gradient_params(names)
    E = np.geomspace(0.5, 10, 21)
    vals = np.array([[0.10, 1.0], [0.15, 0.5]])
    h = p.analytic_batch(E, FD, measure="invE")
    out = opg.cuda_empty((2, 4, 4, 20), devices[0])
    p.avg_path_analytic_batch_device(h, names, vals, out)
    gd = opg.cuda_empty((2, 2), devices[0])
    p.avg_path_analytic_batch_weighted(h, names, vals, out, gd)
    ref = p.avg_path_analytic_batch_weighted(h, names, vals, out.get())
    assert np.allclose(gd.get(), ref, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# DeviceArray validity

def test_device_views_go_stale(devices):
    if not devices:
        with pytest.raises(ValueError):
            set_nominal(opg.Fast(devices=devices)).device_binned()
        return
    p = set_nominal(opg.Fast(devices=devices))
    p.set_gradient_params(["th23"])
    p.set_grid(np.geomspace(1, 30, 11), np.linspace(-1, 0, 7))
    p.calculate()
    v = p.device_probs()
    assert v.valid and np.array_equal(v.get(), p.probs())
    p.calculate()                              # recomputing keeps the buffer
    assert v.valid
    p.set_grid(np.geomspace(1, 30, 13), np.linspace(-1, 0, 7))
    assert not v.valid
    for use in (v.get, v.__dlpack__, lambda: v.__cuda_array_interface__,
                lambda: p.weighted_gradient(v)):
        with pytest.raises(RuntimeError):
            use()
    p.calculate()
    assert p.device_probs().valid

    p.set_bins(np.geomspace(1, 30, 4), np.linspace(-1, 0, 3), 2, 2)
    p.calculate_binned()
    b = p.device_binned()
    assert np.array_equal(b.get(), p.binned())
    p.set_bins(np.geomspace(1, 30, 5), np.linspace(-1, 0, 3), 2, 2)
    assert not b.valid
    with pytest.raises(RuntimeError):
        p.weighted_gradient_binned(b)
    p.calculate_binned()
    b = p.device_binned()
    v = p.device_probs()
    p.set_earth(opg.PremModel())
    assert not b.valid and not v.valid


# ---------------------------------------------------------------------------
# GIL

def test_gil_released_during_computation():
    """The main thread runs while another thread is inside calculate()."""
    p = set_nominal(opg.Fast(devices=[], threads=1))
    p.set_grid(np.geomspace(1, 100, 200), np.linspace(-1, 1, 200))
    span = {}

    def work():
        span["t0"] = time.perf_counter()
        p.calculate()                          # CPU backend: the work is in this call
        span["t1"] = time.perf_counter()

    th = threading.Thread(target=work)
    stamps = []
    th.start()
    while th.is_alive():
        stamps.append(time.perf_counter())
    th.join()
    t0, t1 = span["t0"], span["t1"]
    assert t1 - t0 > 0.01
    inside = sum(t0 < t < t1 for t in stamps)
    # with the GIL held for the whole call, no stamp could be taken inside it
    assert inside > 100
