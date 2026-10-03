"""Gradient API from Python: consistency with finite differences."""

import numpy as np
import pytest

import oscprobgpu as opg
from conftest import TEST_PATH, set_nominal


def test_names_and_switch(devices):
    assert opg.Fast.parameter_names == ["th12", "th13", "th23", "d13", "dm21", "dm31"]
    p = set_nominal(opg.Fast(devices=devices))
    if not opg.Fast.has_gradients:
        pytest.skip("gradients disabled at build time")
    p.set_grid(np.array([1.0, 2.0]), np.array([-0.9, 0.3]))
    with pytest.raises(RuntimeError):
        p.calculate_gradient()          # no parameters selected
    p.set_gradient_params(["dm31", "th23"])
    assert p.gradient_params == ["dm31", "th23"]
    p.calculate_gradient()
    assert p.grad().shape == (2, 2, 3, 3, 2, 2)


@pytest.mark.parametrize("name", ["th12", "th13", "th23", "d13", "dm21", "dm31"])
def test_path_grad_vs_fd(name, devices):
    if not opg.Fast.has_gradients:
        pytest.skip("gradients disabled at build time")
    E = np.geomspace(0.2, 10, 50)
    p = set_nominal(opg.Fast(devices=devices))
    p.set_gradient_params([name])
    P, dP = p.prob_path_grad(E, TEST_PATH, nubar=False)

    # central difference in double precision
    getter = {"th12": (1, 2), "th13": (1, 3), "th23": (2, 3)}
    def setv(q, v):
        if name in getter:
            q.set_angle(*getter[name], v)
        elif name == "d13":
            q.set_delta(1, 3, v)
        else:
            q.set_dm(int(name[2]), v)
    def getv(q):
        if name in getter:
            return q.get_angle(*getter[name])
        if name == "d13":
            return q.get_delta(1, 3)
        return q.get_dm(int(name[2]))
    x0 = getv(p)
    h = 1e-6 * (abs(x0) if name.startswith("dm") else 1.0)
    q = set_nominal(opg.Fast(devices=devices))
    setv(q, x0 + h); Pp = q.prob_path(E, TEST_PATH)
    setv(q, x0 - h); Pm = q.prob_path(E, TEST_PATH)
    fd = (Pp - Pm) / (2 * h)
    scale = max(np.abs(fd).max(), 1e-3)
    assert np.abs(dP[0] - fd).max() / scale < 1e-6


def test_weighted_matches_full(devices):
    if not opg.Fast.has_gradients:
        pytest.skip("gradients disabled at build time")
    p = set_nominal(opg.Fast(devices=devices))
    p.set_gradient_params(opg.Fast.parameter_names)
    p.set_grid(np.geomspace(0.5, 50, 40), np.linspace(-1, 1, 30))
    p.calculate_gradient()
    G = p.grad()
    w = np.random.default_rng(1).normal(size=p.probs().shape)
    g = p.weighted_gradient(w)
    ref = np.einsum("nabce,npabce->p", w, G)
    scale = np.einsum("nabce,npabce->p", np.abs(w), np.abs(G))
    assert np.all(np.abs(g - ref) <= 1e-12 * scale)


def _nsi(devices, eps_emu=0.1):
    p = set_nominal(opg.NSI(devices=devices))
    p.set_eps(0, 1, eps_emu, 0.4)
    p.set_eps(1, 2, 0.05, -1.0)
    p.set_ferm_coup(0.5, 1.0, 0.8)
    return p


def _nunm(devices, alpha_mue=0.03):
    p = set_nominal(opg.NUNM(devices=devices, scale=1))
    p.set_alpha(1, 0, alpha_mue, 0.4)
    p.set_alpha(2, 2, -0.02)
    return p


def _sterile(devices, th24=0.15):
    p = set_nominal(opg.Sterile(devices=devices))
    p.set_dm(4, 0.5)
    p.set_angle(2, 4, th24)
    p.set_angle(1, 4, 0.1)
    return p


def _decay(devices, alpha3=2e-4):
    p = set_nominal(opg.Decay(devices=devices))
    p.set_alpha2(3e-5)
    p.set_alpha3(alpha3)
    return p


@pytest.mark.parametrize("make,name,x0,nflv", [
    (_decay, "alpha3", 2e-4, 3),
    (_nsi, "eps_emu", 0.1, 3),
    (_nunm, "alpha_mue", 0.03, 3),
    (_sterile, "th24", 0.15, 4),
])
def test_g3_path_grad_vs_fd(make, name, x0, nflv, devices):
    cls = type(make(devices))
    if not cls.has_gradients:
        pytest.skip("gradients disabled at build time")
    assert name in cls.parameter_names
    E = np.geomspace(0.2, 10, 50)
    p = make(devices)
    p.set_gradient_params([name])
    P, dP = p.prob_path_grad(E, TEST_PATH, nubar=False)
    assert dP.shape == (1, nflv, nflv, len(E))
    h = 1e-6 * (x0 if make is _decay else 1.0)  # alpha in eV^2: relative step
    fd = (make(devices, x0 + h).prob_path(E, TEST_PATH) -
          make(devices, x0 - h).prob_path(E, TEST_PATH)) / (2 * h)
    scale = max(np.abs(fd).max(), 1e-3)
    assert np.abs(dP[0] - fd).max() / scale < 1e-6


def test_g3_parameter_names():
    if not opg.NSI.has_gradients:
        pytest.skip("gradients disabled at build time")
    assert len(opg.NSI.parameter_names) == 18
    assert len(opg.NUNM.parameter_names) == 16
    assert opg.Sterile.parameter_names[-3:] == ["dm21", "dm31", "dm41"]
    assert len(opg.Sterile.parameter_names) == 12
    assert opg.Decay.parameter_names[-2:] == ["alpha2", "alpha3"]
    # NSI fermion couplings are opt-in
    assert opg.NSI.default_gradient_params == opg.NSI.parameter_names[:15]
    assert opg.Fast.default_gradient_params == opg.Fast.parameter_names
    p = opg.NSI()
    p.set_gradient_params()
    assert p.gradient_params == opg.NSI.parameter_names[:15]
    p.set_gradient_params(["coup_d"])
    assert p.gradient_params == ["coup_d"]


def test_binned_and_earth_gradients(devices):
    if not opg.Fast.has_gradients:
        pytest.skip("gradients disabled at build time")
    p = set_nominal(opg.Fast(devices=devices))
    assert p.earth_parameter_names[0] == "zoa_0"
    assert p.gradient_parameter_names == (opg.Fast.parameter_names +
                                          p.earth_parameter_names)
    p.set_gradient_params(["dm31", "zoa_2"])
    Ee, Ce = np.geomspace(0.6, 20, 9), np.linspace(-1, -0.2, 5)
    p.set_bins(Ee, Ce, 3, 3)
    p.calculate_binned_gradient()
    A, G = p.binned(), p.binned_grad()
    assert G.shape == (2, 2, 3, 3, 4, 8)
    w = np.random.default_rng(2).normal(size=A.shape)
    g = p.weighted_gradient_binned(w)
    ref = np.einsum("nabce,npabce->p", w, G)
    scale = np.einsum("nabce,npabce->p", np.abs(w), np.abs(G))
    assert np.all(np.abs(g - ref) <= 1e-12 * scale)
    # finite differences of the binned averages in dm31
    h = 1e-9
    q = set_nominal(opg.Fast(devices=devices))
    q.set_bins(Ee, Ce, 3, 3)
    q.set_dm(3, 2.507e-3 + h); q.calculate_binned(); Ap = q.binned()
    q.set_dm(3, 2.507e-3 - h); q.calculate_binned(); Am = q.binned()
    fd = (Ap - Am) / (2 * h)
    assert np.abs(G[:, 0] - fd).max() / np.abs(fd).max() < 1e-6
    A1, dA = p.avg_path_grad(np.linspace(0.5, 5, 10), 4, TEST_PATH)
    assert dA.shape == (2, 3, 3, 9)


def test_weighted_points_binned(devices):
    if not opg.Fast.has_gradients:
        pytest.skip("gradients disabled at build time")
    rng = np.random.default_rng(4)
    n = 500
    E, C = rng.uniform(0.5, 20, n), rng.uniform(-1, 1, n)
    nb_ = rng.integers(0, 2, n).astype(np.uint8)
    w = rng.normal(size=(3, 3, n))
    bins = rng.integers(-1, 12, n).astype(np.int32)
    p = set_nominal(opg.Fast(devices=devices))
    p.set_gradient_params()
    P, dP = p.prob_points_grad(E, C, nb_)
    G = p.weighted_gradient_points_binned(E, C, nb_, w, bins, 10)
    assert G.shape == (10, 6)
    for b in range(10):
        m = bins == b
        ref = np.einsum("abi,pabi->p", w[:, :, m], dP[:, :, :, m])
        sc = np.einsum("abi,pabi->p", np.abs(w[:, :, m]), np.abs(dP[:, :, :, m]))
        assert np.all(np.abs(G[b] - ref) <= 1e-12 * sc + 1e-300)


def _model(cls, devices):
    """Each model at a non-trivial point of its extra parameters."""
    if cls is opg.NUNM:
        p = set_nominal(cls(devices=devices, scale=0))
        p.set_alpha(1, 0, 0.03, 0.4)
        return p
    p = set_nominal(cls(devices=devices))
    if cls is opg.NSI:
        p.set_eps(0, 1, 0.1, 0.4)
    elif cls is opg.Sterile:
        p.set_dm(4, 0.5)
        p.set_angle(2, 4, 0.15)
    elif cls is opg.Decay:
        p.set_alpha3(2e-4)
    elif cls is opg.LIV:
        p.set_aT(0, 1, 3, 1.5e-21, 0.7)
        p.set_cT(1, 2, 4, 1e-22, 2.3)
    elif cls is opg.SNSI:
        p.set_lowest_mass(0.05)
        p.set_eps(0, 1, 0.3, 0.4)
    return p


@pytest.mark.parametrize("cls", [opg.Fast, opg.NSI, opg.NUNM, opg.Sterile, opg.Decay,
                                 opg.LIV, opg.SNSI],
                         ids=lambda c: c.__name__)
def test_all_models_grid_points_weighted_binned(cls, devices):
    if not cls.has_gradients:
        pytest.skip("gradients disabled at build time")
    p = _model(cls, devices)
    p.set_gradient_params()                       # model defaults
    names = p.gradient_params
    assert names == cls.default_gradient_params
    n = 4 if cls is opg.Sterile else 3
    E, C = np.geomspace(0.5, 30, 17), np.linspace(-1, 0.4, 11)
    p.set_grid(E, C)
    p.calculate_gradient()
    G = p.grad()
    assert G.shape == (2, len(names), n, n, C.size, E.size)
    # event list at the grid points gives the same gradients
    EE, CC = np.meshgrid(E, C)
    nb_ = np.zeros(EE.size, np.uint8)
    _, dP = p.prob_points_grad(EE.ravel(), CC.ravel(), nb_)
    assert np.max(np.abs(dP.reshape(G[0].shape) - G[0])) <= 1e-13 * np.max(np.abs(G[0]))
    # weighted grid mode
    w = np.random.default_rng(5).normal(size=p.probs().shape)
    ref = np.einsum("nabce,npabce->p", w, G)
    sc = np.einsum("nabce,npabce->p", np.abs(w), np.abs(G))
    assert np.all(np.abs(p.weighted_gradient(w) - ref) <= 1e-12 * sc)
    # binned mode
    p.set_bins(np.geomspace(0.5, 30, 6), np.linspace(-1, 0.4, 4), 3, 3)
    p.calculate_binned_gradient()
    GB = p.binned_grad()
    wb = np.random.default_rng(6).normal(size=p.binned().shape)
    refb = np.einsum("nabce,npabce->p", wb, GB)
    scb = np.einsum("nabce,npabce->p", np.abs(wb), np.abs(GB))
    assert np.all(np.abs(p.weighted_gradient_binned(wb) - refb) <= 1e-12 * scb)


def test_device_weights(devices):
    if not opg.NSI.has_gradients:
        pytest.skip("gradients disabled at build time")
    p = _model(opg.NSI, devices)
    p.set_gradient_params()
    p.set_grid(np.geomspace(0.5, 30, 23), np.linspace(-1, 0.4, 13))
    p.calculate()
    if not devices:
        with pytest.raises(ValueError):
            p.device_probs()
        return
    dw = p.device_probs()                 # CUDA array (DLPack), no copy
    assert dw.shape == p.probs().shape
    assert dw.__dlpack_device__() == (2, devices[0])
    assert dw.__cuda_array_interface__["shape"] == dw.shape
    assert np.array_equal(p.weighted_gradient(dw), p.weighted_gradient(p.probs()))
    p.set_bins(np.geomspace(0.5, 30, 6), np.linspace(-1, 0.4, 4), 3, 3)
    p.calculate_binned()
    db = p.device_binned()
    assert np.array_equal(p.weighted_gradient_binned(db),
                          p.weighted_gradient_binned(p.binned()))
