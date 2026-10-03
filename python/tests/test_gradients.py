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
