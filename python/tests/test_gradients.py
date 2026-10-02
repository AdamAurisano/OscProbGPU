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
