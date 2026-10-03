"""Python API against the OscProb reference values (CPU and GPU backends)."""

import numpy as np
import pytest

import oscprobgpu as opg
from conftest import TEST_PATH, VAC_PATH, load, set_nominal


def make(tag, devices):
    """Model instance configured as the reference variant `tag`."""
    if tag in ("fast", "fast_io"):
        p = set_nominal(opg.Fast(devices=devices))
        if tag == "fast_io":
            p.set_dm(3, -2.465e-3 + 7.41e-5)
            p.set_delta(1, 3, 1.1)
    elif tag.startswith("nsi"):
        p = set_nominal(opg.NSI(devices=devices))
        if tag == "nsi":
            for (i, j), v in {(0, 0): .1, (0, 1): .2, (0, 2): .3,
                              (1, 1): .4, (1, 2): .5, (2, 2): .6}.items():
                p.set_eps(i, j, v, 0)
        else:
            p.set_eps(0, 0, -0.2, 0)
            p.set_eps(0, 1, 0.05, 0.7)
            p.set_eps(0, 2, 0.15, -1.3)
            p.set_eps(1, 1, 0.02, 0)
            p.set_eps(1, 2, 0.03, 2.1)
            p.set_eps(2, 2, 0.1, 0)
            p.set_ferm_coup(0.5, 1.0, 0.8)
    elif tag.startswith("nunm"):
        p = set_nominal(opg.NUNM(devices=devices,
                                 scale=1 if tag == "nunm_high" else 0))
        if tag in ("nunm", "nunm_high"):
            for (i, j), v in {(0, 0): .05, (1, 0): .06, (2, 0): .07,
                              (1, 1): .08, (2, 1): .09, (2, 2): .1}.items():
                p.set_alpha(i, j, v, 0)
        else:
            p.set_alpha(0, 0, -0.03, 0)
            p.set_alpha(1, 0, 0.02, 0.4)
            p.set_alpha(2, 0, 0.04, -2.0)
            p.set_alpha(1, 1, -0.01, 0)
            p.set_alpha(2, 1, 0.05, 1.2)
            p.set_alpha(2, 2, -0.02, 0)
            p.set_frac_vnc(0.7)
    elif tag.startswith("sterile"):
        p = set_nominal(opg.Sterile(devices=devices))
        if tag == "sterile":
            p.set_dm(4, 0.1)
            for i in (1, 2, 3):
                p.set_angle(i, 4, 0.1)
        else:
            p.set_dm(4, 1.3)
            p.set_angle(1, 4, 0.15)
            p.set_angle(2, 4, 0.2)
            p.set_angle(3, 4, 0.3)
            p.set_delta(1, 4, 0.9)
            p.set_delta(2, 4, -1.7)
    elif tag.startswith("decay"):
        p = set_nominal(opg.Decay(devices=devices))
        if tag == "decay":
            p.set_alpha3(1e-4)
        else:
            p.set_alpha2(3e-5)
            p.set_alpha3(2e-4)
    elif tag.startswith("liv"):
        p = set_nominal(opg.LIV(devices=devices))
        if tag == "liv":
            p.set_aT(0, 0, 3, 1e-21); p.set_aT(0, 1, 3, 2e-21)
            p.set_aT(1, 2, 3, -1e-21); p.set_cT(1, 1, 4, 5e-23)
            p.set_cT(0, 2, 4, 1e-22); p.set_aT(1, 2, 5, 1e-24)
            p.set_cT(2, 2, 6, 2e-26); p.set_aT(0, 1, 7, 1e-28)
            p.set_cT(1, 2, 8, 1e-30)
        else:
            p.set_aT(0, 1, 3, 1.5e-21, 0.7); p.set_aT(0, 2, 3, 8e-22, -1.9)
            p.set_aT(2, 2, 3, -6e-22); p.set_cT(1, 2, 4, 1e-22, 2.3)
            p.set_cT(0, 0, 4, -4e-23); p.set_cT(0, 1, 6, 3e-26, 1.1)
    elif tag.startswith("snsi"):
        p = set_nominal(opg.SNSI(devices=devices))
        if tag == "snsi":
            p.set_lowest_mass(0.05)
            for (i, j), v in {(0, 0): .5, (0, 1): .3, (1, 1): -.2,
                              (1, 2): .4, (2, 2): .1}.items():
                p.set_eps(i, j, v, 0)
        else:
            p.set_dm(3, -2.465e-3 + 7.41e-5)
            p.set_lowest_mass(0)
            p.set_eps(0, 1, 0.4, 0.9)
            p.set_eps(0, 2, 0.2, -2.1)
            p.set_eps(1, 1, 0.3, 0)
            p.set_ferm_coup(0.5, 1.0, 0.8)
    elif tag.startswith("deco"):
        p = set_nominal(opg.Deco(devices=devices))
        if tag == "deco":
            p.set_gamma(2, 2e-21)
            p.set_gamma(3, 5e-21)
            p.set_deco_angle(0.3)
        else:
            p.set_dm(3, -2.465e-3 + 7.41e-5)
            p.set_gamma(2, 1e-24)
            p.set_deco_angle(2.0)
            p.set_gamma32(3e-24)
            p.set_power(2)
    else:
        raise ValueError(tag)
    return p


TAGS = ["fast", "fast_io", "nsi", "nsi_phases", "nunm", "nunm_phases",
        "nunm_high", "sterile", "sterile_phases", "decay", "decay_both",
        "liv", "liv_phases", "snsi", "snsi_io", "deco", "deco_power"]


@pytest.mark.parametrize("tag", TAGS)
def test_reference(tag, devices):
    p = make(tag, devices)
    E = load("grid_E_test.npy")
    for path, kind in ((TEST_PATH, "testpath"), (VAC_PATH, "vacuum")):
        ref = load(f"{tag}_{kind}.npy")                      # [nb][iE][a][b]
        got = np.stack([p.prob_path(E, path, nubar=bool(nb)) for nb in (0, 1)])
        got = got.transpose(0, 3, 1, 2)                       # -> [nb][iE][a][b]
        assert np.abs(got - ref).max() < 1e-10, kind

    Ep, C = load("grid_E_prem.npy"), load("grid_cosZ.npy")
    p.set_grid(Ep, C)
    p.calculate()
    got = p.probs().transpose(0, 3, 4, 1, 2)                  # -> [nb][iC][iE][a][b]
    assert np.abs(got - load(f"{tag}_prem.npy")).max() < 1e-10


def test_api_basics(devices):
    p = opg.Fast(devices=devices)
    # OscProb PDG defaults
    assert p.get_dm(3) == pytest.approx(2.52e-3)
    assert p.on_gpu == bool(devices)
    p.set_grid(np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.0]))
    p.calculate()
    P = p.probs()
    assert P.shape == (2, 3, 3, 2, 3)
    # unitarity for standard 3 flavour
    assert np.allclose(P.sum(axis=2), 1, atol=1e-12)
    assert p.prob(1, 0, 0, 2, False) == P[0, 1, 0, 0, 2]
    pts = p.prob_points(np.array([1.0, 3.0]), np.array([-1.0, 0.0]),
                        np.array([0, 1], dtype=np.uint8))
    assert np.allclose(pts[:, :, 0], P[0, :, :, 0, 0], atol=1e-15)
    assert np.allclose(pts[:, :, 1], P[1, :, :, 1, 2], atol=1e-15)
    with pytest.raises(ValueError):
        p.set_delta(1, 2, 0.1)   # rotation 12 is real


def test_premmodel():
    e = opg.PremModel()
    assert e.det_radius == 6368
    seg = e.fill_path(-1.0)
    assert seg.shape[1] == 4
    assert seg[:, 0].sum() == pytest.approx(e.get_total_L(-1.0))
    assert len(e.grazing_cosines()) > 10


def test_binned_vs_bruteforce(devices):
    p = set_nominal(opg.Fast(devices=devices))
    edges, ref = load("avg_fast_bins.npy"), load("avg_fast.npy")
    worst = 0
    for k, (e0, e1, c0, c1) in enumerate(edges):
        p.set_bins(np.array([e0, e1]), np.array([c0, c1]), 24, 12)
        p.calculate_binned()
        A = p.binned()[:, :, :, 0, 0]                         # [nb][a][b]
        worst = max(worst, np.abs(A - ref[k]).max())
    assert worst < 2e-5


def test_avg_path_vs_oscprob_avgprob(devices):
    p = set_nominal(opg.Fast(devices=devices))
    edges, ref = load("avg1d_fast_edges.npy"), load("avg1d_fast.npy")
    for nb in (0, 1):
        A = p.avg_path(edges, 16, TEST_PATH, nubar=bool(nb))  # [a][b][iE]
        assert np.abs(A.transpose(2, 0, 1) - ref[nb]).max() < 5e-4


def test_absorption():
    """OscProb Absorption::Trans on the PREM paths and the test path."""
    xs, ref, C = load("absorption_xsec.npy"), load("absorption_prem.npy"), load("grid_cosZ.npy")
    e = opg.PremModel()
    for k, x in enumerate(xs):
        assert np.array_equal(e.transmission(C, x), ref[:, k])
    assert np.array_equal(e.transmission(C, np.full(C.size, xs[2])), ref[:, 2])
    tp = load("absorption_testpath.npy")
    assert [opg.path_transmission(TEST_PATH, x) for x in xs] == list(tp)
    X = e.column_depth(C)
    assert np.allclose(np.exp(-X * xs[2] / 1.660539066e-24), ref[:, 2], rtol=1e-13, atol=0)
