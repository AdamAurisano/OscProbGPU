"""Python API against the OscProb reference values (CPU and GPU backends)."""

import numpy as np
import pytest

import oscprobgpu as opg
from conftest import TEST_PATH, VAC_PATH, load, set_nominal


# 3+2 and 3+3 (tests/variants.h sterile5_mix, sterile6_mix): nus, dm_j1,
# theta_ij, delta_ij
STERILE_N = {
    "sterile5": (5, {4: 0.5, 5: 1.3},
                 {(1, 4): 0.15, (2, 4): 0.1, (3, 4): 0.2, (1, 5): 0.05, (2, 5): 0.12,
                  (3, 5): 0.1, (4, 5): 0.3},
                 {(1, 4): 0.9, (2, 4): -1.7, (1, 5): 0.4, (2, 5): 2.2, (3, 5): -0.6}),
    "sterile6": (6, {4: 0.1, 5: 0.8, 6: 1.7},
                 {(1, 4): 0.1, (2, 4): 0.1, (3, 4): 0.1, (1, 5): 0.08, (2, 5): 0.06,
                  (3, 5): 0.04, (4, 5): 0.2, (1, 6): 0.05, (2, 6): 0.07, (3, 6): 0.03,
                  (4, 6): 0.15, (5, 6): 0.25},
                 {(1, 4): 0.3, (2, 4): -1.1, (1, 5): 1.9, (2, 5): -0.4, (3, 5): 0.8,
                  (1, 6): -2.5, (2, 6): 1.2, (3, 6): 0.5, (4, 6): -0.9}),
}


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
    elif tag in STERILE_N:
        nus, dms, angles, deltas = STERILE_N[tag]
        p = set_nominal(opg.Sterile(devices=devices, nus=nus))
        assert p.n_flavours == nus
        for j, v in dms.items():
            p.set_dm(j, v)
        for (i, j), v in angles.items():
            p.set_angle(i, j, v)
        for (i, j), v in deltas.items():
            p.set_delta(i, j, v)
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
    elif tag.startswith("sidereal"):
        p = set_nominal(opg.SiderealLIV(devices=devices))
        if tag == "sidereal":
            for (i, j, c), v in {(0, 0, 0): .1e-22, (0, 1, 1): .2e-22, (0, 2, 2): .3e-22,
                                 (1, 1, 0): .4e-22, (1, 2, 1): .5e-22,
                                 (2, 2, 2): .6e-22}.items():
                p.set_a(i, j, c, v)
            for (i, j, c1, c2), v in {(0, 0, 0, 0): .1e-22, (0, 1, 1, 1): .2e-22,
                                      (0, 2, 2, 2): .3e-22, (1, 1, 0, 1): .4e-22,
                                      (1, 2, 1, 2): .5e-22, (2, 2, 0, 2): .6e-22}.items():
                p.set_c(i, j, c1, c2, v)
            p.set_latitude(-89, -59, -24)
            p.set_neutrino_direction(57.3, 28.6)   # fixed paths
            p.set_azimuth(28.6)                     # Earth paths: path zenith
            p.set_time_hours(6.0)
        else:
            p.set_a(0, 1, 0, 3e-22); p.set_a(1, 2, 2, -2e-22); p.set_a(2, 2, 1, 1e-22)
            p.set_c(0, 0, 1, 1, 2e-23); p.set_c(0, 1, 0, 1, -1e-23)
            p.set_c(1, 2, 0, 2, 3e-23); p.set_c(0, 2, 1, 2, 1.5e-23)
            p.set_c(1, 1, 0, 0, -2.5e-23)
            p.set_colatitude(43.5)
            p.set_neutrino_direction(120.0, 250.0)
            p.set_time_hours(17.3)
    elif tag.startswith("oqs"):
        p = set_nominal(opg.OQS(devices=devices))
        if tag == "oqs":
            p.set_deco_element(3, np.sqrt(2e-23))
            p.set_deco_element(8, np.sqrt(4e-23))
            p.set_deco_angle(3, 8, np.arccos(0.5))
        else:
            p.set_dm(3, -2.465e-3 + 7.41e-5)
            a = [0, 1.0, 0.7, 1.3, 0.5, 0.9, 1.1, 0.6, 0.8]
            for i in range(1, 9):
                p.set_deco_element(i, a[i] * 1e-13)
            for i in range(1, 9):
                for j in range(i + 1, 9):
                    p.set_deco_angle(i, j, 0.3 + 0.17 * i + 0.11 * j)
            p.set_power(1)
    else:
        raise ValueError(tag)
    return p


TAGS = ["fast", "fast_io", "nsi", "nsi_phases", "nunm", "nunm_phases",
        "nunm_high", "sterile", "sterile_phases", "sterile5", "sterile6", "decay", "decay_both",
        "liv", "liv_phases", "snsi", "snsi_io", "deco", "deco_power",
        "sidereal", "sidereal_fixed", "oqs", "oqs_full"]


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


@pytest.mark.parametrize("tag", ["sidereal", "sidereal_fixed"])
def test_sidereal_per_event_extra(tag, devices):
    """Per-event azimuth and sidereal time equal the values set on the model."""
    p = make(tag, devices)
    assert opg.SiderealLIV.n_event_extra == 2 and opg.Fast.n_event_extra == 0
    rng = np.random.default_rng(5)
    n = 24
    E = 10 ** rng.uniform(-0.3, 1.7, n)
    C = rng.uniform(-1, 1, n)
    nb = (np.arange(n) % 2).astype(np.uint8)
    extra = np.stack([rng.uniform(0, 360, n), rng.uniform(0, 24, n)])
    grads = opg.SiderealLIV.has_gradients
    names = ["th23", "dm31", "aX_emu", "cXZ_etau"]
    if grads:
        p.set_gradient_params(names)
        P, dP = p.prob_points_grad(E, C, nb, extra)
        assert np.array_equal(P, p.prob_points(E, C, nb, extra=extra))
        w = rng.uniform(-1, 1, P.shape)
        g = p.weighted_gradient_points(E, C, nb, w, extra=extra)
        assert np.allclose(g, np.einsum("pabi,abi->p", dP, w), rtol=1e-12, atol=0)
        bins = (np.arange(n) % 3).astype(np.int32)
        G = p.weighted_gradient_points_binned(E, C, nb, w, bins, 3, extra=extra)
        Ge = np.stack([np.einsum("pabi,abi->p", dP[..., bins == b], w[..., bins == b])
                       for b in range(3)])
        assert np.allclose(G, Ge, rtol=1e-12, atol=0)
    else:
        P = p.prob_points(E, C, nb, extra=extra)
    zen = 120.0 if tag == "sidereal_fixed" else None
    for i in range(n):
        if zen is None:
            p.set_azimuth(extra[0, i])
        else:
            p.set_neutrino_direction(zen, extra[0, i])
        p.set_time_hours(extra[1, i])
        Pi = p.prob_points(E[i:i + 1], C[i:i + 1], nb[i:i + 1])
        assert np.abs(Pi[:, :, 0] - P[:, :, i]).max() <= 1e-12
        if grads:
            _, dPi = p.prob_points_grad(E[i:i + 1], C[i:i + 1], nb[i:i + 1])
            assert np.abs(dPi[..., 0] - dP[..., i]).max() <= 1e-12 * np.abs(dP).max()
    with pytest.raises(ValueError):
        p.prob_points(E, C, nb, extra=extra[:1])
    with pytest.raises(ValueError):
        set_nominal(opg.Fast(devices=devices)).prob_points(E, C, nb, extra=extra)


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


def test_misuse_raises_instead_of_crashing(devices):
    """Getters before a computation, stale gradient layouts and bad indices
    raise (they used to read out of bounds and could segfault)."""
    p = set_nominal(opg.Fast(devices=devices))
    p.set_grid(np.array([1.0, 10.0]), np.array([-1.0, -0.5]))
    with pytest.raises(RuntimeError):
        p.probs()
    p.set_gradient_params()
    with pytest.raises(RuntimeError):
        p.grad()
    p.calculate_gradient()
    assert p.grad().shape[1] == len(p.gradient_params)
    p.set_gradient_params(["th23"])
    with pytest.raises(RuntimeError):
        p.grad()
    with pytest.raises(RuntimeError):
        p.binned_grad()
    with pytest.raises(IndexError):
        p.get_angle(0, 400)
    with pytest.raises(IndexError):
        p.get_dm(7)
    p.calculate()
    with pytest.raises(IndexError):
        p.prob(7, 7, 10**9, 0, False)


def test_default_earth_from_package_data(tmp_path, monkeypatch):
    """PremModel() reads $OPG_DATA_DIR/prem_default.txt when set (installed
    wheels point it at oscprobgpu/data)."""
    import os
    import shutil
    repo_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
    d = tmp_path / "data"
    d.mkdir()
    shutil.copy(os.path.join(repo_data, "prem_15layers.txt"), d / "prem_default.txt")
    monkeypatch.setenv("OPG_DATA_DIR", str(d))
    assert (opg.PremModel().grazing_cosines()
            == opg.PremModel(os.path.join(repo_data, "prem_15layers.txt")).grazing_cosines())
    assert opg.PremModel().grazing_cosines() != opg.PremModel(
        os.path.join(repo_data, "prem_default.txt")).grazing_cosines()
    monkeypatch.setenv("OPG_DATA_DIR", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError):
        opg.PremModel()


def test_version_matches_pyproject():
    """oscprobgpu.__version__ comes from the CMake project version; it must
    match the wheel version in pyproject.toml."""
    import os
    import re
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    with open(os.path.join(root, "pyproject.toml")) as f:
        v = re.search(r'^version = "([^"]+)"', f.read(), re.M).group(1)
    assert opg.__version__ == v


def test_sterile_nus():
    """opg.Sterile(nus=N) as OscProb's PMNS_Sterile(N)."""
    assert type(opg.Sterile()) is opg.Sterile and opg.Sterile().n_flavours == 4
    assert isinstance(opg.Sterile(nus=4), opg.Sterile)
    for nus, cls in ((5, opg.Sterile5), (6, opg.Sterile6)):
        p = opg.Sterile(nus=nus, threads=1)
        assert type(p) is cls and p.n_flavours == nus == cls.n_flavours
        assert cls.parameter_names[-1] == f"dm{nus}1"
        with pytest.raises(ValueError):
            p.set_angle(1, nus + 1, 0.1)
    assert len(opg.Sterile5.parameter_names) == 20
    assert len(opg.Sterile6.parameter_names) == 30
    with pytest.raises(ValueError):
        opg.Sterile(nus=7)
