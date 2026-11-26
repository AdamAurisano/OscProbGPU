"""EarthModelBinned (3D binned Earth) through the Python bindings."""

import os

import numpy as np
import pytest

import oscprobgpu as opg
from conftest import load, set_nominal


@pytest.fixture(scope="module")
def test_table(tmp_path_factory):
    """The reference test table (binned_table.npy) as text, as the dump wrote it."""
    t = load("binned_table.npy")
    f = tmp_path_factory.mktemp("binned") / "table.txt"
    with open(f, "w") as out:
        for r in t:
            out.write(" ".join("%.17g" % x for x in r) + "\n")
    return str(f)


def config(k, table):
    if k == 0:
        e = opg.EarthModelBinned()
        e.set_det_pos(6368, 0, 0)
        e.set_azimuth(30)
        return e
    e = opg.EarthModelBinned(table)
    e.set_det_pos(6368, 42.4, 13.5)
    e.set_region_zoa(7, 0.47)
    e.scale_region_density(2, 1.03)
    e.set_azimuth(250)
    return e


def test_model_basics(test_table):
    e = opg.EarthModelBinned()
    assert (e.n_depth_bins, e.n_lat_bins, e.n_lon_bins) == (44, 3, 4)
    assert e.regions == list(range(7))
    assert e.earth_bins().shape == (528, 7)
    assert e.det_radius == 6368
    t = opg.EarthModelBinned(test_table)
    assert (t.n_lat_bins, t.n_lon_bins) == (6, 8)
    assert np.isclose(t.get_region_zoa(7), 0.497, atol=1e-7)
    p = t.fill_path(-0.4, 100.0)
    assert p.shape[1] == 4 and np.isclose(p[:, 0].sum(), t.get_total_L(-0.4))
    t.set_azimuth(100.0)
    assert np.array_equal(t.fill_path(-0.4), p)
    assert t.column_depth(np.array([-0.4]))[0] == pytest.approx((p[:, 0] * p[:, 1]).sum() * 1e5)
    with pytest.raises(RuntimeError):
        opg.EarthModelBinned("/nonexistent/table.txt")


def test_paths_match_oscprob(test_table):
    cases, off, segs = (load("binned_path_cases.npy"), load("binned_path_offsets.npy"),
                        load("binned_paths.npy"))
    tables = [opg.EarthModelBinned(), opg.EarthModelBinned(test_table)]
    for k, c in enumerate(cases):
        m = tables[int(c[0])]
        m.set_det_pos(c[1], c[2], c[3])
        assert np.array_equal(m.fill_path(c[4], c[5]), segs[off[k]:off[k + 1]]), k


@pytest.mark.parametrize("tag", ["fast", "nsi_phases", "sidereal"])
def test_probabilities_match_oscprob(tag, test_table, devices):
    E, C = load("grid_E_binned.npy"), load("grid_cosZ_binned.npy")
    ref = load(f"{tag}_binned.npy")                     # [cfg][nb][iC][iE][a][b]
    for k in (0, 1):
        e = config(k, test_table)
        if tag == "fast":
            p = set_nominal(opg.Fast(earth=e, devices=devices))
        elif tag == "nsi_phases":
            p = set_nominal(opg.NSI(earth=e, devices=devices))
            for (i, j, v, ph) in [(0, 0, -0.2, 0), (0, 1, 0.05, 0.7), (0, 2, 0.15, -1.3),
                                  (1, 1, 0.02, 0), (1, 2, 0.03, 2.1), (2, 2, 0.1, 0)]:
                p.set_eps(i, j, v, ph)
            p.set_ferm_coup(0.5, 1.0, 0.8)
        else:
            p = set_nominal(opg.SiderealLIV(earth=e, devices=devices))
            for (i, j, c, v) in [(0, 0, 0, 0.1e-22), (0, 1, 1, 0.2e-22), (0, 2, 2, 0.3e-22),
                                 (1, 1, 0, 0.4e-22), (1, 2, 1, 0.5e-22), (2, 2, 2, 0.6e-22)]:
                p.set_a(i, j, c, v)
            for (i, j, c1, c2, v) in [(0, 0, 0, 0, 0.1e-22), (0, 1, 1, 1, 0.2e-22),
                                      (0, 2, 2, 2, 0.3e-22), (1, 1, 0, 1, 0.4e-22),
                                      (1, 2, 1, 2, 0.5e-22), (2, 2, 0, 2, 0.6e-22)]:
                p.set_c(i, j, c1, c2, v)
            p.set_latitude(-89, -59, -24)
            p.set_azimuth(e.azimuth)
            p.set_time_hours(6.0)
        assert p.earth_is_binned and p.earth_azimuth == e.azimuth
        p.set_grid(E, C)
        p.calculate()
        got = p.probs().transpose(0, 3, 4, 1, 2)
        assert np.abs(got - ref[k]).max() < 1e-10
        # event list: per-event azimuths (SiderealLIV: through its azimuth extra)
        Eg, Cg = np.meshgrid(E, C)
        n = Eg.size
        nb = np.zeros(n, np.uint8)
        az = np.full(n, e.azimuth)
        p.set_earth_azimuth(e.azimuth + 77)
        if tag == "sidereal":
            P = p.prob_points(Eg.ravel(), Cg.ravel(), nb, extra=np.stack([az, np.full(n, 6.0)]))
        else:
            P = p.prob_points(Eg.ravel(), Cg.ravel(), nb, azimuth=az)
        got = P.reshape(P.shape[0], P.shape[1], len(C), len(E)).transpose(2, 3, 0, 1)
        assert np.abs(got - ref[k][0]).max() < 1e-10


def test_region_gradients_and_batches(test_table, devices):
    e = config(1, test_table)
    p = set_nominal(opg.Fast(earth=e, devices=devices))
    names = p.earth_parameter_names
    assert names[:8] == [f"zoa_{i}" for i in range(8)] and len(names) == 16
    rng = np.random.default_rng(3)
    n = 40
    E, C = 10 ** rng.uniform(-0.3, 1.5, n), rng.uniform(-1, 0.9, n)
    nb, az = (np.arange(n) % 2).astype(np.uint8), rng.uniform(0, 360, n)
    p.set_gradient_params(["dm31", "zoa_7", "rho_3"])
    P, dP = p.prob_points_grad(E, C, nb, azimuth=az)

    def probs(model):
        q = set_nominal(opg.Fast(earth=model, devices=devices))
        return q.prob_points(E, C, nb, azimuth=az)

    h = 1e-5
    for k, (edit, x0) in enumerate([(lambda m, x: m.set_region_zoa(7, x), 0.47),
                                    (lambda m, x: m.scale_region_density(3, x), 1.0)]):
        mp, mm = config(1, test_table), config(1, test_table)
        edit(mp, x0 + h)
        edit(mm, x0 - h)
        fd = (probs(mp) - probs(mm)) / (2 * h)
        assert np.abs(fd - dP[k + 1]).max() < 1e-6 * max(1, np.abs(dP[k + 1]).max())
    w = rng.uniform(-1, 1, P.shape)
    g = p.weighted_gradient_points(E, C, nb, w, azimuth=az)
    assert np.allclose(g, np.einsum("pabi,abi->p", dP, w), rtol=1e-10, atol=0)
    with pytest.raises(ValueError):
        p.prob_points(E, C, nb, azimuth=az[:3])

    # per-point binned Earths in batches
    p.set_bins(np.geomspace(1, 30, 6), np.linspace(-1, 0.5, 5), 3, 3)
    earths = [config(1, test_table) for _ in range(3)]
    earths[1].set_region_zoa(7, 0.45)
    earths[2].scale_region_density(3, 0.97)
    vals = np.array([[2.5e-3], [2.6e-3], [2.4e-3]])
    A, dA = p.binned_batch(["dm31"], vals, earths=earths, gradient=True)
    for i in range(3):
        q = set_nominal(opg.Fast(earth=earths[i], devices=devices))
        q.set_dm(3, vals[i, 0])
        q.set_bins(np.geomspace(1, 30, 6), np.linspace(-1, 0.5, 5), 3, 3)
        q.calculate_binned()
        assert np.array_equal(q.binned(), A[i])
    with pytest.raises(ValueError):
        p.binned_batch(["dm31"], vals, earths=[opg.PremModel()] * 3)
    with pytest.raises(ValueError):
        p.binned_batch(["dm31"], vals, earths=[opg.PremModel(), earths[0], earths[1]])


def test_azimuth_setter_needs_binned(devices):
    p = opg.Fast(devices=devices)
    assert not p.earth_is_binned
    with pytest.raises(RuntimeError):
        p.set_earth_azimuth(10.0)
    p.set_earth(opg.EarthModelBinned())
    p.set_earth_azimuth(10.0)
    assert p.earth_azimuth == 10.0
    p.set_earth(opg.PremModel())
    assert not p.earth_is_binned


def test_default_binned_table_from_package_data(tmp_path, monkeypatch):
    """EarthModelBinned() reads $OPG_DATA_DIR/earth_binned_default.txt when set."""
    import shutil
    repo_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
    d = tmp_path / "data"
    d.mkdir()
    shutil.copy(os.path.join(repo_data, "earth_binned_default.txt"), d)
    monkeypatch.setenv("OPG_DATA_DIR", str(d))
    assert opg.EarthModelBinned().n_depth_bins == 44
    monkeypatch.setenv("OPG_DATA_DIR", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError):
        opg.EarthModelBinned()
