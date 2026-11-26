// EarthModelBinned (3D binned Earth) on the CPU backend: OscProb paths bit for
// bit, probabilities, per-event azimuths, region gradients, batches.

#include <functional>

#include "doctest.h"

#include "binned.h"

#include "opg/models/all.h"

namespace {

  constexpr double kTolCPU = 1e-11;

  template <class Model>
  void check_probs(opg::Propagator<Model>& prop, const std::string& tag, bool use_extra = false,
                   std::function<void(int)> per_config = nullptr)
  {
    for (int k = 0; k < 2; k++) {
      if (per_config) per_config(k);
      refcmp::Result rg, re;
      bintest::compare_probs(prop, tag, k, rg, re, use_extra);
      MESSAGE(tag << " binned cfg " << k << ": grid max|dP| = " << rg.max_abs << " ("
                  << rg.n_exact << "/" << rg.n << " exact), points " << re.max_abs << " ("
                  << re.n_exact << "/" << re.n << ")");
      CHECK(rg.max_abs < kTolCPU);
      CHECK(re.max_abs < kTolCPU);
#ifdef OPG_OSCPROB_BITWISE
      CHECK(rg.n_exact == rg.n);
      CHECK(re.n_exact == re.n);
#endif
    }
  }

} // namespace

TEST_CASE("EarthModelBinned: loader, detector position, regions")
{
  opg::EarthModelBinned m;
  CHECK(m.GetEarthBins().size() == 528);
  CHECK(m.GetNDepthBins() == 44);
  CHECK(m.GetNLatBins() == 3);
  CHECK(m.GetNLonBins() == 4);
  CHECK(m.GetRadiusMax() == 6386);
  CHECK(m.GetDetRadius() == 6368);
  CHECK(m.GetRegions() == std::vector<int>{0, 1, 2, 3, 4, 5, 6});
  CHECK(m.GetRegionZoA(2) == double(0.4954f));
  CHECK(m.GetRegionZoA(9) == 0);
  m.SetRegionZoA(2, 0.47);
  m.ScaleRegionDensity(2, 2.0);
  for (auto& b : m.GetEarthBins())
    if (b.index == 2) {
      CHECK(b.zoa == 0.47);
      CHECK(b.density > 8);
    }
  // EarthModelBase::SetDetectorCoordinates folding
  m.SetDetPos(-6000, 100, -30);
  CHECK(m.GetDetRadius() == 6000);
  CHECK(m.GetDetLat() == 80 / 180.0 * M_PI);
  CHECK(m.GetDetLon() == 330 / 180.0 * M_PI);

  auto t = bintest::test_table();
  CHECK(t.GetEarthBins().size() == 2112);
  CHECK(t.GetNLatBins() == 6);
  CHECK(t.GetNLonBins() == 8);
  CHECK(t.GetRegions() == std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7});

  CHECK_THROWS_AS(opg::EarthModelBinned("/nonexistent/table.txt"), std::runtime_error);

  // grazing cosines: depth-shell boundaries below the detector and cosZ = 0
  auto g = opg::EarthModelBinned().GetGrazingCosines();
  CHECK(g.size() == 42);
  CHECK(g.back() == 0);
  CHECK(std::is_sorted(g.begin(), g.end()));
}

TEST_CASE("EarthModelBinned paths match OscProb bit for bit")
{
  auto                  cases = npy::load<double>("binned_path_cases.npy");
  auto                  off   = npy::load<int>("binned_path_offsets.npy");
  auto                  segs  = npy::load<double>("binned_paths.npy");
  opg::EarthModelBinned tables[2] = {opg::EarthModelBinned(), bintest::test_table()};
  const size_t          ncase = cases.size() / 6;
  REQUIRE(off.size() == ncase + 1);
  size_t bad = 0, badv = 0, nseg = 0;
  for (size_t k = 0; k < ncase; k++) {
    const double* c = &cases[6 * k];
    auto&         m = tables[int(c[0])];
    m.SetDetPos(c[1], c[2], c[3]);
    const auto   p  = m.FillPath(c[4], c[5]);
    const size_t n0 = size_t(off[k]), n = size_t(off[k + 1]) - n0;
    nseg += n;
    bool ok = p.size() == n;
    for (size_t i = 0; ok && i < n; i++) {
      const double* r = &segs[4 * (n0 + i)];
      ok = p[i].length == r[0] && p[i].density == r[1] && p[i].zoa == r[2] &&
           p[i].layer == int(r[3]);
    }
    if (!ok) {
      if (bad < 5)
        MESSAGE("case " << k << " (table " << c[0] << ", det " << c[1] << " " << c[2] << " "
                        << c[3] << ", cosT " << c[4] << ", phi " << c[5] << "): " << p.size()
                        << " vs " << n << " segments");
      bad++;
    }
    // the same through an EarthView with the azimuth of the path
    const opg::EarthTable<double> t{opg::EarthModel(m)};
    opg::BinnedGeom               g;
    auto                          v = t.view(g);
    v.azimuth                       = c[5];
    std::vector<opg::Segment<double>> q;
    opg::for_each_segment(v, c[4], [&](const opg::Segment<double>& s) { q.push_back(s); });
    bool same = q.size() == p.size();
    for (size_t i = 0; same && i < q.size(); i++)
      same = q[i].length == p[i].length && q[i].density == p[i].density &&
             q[i].zoa == p[i].zoa && q[i].layer == p[i].layer;
    badv += !same;
  }
  MESSAGE(ncase << " paths (" << nseg << " segments): " << bad
                << " differ from OscProb, " << badv << " differ through the view");
  CHECK(bad == 0);
  CHECK(badv == 0);
}

TEST_CASE("EarthModelBinned probabilities match OscProb (CPU)")
{
  {
    opg::Propagator<opg::Fast<>> prop;
    opg::Fast<>::Params          p;
    p.mix = variants::nominal_mix<3>();
    prop.set_params(p);
    check_probs(prop, "fast");
  }
  {
    opg::Propagator<opg::NSI<>> prop;
    prop.set_params(variants::nsi_phases());
    check_probs(prop, "nsi_phases");
  }
  {
    // the direction's azimuth is that of the path; event lists share the
    // azimuth extra with the Earth
    opg::Propagator<opg::SiderealLIV<>> prop;
    check_probs(prop, "sidereal", true, [&](int k) {
      auto p = variants::sidereal();
      p.SetAzimuth(k == 0 ? 30 : 250);
      prop.set_params(p);
    });
  }
}

TEST_CASE("EarthModelBinned: Earth accessors and errors")
{
  opg::Propagator<opg::Fast<>> prem;
  CHECK_THROWS_AS(prem.set_earth_azimuth(10), std::logic_error);
  CHECK_NOTHROW(prem.earth());
  CHECK(!prem.earth_is_binned());

  opg::Propagator<opg::Fast<>> prop(opg::EarthModelBinned{});
  CHECK(prop.earth_is_binned());
  CHECK_THROWS_AS(prop.earth(), std::logic_error);
  prop.set_earth_azimuth(123);
  CHECK(prop.earth_azimuth() == 123);
  CHECK(prop.earth_model().binned().GetAzimuth() == 123);
  CHECK(prop.earth_parameter_names().size() == 14);  // regions 0..6

  opg::Fast<>::Params p;
  p.mix = variants::nominal_mix<3>();
  prop.set_params(p);
  std::vector<double>  e = {1, 2}, c = {-0.5, 0.3}, az = {10};
  std::vector<uint8_t> nb = {0, 1};
  CHECK_THROWS_AS(prop.prob_points(e, c, nb, {}, az), std::invalid_argument);

  // PREM ignores event azimuths
  prem.set_params(p);
  CHECK(prem.prob_points(e, c, nb, {}, {10, 20}) == prem.prob_points(e, c, nb));

  // a selected region parameter must exist in the new Earth
  prop.set_earth(bintest::test_table());
  prop.set_gradient_params({"zoa_7"});
  CHECK_THROWS_AS(prop.set_earth(opg::EarthModelBinned{}), std::invalid_argument);
  CHECK(prop.earth_model().binned().GetNLatBins() == 6);
  prop.set_gradient_params({});

  // detector above the top of the model
  opg::EarthModelBinned high;
  high.SetDetPos(7000);
  CHECK_THROWS_AS(prop.set_earth(high), std::invalid_argument);
}

TEST_CASE_TEMPLATE("Binned Earth: per-event azimuths equal per-propagator azimuths (CPU)", M,
                   gradtest::Fast, gradtest::NSI)
{
  opg::Propagator<M> prop;
  bintest::check_event_azimuth(prop);
}

TEST_CASE_TEMPLATE("Binned Earth region gradients match long-double FD (CPU)", M,
                   gradtest::Fast, gradtest::NSI, gradtest::Sterile)
{
  opg::Propagator<M> prop;
  bintest::check_region_grads(prop, 1e-9);
}

TEST_CASE_TEMPLATE("Binned Earth: adjoint equals forward, grid/event/weighted gradients "
                   "consistent (CPU)",
                   M, gradtest::Fast, gradtest::NSI, gradtest::Sterile)
{
  opg::Propagator<M> prop(bintest::config(1));
  // Sterile (Jacobi, no affine reverse path) accumulates a little more
  // round-off over the longer binned paths than through PREM
  gradtest::check_adjoint(prop, std::is_same<M, gradtest::Sterile>::value ? 1e-10 : 1e-11);
  gradtest::check_grid_and_weighted(prop, 1e-13);
  gradtest::check_values_unchanged(prop);
}

TEST_CASE("Binned Earth bin averages and per-point batches (CPU)")
{
  using M = opg::Fast<>;
  opg::Propagator<M> prop(bintest::config(1));
  M::Params          p;
  p.mix = variants::nominal_mix<3>();
  prop.set_params(p);
  const std::vector<double> Ee = {1.0, 2.0, 4.0, 10.0}, Ce = {-1.0, -0.8, -0.4, 0.0, 0.5};
  // Layer-adapted nodes split the cosZ bins at the depth-shell grazing
  // cosines only (lateral boundaries are ignored): the convergence in the
  // number of cosZ nodes is that of PREM's for a laterally uniform table
  // (cfg 0) and of the same order with lateral steps (cfg 1).
  double d[3][2];
  for (int k = 0; k < 3; k++) {
    if (k < 2)
      prop.set_earth(bintest::config(k));
    else
      prop.set_earth(opg::PremModel());
    std::vector<std::vector<double>> B;
    for (int n : {6, 12, 24}) {
      prop.set_bins(Ee, Ce, 8, n);
      prop.calculate_binned();
      B.emplace_back(prop.binned(), prop.binned() + 2 * 9 * 4 * 3);
    }
    for (int j = 0; j < 2; j++) {
      d[k][j] = 0;
      for (size_t i = 0; i < B[0].size(); i++)
        d[k][j] = std::max(d[k][j], std::fabs(B[size_t(j)][i] - B[2][i]));
    }
    MESSAGE((k < 2 ? "binned Earth cfg " + std::to_string(k) : std::string("PREM"))
            << " bin averages, 6 and 12 vs 24 cosZ nodes per piece: " << d[k][0] << ", "
            << d[k][1]);
  }
  for (int k = 0; k < 2; k++)
    for (int j = 0; j < 2; j++) CHECK(d[k][j] < 10 * d[2][j]);
  prop.set_earth(bintest::config(1));
  prop.set_bins(Ee, Ce, 8, 12);

  // per-point Earths: only densities and Z/A differ
  std::vector<M::Params>             pts(3, p);
  std::vector<opg::EarthModelBinned> earths(3, bintest::config(1));
  pts[1].mix.dm[2] *= 1.01;
  earths[1].SetRegionZoA(7, 0.45);
  earths[2].ScaleRegionDensity(3, 0.97);
  prop.set_gradient_params({"dm31", "zoa_7", "rho_3"});
  std::vector<double> out, dout;
  prop.binned_batch(pts, &earths, out, &dout);
  const auto   e0  = prop.earth_model();
  const size_t per = 2 * 9 * prop.n_cosine_bins() * prop.n_energy_bins();
  for (size_t k = 0; k < 3; k++) {
    opg::Propagator<M> q(earths[k]);
    q.set_params(pts[k]);
    q.set_gradient_params({"dm31", "zoa_7", "rho_3"});
    q.set_bins(Ee, Ce, 8, 12);
    q.calculate_binned(opg::Flavor::Both, true);
    CHECK(std::equal(q.binned(), q.binned() + per, out.begin() + long(k * per)));
    CHECK(std::equal(q.binned_grad(), q.binned_grad() + 3 * per,
                     dout.begin() + long(k * 3 * per)));
  }
  CHECK(prop.earth_model().binned().GetEarthBins()[0].density ==
        e0.binned().GetEarthBins()[0].density);
  // a different geometry (detector, azimuth or table) is refused
  std::vector<opg::EarthModelBinned> bad(3, bintest::config(1));
  bad[2].SetAzimuth(10);
  CHECK_THROWS_AS(prop.binned_batch(pts, &bad, out), std::invalid_argument);
  bad[2] = bintest::config(1);
  bad[2].SetDetPos(6368, 42.4, 13.6);
  CHECK_THROWS_AS(prop.binned_batch(pts, &bad, out), std::invalid_argument);
  std::vector<opg::PremModel> prems(3);
  CHECK_THROWS_AS(prop.binned_batch(pts, &prems, out), std::invalid_argument);
}
