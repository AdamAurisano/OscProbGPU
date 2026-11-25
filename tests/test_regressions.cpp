// Regression tests for defects found in the review of 2026-11-24 (CPU).

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "averaging.h"
#include "opg/linalg/eigh3.h"

namespace {
  using Fast = opg::Fast<double>;

  std::string data_file(const char* name)
  {
    std::string f = opg::PremModel::DefaultFile();
    return f.substr(0, f.rfind('/') + 1) + name;
  }

  void small_bins(opg::Propagator<Fast>& prop,
                  opg::CosZRule rule = opg::CosZRule::LayerAdapted)
  {
    std::vector<double> Ee, Ce;
    for (int i = 0; i <= 6; i++) Ee.push_back(std::pow(10.0, 0.5 + 1.5 * i / 6));
    for (int i = 0; i <= 5; i++) Ce.push_back(-1 + 1.0 * i / 5);
    prop.set_bins(Ee, Ce, 2, 2, opg::EMeasure::Log, rule);
  }

  std::vector<double> copy_binned(opg::Propagator<Fast>& prop)
  {
    const size_t n = 2 * 9 * prop.n_cosine_bins() * prop.n_energy_bins();
    const double* b = prop.binned();
    return std::vector<double>(b, b + n);
  }
} // namespace

TEST_CASE("eigh3: NaN in, NaN out (an all-NaN matrix took the zero-matrix exit)")
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (int all = 0; all < 2; all++) {
    opg::Mat<3, double> A = opg::Mat<3, double>::zero(), V;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        if (all || (i == 0 && j == 1)) A(i, j) = opg::Complex<double>(nan, all ? nan : 0);
    double w[3];
    opg::eigh3<double>(A, V, w);
    CAPTURE(all);
    CHECK(std::isnan(w[0]));
    CHECK(std::isnan(w[1]));
    CHECK(std::isnan(w[2]));
  }
}

TEST_CASE("CPU binned gradients: nu then nubar in two calls keep both halves")
{
  opg::Propagator<Fast> prop;
  avgtest::setup(prop);
  small_bins(prop);
  prop.set_gradient_params({"th23", "dm31"});
  prop.calculate_binned(opg::Flavor::Both, true);
  const size_t        n = 2 * 2 * 9 * prop.n_cosine_bins() * prop.n_energy_bins();
  std::vector<double> both(prop.binned_grad(), prop.binned_grad() + n);
  prop.calculate_binned(opg::Flavor::Neutrino, true);
  prop.calculate_binned(opg::Flavor::Antineutrino, true);
  std::vector<double> two(prop.binned_grad(), prop.binned_grad() + n);
  CHECK(two == both);
}

TEST_CASE("Propagator getters throw instead of reading missing or stale results")
{
  opg::Propagator<Fast> prop;
  avgtest::setup(prop);
  prop.set_grid({1.0, 10.0}, {-1.0, -0.5, 0.0});
  CHECK_THROWS_AS(prop.probs(), std::logic_error);
  prop.calculate();
  CHECK_NOTHROW(prop.probs());
  CHECK_THROWS_AS(prop.prob(0, 3, 0, 0, false), std::out_of_range);
  CHECK_THROWS_AS(prop.prob(0, 0, 3, 0, false), std::out_of_range);
  CHECK_THROWS_AS(prop.grad(), std::logic_error);

  prop.set_gradient_params({"th23"});
  prop.calculate(opg::Flavor::Both, true);
  CHECK_NOTHROW(prop.grad());
  CHECK_THROWS_AS(prop.grad(1, 0, 0, 0, 0, false), std::out_of_range);
  prop.set_gradient_params({"th23", "dm31"});  // layout of grad() changes
  CHECK_THROWS_AS(prop.grad(), std::logic_error);
  prop.calculate(opg::Flavor::Both, true);
  prop.set_grid({1.0}, {-1.0});
  CHECK_THROWS_AS(prop.probs(), std::logic_error);
  CHECK_THROWS_AS(prop.grad(), std::logic_error);

  CHECK_THROWS_AS(prop.binned(), std::logic_error);
  small_bins(prop);
  CHECK_THROWS_AS(prop.binned(), std::logic_error);
  CHECK_THROWS_AS(prop.binned_grad(), std::logic_error);
  prop.calculate_binned(opg::Flavor::Both, true);
  CHECK_NOTHROW(prop.binned_grad());
  CHECK_THROWS_AS(prop.binned(0, 0, 99, 0, false), std::out_of_range);
  prop.set_gradient_params({"th23"});
  CHECK_THROWS_AS(prop.binned_grad(), std::logic_error);
  CHECK_NOTHROW(prop.binned());
}

TEST_CASE("set_earth that throws leaves the propagator unchanged")
{
  opg::Propagator<Fast> prop;
  avgtest::setup(prop);
  small_bins(prop);
  const auto en = prop.earth_parameter_names();
  REQUIRE(std::find(en.begin(), en.end(), "zoa_6") != en.end());
  prop.set_gradient_params({"th23", "zoa_6"});
  prop.calculate_binned(opg::Flavor::Both, true);
  const auto before = copy_binned(prop);
  const size_t nl = prop.earth().GetLayers().size();

  const opg::PremModel lbl(data_file("prem_lbl.txt"));  // layer types 0-4 only
  CHECK_THROWS_AS(prop.set_earth(lbl), std::invalid_argument);
  CHECK(prop.earth().GetLayers().size() == nl);
  CHECK(prop.earth_parameter_names() == en);
  CHECK(prop.gradient_params() == std::vector<std::string>{"th23", "zoa_6"});
  prop.calculate_binned(opg::Flavor::Both, true);
  CHECK(copy_binned(prop) == before);
}

TEST_CASE("Layer-adapted cosZ nodes follow set_earth")
{
  const opg::PremModel e2(data_file("prem_15layers.txt"));
  REQUIRE(e2.GetGrazingCosines() != opg::PremModel().GetGrazingCosines());
  opg::Propagator<Fast> a, b(e2);
  avgtest::setup(a);
  avgtest::setup(b);
  small_bins(a);
  a.set_earth(e2);
  small_bins(b);
  a.calculate_binned();
  b.calculate_binned();
  CHECK(copy_binned(a) == copy_binned(b));
}

TEST_CASE("binned_batch equals set_params/set_earth + calculate_binned (CPU) and restores state")
{
  using Params = Fast::Params;
  opg::Propagator<Fast> prop;
  avgtest::setup(prop);
  for (int rule = 0; rule < 2; rule++) {
    CAPTURE(rule);
    small_bins(prop, rule ? opg::CosZRule::LayerAdapted : opg::CosZRule::Plain);
    std::vector<std::string> g = {"th23", "dm31", "zoa_1", "rho_6"};
    prop.set_gradient_params(g);
    Params p0;
    p0.mix = variants::nominal_mix<3>();  // as avgtest::setup
    prop.calculate_binned(opg::Flavor::Both, true);
    const auto ref0 = copy_binned(prop);

    const int                   np = 3;
    std::vector<Params>         pts(np, p0);
    std::vector<opg::PremModel> earths(np, prop.earth());
    for (int p = 0; p < np; p++) {
      pts[size_t(p)].mix.dm[2] *= 1 + 0.02 * (p - 1);
      pts[size_t(p)].mix.th[1][2] += 0.01 * p;
      earths[size_t(p)].ScaleLayerDensity(6, 1 + 0.03 * (p - 1));
      earths[size_t(p)].SetLayerZoA(1, 0.47 + 0.005 * p);
    }
    std::vector<double> out, dout;
    prop.binned_batch(pts, &earths, out, &dout);

    const size_t per = 2 * 9 * prop.n_cosine_bins() * prop.n_energy_bins(), nq = g.size();
    REQUIRE(out.size() == np * per);
    REQUIRE(dout.size() == np * per * nq);
    opg::Propagator<Fast> ref;
    small_bins(ref, rule ? opg::CosZRule::LayerAdapted : opg::CosZRule::Plain);
    ref.set_gradient_params(g);
    for (int p = 0; p < np; p++) {
      ref.set_params(pts[size_t(p)]);
      ref.set_earth(earths[size_t(p)]);
      ref.calculate_binned(opg::Flavor::Both, true);
      const double* b  = ref.binned();
      const double* gb = ref.binned_grad();
      CHECK(std::equal(b, b + per, out.begin() + long(p * per)));
      CHECK(std::equal(gb, gb + per * nq, dout.begin() + long(p * per * nq)));
    }

    // parameters and Earth restored, also when the batch throws
    prop.calculate_binned(opg::Flavor::Both, true);
    CHECK(copy_binned(prop) == ref0);
    prop.set_gradient_params(std::vector<std::string>{});
    CHECK_THROWS(prop.binned_batch(pts, &earths, out, &dout));
    prop.calculate_binned();
    CHECK(copy_binned(prop) == ref0);
  }
}
