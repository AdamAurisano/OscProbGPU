// Production-height averaging (avg/prod_height.h) on the CPU backend.

#include <cmath>
#include <complex>

#include "doctest.h"

#include "binned.h"
#include "prod_height.h"

TEST_CASE("Production heights: Legendre-Fourier integrals of P_k")
{
  // int_{-1}^{1} P_k(t) exp(-i theta t) dt by 400-point Gauss-Legendre
  opg::GaussLegendre gl(400);
  double             worst = 0;
  for (double th : {0.0, 1e-9, 1e-3, 0.3, 0.999, 1.0, 2.5, 7.0, 11.0, 19.5, 20.0, 45.0, 200.0,
                    -3.0, -60.0}) {
    double jb[opg::prodh::kMaxOrder + 1];
    opg::prodh::sph_bessel(th, opg::prodh::kMaxOrder, jb);
    for (int k = 0; k <= opg::prodh::kMaxOrder; k++) {
      double c[opg::prodh::kMaxOrder + 1] = {};
      c[k]                                = 1;
      auto                 f              = opg::prodh::legendre_fourier(c, k, jb);
      std::complex<double> ref(0, 0);
      for (size_t q = 0; q < gl.x.size(); q++) {
        double t = gl.x[q], p0 = 1, p1 = t, pk = k == 0 ? 1 : t;
        for (int m = 2; m <= k; m++) {
          pk = ((2 * m - 1) * t * p1 - (m - 1) * p0) / m;
          p0 = p1;
          p1 = pk;
        }
        ref += gl.w[q] * pk * std::exp(std::complex<double>(0, -th * t));
      }
      worst = std::max(worst, std::abs(std::complex<double>(f.re, f.im) - ref));
    }
  }
  MESSAGE("max |F_k(theta) - GL| = " << worst);
  CHECK(worst < 1e-14);
}

TEST_CASE("Production heights: atmosphere geometry matches the Earth model's paths")
{
  opg::PremModel       prem;
  opg::ProdHeightTable t;
  t.set_earth(opg::EarthModel(prem));
  CHECK(t.r_surf == 6371);
  CHECK(t.r_det == 6368);
  CHECK(t.atm_type == 6);
  double worst = 0;
  for (double c : {-1.0, -0.7, -0.2, -0.01, 0.0, 0.05, 0.6, 1.0}) {
    // PREM's own top: 15 km
    worst = std::max(worst, std::fabs(t.path_length(c, 15) - prem.FillPath(c)[0].length));
    opg::PremModel p2 = prem;
    p2.SetTopLayerSize(42.5);
    worst = std::max(worst, std::fabs(t.path_length(c, 42.5) - p2.FillPath(c)[0].length));
  }
  MESSAGE("max |L(h) - PremModel::FillPath| = " << worst << " km");
  CHECK(worst < 1e-10);
  CHECK(t.path_length(1.0, 20) == doctest::Approx(20).epsilon(1e-14));

  // binned Earth: the top depth shell
  opg::ProdHeightTable tb;
  tb.set_earth(opg::EarthModel(opg::EarthModelBinned()));
  CHECK(tb.r_surf == 6371);
  CHECK(tb.atm_type == 6);

  // the detector must be below the atmosphere
  opg::PremModel high = prem;
  high.SetDetPos(6380);
  CHECK_THROWS_AS(t.set_earth(opg::EarthModel(high)), std::invalid_argument);
  // bad distributions
  CHECK_THROWS_AS(t.add({0, 10}, {1, 2}), std::invalid_argument);
  CHECK_THROWS_AS(t.add({10, 5}, {1}), std::invalid_argument);
  CHECK_THROWS_AS(t.add({-1, 5}, {1}), std::invalid_argument);
  CHECK_THROWS_AS(t.add({0, 5}, {0}), std::invalid_argument);
}

TEST_CASE_TEMPLATE("Production heights: exact vs brute force (PREM and binned Earth)", Model,
                   opg::Fast<double>, opg::NSI<double>, opg::NUNM<double>,
                   opg::Sterile<double>, opg::Sterile5<double>, opg::Sterile6<double>,
                   opg::LIV<double>, opg::SNSI<double>)
{
  opg::Propagator<Model> prop;
  phtest::check_values(prop, opg::EarthModel(opg::PremModel()), 1e-11, "PREM");
  phtest::check_values(prop, opg::EarthModel(bintest::config(0)), 1e-11, "binned");
}

TEST_CASE("Production heights: refused for models without a segment eigensystem")
{
  opg::Propagator<opg::Decay<double>> d;
  CHECK_THROWS_AS(d.set_prod_heights(phtest::edges(), phtest::probs()), std::logic_error);
  opg::Propagator<opg::SiderealLIV<double>> s;
  CHECK_THROWS_AS(s.set_prod_heights(phtest::edges(), phtest::probs()), std::logic_error);
  opg::Propagator<opg::OQS<double>> o;
  CHECK_THROWS_AS(o.set_prod_heights(phtest::edges(), phtest::probs()), std::logic_error);
  opg::Propagator<opg::Fast<double>> f;
  CHECK_THROWS_AS(f.prob_points_prod_height({1}, {0}, {0}, {0}), std::logic_error);
  f.set_params(phtest::params<opg::Fast<double>>());
  f.set_prod_heights(phtest::edges(), phtest::probs());
  CHECK_THROWS_AS(f.prob_points_prod_height({1}, {0}, {0}, {5}), std::out_of_range);
  f.set_grid({1, 2}, {-0.5, 0.5});
  CHECK_THROWS_AS(f.calculate_prod_height(), std::logic_error);
}

TEST_CASE_TEMPLATE("Production heights: gradients", Model, opg::Fast<double>,
                   opg::NSI<double>, opg::NUNM<double>, opg::Sterile<double>, opg::LIV<double>,
                   opg::SNSI<double>)
{
  opg::Propagator<Model> prop;
  phtest::check_grads(prop, 1e-8, 1e-10);
}

TEST_CASE_TEMPLATE("Production heights: common height scale", Model, opg::Fast<double>,
                   opg::NSI<double>, opg::NUNM<double>, opg::Sterile<double>,
                   opg::Sterile5<double>, opg::Sterile6<double>, opg::LIV<double>,
                   opg::SNSI<double>)
{
  opg::Propagator<Model> prop;
  phtest::check_scale(prop, opg::EarthModel(opg::PremModel()), "PREM");
  phtest::check_scale(prop, opg::EarthModel(bintest::config(0)), "binned");
  CHECK_THROWS_AS(prop.set_prod_height_scale(0), std::invalid_argument);
  CHECK_THROWS_AS(prop.set_prod_height_scale(-1), std::invalid_argument);
}
