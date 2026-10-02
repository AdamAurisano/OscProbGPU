// Phase 8: Gauss-Legendre bin averaging (CPU backend).

#include <cstdio>

#include "averaging.h"

TEST_CASE("Binned averages equal the weighted sum over GL nodes (CPU)")
{
  opg::Propagator<avgtest::Fast> prop;
  avgtest::setup(prop);
  for (auto m : {opg::EMeasure::Linear, opg::EMeasure::Log, opg::EMeasure::InvE})
    avgtest::check_reduction(prop, m);
}

TEST_CASE("GL bin averages converge to the brute-force OscProb averages")
{
  opg::Propagator<avgtest::Fast> prop;
  avgtest::setup(prop);
  avgtest::BruteForce bf;

  // High-order layer-adapted rule as the "exact" average.
  std::vector<double> ref;
  for (size_t k = 0; k < bf.n(); k++) {
    const double* e = &bf.edges.data[k * 4];
    prop.set_bins({e[0], e[1]}, {e[2], e[3]}, 24, 24);
    prop.calculate_binned();
    ref.insert(ref.end(), prop.binned(), prop.binned() + 18);
  }

  auto self_err = [&](int nE, int nC, opg::CosZRule rule) {
    double m = 0;
    for (size_t k = 0; k < bf.n(); k++) {
      const double* e = &bf.edges.data[k * 4];
      prop.set_bins({e[0], e[1]}, {e[2], e[3]}, nE, nC, opg::EMeasure::Linear, rule);
      prop.calculate_binned();
      for (int q = 0; q < 18; q++)
        m = std::max(m, std::fabs(prop.binned()[q] - ref[k * 18 + q]));
    }
    return m;
  };

  // Each direction separately (the other one at order 24).
  std::printf("  order   E-only error   cosZ-only plain   cosZ-only layer-adapted\n");
  double c_plain12 = 0, c_adapt8 = 0, c_adapt12 = 0, e8 = 0, e16 = 0;
  for (int n : {1, 2, 3, 4, 5, 6, 8, 12, 16}) {
    double ee = self_err(n, 24, opg::CosZRule::LayerAdapted);
    double cp = self_err(24, n, opg::CosZRule::Plain);
    double ca = self_err(24, n, opg::CosZRule::LayerAdapted);
    std::printf("  %5d   %12.3e   %15.3e   %23.3e\n", n, ee, cp, ca);
    if (n == 8) { c_adapt8 = ca; e8 = ee; }
    if (n == 12) { c_adapt12 = ca; c_plain12 = cp; }
    if (n == 16) e16 = ee;
  }
  size_t nk = prop.earth().GetGrazingCosines().size();
  MESSAGE("default PREM has " << nk << " layer-grazing cosines in [-1, 0]");
  // Plain GL in cosZ stalls at ~1e-3 because of the square-root kinks at
  // layer-grazing directions; the layer-adapted rule converges exponentially
  // (use >= 6 nodes per piece).
  CHECK(c_adapt8 < 1e-5);
  CHECK(c_adapt12 < 1e-8);
  CHECK(c_adapt12 < c_plain12 * 1e-3);
  CHECK(e16 < 1e-8);  // E converges exponentially once nodes resolve the phase
  CHECK(e16 < e8 * 1e-3);

  // The brute-force midpoint rule (400^2 points) has its own O(h^2) error,
  // so it is only a loose check of the absolute normalisation.
  double d = 0;
  for (size_t k = 0; k < bf.n(); k++)
    for (int q = 0; q < 18; q++)
      d = std::max(d, std::fabs(ref[k * 18 + q] - bf.avg[k * 18 + q]));
  MESSAGE("layer-adapted GL-24 vs brute force (400^2 midpoint): " << d);
  CHECK(d < 2e-4);
}

TEST_CASE("1D path averages agree with OscProb AvgProb (CPU)")
{
  opg::Propagator<avgtest::Fast> prop;
  avgtest::setup(prop);
  auto edges = npy::load<double>("avg1d_fast_edges.npy");
  auto ref   = npy::load<double>("avg1d_fast.npy");
  const size_t nb = edges.size() - 1;
  double       m  = 0;
  for (int nub = 0; nub < 2; nub++) {
    auto out = prop.avg_path(edges.data, 16, refcmp::test_path(), nub);
    for (size_t i = 0; i < nb; i++)
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
          m = std::max(m, std::fabs(out[(a * 3 + b) * nb + i] -
                                    ref[((nub * nb + i) * 3 + a) * 3 + b]));
  }
  MESSAGE("max |GL-16 avg - OscProb AvgProb| = " << m);
  // OscProb's Maltoni averaging targets a precision of 1e-4 (its own test
  // accepts 5x that against a 1000-point average).
  CHECK(m < 5e-4);
}
