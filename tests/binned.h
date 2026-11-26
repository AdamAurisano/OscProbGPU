// EarthModelBinned checks shared by the CPU and GPU test suites: OscProb
// reference paths and probabilities (reference/dump_reference.cxx, tag
// "binned"), per-event azimuths, region gradients against long-double finite
// differences.

#ifndef OPG_TESTS_BINNED_H
#define OPG_TESTS_BINNED_H

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "doctest.h"

#include "gradients.h"
#include "npy.h"
#include "ref_compare.h"
#include "variants.h"

#include "opg/models/all.h"
#include "opg/propagator.h"

namespace bintest {

  /// The test table with lateral structure (binned_table.npy), written as
  /// text with %.17g as the reference dump did, and loaded.
  inline opg::EarthModelBinned test_table()
  {
    static std::string fname;
    if (fname.empty()) {
      auto t = npy::load<double>("binned_table.npy");
      fname  = (std::filesystem::temp_directory_path() /
               ("opg_binned_table_" + std::to_string(std::random_device{}()) + ".txt"))
                  .string();
      FILE* f = std::fopen(fname.c_str(), "w");
      REQUIRE(f);
      for (size_t i = 0; i + 5 < t.size(); i += 6)
        std::fprintf(f, "%.17g %.17g %.17g %.17g %.17g %.17g\n", t[i], t[i + 1], t[i + 2],
                     t[i + 3], t[i + 4], t[i + 5]);
      std::fclose(f);
    }
    opg::EarthModelBinned m(fname);
    return m;
  }

  /// The reference configurations: 0: default table and detector, azimuth
  /// 30; 1: test table, detector (6368 km, 42.4, 13.5), azimuth 250, region
  /// 7 Z/A 0.47, region 2 density x 1.03.
  inline opg::EarthModelBinned config(int k)
  {
    if (k == 0) {
      opg::EarthModelBinned m;
      m.SetDetPos(6368, 0, 0);
      m.SetAzimuth(30);
      return m;
    }
    opg::EarthModelBinned m = test_table();
    m.SetDetPos(6368, 42.4, 13.5);
    m.SetRegionZoA(7, 0.47);
    m.ScaleRegionDensity(2, 1.03);
    m.SetAzimuth(250);
    return m;
  }

  /// Probabilities on the reference grid through config k against
  /// <tag>_binned.npy [2 cfg][2 nubar][nC][nE][N][N]: grid mode, and event
  /// list mode with the azimuth per event (explicit, or through the
  /// model's azimuth extra if use_extra).
  template <class Model>
  void compare_probs(opg::Propagator<Model>& prop, const std::string& tag, int k,
                     refcmp::Result& rg, refcmp::Result& re, bool use_extra = false)
  {
    constexpr int N   = Model::N;
    auto          E   = npy::load<double>("grid_E_binned.npy");
    auto          C   = npy::load<double>("grid_cosZ_binned.npy");
    auto          ref = npy::load<double>(tag + "_binned.npy");
    const size_t  nE = E.size(), nC = C.size();
    const auto    m   = config(k);
    prop.set_earth(m);
    prop.set_grid(E.data, C.data);
    prop.calculate(opg::Flavor::Both);
    const double* P = prop.probs();
    auto R = [&](int nb, size_t ic, size_t ie, int a, int b) {
      return ref[(((((size_t(k) * 2 + nb) * nC + ic) * nE + ie) * N + a) * N + b)];
    };
    for (int nb = 0; nb < 2; nb++)
      for (size_t ic = 0; ic < nC; ic++)
        for (size_t ie = 0; ie < nE; ie++)
          for (int a = 0; a < N; a++)
            for (int b = 0; b < N; b++)
              refcmp::accumulate(rg, P[(((size_t(nb) * N + a) * N + b) * nC + ic) * nE + ie],
                                 R(nb, ic, ie, a, b));

    // event list: every 3rd grid point; the Earth's own azimuth is moved
    // away so that only the per-event one can give the reference
    std::vector<double>  e, c, az, extra;
    std::vector<uint8_t> nbv;
    std::vector<size_t>  idx;
    for (size_t q = 0; q < 2 * nC * nE; q += 3) {
      e.push_back(E[q % nE]);
      c.push_back(C[(q / nE) % nC]);
      nbv.push_back(uint8_t(q / (nC * nE)));
      az.push_back(m.GetAzimuth());
      idx.push_back(q);
    }
    prop.set_earth_azimuth(m.GetAzimuth() + 77);
    std::vector<double> out;
    if constexpr (opg::extra_azimuth_v<Model> == 0) {
      if (use_extra) {  // azimuth extra, then the sidereal time
        constexpr int NX = opg::Propagator<Model>::n_event_extra();
        extra            = az;
        for (int x = 1; x < NX; x++)
          for (size_t i = 0; i < e.size(); i++) extra.push_back(variants::sidereal().time);
      }
    }
    else
      REQUIRE(!use_extra);
    out = use_extra ? prop.prob_points(e, c, nbv, extra) : prop.prob_points(e, c, nbv, {}, az);
    for (size_t i = 0; i < idx.size(); i++) {
      const size_t q = idx[i];
      for (int a = 0; a < N; a++)
        for (int b = 0; b < N; b++)
          refcmp::accumulate(re, out[(a * N + b) * idx.size() + i],
                             R(int(q / (nC * nE)), (q / nE) % nC, q % nE, a, b));
    }
  }

  /// Events with random directions and azimuths.
  struct AzEvents {
      std::vector<double>  E, C, az;
      std::vector<uint8_t> nb;
      std::vector<int>     nbi;
      explicit AzEvents(size_t n = 60, unsigned seed = 11)
      {
        std::mt19937_64                        rng(seed);
        std::uniform_real_distribution<double> u(0, 1);
        const double                           azs[5] = {0, 45, 137.5, 250, 333.3};
        for (size_t i = 0; i < n; i++) {
          E.push_back(std::pow(10.0, -0.3 + 2.0 * u(rng)));
          C.push_back(-1 + 1.9 * u(rng));
          az.push_back(i % 7 == 6 ? 360 * u(rng) : azs[i % 5]);
          nb.push_back(uint8_t(i % 2));
          nbi.push_back(int(i % 2));
        }
      }
  };

  /// Event lists with per-event azimuths give, bit for bit, what a propagator
  /// with that azimuth set gives (values, gradients, weighted gradients).
  template <class Model> void check_event_azimuth(opg::Propagator<Model>& prop)
  {
    constexpr int NN = Model::N * Model::N;
    prop.set_earth(config(1));
    prop.set_params(gradtest::param_points<Model>()[0].second);
    prop.set_gradient_params({"dm31", "th23", "zoa_7", "rho_3"});
    AzEvents            ev;
    const size_t        n = ev.E.size();
    std::vector<double> P, dP, w(NN * n);
    for (size_t i = 0; i < w.size(); i++) w[i] = std::sin(1.0 + 0.37 * double(i));
    prop.prob_points_grad(ev.E, ev.C, ev.nb, P, dP, {}, ev.az);
    auto Pv = prop.prob_points(ev.E, ev.C, ev.nb, {}, ev.az);
    CHECK(Pv == P);
    size_t bad = 0, badg = 0;
    for (size_t i = 0; i < n; i++) {
      prop.set_earth_azimuth(ev.az[i]);
      std::vector<double> e1 = {ev.E[i]}, c1 = {ev.C[i]}, P1, dP1;
      std::vector<uint8_t> b1 = {ev.nb[i]};
      prop.prob_points_grad(e1, c1, b1, P1, dP1);
      for (int ab = 0; ab < NN; ab++) bad += P1[size_t(ab)] != P[size_t(ab) * n + i];
      for (size_t q = 0; q < 4; q++)
        for (int ab = 0; ab < NN; ab++)
          badg += dP1[q * NN + size_t(ab)] != dP[(q * NN + size_t(ab)) * n + i];
    }
    MESSAGE(std::string(Model::name) << " per-event vs per-propagator azimuth: " << bad
                                     << " values and " << badg << " derivatives differ");
    CHECK(bad == 0);
    CHECK(badg == 0);
    // weighted (forward and reverse mode) against the contraction of dP
    std::vector<double> ref(4, 0.0);
    for (size_t q = 0; q < 4; q++)
      for (size_t k = 0; k < NN * n; k++) ref[q] += w[k] * dP[q * NN * n + k];
    for (bool adj : {false, true}) {
      prop.set_adjoint_gradients(adj);
      auto g = prop.weighted_gradient_points(ev.E, ev.C, ev.nb, w, {}, ev.az);
      std::vector<int> bin(n, 0);
      auto G = prop.weighted_gradient_points_binned(ev.E, ev.C, ev.nb, w, bin, 1, {}, ev.az);
      for (size_t q = 0; q < 4; q++) {
        CHECK(g[q] == doctest::Approx(ref[q]).epsilon(1e-10));
        CHECK(G[q] == doctest::Approx(ref[q]).epsilon(1e-10));
      }
    }
    prop.set_adjoint_gradients(true);
  }

  /// Region Z/A and density-scale gradients (and dm31) through the test
  /// table, event list with per-event azimuths, against long-double finite
  /// differences of the model edited with SetRegionZoA / ScaleRegionDensity.
  template <class Model> void check_region_grads(opg::Propagator<Model>& prop, double tol)
  {
    using LD           = long double;
    using FL           = typename gradtest::LongDouble<Model>::type;
    constexpr int NN   = Model::N * Model::N;
    const auto    base = config(1);
    prop.set_earth(base);
    auto par = gradtest::param_points<Model>()[0].second;
    prop.set_params(par);
    auto earth = prop.earth_parameter_names();
    REQUIRE(earth.size() == 2 * base.GetRegions().size());
    std::vector<std::string> names = {"dm31"};
    for (auto& z : earth) names.push_back(z);
    prop.set_gradient_params(names);
    AzEvents            ev(40, 5);
    std::vector<double> P, dP;
    prop.prob_points_grad(ev.E, ev.C, ev.nb, P, dP, {}, ev.az);

    const auto PL     = FL::template prepare_generic<LD>(Model::template cast<LD>(par));
    auto       probsL = [&](const opg::EarthModelBinned& m, const typename FL::Prepared& PP) {
      const opg::EarthTable<LD> t{opg::EarthModel(m)};
      opg::BinnedGeom           g;
      const auto                v = t.view(g);
      std::vector<LD>           out;
      for (size_t i = 0; i < ev.E.size(); i++) {
        auto vi    = v;
        vi.azimuth = ev.az[i];
        auto S = opg::evolve_prem<FL, LD>(PP, vi, LD(ev.E[i]), LD(ev.C[i]), ev.nb[i] != 0);
        LD pr[NN];
        opg::store_model_probs<FL, LD>(S, pr, 1);
        for (int ab = 0; ab < NN; ab++) out.push_back(pr[ab]);
      }
      return out;
    };
    auto fd = [&](auto&& at, LD h) {
      auto p1 = at(h), m1 = at(-h), p2 = at(2 * h), m2 = at(-2 * h), p3 = at(3 * h),
           m3 = at(-3 * h);
      std::vector<LD> g(p1.size());
      for (size_t i = 0; i < g.size(); i++)
        g[i] = (45 * (p1[i] - m1[i]) - 9 * (p2[i] - m2[i]) + (p3[i] - m3[i])) / (60 * h);
      return g;
    };
    std::vector<std::vector<LD>> ref;
    {
      const auto all = Model::param_names();
      const int  idx = int(std::find(all.begin(), all.end(), "dm31") - all.begin());
      auto       q0  = Model::template cast<LD>(par);
      const LD   x0  = FL::param_ref(q0, idx);
      ref.push_back(fd(
          [&](LD dx) {
            auto q                 = q0;
            FL::param_ref(q, idx)  = x0 + dx;
            return probsL(base, FL::template prepare_generic<LD>(q));
          },
          gradtest::step<Model>("dm31", x0)));
    }
    for (auto& z : earth) {
      const int    t   = std::stoi(z.substr(4));
      const bool   rho = z.rfind("rho_", 0) == 0;
      const double z0  = base.GetRegionZoA(t);
      ref.push_back(fd(
          [&](LD dx) {
            opg::EarthModelBinned m = base;
            if (rho)
              m.ScaleRegionDensity(t, double(1 + dx));
            else
              m.SetRegionZoA(t, double(z0 + dx));
            return probsL(m, PL);
          },
          1e-4L));
    }
    size_t       w = 0;
    const double e = gradtest::rel_err<Model>(dP, ref, ev.E.size(), NN, names, &w);
    MESSAGE(std::string(Model::name) << " binned Earth region gradients vs long-double FD: " << e
                                     << " (" << names[w] << ")");
    CHECK(e < tol);
  }

} // namespace bintest

#endif
