// Production-height averaging checks shared by the CPU and GPU test suites:
// brute-force averages over heights (Gauss-Legendre in h, fixed-height paths),
// fixed heights, gradients against long-double finite differences.

#ifndef OPG_TESTS_PROD_HEIGHT_H
#define OPG_TESTS_PROD_HEIGHT_H

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "doctest.h"

#include "gradients.h"
#include "variants.h"

#include "opg/avg/gauss_legendre.h"
#include "opg/avg/prod_height.h"
#include "opg/models/all.h"
#include "opg/propagator.h"

namespace phtest {

  /// Test distributions (km above the atmosphere's base): 0 a Honda-like
  /// histogram up to 70 km, 1 one wide bin [0, 80], 2 a fixed height of
  /// 15 km (PREM's own atmosphere), 3 a mixture of a fixed height and bins,
  /// 4 narrow bins near the ground.
  inline std::vector<std::vector<double>> edges()
  {
    return {{0, 5, 10, 15, 20, 25, 30, 40, 50, 70},
            {0, 80},
            {15, 15},
            {10, 10, 20, 30},
            {0, 0.5, 1, 2}};
  }
  inline std::vector<std::vector<double>> probs()
  {
    return {{0.02, 0.08, 0.18, 0.22, 0.18, 0.12, 0.12, 0.06, 0.02},
            {1},
            {1},
            {0.3, 0.5, 0.2},
            {1, 2, 1}};
  }

  /// Events: directions from straight up-going through the horizon to down-
  /// going, energies 0.3 - 60 GeV, nu and nubar, every distribution.
  struct Events {
      std::vector<double>  E, C;
      std::vector<uint8_t> nb;
      std::vector<int>     dist;
      explicit Events(int ndist = 5, int nE = 6)
      {
        const std::vector<double> cs = {-1.0, -0.83, -0.4, -0.08, -0.01, 0.0, 0.02, 0.15, 0.5, 1.0};
        int                       d  = 0;
        for (int b = 0; b < 2; b++)
          for (double c : cs)
            for (int i = 0; i < nE; i++) {
              E.push_back(0.3 * std::pow(200.0, double(i) / (nE - 1)));
              C.push_back(c);
              nb.push_back(uint8_t(b));
              dist.push_back(d++ % ndist);
            }
      }
  };

  /// The rest of the path after the atmosphere, for direction cosZ.
  inline std::vector<opg::Segment<double>> rest_path(const opg::EarthModel& earth,
                                                     const opg::ProdHeightTable& t, double c)
  {
    auto                          full = earth.FillPath(c);
    std::vector<opg::Segment<double>> out;
    bool                          lead = true;
    for (auto& s : full) {
      if (lead && s.layer == t.atm_type) continue;
      lead = false;
      out.push_back(s);
    }
    return out;
  }

  /// Brute-force average: each bin split into nsub pieces with nq
  /// Gauss-Legendre nodes in h, a fixed-height path (atmosphere segment of
  /// length L(h), then the rest) at each node. out[a][b][i]; events with
  /// dist < 0 are left at 0.
  template <class Model>
  std::vector<double> brute(const typename Model::Params& par, const opg::EarthModel& earth,
                            const Events& ev, int nsub = 2, int nq = 32, double scale = 1)
  {
    constexpr int     N = Model::N;
    opg::ProdHeightTable t;
    t.set_earth(earth);
    auto ed = edges(), pr = probs();
    for (size_t d = 0; d < ed.size(); d++) t.add(ed[d], pr[d]);
    const auto        P = Model::prepare(par);
    opg::GaussLegendre gl(nq);
    const size_t      n = ev.E.size();
    std::vector<double> out(size_t(N) * N * n, 0.0);
    for (size_t i = 0; i < n; i++) {
      const int d = ev.dist[i];
      if (d < 0) continue;
      const auto rest = rest_path(earth, t, ev.C[i]);
      std::vector<opg::Segment<double>> path(rest.size() + 1);
      std::copy(rest.begin(), rest.end(), path.begin() + 1);
      double acc[N * N] = {};
      auto   add        = [&](double h, double wt) {
        path[0] = opg::Segment<double>{t.path_length(ev.C[i], h), t.atm_density, t.atm_zoa,
                                       t.atm_type};
        auto   S = opg::evolve_path<Model, double>(P, path.data(), int(path.size()), ev.E[i],
                                                   ev.nb[i] != 0);
        double p[N * N];
        opg::store_model_probs<Model, double>(S, p, 1);
        for (int ab = 0; ab < N * N; ab++) acc[ab] += wt * p[ab];
      };
      for (int b = t.off[size_t(d)]; b < t.off[size_t(d) + 1]; b++) {
        const double lo = scale * t.lo[size_t(b)], hi = scale * t.hi[size_t(b)];
        const double p  = t.prob[size_t(b)];
        if (hi == lo) {
          add(lo, p);
          continue;
        }
        // pieces of at most ~15 rad of the largest phase difference (splittings
        // up to 2 eV^2 with steriles, else 0.01 eV^2 including matter terms)
        const double span  = (N > 3 ? 2.0 : 0.01) / (2e9 * ev.E[i]);  // eV
        const double phase = span * opg::constants::kKm2eV *
                             (t.path_length(ev.C[i], hi) - t.path_length(ev.C[i], lo));
        const int ns = std::max(nsub, int(std::ceil(phase / 15)));
        for (int s = 0; s < ns; s++) {
          const double a = lo + (hi - lo) * s / ns, c = lo + (hi - lo) * (s + 1) / ns;
          for (int q = 0; q < nq; q++)
            add(0.5 * (a + c) + 0.5 * (c - a) * gl.x[size_t(q)], p * gl.w[size_t(q)] / (2 * ns));
        }
      }
      for (int ab = 0; ab < N * N; ab++) out[size_t(ab) * n + i] = acc[ab];
    }
    return out;
  }

  inline double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b)
  {
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
  }

  /// Model parameters for the checks (one point per model).
  template <class Model> typename Model::Params params()
  {
    return gradtest::param_points<Model>().front().second;
  }

  /// Long-double finite-difference reference of the averaged probabilities
  /// (PREM): ref[p][i][a][b] (layout of gradtest::rel_err).
  template <class Model>
  std::vector<std::vector<long double>> ld_grads(const typename Model::Params& p0,
                                                 const std::vector<std::string>& names,
                                                 const opg::PremModel& prem, const Events& ev)
  {
    using LD = long double;
    using FL = typename gradtest::LongDouble<Model>::type;
    constexpr int N = Model::N;
    opg::ProdHeightTable t;
    t.set_earth(opg::EarthModel(prem));
    auto ed = edges(), pr = probs();
    for (size_t d = 0; d < ed.size(); d++) t.add(ed[d], pr[d]);
    const opg::PremModel::HostTable<LD> table(prem);
    const auto                         earth = table.view();
    const auto                         hv    = t.view<LD>();
    auto probs_at = [&](const typename FL::Prepared& P) {
      std::vector<LD> out;
      for (size_t i = 0; i < ev.E.size(); i++) {
        LD pr_[N * N];
        opg::prod_height_probs<FL, LD>(P, earth, hv, ev.dist[i], LD(ev.E[i]), LD(ev.C[i]),
                                       ev.nb[i] != 0, pr_, 1);
        for (int ab = 0; ab < N * N; ab++) out.push_back(pr_[ab]);
      }
      return out;
    };
    auto                         all = Model::param_names();
    std::vector<std::vector<LD>> out;
    for (auto& nm : names) {
      const int idx  = int(std::find(all.begin(), all.end(), nm) - all.begin());
      auto      base = Model::template cast<LD>(p0);
      const LD  x0   = FL::param_ref(base, idx);
      const LD  h    = gradtest::step<Model>(nm, x0);
      auto      at   = [&](LD dx) {
        auto q                = base;
        FL::param_ref(q, idx) = x0 + dx;
        return probs_at(FL::template prepare_generic<LD>(q));
      };
      auto p1 = at(h), m1 = at(-h), p2 = at(2 * h), m2 = at(-2 * h), p3 = at(3 * h),
           m3 = at(-3 * h);
      std::vector<LD> g(p1.size());
      for (size_t i = 0; i < g.size(); i++)
        g[i] = (45 * (p1[i] - m1[i]) - 9 * (p2[i] - m2[i]) + (p3[i] - m3[i])) / (60 * h);
      out.push_back(g);
    }
    return out;
  }

  /// Exact averages against the brute-force reference, fixed heights
  /// against the plain event list, dist = -1 bit-identical, grid against
  /// events, for one propagator and Earth model.
  template <class Model>
  void check_values(opg::Propagator<Model>& prop, const opg::EarthModel& earth,
                    double tol_brute, const std::string& label, double exact_tol = 0)
  {
    constexpr int N = Model::N;
    prop.set_earth(earth);
    prop.set_params(params<Model>());
    prop.set_prod_heights(edges(), probs());
    Events ev(5, N > 4 ? 2 : 6);  // (3+2, 3+3: brute force is slow)
    auto   P   = prop.prob_points_prod_height(ev.E, ev.C, ev.nb, ev.dist);
    auto   ref = brute<Model>(params<Model>(), earth, ev, 2);
    double unit = 0;
    for (size_t i = 0; i < ev.E.size(); i++)
      for (int a = 0; a < N; a++) {
        double s = 0;
        for (int b = 0; b < N; b++) s += P[size_t(a * N + b) * ev.E.size() + i];
        if (Model::name != std::string("NUNM")) unit = std::max(unit, std::fabs(s - 1));
      }
    const double eb = max_abs_diff(P, ref);

    // fixed height at the Earth model's own top: the plain event list
    opg::ProdHeightTable t;
    t.set_earth(earth);
    const double htop = earth.GetTotalL(1.0) + t.r_det - t.r_surf;  // top radius - base
    prop.set_prod_heights({{htop, htop}}, {{1}});
    std::vector<int> d0(ev.E.size(), 0), dm(ev.E.size(), -1);
    auto             Pf = prop.prob_points_prod_height(ev.E, ev.C, ev.nb, d0);
    auto             Pp = prop.prob_points(ev.E, ev.C, ev.nb);
    auto             Pm = prop.prob_points_prod_height(ev.E, ev.C, ev.nb, dm);
    const double     ef = max_abs_diff(Pf, Pp);
    MESSAGE(label << " " << std::string(Model::name) << " N=" << N << ": max |exact - brute| = " << eb
                  << ", fixed height " << htop << " km vs plain = " << ef
                  << ", max |sum_b P - 1| = " << unit);
    CHECK(eb < tol_brute);
    CHECK(ef < 5e-12);  // (round-off in L times phases ~1e4 rad with steriles)
    CHECK(unit < 1e-13);
    CHECK(max_abs_diff(Pm, Pp) <= exact_tol);  // no averaging: bit-identical on the CPU

    // grid: per cosine and per point, against the event list
    prop.set_prod_heights(edges(), probs());
    std::vector<double> Eg = {0.4, 2.0, 9.0, 40.0}, Cg = {-0.9, -0.2, 0.0, 0.3, 1.0};
    prop.set_grid(Eg, Cg);
    std::vector<int> dc = {0, 1, 3, -1, 4}, dp;
    for (size_t ic = 0; ic < Cg.size(); ic++)
      for (size_t ie = 0; ie < Eg.size(); ie++) dp.push_back(int((ic + ie) % 6) - 1);
    for (int mode = 0; mode < 2; mode++) {
      prop.set_grid_prod_heights(mode == 0 ? dc : dp);
      prop.calculate_prod_height();
      const double* G = prop.probs();
      double        e = 0;
      for (int nb = 0; nb < 2; nb++) {
        Events g(0, 0);
        for (size_t ic = 0; ic < Cg.size(); ic++)
          for (size_t ie = 0; ie < Eg.size(); ie++) {
            g.E.push_back(Eg[ie]);
            g.C.push_back(Cg[ic]);
            g.nb.push_back(uint8_t(nb));
            g.dist.push_back(mode == 0 ? dc[ic] : dp[ic * Eg.size() + ie]);
          }
        auto Pe = prop.prob_points_prod_height(g.E, g.C, g.nb, g.dist);
        for (int ab = 0; ab < N * N; ab++)
          for (size_t k = 0; k < g.E.size(); k++)
            e = std::max(e, std::fabs(Pe[size_t(ab) * g.E.size() + k] -
                                      G[(size_t(nb) * N * N + ab) * g.E.size() + k]));
      }
      CHECK(e <= std::max(1e-15, exact_tol));
    }
  }

  /// Gradients: model parameters against long-double finite differences of
  /// the averages (PREM), Earth parameters of the atmosphere and of deeper
  /// layers against double finite differences, probabilities bit-identical
  /// with gradients on, reverse mode against forward mode (events, grid),
  /// grid gradients against the event list.
  template <class Model>
  void check_grads(opg::Propagator<Model>& prop, double tol_fd, double tol_adj,
                   double exact_tol = 0)
  {
    constexpr int N = Model::N, NN = N * N;
    opg::PremModel prem;
    prop.set_earth(opg::EarthModel(prem));
    const auto par = params<Model>();
    prop.set_params(par);
    prop.set_prod_heights(edges(), probs());
    auto names = Model::param_names();
    prop.set_gradient_params(names);
    Events              ev(5, 4);
    const size_t        n = ev.E.size();
    std::vector<double> P, dP;
    prop.prob_points_grad_prod_height(ev.E, ev.C, ev.nb, ev.dist, P, dP);
    CHECK(max_abs_diff(P, prop.prob_points_prod_height(ev.E, ev.C, ev.nb, ev.dist)) <=
          exact_tol);
    auto         ref = ld_grads<Model>(par, names, prem, ev);
    size_t       iw  = 0;
    const double e1  = gradtest::rel_err<Model>(dP, ref, n, NN, names, &iw);

    // Earth parameters: Z/A and density of the atmosphere (type 6) and of
    // the outer core (2) and upper mantle (4), double central differences
    const std::vector<std::string> en = {"zoa_6", "rho_6", "zoa_2", "rho_4"};
    prop.set_gradient_params(en);
    prop.prob_points_grad_prod_height(ev.E, ev.C, ev.nb, ev.dist, P, dP);
    double e2 = 0;
    for (size_t p = 0; p < en.size(); p++) {
      const bool zoa = en[p][0] == 'z';
      const int  t   = std::stoi(en[p].substr(4));
      const double h = zoa ? 1e-5 : 1e-5;
      auto at = [&](double dx) {
        opg::PremModel m = prem;
        if (zoa)
          m.SetLayerZoA(t, prem.GetLayerZoA(t) + dx);
        else
          m.ScaleLayerDensity(t, std::exp(dx));
        opg::Propagator<Model> q{opg::EarthModel(m)};
        q.set_params(par);
        q.set_prod_heights(edges(), probs());
        return q.prob_points_prod_height(ev.E, ev.C, ev.nb, ev.dist);
      };
      const auto pp = at(h), pm = at(-h);
      double     m = 0, r = 0;
      for (size_t k = 0; k < pp.size(); k++) {
        const double fd = (pp[k] - pm[k]) / (2 * h);
        m               = std::max(m, std::fabs(dP[p * NN * n + k] - fd));
        r               = std::max(r, std::fabs(fd));
      }
      e2 = std::max(e2, m / std::max(r, 1e-3));
    }
    MESSAGE(std::string(Model::name) << " N=" << N << " production heights: gradients vs "
            "long-double FD " << e1 << " (" << names[iw] << "), Earth parameters vs FD " << e2);
    CHECK(e1 < tol_fd);
    CHECK(e2 < 1e-6);

    // grid gradients against the event list (identical per-point code)
    auto all = prop.gradient_parameter_names();
    prop.set_gradient_params(all);
    std::vector<double> Eg = {0.4, 3.0, 25.0}, Cg = {-0.95, -0.1, 0.0, 0.4};
    prop.set_grid(Eg, Cg);
    std::vector<int> dg;
    for (size_t ic = 0; ic < Cg.size(); ic++)
      for (size_t ie = 0; ie < Eg.size(); ie++) dg.push_back(int((ic * 3 + ie) % 6) - 1);
    prop.set_grid_prod_heights(dg);
    prop.calculate_prod_height(opg::Flavor::Both, true);
    const double* Gg   = prop.grad();
    const size_t  npar = all.size(), ng = Eg.size() * Cg.size();
    double        e3   = 0;
    for (int nb = 0; nb < 2; nb++) {
      Events g(0, 0);
      for (size_t ic = 0; ic < Cg.size(); ic++)
        for (size_t ie = 0; ie < Eg.size(); ie++) {
          g.E.push_back(Eg[ie]);
          g.C.push_back(Cg[ic]);
          g.nb.push_back(uint8_t(nb));
          g.dist.push_back(dg[ic * Eg.size() + ie]);
        }
      prop.prob_points_grad_prod_height(g.E, g.C, g.nb, g.dist, P, dP);
      for (size_t p = 0; p < npar; p++)
        for (int ab = 0; ab < NN; ab++)
          for (size_t k = 0; k < ng; k++)
            e3 = std::max(e3, std::fabs(dP[(p * NN + ab) * ng + k] -
                                        Gg[((nb * npar + p) * NN + ab) * ng + k]));
    }
    CHECK(e3 <= exact_tol * 1e3);  // (derivatives up to ~1e3)

    // reverse vs forward mode, weighted (events and grid)
    if constexpr (opg::Propagator<Model>::has_adjoint_gradients()) {
      std::mt19937_64                        rng(3);
      std::uniform_real_distribution<double> u(-1, 1);
      std::vector<double>                    w(NN * n), wg(2 * NN * ng);
      for (auto& x : w) x = u(rng);
      for (auto& x : wg) x = u(rng);
      std::string worst_name;
      auto cmp = [&](const std::vector<double>& a, const std::vector<double>& b) {
        double sc = 0, worst = 0;
        for (double x : a) sc = std::max(sc, std::fabs(x));
        for (size_t i = 0; i < a.size(); i++) {
          const double r =
              std::fabs(a[i] - b[i]) / std::max(std::fabs(a[i]), 1e-6 * sc + 1e-300);
          if (r > worst) worst = r, worst_name = all[i];
        }
        return worst;
      };
      prop.set_adjoint_gradients(false);
      auto gf  = prop.weighted_gradient_points_prod_height(ev.E, ev.C, ev.nb, ev.dist, w);
      auto ggf = prop.weighted_gradient_prod_height(wg);
      prop.set_adjoint_gradients(true);
      auto         ga  = prop.weighted_gradient_points_prod_height(ev.E, ev.C, ev.nb, ev.dist, w);
      auto         gga = prop.weighted_gradient_prod_height(wg);
      const double eap = cmp(gf, ga);
      const std::string wp = worst_name;
      const double eag = cmp(ggf, gga);
      const double ea  = std::max(eap, eag);
      // forward weighted grid against the stored grid gradients
      std::vector<double> gs(npar, 0.0);
      for (int nb = 0; nb < 2; nb++)
        for (size_t p = 0; p < npar; p++)
          for (int ab = 0; ab < NN; ab++)
            for (size_t k = 0; k < ng; k++)
              gs[p] += wg[(size_t(nb) * NN + ab) * ng + k] *
                       Gg[((nb * npar + p) * NN + ab) * ng + k];
      const double es = cmp(gs, ggf);
      MESSAGE(std::string(Model::name) << " N=" << N << " production heights: adjoint vs "
              "forward: events " << eap << " (" << wp << "), grid " << eag << " ("
              << worst_name << "); weighted grid vs stored " << es);
      CHECK(ea < tol_adj);
      CHECK(es < 1e-12);
    }
  }

  /// The common height scale s (prod_height_scale): values at s = 1.3
  /// against the brute force with scaled heights, d/ds against central
  /// finite differences (events and grid, stored gradients), forward
  /// against reverse mode (weighted, events and grid), at s = 1.3.
  template <class Model>
  void check_scale(opg::Propagator<Model>& prop, const opg::EarthModel& earth,
                   const std::string& label)
  {
    constexpr int N = Model::N, NN = N * N;
    prop.set_earth(earth);
    prop.set_params(params<Model>());
    prop.set_prod_heights(edges(), probs());
    const double s0 = 1.3;
    prop.set_prod_height_scale(s0);
    Events       ev(5, N > 4 ? 2 : 4);
    const size_t n   = ev.E.size();
    auto         P   = prop.prob_points_prod_height(ev.E, ev.C, ev.nb, ev.dist);
    auto         ref = brute<Model>(params<Model>(), earth, ev, 2, 32, s0);
    const double eb  = max_abs_diff(P, ref);

    prop.set_gradient_params({"th23", "prod_height_scale", "dm31"});
    std::vector<double> Pg, dP;
    prop.prob_points_grad_prod_height(ev.E, ev.C, ev.nb, ev.dist, Pg, dP);
    const double h = 1e-6;
    prop.set_prod_height_scale(s0 + h);
    auto pp = prop.prob_points_prod_height(ev.E, ev.C, ev.nb, ev.dist);
    prop.set_prod_height_scale(s0 - h);
    auto pm = prop.prob_points_prod_height(ev.E, ev.C, ev.nb, ev.dist);
    prop.set_prod_height_scale(s0);
    double m = 0, r = 0;
    for (size_t k = 0; k < pp.size(); k++) {
      const double fd = (pp[k] - pm[k]) / (2 * h);
      m               = std::max(m, std::fabs(dP[NN * n + k] - fd));
      r               = std::max(r, std::fabs(fd));
    }
    const double efd = m / std::max(r, 1e-3);

    // grid (one distribution per point), stored gradients against the FD of the grid
    std::vector<double> Eg = {0.4, 3.0, 25.0}, Cg = {-0.95, -0.1, 0.0, 0.4};
    prop.set_grid(Eg, Cg);
    std::vector<int> dg;
    for (size_t ic = 0; ic < Cg.size(); ic++)
      for (size_t ie = 0; ie < Eg.size(); ie++) dg.push_back(int((ic * 3 + ie) % 6) - 1);
    prop.set_grid_prod_heights(dg);
    prop.calculate_prod_height(opg::Flavor::Both, true);
    const size_t        ng = 2 * NN * Eg.size() * Cg.size();
    std::vector<double> gg(prop.grad(), prop.grad() + 3 * ng);
    auto grid_at = [&](double sc) {
      prop.set_prod_height_scale(sc);
      prop.calculate_prod_height();
      return std::vector<double>(prop.probs(), prop.probs() + ng);
    };
    auto gp = grid_at(s0 + h), gm = grid_at(s0 - h);
    prop.set_prod_height_scale(s0);
    m = r = 0;
    const size_t per = NN * Eg.size() * Cg.size();  // [nb][p][ab][iC][iE]
    for (int nb = 0; nb < 2; nb++)
      for (size_t k = 0; k < per; k++) {
        const double fd = (gp[nb * per + k] - gm[nb * per + k]) / (2 * h);
        m               = std::max(m, std::fabs(gg[(size_t(nb) * 3 + 1) * per + k] - fd));
        r               = std::max(r, std::fabs(fd));
      }
    const double efg = m / std::max(r, 1e-3);

    // forward vs reverse, weighted
    double ea = 0;
    if constexpr (opg::Propagator<Model>::has_adjoint_gradients()) {
      std::vector<double> w(NN * n), wg(ng);
      for (size_t i = 0; i < w.size(); i++) w[i] = std::sin(0.3 * double(i) + 1);
      for (size_t i = 0; i < wg.size(); i++) wg[i] = std::cos(0.7 * double(i));
      prop.set_adjoint_gradients(false);
      auto f1 = prop.weighted_gradient_points_prod_height(ev.E, ev.C, ev.nb, ev.dist, w);
      auto f2 = prop.weighted_gradient_prod_height(wg);
      prop.set_adjoint_gradients(true);
      auto a1 = prop.weighted_gradient_points_prod_height(ev.E, ev.C, ev.nb, ev.dist, w);
      auto a2 = prop.weighted_gradient_prod_height(wg);
      ea = std::max(std::fabs(f1[1] - a1[1]) / std::max(std::fabs(f1[1]), 1e-12),
                    std::fabs(f2[1] - a2[1]) / std::max(std::fabs(f2[1]), 1e-12));
    }
    prop.set_prod_height_scale(1);
    MESSAGE(label << " " << std::string(Model::name) << " N=" << N
                  << " height scale 1.3: exact vs brute " << eb << ", d/ds vs FD events " << efd
                  << " grid " << efg << ", reverse vs forward " << ea);
    CHECK(eb < 1e-11);
    CHECK(efd < 1e-6);
    CHECK(efg < 1e-6);
    CHECK(ea < 1e-10);
  }

} // namespace phtest

#endif
