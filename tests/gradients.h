// Shared gradient checks (CPU and GPU backends).

#ifndef OPG_TESTS_GRADIENTS_H
#define OPG_TESTS_GRADIENTS_H

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "doctest.h"

#include "ref_compare.h"
#include "variants.h"

namespace gradtest {

  using Fast = opg::Fast<double>;
  using LD   = long double;
  using FL   = opg::Fast<LD>;

  /// Parameter points to test.
  inline std::vector<std::pair<std::string, Fast::Params>> param_points()
  {
    std::vector<std::pair<std::string, Fast::Params>> v;
    Fast::Params                                     p;
    p.mix = variants::nominal_mix<3>();
    v.push_back({"nominal", p});
    p.mix = variants::fast_io_mix();
    v.push_back({"IO", p});
    p.mix = variants::nominal_mix<3>();
    p.mix.SetAngle(1, 3, 0);
    v.push_back({"th13=0", p});
    p.mix = variants::nominal_mix<3>();
    p.mix.SetAngle(1, 2, 0);
    p.mix.SetDelta(1, 3, 0);
    v.push_back({"th12=0,d=0", p});
    p.mix = variants::nominal_mix<3>();
    p.mix.SetDelta(1, 3, M_PI / 2);
    v.push_back({"d=pi/2", p});
    return v;
  }

  /// Relative step for each parameter (angles absolute, dm relative).
  inline LD step(const std::string& name, LD value)
  {
    if (name.rfind("dm", 0) == 0) return std::fabs(value) * 1e-5L;
    return 1e-5L;
  }

  /// P[a][b] for a list of points through either the PREM model or a
  /// fixed path, computed in long double.
  struct LDEval {
      opg::PremModel::HostTable<LD>   table;
      std::vector<opg::Segment<LD>>   path;
      bool                            use_path;

      explicit LDEval(const opg::PremModel& m) : table(m), use_path(false) {}
      LDEval(const opg::PremModel& m, const std::vector<opg::Segment<double>>& p)
          : table(m), use_path(true)
      {
        for (auto& s : p) path.push_back({s.length, s.density, s.zoa, s.layer});
      }

      std::vector<LD> probs(const FL::Prepared& P, const std::vector<double>& E,
                            const std::vector<double>& C, const std::vector<int>& nb) const
      {
        std::vector<LD> out;
        auto            ev = table.view();
        for (size_t i = 0; i < E.size(); i++) {
          auto S = use_path ? opg::evolve_path<FL, LD>(P, path.data(), int(path.size()),
                                                       LD(E[i]), nb[i] != 0)
                            : opg::evolve_prem<FL, LD>(P, ev, LD(E[i]), LD(C[i]),
                                                       nb[i] != 0);
          for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++) out.push_back(opg::norm(S(b, a)));
        }
        return out;
      }

      /// dP/dp by 4-point central differences: ref[p][i][a][b]
      std::vector<std::vector<LD>> grads(const Fast::Params& p0,
                                         const std::vector<std::string>& names,
                                         const std::vector<double>& E,
                                         const std::vector<double>& C,
                                         const std::vector<int>& nb) const
      {
        auto                         all = Fast::param_names();
        std::vector<std::vector<LD>> out;
        for (auto& n : names) {
          int  idx = int(std::find(all.begin(), all.end(), n) - all.begin());
          auto base = Fast::cast<LD>(p0);
          LD   x0   = FL::param_ref(base, idx);
          LD   h    = step(n, x0);
          auto at   = [&](LD dx) {
            auto q                 = base;
            FL::param_ref(q, idx)  = x0 + dx;
            return probs(FL::prepare_generic<LD>(q), E, C, nb);
          };
          auto p1 = at(h), m1 = at(-h), p2 = at(2 * h), m2 = at(-2 * h);
          std::vector<LD> g(p1.size());
          for (size_t i = 0; i < g.size(); i++)
            g[i] = (8 * (p1[i] - m1[i]) - (p2[i] - m2[i])) / (12 * h);
          out.push_back(g);
        }
        return out;
      }
  };

  /// max_i |g - ref| / max(max_i |ref|, 1e-3), per parameter, maximised over
  /// parameters. The floor handles derivatives that vanish identically
  /// (e.g. d/d delta at theta12 = 0), where both sides are round-off noise
  /// (~1e-13 absolute, against typical derivatives of O(1e-2..1)).
  inline double rel_err(const std::vector<double>& g /*[p][a][b][i]*/,
                        const std::vector<std::vector<LD>>& ref /*[p][i][a][b]*/,
                        size_t n)
  {
    double worst = 0;
    for (size_t p = 0; p < ref.size(); p++) {
      double m = 0, r = 0;
      for (size_t i = 0; i < n; i++)
        for (int ab = 0; ab < 9; ab++) {
          double gr = double(ref[p][i * 9 + ab]);
          double gg = g[(p * 9 + ab) * n + i];
          m         = std::max(m, std::fabs(gg - gr));
          r         = std::max(r, std::fabs(gr));
        }
      worst = std::max(worst, m / std::max(r, 1e-3));
    }
    return worst;
  }

  /// Points: a PREM sub-grid (both nu and nubar).
  struct Points {
      std::vector<double>  E, C;
      std::vector<uint8_t> nb;
      std::vector<int>     nbi;
      Points()
      {
        std::vector<double> cs = {-1.0, -0.97, -0.85, -0.6, -0.3, -0.05, 0.2, 0.9};
        for (int b = 0; b < 2; b++)
          for (double c : cs)
            for (int i = 0; i < 25; i++) {
              E.push_back(std::pow(10.0, -0.5 + 2.0 * i / 24));
              C.push_back(c);
              nb.push_back(uint8_t(b));
              nbi.push_back(b);
            }
      }
  };

  /// Full gradient checks for one propagator (CPU or GPU).
  inline void check_against_ld(opg::Propagator<Fast>& prop, double tol)
  {
    auto names = Fast::param_names();
    prop.set_gradient_params(names);
    opg::PremModel prem;
    LDEval         ldprem(prem);
    LDEval         ldpath(prem, refcmp::test_path());
    LDEval         ldvac(prem, refcmp::vacuum_path());
    Points         pts;
    std::vector<double> Ep;
    for (int i = 0; i < 60; i++) Ep.push_back(std::pow(10.0, -1 + 2.0 * i / 59));

    for (auto& [label, par] : param_points()) {
      prop.set_params(par);
      // PREM event list
      std::vector<double> P, dP;
      prop.prob_points_grad(pts.E, pts.C, pts.nb, P, dP);
      auto   ref = ldprem.grads(par, names, pts.E, pts.C, pts.nbi);
      double e1  = rel_err(dP, ref, pts.E.size());
      // fixed paths (test path, vacuum), nu and nubar
      double e2 = 0, e3 = 0;
      for (int b = 0; b < 2; b++) {
        std::vector<int> nbv(Ep.size(), b);
        std::vector<double> Cdummy(Ep.size(), 0.0);
        prop.prob_path_grad(Ep, refcmp::test_path(), b, P, dP);
        e2 = std::max(e2, rel_err(dP, ldpath.grads(par, names, Ep, Cdummy, nbv), Ep.size()));
        prop.prob_path_grad(Ep, refcmp::vacuum_path(), b, P, dP);
        e3 = std::max(e3, rel_err(dP, ldvac.grads(par, names, Ep, Cdummy, nbv), Ep.size()));
      }
      MESSAGE("Fast gradients [" << label << "] vs long-double FD: PREM "
                                 << e1 << ", test path " << e2 << ", vacuum " << e3);
      CHECK(e1 < tol);
      CHECK(e2 < tol);
      CHECK(e3 < tol);
    }
  }

  /// Gradients on must not change probabilities.
  inline void check_values_unchanged(opg::Propagator<Fast>& prop)
  {
    Fast::Params par;
    par.mix = variants::nominal_mix<3>();
    prop.set_params(par);
    prop.set_gradient_params({"th23", "dm31"});
    Points pts;
    auto   P0 = prop.prob_points(pts.E, pts.C, pts.nb);
    std::vector<double> P1, dP;
    prop.prob_points_grad(pts.E, pts.C, pts.nb, P1, dP);
    CHECK(P0 == P1);

    std::vector<double> E, C;
    for (int i = 0; i < 37; i++) E.push_back(0.5 + 0.3 * i);
    for (int i = 0; i < 23; i++) C.push_back(-1 + 2.0 * i / 22);
    prop.set_grid(E, C);
    prop.calculate();
    std::vector<double> g0(prop.probs(), prop.probs() + 2 * 9 * E.size() * C.size());
    prop.calculate(opg::Flavor::Both, true);
    std::vector<double> g1(prop.probs(), prop.probs() + g0.size());
    CHECK(g0 == g1);
  }

  /// Grid gradients equal the event-list gradients at the same points, and
  /// the weighted mode equals the explicit contraction of full gradients.
  inline void check_grid_and_weighted(opg::Propagator<Fast>& prop, double tol_w,
                                      double tol_same = 0)
  {
    Fast::Params par;
    par.mix = variants::nominal_mix<3>();
    prop.set_params(par);
    std::vector<std::string> names = {"dm31", "th23", "d13", "th13", "dm21"};
    prop.set_gradient_params(names);
    const size_t        np = names.size();
    std::vector<double> E, C;
    for (int i = 0; i < 41; i++) E.push_back(std::pow(10.0, -0.3 + 2.0 * i / 40));
    for (int i = 0; i < 29; i++) C.push_back(-1 + 2.0 * i / 28);
    const size_t nE = E.size(), nC = C.size(), npt = nE * nC;
    prop.set_grid(E, C);
    prop.calculate(opg::Flavor::Both, true);
    std::vector<double> G(prop.grad(), prop.grad() + 2 * np * 9 * npt);

    // event list at the same points
    std::vector<double>  e, c;
    std::vector<uint8_t> nb;
    for (int b = 0; b < 2; b++)
      for (size_t ic = 0; ic < nC; ic++)
        for (size_t ie = 0; ie < nE; ie++) {
          e.push_back(E[ie]);
          c.push_back(C[ic]);
          nb.push_back(uint8_t(b));
        }
    std::vector<double> P, dP;
    prop.prob_points_grad(e, c, nb, P, dP);
    double d = 0;
    for (int b = 0; b < 2; b++)
      for (size_t p = 0; p < np; p++)
        for (int ab = 0; ab < 9; ab++)
          for (size_t k = 0; k < npt; k++) {
            double gg = G[((b * np + p) * 9 + ab) * npt + k];
            double gp = dP[(p * 9 + ab) * (2 * npt) + b * npt + k];
            d         = std::max(d, std::fabs(gg - gp));
          }
    MESSAGE("grid vs event-list gradients: max diff " << d);
    CHECK(d <= tol_same);

    // weighted mode (grid)
    std::mt19937_64                        rng(3);
    std::uniform_real_distribution<double> u(-1, 1);
    std::vector<double>                    w(2 * 9 * npt);
    for (auto& x : w) x = u(rng);
    auto           gw = prop.weighted_gradient(w);
    std::vector<double> ge(np, 0), scale(np, 0);
    for (int b = 0; b < 2; b++)
      for (size_t p = 0; p < np; p++)
        for (int ab = 0; ab < 9; ab++)
          for (size_t k = 0; k < npt; k++) {
            double t = w[(b * 9 + ab) * npt + k] * G[((b * np + p) * 9 + ab) * npt + k];
            ge[p] += t;
            scale[p] += std::fabs(t);
          }
    double rw = 0;
    for (size_t p = 0; p < np; p++) rw = std::max(rw, std::fabs(gw[p] - ge[p]) / scale[p]);
    MESSAGE("weighted grid gradient vs explicit contraction: " << rw);
    CHECK(rw < tol_w);
    CHECK(prop.weighted_gradient(w) == gw);  // deterministic

    // only antineutrinos
    auto gnb = prop.weighted_gradient(w, opg::Flavor::Antineutrino);
    std::vector<double> gnbe(np, 0);
    for (size_t p = 0; p < np; p++)
      for (int ab = 0; ab < 9; ab++)
        for (size_t k = 0; k < npt; k++)
          gnbe[p] += w[(9 + ab) * npt + k] * G[((np + p) * 9 + ab) * npt + k];
    for (size_t p = 0; p < np; p++)
      CHECK(std::fabs(gnb[p] - gnbe[p]) <= tol_w * scale[p]);

    // weighted mode (event list)
    std::vector<double> wp(9 * e.size());
    for (auto& x : wp) x = u(rng);
    auto gwp = prop.weighted_gradient_points(e, c, nb, wp);
    for (size_t p = 0; p < np; p++) {
      double s = 0, sc = 0;
      for (int ab = 0; ab < 9; ab++)
        for (size_t i = 0; i < e.size(); i++) {
          double t = wp[ab * e.size() + i] * dP[(p * 9 + ab) * e.size() + i];
          s += t;
          sc += std::fabs(t);
        }
      CHECK(std::fabs(gwp[p] - s) <= tol_w * sc);
    }
  }

} // namespace gradtest

#endif
