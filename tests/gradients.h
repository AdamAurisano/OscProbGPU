// Shared gradient checks (CPU and GPU backends).

#ifndef OPG_TESTS_GRADIENTS_H
#define OPG_TESTS_GRADIENTS_H

#include <algorithm>
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

  /// The same model in long double (for the finite-difference reference).
  template <class Model> struct LongDouble;
  template <template <class> class M, class R> struct LongDouble<M<R>> {
      using type = M<LD>;
  };

  /// Number of mixing parameters (they come first in param_names()).
  template <class Model> size_t MixingCount()
  {
    return size_t(opg::MixingRegistry<Model::N>::count());
  }

  /// Parameter points to test, per model.
  template <class Model>
  std::vector<std::pair<std::string, typename Model::Params>> param_points();

  template <> inline std::vector<std::pair<std::string, Fast::Params>> param_points<Fast>()
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

  using NSI = opg::NSI<double>;
  template <> inline std::vector<std::pair<std::string, NSI::Params>> param_points<NSI>()
  {
    std::vector<std::pair<std::string, NSI::Params>> v;
    v.push_back({"nsi", variants::nsi()});
    v.push_back({"nsi_phases", variants::nsi_phases()});
    NSI::Params p;
    p.mix = variants::nominal_mix<3>();
    v.push_back({"eps=0", p});
    p.SetFermCoup(0, 0, 0);
    v.push_back({"eps=0,coup=0", p});
    p = variants::nsi_phases();
    p.SetEps(0, 1, 0.0, 0.7);  // zero magnitude, nonzero phase
    p.SetEps(1, 2, 0.3, M_PI);
    v.push_back({"emu=0,mutau@pi", p});
    return v;
  }

  using NUNM = opg::NUNM<double>;
  template <> inline std::vector<std::pair<std::string, NUNM::Params>> param_points<NUNM>()
  {
    std::vector<std::pair<std::string, NUNM::Params>> v;
    v.push_back({"nunm", variants::nunm(0)});
    v.push_back({"nunm_phases", variants::nunm_phases()});
    v.push_back({"nunm_high", variants::nunm(1)});
    auto ph = variants::nunm_phases();
    ph.scale = 1;
    v.push_back({"nunm_phases_high", ph});
    NUNM::Params p;
    p.mix = variants::nominal_mix<3>();
    v.push_back({"alpha=0", p});
    p.scale = 1;
    v.push_back({"alpha=0,high", p});
    return v;
  }

  using Sterile = opg::Sterile<double>;
  template <>
  inline std::vector<std::pair<std::string, Sterile::Params>> param_points<Sterile>()
  {
    std::vector<std::pair<std::string, Sterile::Params>> v;
    Sterile::Params p;
    p.mix = variants::sterile_mix();
    v.push_back({"sterile", p});
    p.mix = variants::sterile_phases_mix();
    v.push_back({"sterile_phases", p});
    p.mix = variants::sterile_mix();
    p.mix.SetAngle(1, 4, 0);
    p.mix.SetAngle(2, 4, 0);
    p.mix.SetAngle(3, 4, 0);
    v.push_back({"th14=th24=th34=0", p});
    p.mix = variants::sterile_phases_mix();
    p.mix.SetDm(4, 2.0e-3);  // near the atmospheric splitting
    v.push_back({"dm41~dm31", p});
    return v;
  }

  using Decay = opg::Decay<double>;
  template <> inline std::vector<std::pair<std::string, Decay::Params>> param_points<Decay>()
  {
    std::vector<std::pair<std::string, Decay::Params>> v;
    v.push_back({"decay", variants::decay()});
    v.push_back({"decay_both", variants::decay_both()});
    Decay::Params p;
    p.mix = variants::nominal_mix<3>();
    v.push_back({"alpha=0", p});
    p.mix = variants::fast_io_mix();
    p.SetAlpha2(5e-5);
    v.push_back({"IO,alpha2", p});
    p = variants::decay_both();
    p.mix.SetAngle(1, 3, 0);
    v.push_back({"decay_both,th13=0", p});
    return v;
  }

  /// Parameters in eV^2 (mass splittings, Decay's alpha_j).
  inline bool is_ev2(const std::string& name)
  {
    return name.rfind("dm", 0) == 0 || name == "alpha2" || name == "alpha3";
  }

  /// Step for each parameter: 1e-4 absolute for angles, phases and
  /// couplings (smaller steps amplify the long-double round-off of the
  /// O(1e5) rad phases at dm41 ~ 1 eV^2); relative 1e-5 for mass
  /// splittings, capped at 3e-8 eV^2 so that the oscillation phase changes
  /// by at most ~1e-3 rad over the Earth even for dm41 ~ 1 eV^2; 3e-8 eV^2
  /// absolute for decay constants.
  inline LD step(const std::string& name, LD value)
  {
    if (name.rfind("dm", 0) == 0) return std::min(std::fabs(value) * 1e-5L, 3e-8L);
    if (is_ev2(name)) return 3e-8L;  // decay constants (may be 0)
    return 1e-4L;
  }

  /// P[a][b] for a list of points through either the PREM model or a
  /// fixed path, computed in long double.
  template <class Model> struct LDEval {
      using FL               = typename LongDouble<Model>::type;
      static constexpr int N = Model::N;
      opg::PremModel::HostTable<LD>   table;
      std::vector<opg::Segment<LD>>   path;
      bool                            use_path;

      explicit LDEval(const opg::PremModel& m) : table(m), use_path(false) {}
      LDEval(const opg::PremModel& m, const std::vector<opg::Segment<double>>& p)
          : table(m), use_path(true)
      {
        for (auto& s : p) path.push_back({s.length, s.density, s.zoa, s.layer});
      }

      std::vector<LD> probs(const typename FL::Prepared& P, const std::vector<double>& E,
                            const std::vector<double>& C, const std::vector<int>& nb) const
      {
        std::vector<LD> out;
        auto            ev = table.view();
        for (size_t i = 0; i < E.size(); i++) {
          auto S = use_path ? opg::evolve_path<FL, LD>(P, path.data(), int(path.size()),
                                                       LD(E[i]), nb[i] != 0)
                            : opg::evolve_prem<FL, LD>(P, ev, LD(E[i]), LD(C[i]),
                                                       nb[i] != 0);
          for (int a = 0; a < N; a++)
            for (int b = 0; b < N; b++) out.push_back(opg::norm(S(b, a)));
        }
        return out;
      }

      /// dP/dp by 6-point central differences (error O(h^6)): ref[p][i][a][b]
      std::vector<std::vector<LD>> grads(const typename Model::Params& p0,
                                         const std::vector<std::string>& names,
                                         const std::vector<double>& E,
                                         const std::vector<double>& C,
                                         const std::vector<int>& nb) const
      {
        auto                         all = Model::param_names();
        std::vector<std::vector<LD>> out;
        for (auto& n : names) {
          int  idx = int(std::find(all.begin(), all.end(), n) - all.begin());
          auto base = Model::template cast<LD>(p0);
          LD   x0   = FL::param_ref(base, idx);
          LD   h    = step(n, x0);
          auto at   = [&](LD dx) {
            auto q                 = base;
            FL::param_ref(q, idx)  = x0 + dx;
            return probs(FL::template prepare_generic<LD>(q), E, C, nb);
          };
          auto p1 = at(h), m1 = at(-h), p2 = at(2 * h), m2 = at(-2 * h);
          auto p3 = at(3 * h), m3 = at(-3 * h);
          std::vector<LD> g(p1.size());
          for (size_t i = 0; i < g.size(); i++)
            g[i] = (45 * (p1[i] - m1[i]) - 9 * (p2[i] - m2[i]) + (p3[i] - m3[i])) /
                   (60 * h);
          out.push_back(g);
        }
        return out;
      }
  };

  /// max_i |g - ref| / max(max_i |ref|, floor), per parameter, maximised
  /// over parameters. The floor handles derivatives that vanish identically
  /// (e.g. d/d delta at theta12 = 0, or d/d dm41 of P(s -> s) = 1 at zero
  /// sterile mixing), where both sides are round-off noise. It is 1e-3 for
  /// angles, phases and couplings (typical derivatives O(1e-2..1)) and 10
  /// eV^-2 for mass splittings, whose derivatives scale with L/E (up to
  /// ~5e4 eV^-2 through the Earth at 0.3 GeV); likewise for decay constants.
  inline double grad_floor(const std::string& name)
  {
    return is_ev2(name) ? 10.0 : 1e-3;
  }

  inline double rel_err(const std::vector<double>& g /*[p][a][b][i]*/,
                        const std::vector<std::vector<LD>>& ref /*[p][i][a][b]*/,
                        size_t n, int NN, const std::vector<std::string>& names,
                        size_t* which = nullptr)
  {
    double worst = 0;
    for (size_t p = 0; p < ref.size(); p++) {
      double m = 0, r = 0;
      for (size_t i = 0; i < n; i++)
        for (int ab = 0; ab < NN; ab++) {
          double gr = double(ref[p][i * NN + ab]);
          double gg = g[(p * NN + ab) * n + i];
          m         = std::max(m, std::fabs(gg - gr));
          r         = std::max(r, std::fabs(gr));
        }
      const double e = m / std::max(r, grad_floor(names[p]));
      if (e > worst && which) *which = p;
      worst = std::max(worst, e);
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

  /// Full gradient checks for one propagator (CPU or GPU): all parameters
  /// at every point of param_points<Model>().
  template <class Model>
  void check_against_ld(opg::Propagator<Model>& prop, double tol)
  {
    constexpr int NN    = Model::N * Model::N;
    auto          names = Model::param_names();
    prop.set_gradient_params(names);
    opg::PremModel prem;
    LDEval<Model>  ldprem(prem);
    LDEval<Model>  ldpath(prem, refcmp::test_path());
    LDEval<Model>  ldvac(prem, refcmp::vacuum_path());
    Points         pts;
    std::vector<double> Ep;
    for (int i = 0; i < 60; i++) Ep.push_back(std::pow(10.0, -1 + 2.0 * i / 59));

    for (auto& [label, par] : param_points<Model>()) {
      prop.set_params(par);
      // PREM event list
      std::vector<double> P, dP;
      prop.prob_points_grad(pts.E, pts.C, pts.nb, P, dP);
      auto   ref = ldprem.grads(par, names, pts.E, pts.C, pts.nbi);
      size_t w1 = 0;
      double e1  = rel_err(dP, ref, pts.E.size(), NN, names, &w1);
      // fixed paths (test path, vacuum), nu and nubar
      double e2 = 0, e3 = 0;
      for (int b = 0; b < 2; b++) {
        std::vector<int> nbv(Ep.size(), b);
        std::vector<double> Cdummy(Ep.size(), 0.0);
        prop.prob_path_grad(Ep, refcmp::test_path(), b, P, dP);
        e2 = std::max(e2, rel_err(dP, ldpath.grads(par, names, Ep, Cdummy, nbv),
                                  Ep.size(), NN, names));
        prop.prob_path_grad(Ep, refcmp::vacuum_path(), b, P, dP);
        e3 = std::max(e3, rel_err(dP, ldvac.grads(par, names, Ep, Cdummy, nbv),
                                  Ep.size(), NN, names));
      }
      MESSAGE(std::string(Model::name)
              << " gradients [" << label << "] vs long-double FD: PREM " << e1 << " ("
              << names[w1] << "), test path " << e2 << ", vacuum " << e3);
      CHECK(e1 < tol);
      CHECK(e2 < tol);
      CHECK(e3 < tol);
    }
  }

  /// A few parameters to differentiate in the consistency checks (all
  /// kinds, more than one gradient pass).
  template <class Model> std::vector<std::string> some_params()
  {
    auto all = Model::param_names();
    std::vector<std::string> v = {"dm31", "th23", "d13", "th13", "dm21"};
    for (size_t i = MixingCount<Model>(); i < all.size(); i += 2) v.push_back(all[i]);
    if (all.size() > MixingCount<Model>()) v.push_back(all.back());
    return v;
  }

  /// Gradients on must not change probabilities.
  template <class Model> void check_values_unchanged(opg::Propagator<Model>& prop)
  {
    constexpr int NN  = Model::N * Model::N;
    auto          par = param_points<Model>()[0].second;
    prop.set_params(par);
    std::vector<std::string> sel = {"th23", "dm31"};
    if (Model::param_names().back() != "dm31") sel.push_back(Model::param_names().back());
    prop.set_gradient_params(sel);
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
    std::vector<double> g0(prop.probs(), prop.probs() + 2 * NN * E.size() * C.size());
    prop.calculate(opg::Flavor::Both, true);
    std::vector<double> g1(prop.probs(), prop.probs() + g0.size());
    CHECK(g0 == g1);
  }

  /// Grid gradients equal the event-list gradients at the same points, and
  /// the weighted mode equals the explicit contraction of full gradients.
  template <class Model>
  void check_grid_and_weighted(opg::Propagator<Model>& prop, double tol_w,
                               double tol_same = 0)
  {
    constexpr int NN  = Model::N * Model::N;
    auto          par = param_points<Model>()[0].second;
    prop.set_params(par);
    std::vector<std::string> names = some_params<Model>();
    prop.set_gradient_params(names);
    const size_t        np = names.size();
    std::vector<double> E, C;
    for (int i = 0; i < 41; i++) E.push_back(std::pow(10.0, -0.3 + 2.0 * i / 40));
    for (int i = 0; i < 29; i++) C.push_back(-1 + 2.0 * i / 28);
    const size_t nE = E.size(), nC = C.size(), npt = nE * nC;
    prop.set_grid(E, C);
    prop.calculate(opg::Flavor::Both, true);
    std::vector<double> G(prop.grad(), prop.grad() + 2 * np * NN * npt);

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
        for (int ab = 0; ab < NN; ab++)
          for (size_t k = 0; k < npt; k++) {
            double gg = G[((b * np + p) * NN + ab) * npt + k];
            double gp = dP[(p * NN + ab) * (2 * npt) + b * npt + k];
            d         = std::max(d, std::fabs(gg - gp));
          }
    MESSAGE(std::string(Model::name) << " grid vs event-list gradients: max diff " << d);
    CHECK(d <= tol_same);

    // weighted mode (grid)
    std::mt19937_64                        rng(3);
    std::uniform_real_distribution<double> u(-1, 1);
    std::vector<double>                    w(2 * NN * npt);
    for (auto& x : w) x = u(rng);
    auto           gw = prop.weighted_gradient(w);
    std::vector<double> ge(np, 0), scale(np, 0);
    for (int b = 0; b < 2; b++)
      for (size_t p = 0; p < np; p++)
        for (int ab = 0; ab < NN; ab++)
          for (size_t k = 0; k < npt; k++) {
            double t = w[(b * NN + ab) * npt + k] * G[((b * np + p) * NN + ab) * npt + k];
            ge[p] += t;
            scale[p] += std::fabs(t);
          }
    double rw = 0;
    for (size_t p = 0; p < np; p++)
      rw = std::max(rw, std::fabs(gw[p] - ge[p]) / std::max(scale[p], 1e-300));
    MESSAGE(std::string(Model::name) << " weighted grid gradient vs explicit contraction: " << rw);
    CHECK(rw < tol_w);
    CHECK(prop.weighted_gradient(w) == gw);  // deterministic

    // only antineutrinos
    auto gnb = prop.weighted_gradient(w, opg::Flavor::Antineutrino);
    std::vector<double> gnbe(np, 0);
    for (size_t p = 0; p < np; p++)
      for (int ab = 0; ab < NN; ab++)
        for (size_t k = 0; k < npt; k++)
          gnbe[p] += w[(NN + ab) * npt + k] * G[((np + p) * NN + ab) * npt + k];
    for (size_t p = 0; p < np; p++)
      CHECK(std::fabs(gnb[p] - gnbe[p]) <= tol_w * scale[p]);

    // weighted mode (event list)
    std::vector<double> wp(NN * e.size());
    for (auto& x : wp) x = u(rng);
    auto gwp = prop.weighted_gradient_points(e, c, nb, wp);
    for (size_t p = 0; p < np; p++) {
      double s = 0, sc = 0;
      for (int ab = 0; ab < NN; ab++)
        for (size_t i = 0; i < e.size(); i++) {
          double t = wp[ab * e.size() + i] * dP[(p * NN + ab) * e.size() + i];
          s += t;
          sc += std::fabs(t);
        }
      CHECK(std::fabs(gwp[p] - s) <= tol_w * sc);
    }
  }

} // namespace gradtest

#endif
