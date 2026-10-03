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
    // degenerate vacuum eigenvalues (exercises the dual-expm fallback)
    p = Decay::Params();
    p.mix.SetDm(3, 2.5e-3);
    p.SetAlpha3(1e-4);
    v.push_back({"no mixing,dm21=0", p});
    return v;
  }

  /// Natural size of a parameter: 1 except for the LIV coefficients, whose
  /// dimension-d magnitudes are of order 10^(-21 - ...) GeV^(4-d).
  inline double param_scale(const std::string& name)
  {
    if (name.size() > 3 && (name.rfind("aT", 0) == 0 || name.rfind("cT", 0) == 0)) {
      static const double s[6] = {1e-21, 1e-22, 1e-24, 1e-26, 1e-28, 1e-30};
      return s[name[2] - '3'];
    }
    return 1.0;
  }

  /// Natural size of a parameter for a given model (finite-difference step
  /// and error floor scale with it).
  template <class Model> double model_scale(const std::string& name)
  {
    return param_scale(name);
  }
  /// SiderealLIV: P varies on scales of aT ~ 1e-22 GeV and (as cT enters
  /// as E[GeV] cT, without the GeV -> eV factor, as OscProb) cT ~ 1e-14.
  template <> inline double model_scale<opg::SiderealLIV<double>>(const std::string& name)
  {
    if (name.size() > 2 && name[0] == 'a' && name[2] == '_') return 1e-22;
    if (name.size() > 3 && name[0] == 'c' && name[3] == '_') return 1e-14;
    return param_scale(name);
  }

  /// Deco: Gamma_ij of order 1e-21 GeV (power 0) to 1e-24 GeV (power 2).
  template <> inline double model_scale<opg::Deco<double>>(const std::string& name)
  {
    if (name == "gamma21" || name == "gamma31") return 1e-24;
    return param_scale(name);
  }

  /// OQS: |a_i| matter on scales ~1e-12 (D ~ a_i a_j GeV with D L ~ 1).
  template <> inline double model_scale<opg::OQS<double>>(const std::string& name)
  {
    if (name.size() == 2 && name[0] == 'a') return 1e-12;
    return param_scale(name);
  }

  /// SNSI: with absolute masses of a few 1e-2 eV, the probabilities vary on
  /// scales ~1e-2 of eps (MeV^-2), the couplings and mlight (eV).
  template <> inline double model_scale<opg::SNSI<double>>(const std::string& name)
  {
    if (name == "mlight" || name.rfind("eps_", 0) == 0 || name.rfind("coup_", 0) == 0)
      return 1e-2;
    return param_scale(name);
  }

  /// Parameters in eV^2 (mass splittings, Decay's alpha_j).
  inline bool is_ev2(const std::string& name)
  {
    return name.rfind("dm", 0) == 0 || name == "alpha2" || name == "alpha3";
  }

  using LIV = opg::LIV<double>;
  template <> inline std::vector<std::pair<std::string, LIV::Params>> param_points<LIV>()
  {
    std::vector<std::pair<std::string, LIV::Params>> v;
    v.push_back({"liv", variants::liv()});
    v.push_back({"liv_phases", variants::liv_phases()});
    LIV::Params p;
    p.mix = variants::nominal_mix<3>();
    v.push_back({"no LIV", p});
    return v;
  }

  using SNSI = opg::SNSI<double>;
  template <> inline std::vector<std::pair<std::string, SNSI::Params>> param_points<SNSI>()
  {
    std::vector<std::pair<std::string, SNSI::Params>> v;
    v.push_back({"snsi", variants::snsi()});
    auto io = variants::snsi_io();
    io.SetLowestMass(0.02);  // (at 0 the mlight derivative is one-sided)
    v.push_back({"snsi_io,m=0.02", io});
    SNSI::Params p;
    p.mix = variants::nominal_mix<3>();
    p.SetLowestMass(0.05);
    v.push_back({"eps=0", p});
    return v;
  }

  using OQS = opg::OQS<double>;
  template <> inline std::vector<std::pair<std::string, OQS::Params>> param_points<OQS>()
  {
    std::vector<std::pair<std::string, OQS::Params>> v;
    v.push_back({"oqs", variants::oqs()});
    v.push_back({"oqs_full", variants::oqs_full()});
    OQS::Params p;
    p.mix = variants::nominal_mix<3>();
    v.push_back({"a=0", p});
    p = variants::oqs();
    p.mix = variants::fast_io_mix();
    p.SetPower(2);
    p.SetDecoElement(3, 3e-14);
    p.SetDecoElement(8, 5e-14);
    v.push_back({"IO,power=2", p});
    return v;
  }

  using Deco = opg::Deco<double>;
  template <> inline std::vector<std::pair<std::string, Deco::Params>> param_points<Deco>()
  {
    std::vector<std::pair<std::string, Deco::Params>> v;
    v.push_back({"deco", variants::deco()});
    v.push_back({"deco_power", variants::deco_power()});
    Deco::Params p;
    p.mix = variants::nominal_mix<3>();
    v.push_back({"Gamma=0", p});
    return v;
  }

  using SiderealLIV = opg::SiderealLIV<double>;
  template <>
  inline std::vector<std::pair<std::string, SiderealLIV::Params>> param_points<SiderealLIV>()
  {
    std::vector<std::pair<std::string, SiderealLIV::Params>> v;
    auto p = variants::sidereal();
    // sizes with visible effects (see model_scale)
    p.SetA(0, 1, 1, 2e-20);
    p.SetA(1, 2, 0, -1e-20);
    p.SetC(0, 1, 0, 2, 3e-12);
    p.SetC(1, 1, 1, 2, -2e-12);
    v.push_back({"sidereal (path zenith)", p});
    v.push_back({"sidereal_fixed", variants::sidereal_fixed()});
    SiderealLIV::Params q;
    q.mix = variants::nominal_mix<3>();
    q.SetColatitude(43.5);
    q.SetAzimuth(30.0);
    q.SetTimeHours(3.0);
    v.push_back({"no LIV", q});
    return v;
  }

  /// Step for each parameter: 1e-4 absolute for angles, phases and
  /// couplings (smaller steps amplify the long-double round-off of the
  /// O(1e5) rad phases at dm41 ~ 1 eV^2); relative 1e-5 for mass
  /// splittings, capped at 3e-8 eV^2 so that the oscillation phase changes
  /// by at most ~1e-3 rad over the Earth even for dm41 ~ 1 eV^2; 3e-8 eV^2
  /// absolute for decay constants.
  template <class Model> LD step(const std::string& name, LD value)
  {
    if (name.rfind("dm", 0) == 0) return std::min(std::fabs(value) * 1e-5L, 3e-8L);
    if (is_ev2(name)) return 3e-8L;  // decay constants (may be 0)
    return 1e-4L * LD(model_scale<Model>(name));
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
          LD pr[N * N];
          opg::store_model_probs<FL, LD>(S, pr, 1);
          for (int ab = 0; ab < N * N; ab++) out.push_back(pr[ab]);
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
          LD   h    = step<Model>(n, x0);
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
  template <class Model = opg::Fast<double>> double grad_floor(const std::string& name)
  {
    return is_ev2(name) ? 10.0 : 1e-3 / model_scale<Model>(name);
  }

  template <class Model = opg::Fast<double>>
  double rel_err(const std::vector<double>& g /*[p][a][b][i]*/,
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
      const double e = m / std::max(r, grad_floor<Model>(names[p]));
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
      double e1  = rel_err<Model>(dP, ref, pts.E.size(), NN, names, &w1);
      // fixed paths (test path, vacuum), nu and nubar
      double e2 = 0, e3 = 0;
      for (int b = 0; b < 2; b++) {
        std::vector<int> nbv(Ep.size(), b);
        std::vector<double> Cdummy(Ep.size(), 0.0);
        prop.prob_path_grad(Ep, refcmp::test_path(), b, P, dP);
        e2 = std::max(e2, rel_err<Model>(dP, ldpath.grads(par, names, Ep, Cdummy, nbv),
                                  Ep.size(), NN, names));
        prop.prob_path_grad(Ep, refcmp::vacuum_path(), b, P, dP);
        e3 = std::max(e3, rel_err<Model>(dP, ldvac.grads(par, names, Ep, Cdummy, nbv),
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

    // per-analysis-bin weighted mode (event list); bins -1 and >= nbins are
    // ignored
    const int        nbins = 37;
    std::vector<int> bin(e.size());
    for (size_t i = 0; i < e.size(); i++) bin[i] = int((i * 7919) % (nbins + 3)) - 1;
    auto Gb = prop.weighted_gradient_points_binned(e, c, nb, wp, bin, nbins);
    REQUIRE(Gb.size() == size_t(nbins) * np);
    double rb = 0;
    for (int b = 0; b < nbins; b++)
      for (size_t p = 0; p < np; p++) {
        double s = 0, sc = 0;
        for (size_t i = 0; i < e.size(); i++) {
          if (bin[i] != b) continue;
          for (int ab = 0; ab < NN; ab++) {
            double t = wp[ab * e.size() + i] * dP[(p * NN + ab) * e.size() + i];
            s += t;
            sc += std::fabs(t);
          }
        }
        rb = std::max(rb, std::fabs(Gb[size_t(b) * np + p] - s) / std::max(sc, 1e-300));
      }
    MESSAGE(std::string(Model::name) << " per-bin weighted gradients vs explicit: " << rb);
    CHECK(rb < tol_w);
    CHECK(prop.weighted_gradient_points_binned(e, c, nb, wp, bin, nbins) == Gb);
  }

  /// Bin-averaged gradients: values unchanged, finite differences of
  /// binned() (4-point, double precision), weighted mode, and avg_path_grad.
  template <class Model>
  void check_binned(opg::Propagator<Model>& prop, double tol_fd, double tol_w)
  {
    constexpr int NN    = Model::N * Model::N;
    auto          par   = param_points<Model>()[0].second;
    auto          all   = Model::param_names();
    std::vector<std::string> names = {"th23", "dm31", all.back()};
    prop.set_params(par);
    prop.set_gradient_params(names);
    const size_t np = names.size();
    std::vector<double> Ee, Ce;
    for (int i = 0; i <= 12; i++) Ee.push_back(std::pow(10.0, -0.2 + 1.6 * i / 12));
    for (int i = 0; i <= 7; i++) Ce.push_back(-1 + 1.2 * i / 7);
    prop.set_bins(Ee, Ce, 3, 3, opg::EMeasure::Log);
    const size_t nbin = prop.n_energy_bins() * prop.n_cosine_bins();
    prop.calculate_binned();
    std::vector<double> B0(prop.binned(), prop.binned() + 2 * NN * nbin);
    prop.calculate_binned(opg::Flavor::Both, true);
    std::vector<double> B1(prop.binned(), prop.binned() + 2 * NN * nbin);
    CHECK(B0 == B1);
    std::vector<double> G(prop.binned_grad(), prop.binned_grad() + 2 * np * NN * nbin);

    // central differences of binned() in double precision
    double worst = 0;
    for (size_t p = 0; p < np; p++) {
      const int idx = int(std::find(all.begin(), all.end(), names[p]) - all.begin());
      auto      at  = [&](double dx) {
        auto q                         = Model::template cast<double>(par);
        Model::param_ref(q, idx)      += dx;
        opg::Propagator<Model> pp(opg::PremModel(), {}, 0);
        pp.set_params(q);
        pp.set_bins(Ee, Ce, 3, 3, opg::EMeasure::Log);
        pp.calculate_binned();
        return std::vector<double>(pp.binned(), pp.binned() + 2 * NN * nbin);
      };
      auto         q0 = Model::template cast<double>(par);
      const double x0 = Model::param_ref(q0, idx);
      const double h  = is_ev2(names[p]) ? std::min(std::max(std::fabs(x0), 1e-4) * 1e-6, 3e-9)
                                         : 1e-4 * model_scale<Model>(names[p]);
      auto         pl = at(h), mi = at(-h), p2 = at(2 * h), m2 = at(-2 * h);
      double       m = 0, r = 0;
      for (int b = 0; b < 2; b++)
        for (int ab = 0; ab < NN; ab++)
          for (size_t k = 0; k < nbin; k++) {
            const size_t i  = (b * NN + ab) * nbin + k;
            const double fd = (8 * (pl[i] - mi[i]) - (p2[i] - m2[i])) / (12 * h);
            const double g  = G[((b * np + p) * NN + ab) * nbin + k];
            m               = std::max(m, std::fabs(g - fd));
            r               = std::max(r, std::fabs(fd));
          }
      worst = std::max(worst, m / std::max(r, grad_floor<Model>(names[p])));
    }
    MESSAGE(std::string(Model::name) << " binned gradients vs double FD: " << worst);
    CHECK(worst < tol_fd);

    // weighted binned mode
    std::vector<double> w(2 * NN * nbin);
    for (size_t i = 0; i < w.size(); i++) w[i] = std::sin(0.7 * i + 0.3);
    auto                gw = prop.weighted_gradient_binned(w);
    for (size_t p = 0; p < np; p++) {
      double s = 0, sc = 0;
      for (int b = 0; b < 2; b++)
        for (int ab = 0; ab < NN; ab++)
          for (size_t k = 0; k < nbin; k++) {
            const double t = w[(b * NN + ab) * nbin + k] * G[((b * np + p) * NN + ab) * nbin + k];
            s += t;
            sc += std::fabs(t);
          }
      CHECK(std::fabs(gw[p] - s) <= tol_w * sc);
    }

    // avg_path_grad: GL average of prob_path_grad, and finite differences
    std::vector<double> edges;
    for (int i = 0; i <= 15; i++) edges.push_back(0.5 + 0.3 * i);
    std::vector<double> A, dA;
    prop.avg_path_grad(edges, 5, refcmp::test_path(), false, A, dA);
    CHECK(A == prop.avg_path(edges, 5, refcmp::test_path(), false));
    const size_t nb1 = edges.size() - 1;
    double       wa  = 0;
    for (size_t p = 0; p < np; p++) {
      const int idx = int(std::find(all.begin(), all.end(), names[p]) - all.begin());
      auto         q0 = Model::template cast<double>(par);
      const double x0 = Model::param_ref(q0, idx);
      const double h  = is_ev2(names[p]) ? std::min(std::max(std::fabs(x0), 1e-4) * 1e-6, 3e-9)
                                         : 1e-4 * model_scale<Model>(names[p]);
      auto         at = [&](double dx) {
        auto q                    = Model::template cast<double>(par);
        Model::param_ref(q, idx) += dx;
        opg::Propagator<Model> pp(opg::PremModel(), {}, 0);
        pp.set_params(q);
        return pp.avg_path(edges, 5, refcmp::test_path(), false);
      };
      auto   pl = at(h), mi = at(-h), p2 = at(2 * h), m2 = at(-2 * h);
      double m = 0, r = 0;
      for (size_t k = 0; k < NN * nb1; k++) {
        const double fd = (8 * (pl[k] - mi[k]) - (p2[k] - m2[k])) / (12 * h);
        m               = std::max(m, std::fabs(dA[p * NN * nb1 + k] - fd));
        r               = std::max(r, std::fabs(fd));
      }
      wa = std::max(wa, m / std::max(r, grad_floor<Model>(names[p])));
    }
    MESSAGE(std::string(Model::name) << " avg_path gradients vs double FD: " << wa);
    CHECK(wa < tol_fd);
  }

  /// Earth Z/A gradients (zoa_<type>) against long-double finite
  /// differences of the layer Z/A (PREM event list) and of the segment Z/A
  /// (fixed test path, all segments of type 0), together with a model
  /// parameter in the same pass.
  template <class Model> void check_zoa(opg::Propagator<Model>& prop, double tol)
  {
    using FL           = typename LongDouble<Model>::type;
    constexpr int NN   = Model::N * Model::N;
    auto          par  = param_points<Model>()[0].second;
    auto          zoas = prop.earth_parameter_names();
    REQUIRE(!zoas.empty());
    std::vector<std::string> names = {"dm31"};
    for (auto& z : zoas) names.push_back(z);
    prop.set_params(par);
    prop.set_gradient_params(names);
    const auto PL = FL::template prepare_generic<LD>(Model::template cast<LD>(par));
    const LD   h  = 1e-4L;
    auto       fd = [&](auto&& at) {
      auto p1 = at(h), m1 = at(-h), p2 = at(2 * h), m2 = at(-2 * h), p3 = at(3 * h),
           m3 = at(-3 * h);
      std::vector<LD> g(p1.size());
      for (size_t i = 0; i < g.size(); i++)
        g[i] = (45 * (p1[i] - m1[i]) - 9 * (p2[i] - m2[i]) + (p3[i] - m3[i])) / (60 * h);
      return g;
    };

    // PREM event list
    opg::PremModel      prem;
    Points              pts;
    std::vector<double> P, dP;
    prop.prob_points_grad(pts.E, pts.C, pts.nb, P, dP);
    LDEval<Model> base(prem);
    std::vector<std::vector<LD>> ref;
    ref.push_back(base.grads(par, {"dm31"}, pts.E, pts.C, pts.nbi)[0]);
    for (auto& z : zoas) {
      const int    t  = std::stoi(z.substr(4));
      const double z0 = prem.GetLayerZoA(t);
      ref.push_back(fd([&](LD dx) {
        opg::PremModel m = prem;
        m.SetLayerZoA(t, double(z0 + dx));
        return LDEval<Model>(m).probs(PL, pts.E, pts.C, pts.nbi);
      }));
    }
    size_t       w1 = 0;
    const double e1 = rel_err<Model>(dP, ref, pts.E.size(), NN, names, &w1);

    // fixed test path (all segments of layer type 0)
    std::vector<double> Ep;
    for (int i = 0; i < 60; i++) Ep.push_back(std::pow(10.0, -1 + 2.0 * i / 59));
    std::vector<int>    nbv(Ep.size(), 0);
    std::vector<double> Cd(Ep.size(), 0.0);
    prop.set_gradient_params({"zoa_0"});
    prop.prob_path_grad(Ep, refcmp::test_path(), false, P, dP);
    std::vector<std::vector<LD>> refp = {fd([&](LD dx) {
      auto path = refcmp::test_path();
      for (auto& sg : path) sg.zoa = double(LD(sg.zoa) + dx);
      return LDEval<Model>(prem, path).probs(PL, Ep, Cd, nbv);
    })};
    const double e2 = rel_err<Model>(dP, refp, Ep.size(), NN, {"zoa_0"});
    prop.prob_path_grad(Ep, refcmp::vacuum_path(), false, P, dP);
    double vmax = 0;
    for (double x : dP) vmax = std::max(vmax, std::fabs(x));

    MESSAGE(std::string(Model::name) << " Z/A gradients vs long-double FD: PREM " << e1
                                     << " (" << names[w1] << "), test path " << e2
                                     << ", vacuum max |dP| " << vmax);
    CHECK(e1 < tol);
    CHECK(e2 < tol);
    CHECK(vmax == 0);
  }

} // namespace gradtest

#endif
