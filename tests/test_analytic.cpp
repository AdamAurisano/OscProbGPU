// Analytic bin averages (avg/analytic.h, after OscProb's PMNS_Maltoni).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "doctest.h"

#include "maltoni_reference.h"
#include "opg/propagator.h"
#include "variants.h"

namespace antest {

  using Sterile = opg::Sterile<double>;
  using Fast    = opg::Fast<double>;
  using NSI     = opg::NSI<double>;
  using opg::BinVar;
  using opg::EMeasure;
  template <class R> using Path = std::vector<opg::Segment<R>>;

  /// NOvA-like 3+1 point (PISCES validation set).
  inline Sterile::Params nova_sterile(double dm41, double th14, bool io = false)
  {
    Sterile::Params p;
    p.mix.SetAngle(1, 2, 0.5872);
    p.mix.SetAngle(1, 3, 0.1485);
    p.mix.SetAngle(2, 3, 0.8);
    p.mix.SetAngle(1, 4, th14);
    p.mix.SetAngle(2, 4, std::asin(std::sqrt(0.05)));
    p.mix.SetAngle(3, 4, std::asin(std::sqrt(0.1)));
    p.mix.SetDelta(1, 3, 1.0);
    p.mix.SetDelta(1, 4, 0.3);
    p.mix.SetDelta(2, 4, M_PI / 2);
    p.mix.SetDm(2, 7.5e-5);
    p.mix.SetDm(3, io ? -2.45e-3 : 2.5e-3);
    p.mix.SetDm(4, dm41);
    return p;
  }

  /// Weight of the averaging measure as a function of u = 1/E.
  inline double measure_weight(EMeasure m, double u)
  {
    return m == EMeasure::Linear ? 1 / (u * u) : m == EMeasure::Log ? 1 / u : 1.0;
  }

  /// Brute-force averages out[a][b][bin] over 1/E bins [ulo, uhi]: composite
  /// 16-node GL in u with panels of at most dphi = 0.5 rad of the fastest
  /// phase (rate = max |dm^2| L / 2 per unit u).
  template <class Model>
  std::vector<double> brute(opg::Propagator<Model>& prop, const std::vector<double>& ulo,
                            const std::vector<double>& uhi, const Path<double>& path,
                            bool nubar, EMeasure m, double maxdm)
  {
    double L = 0;
    for (auto& s : path) L += s.length;
    const double         rate = maxdm * opg::length_in_eV(L) / (2 * opg::constants::kGeV2eV);
    opg::GaussLegendre   gl(16);
    std::vector<double>  x, w;
    const size_t         nb = ulo.size(), nch = size_t(Model::N) * Model::N;
    std::vector<double>  out(nch * nb);
    for (size_t b = 0; b < nb; b++) {
      const int np = int(std::ceil((uhi[b] - ulo[b]) * rate / 0.5)) + 4;
      std::vector<double> E, W;
      double              wsum = 0;
      for (int k = 0; k < np; k++) {
        gl.map(ulo[b] + (uhi[b] - ulo[b]) * k / np, ulo[b] + (uhi[b] - ulo[b]) * (k + 1) / np,
               x, w);
        for (int i = 0; i < 16; i++) {
          E.push_back(1 / x[i]);
          W.push_back(w[i] * measure_weight(m, x[i]));
          wsum += W.back();
        }
      }
      auto P = prop.prob_path(E, path, nubar);
      for (size_t ch = 0; ch < nch; ch++) {
        double acc = 0;
        for (size_t i = 0; i < E.size(); i++) acc += W[i] * P[ch * E.size() + i];
        out[ch * nb + b] = acc / wsum;
      }
    }
    return out;
  }

  inline double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b)
  {
    REQUIRE(a.size() == b.size());
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
  }

  /// NOvA FD true-energy bins (E edges), without the clamped first bin.
  inline std::vector<double> fd_edges()
  {
    std::vector<double> e;
    for (double ie = 1 / 0.3; ie > 1 / 44.7; ie -= 0.02237) e.push_back(1 / ie);
    e.push_back(44.7);
    e.push_back(120);
    return e;
  }

  /// u = 1/E bins of ascending E edges.
  inline void e_to_u(const std::vector<double>& e, std::vector<double>& lo,
                     std::vector<double>& hi)
  {
    lo.clear();
    hi.clear();
    for (size_t b = 0; b + 1 < e.size(); b++) {
      lo.push_back(1 / e[b + 1]);
      hi.push_back(1 / e[b]);
    }
  }

} // namespace antest

using namespace antest;

TEST_CASE("Analytic averages agree with OscProb PMNS_Maltoni on its sub-bins")
{
  // OscProb's AvgProbLoE: n = ceil(3 (dLoE/LoE)^0.8 LoE^0.3) sub-bins uniform
  // in L/E, weights 1 / LoE_k^2 (default precision 1e-4).
  auto run = [](auto& prop, double L, double rho, const double* loe, const double* dloe,
                size_t nb, const double* ref, const char* name) {
    using Prop = std::remove_reference_t<decltype(prop)>;
    constexpr int N = Prop::N;
    std::vector<opg::AnalyticSubBin> sub;
    for (size_t b = 0; b < nb; b++) {
      const int n = int(std::ceil(3 * std::pow(dloe[b] / loe[b], 0.8) * std::pow(loe[b], 0.3)));
      for (int k = 0; k < n; k++) {
        const double c = loe[b] - dloe[b] / 2 + (k + 0.5) * dloe[b] / n;
        sub.push_back({int(b), c / L, dloe[b] / n / (2 * L), 1 / (c * c), 0});
      }
    }
    const Path<double> path{{L, rho, 0.5, 0}};
    double             m = 0;
    for (int nubar = 0; nubar < 2; nubar++) {
      auto P = prop.avg_path_analytic_subbins(sub, nb, path, nubar);
      for (size_t b = 0; b < nb; b++)
        for (int a = 0; a < N; a++)
          for (int c = 0; c < N; c++)
            m = std::max(m, std::fabs(P[(a * N + c) * nb + b] -
                                      ref[((nubar * nb + b) * N + a) * N + c]));
    }
    MESSAGE(std::string(name) << ": max |opg - OscProb Maltoni| = " << m);
    CHECK(m < 1e-12);
  };

  opg::Propagator<Sterile> ps;
  Sterile::Params          sp;
  sp.mix = variants::sterile_phases_mix();
  ps.set_params(sp);
  run(ps, 810, 2.84, maltoni_ref::sterile_fd_loe, maltoni_ref::sterile_fd_dloe, 5,
      maltoni_ref::sterile_fd, "sterile_fd");
  sp.mix = variants::sterile_mix();
  ps.set_params(sp);
  run(ps, 1, 2.84, maltoni_ref::sterile_nd_loe, maltoni_ref::sterile_nd_dloe, 3,
      maltoni_ref::sterile_nd, "sterile_nd");
  opg::Propagator<Fast> pf;
  Fast::Params          fp;
  fp.mix = variants::nominal_mix<3>();
  pf.set_params(fp);
  run(pf, 1300, 2.8, maltoni_ref::fast_1300_loe, maltoni_ref::fast_1300_dloe, 4,
      maltoni_ref::fast_1300, "fast_1300");
}

TEST_CASE("Analytic averages match brute-force GL (NOvA FD and ND, 3+1)")
{
  opg::Propagator<Sterile> prop;
  const auto               fe = fd_edges();
  std::vector<double>      flo, fhi;
  e_to_u(fe, flo, fhi);
  std::vector<double> nde, nlo, nhi;  // ND: 400 log bins in L/E, L = 1 km
  for (int i = 0; i <= 400; i++) nde.push_back(0.005 * std::pow(1000.0, i / 400.0));
  for (int i = 0; i < 400; i++) { nlo.push_back(nde[i]); nhi.push_back(nde[i + 1]); }
  const Path<double> fd{{810, 2.84, 0.5, 0}}, nd{{1, 2.84, 0.5, 0}};
  for (double dm41 : {1e-3, 0.3, 30.0}) {
    for (int nubar = 0; nubar < 2; nubar++) {
      prop.set_params(nova_sterile(dm41, 0.1, nubar));  // IO for antineutrinos
      const double maxdm = std::max(dm41, 2.5e-3);
      auto a = prop.avg_path_analytic(fe, fd, nubar, EMeasure::InvE, BinVar::E);
      auto r = brute(prop, flo, fhi, fd, nubar, EMeasure::InvE, maxdm);
      const double efd = max_abs_diff(a, r);
      a = prop.avg_path_analytic(nde, nd, nubar, EMeasure::InvE, BinVar::LoE);
      r = brute(prop, nlo, nhi, nd, nubar, EMeasure::InvE, maxdm);
      const double end = max_abs_diff(a, r);
      MESSAGE("dm41 = " << dm41 << " nubar = " << nubar << ": FD " << efd << ", ND " << end);
      CHECK(efd < 3e-6);
      CHECK(end < 1e-8);
    }
  }
}

TEST_CASE("Analytic averages: measures, multi-segment paths, vacuum, other models")
{
  const std::vector<double> E{0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 20.0};
  std::vector<double>       lo, hi;
  e_to_u(E, lo, hi);

  SUBCASE("Fast: E, log E and 1/E measures, 3 segments")
  {
    opg::Propagator<Fast> prop;
    Fast::Params          p;
    p.mix = variants::nominal_mix<3>();
    prop.set_params(p);
    const Path<double> path{{1000, 2, 0.5, 0}, {1000, 4, 0.5, 1}, {1000, 2, 0.5, 0}};
    for (auto m : {EMeasure::Linear, EMeasure::Log, EMeasure::InvE})
      for (int nubar = 0; nubar < 2; nubar++) {
        auto   a = prop.avg_path_analytic(E, path, nubar, m);
        double d = max_abs_diff(a, brute(prop, lo, hi, path, nubar, m, 2.6e-3));
        MESSAGE("measure " << int(m) << " nubar " << nubar << ": " << d);
        CHECK(d < 3e-6);
      }
  }
  SUBCASE("NSI on 1300 km")
  {
    opg::Propagator<NSI> prop;
    prop.set_params(variants::nsi());
    const Path<double> path{{1300, 2.8, 0.5, 0}};
    auto   a = prop.avg_path_analytic(E, path, false, EMeasure::Linear);
    double d = max_abs_diff(a, brute(prop, lo, hi, path, false, EMeasure::Linear, 2.6e-3));
    MESSAGE("NSI: " << d);
    CHECK(d < 3e-6);
  }
  SUBCASE("Vacuum: exact with one sub-bin per bin")
  {
    opg::Propagator<Sterile> prop;
    prop.set_params(nova_sterile(1.0, 0.1));
    const Path<double>      path{{810, 0, 0.5, 0}};
    opg::AnalyticAvgOptions opt;
    opt.nsub = 1;
    auto   a = prop.avg_path_analytic(E, path, false, EMeasure::InvE, BinVar::E, opt);
    double d = max_abs_diff(a, brute(prop, lo, hi, path, false, EMeasure::InvE, 1.0));
    MESSAGE("vacuum: " << d);
    CHECK(d < 1e-11);
  }
  SUBCASE("Sub-bins, edges in L/E, threads")
  {
    opg::Propagator<Sterile> prop;
    prop.set_params(nova_sterile(1.0, 0.1));
    const Path<double>  path{{810, 2.84, 0.5, 0}};
    std::vector<double> X;  // the same bins in L/E
    for (size_t i = E.size(); i-- > 0;) X.push_back(810 / E[i]);
    auto a = prop.avg_path_analytic(E, path, false, EMeasure::InvE, BinVar::E);
    auto b = prop.avg_path_analytic(X, path, false, EMeasure::InvE, BinVar::LoE);
    const size_t nb = E.size() - 1;
    double       d  = 0;
    for (size_t ch = 0; ch < 16; ch++)
      for (size_t i = 0; i < nb; i++)
        d = std::max(d, std::fabs(a[ch * nb + i] - b[ch * nb + (nb - 1 - i)]));
    CHECK(d < 1e-14);
    opg::AnalyticAvgOptions one;
    one.threads = 1;
    CHECK(prop.avg_path_analytic(E, path, false, EMeasure::InvE, BinVar::E, one) == a);
    // r = sqrt(1e-6 / (0.01 sum V L)), geometric
    auto sub = opg::Propagator<Sterile>::analytic_subbins(E, path, EMeasure::InvE, BinVar::E);
    CHECK(sub.size() > E.size());
    CHECK_THROWS_AS(prop.avg_path_analytic({1.0, 0.5}, path, false), std::invalid_argument);
  }
}

#ifndef OPG_DISABLE_GRADIENTS
namespace antest {

  template <class Model> struct LongDouble;
  template <template <class> class M, class R> struct LongDouble<M<R>> {
      using type = M<long double>;
  };

  /// dP/dp of the analytic averages vs 4-point central differences in long
  /// double (prepared state built in long double from the shifted
  /// parameters). Steps: 1e-3 for angles and phases, 1e-3 / (1.267 xmax)
  /// for splittings (xmax = largest L/E in the bins). Returns the largest
  /// error relative to max(max |dP/dp|, floor) over parameters.
  template <class Model>
  double check_grad(const typename Model::Params& p, const std::vector<opg::AnalyticSubBin>& sub,
                    size_t nb, const Path<double>& path, bool nubar, double xmax,
                    const char* label)
  {
    using ML = typename LongDouble<Model>::type;
    opg::Propagator<Model> prop;
    prop.set_params(p);
    prop.set_gradient_params();
    std::vector<double> P, dP;
    prop.avg_path_analytic_subbins_grad(sub, nb, path, nubar, P, dP);
    CHECK(prop.avg_path_analytic_subbins(sub, nb, path, nubar) == P);

    Path<long double> pl;
    for (auto& s : path) pl.push_back({s.length, s.density, s.zoa, s.layer});
    const auto   all   = Model::param_names();
    const auto   names = prop.gradient_params();
    const size_t n     = size_t(Model::N) * Model::N * nb;
    double       worst = 0;
    for (size_t ip = 0; ip < names.size(); ip++) {
      const int idx = int(std::find(all.begin(), all.end(), names[ip]) - all.begin());
      auto base        = ML::template cast<long double>(p);
      const long double v = ML::template param_ref<long double>(base, idx);
      const bool        dm = names[ip].rfind("dm", 0) == 0;
      const long double h  = dm ? 1e-3 / (1.267 * xmax) : 1e-3;
      auto ev = [&](int s) {
        auto q = base;
        ML::template param_ref<long double>(q, idx) = v + s * h;
        std::vector<long double> o;
        opg::analytic::average<ML>(ML::template prepare_generic<long double>(q), nullptr, 0,
                                   sub, nb, pl, nubar, opg::AnalyticAvgOptions{}, o, nullptr);
        return o;
      };
      auto   a = ev(-2), b = ev(-1), c = ev(1), d = ev(2);
      double emax = 0, gmax = 0;
      for (size_t i = 0; i < n; i++) {
        const double fd = double((a[i] - 8 * b[i] + 8 * c[i] - d[i]) / (12 * h));
        emax            = std::max(emax, std::fabs(fd - dP[ip * n + i]));
        gmax            = std::max(gmax, std::fabs(fd));
      }
      const double rel = emax / std::max(gmax, dm ? 10.0 : 1e-3);
      worst            = std::max(worst, rel);
      if (rel > 1e-6)
        MESSAGE(std::string(label) << " " << names[ip] << ": max|dP| " << gmax << ", error " << emax);
    }
    MESSAGE(std::string(label) << ": worst relative gradient error " << worst);
    return worst;
  }

} // namespace antest

TEST_CASE("Analytic average gradients match long-double finite differences")
{
  const auto          fe = fd_edges();
  std::vector<double> fsub(fe.begin(), fe.begin() + 30);  // 0.3 .. ~0.6 GeV
  fsub.push_back(5);
  fsub.push_back(44.7);
  fsub.push_back(120);
  const Path<double> fd{{810, 2.84, 0.5, 0}}, nd{{1, 2.84, 0.5, 0}};
  auto fs = opg::Propagator<Sterile>::analytic_subbins(fsub, fd, EMeasure::InvE, BinVar::E);
  std::vector<double> nde;
  for (int i = 0; i <= 40; i++) nde.push_back(0.005 * std::pow(1000.0, i / 40.0));
  auto ns = opg::Propagator<Sterile>::analytic_subbins(nde, nd, EMeasure::InvE, BinVar::LoE);

  for (double dm41 : {1e-3, 2.5e-3, 1.0, 100.0})
    for (double th14 : {0.0, 0.1}) {
      char label[64];
      std::snprintf(label, sizeof label, "FD dm41=%g th14=%g", dm41, th14);
      CHECK(check_grad<Sterile>(nova_sterile(dm41, th14), fs, fsub.size() - 1, fd, false,
                                810 / 0.3, label) < 1e-6);
      std::snprintf(label, sizeof label, "ND dm41=%g th14=%g", dm41, th14);
      CHECK(check_grad<Sterile>(nova_sterile(dm41, th14, true), ns, nde.size() - 1, nd, true,
                                5, label) < 1e-6);
    }

  // Linear-in-E weight (beta != 0), multi-segment path, 3 flavours
  const std::vector<double> E{0.5, 1.0, 2.0, 5.0, 20.0};
  const Path<double> tp{{1000, 2, 0.5, 0}, {1000, 4, 0.5, 1}, {1000, 2, 0.5, 0}};
  auto sub = opg::Propagator<Fast>::analytic_subbins(E, tp, EMeasure::Linear, BinVar::E);
  Fast::Params fp;
  fp.mix = variants::nominal_mix<3>();
  CHECK(check_grad<Fast>(fp, sub, E.size() - 1, tp, false, 3000 / 0.5, "Fast 3 segments") <
        1e-6);
  CHECK(check_grad<NSI>(variants::nsi_phases(), sub, E.size() - 1, tp, true, 3000 / 0.5,
                        "NSI 3 segments") < 1e-6);
}

TEST_CASE("Analytic averages are smooth through the fade-out of fast pairs")
{
  // Clamped FD bin [1e-5, 0.3] GeV (L/E up to 8.1e7 km/GeV): sterile pairs
  // pass through the fade region. Check the dm41 derivative there.
  const Path<double>        fd{{810, 2.84, 0.5, 0}};
  const std::vector<double> e{1e-5, 0.3};
  auto sub = opg::Propagator<Sterile>::analytic_subbins(e, fd, EMeasure::InvE, BinVar::E);
  CHECK(check_grad<Sterile>(nova_sterile(10, 0.1), sub, 1, fd, false, 8.1e7,
                            "clamped bin dm41=10") < 1e-4);
}
#endif

#ifndef OPG_DISABLE_GRADIENTS
TEST_CASE("Analytic average gradients with respect to Earth density and Z/A")
{
  // Two segments of layer types 0 and 1: d/d ln rho_t and d/d zoa_t.
  opg::Propagator<Sterile> prop;
  prop.set_params(nova_sterile(1.0, 0.1));
  prop.set_gradient_params({"th24", "zoa_0", "rho_0", "zoa_1", "rho_1"});
  const Path<double>        path{{400, 2.6, 0.49, 0}, {410, 3.1, 0.5, 1}};
  const std::vector<double> E{0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 20.0};
  const size_t              nb = E.size() - 1, n = 16 * nb;
  std::vector<double>       P, dP;
  prop.avg_path_analytic_grad(E, path, true, P, dP, EMeasure::InvE);
  const auto sub = opg::Propagator<Sterile>::analytic_subbins(E, path, EMeasure::InvE, BinVar::E);
  using ML       = opg::Sterile<long double>;
  const auto pr  = ML::prepare(nova_sterile(1.0, 0.1));
  for (int q = 1; q < 5; q++) {
    const int  layer = (q - 1) / 2;
    const bool rho   = q % 2 == 0;
    auto       ev    = [&](int s) {
      const long double h = 1e-4;
      Path<long double> pl;
      for (auto& g : path) pl.push_back({g.length, g.density, g.zoa, g.layer});
      if (rho) pl[layer].density *= 1 + s * h;  // relative scale (d/d ln rho)
      else pl[layer].zoa += s * h;
      std::vector<long double> o;
      opg::analytic::average<ML>(pr, nullptr, 0, sub, nb, pl, true, opg::AnalyticAvgOptions{},
                                 o, nullptr);
      return o;
    };
    auto   a = ev(-2), b = ev(-1), c = ev(1), d = ev(2);
    double emax = 0, gmax = 0;
    for (size_t i = 0; i < n; i++) {
      const double fd = double((a[i] - 8 * b[i] + 8 * c[i] - d[i]) / (12 * 1e-4L));
      emax            = std::max(emax, std::fabs(fd - dP[q * n + i]));
      gmax            = std::max(gmax, std::fabs(fd));
    }
    MESSAGE(prop.gradient_params()[q] << ": max|dP| " << gmax << ", error " << emax);
    CHECK(emax < 1e-9 * std::max(gmax, 1e-3));
  }
}
#endif

#ifndef OPG_DISABLE_GRADIENTS
TEST_CASE("Batched analytic averages on the host equal per-point calls")
{
  opg::Propagator<Sterile> prop;
  prop.set_gradient_params({"th23", "th24", "th34", "d24", "dm31", "dm41"});
  const Path<double>        fd{{810, 2.84, 0.5, 0}};
  const std::vector<double> E{1e-5, 0.3, 0.5, 1.0, 2.0, 5.0, 44.7};
  std::vector<Sterile::Params> pts;
  for (double dm41 : {1e-3, 0.5, 30.0}) pts.push_back(nova_sterile(dm41, 0.1, dm41 > 1));
  auto b = prop.analytic_batch(E, fd, true, EMeasure::InvE, BinVar::E);
  CHECK_FALSE(b.on_device());
  std::vector<double> out, dout;
  prop.avg_path_analytic_batch(b, pts, out, &dout);
  const size_t nb = E.size() - 1, n = 16 * nb, nq = 6;
  for (size_t p = 0; p < pts.size(); p++) {
    prop.set_params(pts[p]);
    std::vector<double> P, dP;
    prop.avg_path_analytic_grad(E, fd, true, P, dP, EMeasure::InvE, BinVar::E);
    CHECK(std::equal(P.begin(), P.end(), out.begin() + p * n));
    CHECK(std::equal(dP.begin(), dP.end(), dout.begin() + p * nq * n));
  }
  std::vector<double> v;
  prop.avg_path_analytic_batch(b, pts, v);  // values only
  CHECK(v == out);
  double* none = nullptr;  // device output needs a handle on a GPU
  CHECK_THROWS_AS(prop.avg_path_analytic_batch(b, pts, out.data(), none), std::logic_error);
}
#endif
