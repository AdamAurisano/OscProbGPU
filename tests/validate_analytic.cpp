// Validation of the analytic bin averages (avg/analytic.h) for 3+1 in the
// NOvA configurations: accuracy against a converged reference, gradients
// against finite differences, timings. Prints markdown tables.
//
//   opg_validate_analytic [quick | timing]
//
// Reference (uniform in L/E over each bin, single constant-density
// segment): at each node u = 1/E the eigensystem of H gives
// P_ab(u) = sum_ij X_i conj(X_j), X_i = V_bi conj(V_ai) exp(-i lam_i L).
// Pairs whose phase sweeps more than 1e6 rad across the bin are integrated
// asymptotically (first boundary term [c e^{-i phi} / (-i phi')], phi' from
// Hellmann-Feynman); all other terms by composite 10-node Gauss-Legendre
// with panels of <= 0.5 rad of the fastest kept phase. Convergence: the
// same with twice the panels. Only the clamped FD bin [1e-5, 0.3] GeV has
// asymptotic pairs.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "opg/propagator.h"

using namespace opg;
using M  = Sterile<double>;
using ML = Sterile<long double>;

namespace {

  M::Params nova(double dm41, double th14, bool io)
  {
    M::Params p;
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

  struct Eig {
      double      lam[4];
      Mat<4, double> V;
      double      dlam[4];  // d lam / du (Hellmann-Feynman)
  };

  Eig eig_sorted(const M::Prepared& P, const Mat<4, double>& A, double u, bool nubar,
                 const Segment<double>& s)
  {
    Mat<4, double> H, V;
    double         lam[4];
    M::hamiltonian(P, 1 / u, nubar, s, H);
    jacobi_hermitian<4, double>(H, V, lam);
    int idx[4] = {0, 1, 2, 3};
    std::sort(idx, idx + 4, [&](int a, int b) { return lam[a] < lam[b]; });
    Eig e;
    for (int k = 0; k < 4; k++) {
      e.lam[k] = lam[idx[k]];
      for (int i = 0; i < 4; i++) e.V(i, k) = V(i, idx[k]);
      Complex<double> d(0, 0);
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) d += conj(e.V(i, k)) * A(i, j) * e.V(j, k);
      e.dlam[k] = d.re;
    }
    return e;
  }

  /// X[a][b][i] = V_bi conj(V_ai) exp(-i lam_i L)
  void amplitudes(const Eig& e, double L, Complex<double> X[4][4][4])
  {
    Complex<double> ph[4];
    for (int i = 0; i < 4; i++) ph[i] = expi(-e.lam[i] * L);
    for (int a = 0; a < 4; a++)
      for (int b = 0; b < 4; b++)
        for (int i = 0; i < 4; i++) X[a][b][i] = e.V(b, i) * conj(e.V(a, i)) * ph[i];
  }

  struct RefResult {
      std::array<double, 16> P;
      bool                   asym = false;
  };

  RefResult ref_bin(const M::Prepared& P, const Mat<4, double>& A, const Segment<double>& s,
                    bool nubar, double ulo, double uhi, int scale)
  {
    const double L  = length_in_eV(s.length);
    const Eig    e0 = eig_sorted(P, A, ulo, nubar, s), e1 = eig_sorted(P, A, uhi, nubar, s);
    bool         fast[4][4] = {};
    double       maxrate    = 0;
    RefResult    res;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) {
        if (i == j) continue;
        const double r0 = std::fabs(e0.dlam[i] - e0.dlam[j]) * L;
        const double r1 = std::fabs(e1.dlam[i] - e1.dlam[j]) * L;
        if (std::min(r0, r1) * (uhi - ulo) > 1e6) { fast[i][j] = true; res.asym = true; }
        else maxrate = std::max(maxrate, std::max(r0, r1));
      }
    const long np = (long(std::ceil(maxrate * (uhi - ulo) / 0.5)) + 4) * scale;
    GaussLegendre       gl(10);
    std::vector<double> gx, gw;
    gl.map(0, 1, gx, gw);
    double acc[16] = {};
#pragma omp parallel
    {
      double loc[16] = {};
#pragma omp for schedule(static)
      for (long k = 0; k < np; k++) {
        const double a0 = ulo + (uhi - ulo) * double(k) / np;
        const double a1 = ulo + (uhi - ulo) * double(k + 1) / np;
        for (int q = 0; q < 10; q++) {
          const double    u = a0 + (a1 - a0) * gx[q], w = (a1 - a0) * gw[q];
          Complex<double> X[4][4][4];
          amplitudes(eig_sorted(P, A, u, nubar, s), L, X);
          for (int a = 0; a < 4; a++)
            for (int b = 0; b < 4; b++) {
              double v = 0;
              for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                  if (!fast[i][j]) v += (X[a][b][i] * conj(X[a][b][j])).re;
              loc[a * 4 + b] += w * v;
            }
        }
      }
#pragma omp critical
      for (int c = 0; c < 16; c++) acc[c] += loc[c];
    }
    // asymptotic boundary terms of the fast pairs
    for (int end = 0; end < 2; end++) {
      const Eig&      e = end ? e1 : e0;
      Complex<double> X[4][4][4];
      amplitudes(e, L, X);
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
          if (!fast[i][j]) continue;
          const double dphi = (e.dlam[i] - e.dlam[j]) * L;
          for (int a = 0; a < 4; a++)
            for (int b = 0; b < 4; b++) {
              // int c e^{-i phi} du ~ [c e^{-i phi} / (-i phi')]
              const Complex<double> t = X[a][b][i] * conj(X[a][b][j]) / Complex<double>(0, -dphi);
              acc[a * 4 + b] += (end ? 1.0 : -1.0) * t.re;
            }
        }
    }
    for (int c = 0; c < 16; c++) res.P[c] = acc[c] / (uhi - ulo);
    return res;
  }

  std::vector<double> fd_edges()  // E edges, with the clamped first bin
  {
    std::vector<double> e{1e-5};
    for (double ie = 1 / 0.3; ie > 1 / 44.7; ie -= 0.02237) e.push_back(1 / ie);
    e.push_back(44.7);
    e.push_back(120);
    return e;
  }
  std::vector<double> nd_edges()  // L/E edges, L = 1 km
  {
    std::vector<double> e;
    for (int i = 0; i <= 400; i++) e.push_back(0.005 * std::pow(1000.0, i / 400.0));
    return e;
  }

  struct Errs {
      double fd = 0, clamp = 0, nd = 0, conv = 0, gl5 = 0, gl20 = 0, fd8 = 0, nd8 = 0;
  };

  double now_ms()
  {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

} // namespace

int main(int argc, char** argv)
{
  const bool quick  = argc > 1 && std::strcmp(argv[1], "quick") == 0;
  const bool timing = argc > 1 && std::strcmp(argv[1], "timing") == 0;
  std::vector<double> dms = {1e-3, 1e-2, 0.1, 0.3, 1, 3, 10, 30, 100};
  if (quick) dms = {1e-3, 1, 100};

  const auto                   fe = fd_edges(), ne = nd_edges();
  const std::vector<Segment<double>> fd{{810, 2.84, 0.5, 0}}, nd{{1, 2.84, 0.5, 0}};
  const size_t                 nfb = fe.size() - 1, nnb = ne.size() - 1;
  Propagator<M>                prop;
  AnalyticAvgOptions           o8;
  o8.tol = 1e-8;

  std::printf("FD: %zu bins (first = clamped [1e-5, 0.3] GeV), %zu sub-bins (tol 1e-6), "
              "%zu (tol 1e-8); ND: %zu bins, %zu sub-bins\n\n",
              nfb, Propagator<M>::analytic_subbins(fe, fd, EMeasure::InvE, BinVar::E).size(),
              Propagator<M>::analytic_subbins(fe, fd, EMeasure::InvE, BinVar::E, o8).size(),
              nnb, Propagator<M>::analytic_subbins(ne, nd, EMeasure::InvE, BinVar::LoE).size());

  // ---- (b) accuracy ---------------------------------------------------------
  if (!timing) {
  std::printf("## Accuracy: max |Pbar - reference| over all channels, bins, nu/nubar, "
              "NO/IO, th14 in {0, 0.1}\n\n");
  std::printf("| dm41 | FD (150 bins) | FD clamped bin | ND (400 bins) | FD tol=1e-8 | "
              "ND tol=1e-8 | GL5 FD | GL20 FD | ref. convergence |\n");
  std::printf("|---|---|---|---|---|---|---|---|---|\n");
  for (double dm41 : dms) {
    Errs E;
    for (int nubar = 0; nubar < 2; nubar++)
      for (int io = 0; io < 2; io++)
        for (double th14 : {0.0, 0.1}) {
          prop.set_params(nova(dm41, th14, io));
          const M::Prepared P = M::prepare(nova(dm41, th14, io));
          Mat<4, double>    A;
          M::hamiltonian(P, 1.0, nubar, Segment<double>{0, 0, 0.5, -1}, A);
          hermitize_from_upper(A);
          auto af  = prop.avg_path_analytic(fe, fd, nubar, EMeasure::InvE, BinVar::E);
          auto an  = prop.avg_path_analytic(ne, nd, nubar, EMeasure::InvE, BinVar::LoE);
          auto af8 = prop.avg_path_analytic(fe, fd, nubar, EMeasure::InvE, BinVar::E, o8);
          auto an8 = prop.avg_path_analytic(ne, nd, nubar, EMeasure::InvE, BinVar::LoE, o8);
          // fixed-node GL in 1/E (regular FD bins only)
          std::vector<double> fe1(fe.begin() + 1, fe.end());
          auto g5  = prop.avg_path(fe1, 5, fd, nubar, EMeasure::InvE);
          auto g20 = prop.avg_path(fe1, 20, fd, nubar, EMeasure::InvE);
          for (size_t b = 0; b < nfb; b++) {
            const double ulo = 1 / fe[b + 1], uhi = 1 / fe[b];
            auto         r1  = ref_bin(P, A, fd[0], nubar, ulo, uhi, 1);
            auto         r2  = ref_bin(P, A, fd[0], nubar, ulo, uhi, 2);
            for (int c = 0; c < 16; c++) {
              const double ref = r2.P[c], ea = std::fabs(af[c * nfb + b] - ref);
              E.conv = std::max(E.conv, std::fabs(r1.P[c] - r2.P[c]));
              if (b == 0) E.clamp = std::max(E.clamp, ea);
              else {
                E.fd   = std::max(E.fd, ea);
                E.gl5  = std::max(E.gl5, std::fabs(g5[c * (nfb - 1) + b - 1] - ref));
                E.gl20 = std::max(E.gl20, std::fabs(g20[c * (nfb - 1) + b - 1] - ref));
              }
              E.fd8 = std::max(E.fd8, std::fabs(af8[c * nfb + b] - ref));
            }
          }
          for (size_t b = 0; b < nnb; b++) {
            auto r1 = ref_bin(P, A, nd[0], nubar, ne[b], ne[b + 1], 1);
            auto r2 = ref_bin(P, A, nd[0], nubar, ne[b], ne[b + 1], 2);
            for (int c = 0; c < 16; c++) {
              E.conv = std::max(E.conv, std::fabs(r1.P[c] - r2.P[c]));
              E.nd   = std::max(E.nd, std::fabs(an[c * nnb + b] - r2.P[c]));
              E.nd8  = std::max(E.nd8, std::fabs(an8[c * nnb + b] - r2.P[c]));
            }
          }
        }
    std::printf("| %g | %.1e | %.1e | %.1e | %.1e | %.1e | %.1e | %.1e | %.1e |\n", dm41, E.fd,
                E.clamp, E.nd, E.fd8, E.nd8, E.gl5, E.gl20, E.conv);
    std::fflush(stdout);
  }

  // ---- (c) gradients ----------------------------------------------------------
  std::printf("\n## Gradients vs 4-point central differences (long double)\n\n");
  std::printf("Relative error per parameter = max |dPbar/dp - FD| / max(max |FD|, floor) "
              "over channels and bins (floor: 1e-3 for angles and phases, 10 eV^-2 for "
              "splittings, as the repository's gradient tests; d14 has no effect at th14 = 0); "
              "worst over nu/nubar, NO/IO, th14 in {0, 0.1}. Steps: 1e-3 for "
              "angles/phases; dm: 1.27 h (L/E)max = 1e-3 (FD regular bins: (L/E)max = 2700, "
              "clamped bin: 8.1e7; ND: 5 km/GeV).\n\n");
  const auto names = M::param_names();
  std::printf("| dm41 | region |");
  for (auto& n : names) std::printf(" %s |", n.c_str());
  std::printf("\n|---|---|");
  for (size_t i = 0; i < names.size(); i++) std::printf("---|");
  std::printf("\n");
  prop.set_gradient_params();
  for (double dm41 : dms) {
    // regions: FD regular, FD clamped, ND
    double rel[3][12] = {}, absmax[3][12] = {};
    for (int nubar = 0; nubar < 2; nubar++)
      for (int io = 0; io < 2; io++)
        for (double th14 : {0.0, 0.1}) {
          const auto p = nova(dm41, th14, io);
          prop.set_params(p);
          for (int reg = 0; reg < 3; reg++) {
            std::vector<double> edges;
            if (reg == 0) edges.assign(fe.begin() + 1, fe.end());
            else if (reg == 1) edges.assign(fe.begin(), fe.begin() + 2);
            else edges = ne;
            const BinVar var  = reg == 2 ? BinVar::LoE : BinVar::E;
            const auto&  path = reg == 2 ? nd : fd;
            const double xmax = reg == 0 ? 2700 : reg == 1 ? 8.1e7 : 5;
            const auto   sub  = Propagator<M>::analytic_subbins(edges, path, EMeasure::InvE, var);
            const size_t nb   = edges.size() - 1, n = 16 * nb;
            std::vector<double> Pv, dP;
            prop.avg_path_analytic_subbins_grad(sub, nb, path, nubar, Pv, dP);
            std::vector<Segment<long double>> pl{
                {path[0].length, path[0].density, path[0].zoa, 0}};
            for (size_t ip = 0; ip < names.size(); ip++) {
              auto              base = ML::cast<long double>(p);
              const long double v    = ML::param_ref<long double>(base, int(ip));
              const bool        dm   = names[ip].rfind("dm", 0) == 0;
              const long double h    = dm ? 1e-3 / (1.267 * xmax) : 1e-3;
              auto ev = [&](int s) {
                auto q                                = base;
                ML::param_ref<long double>(q, int(ip)) = v + s * h;
                std::vector<long double> o;
                analytic::average<ML>(ML::prepare_generic<long double>(q), nullptr, 0, sub, nb,
                                      pl, nubar, AnalyticAvgOptions{}, o, nullptr);
                return o;
              };
              auto   a = ev(-2), b = ev(-1), c = ev(1), d = ev(2);
              double emax = 0, gmax = 0;
              for (size_t i = 0; i < n; i++) {
                const double f = double((a[i] - 8 * b[i] + 8 * c[i] - d[i]) / (12 * h));
                emax           = std::max(emax, std::fabs(f - dP[ip * n + i]));
                gmax           = std::max(gmax, std::fabs(f));
              }
              absmax[reg][ip] = std::max(absmax[reg][ip], gmax);
              rel[reg][ip]    = std::max(rel[reg][ip], emax / std::max(gmax, dm ? 10.0 : 1e-3));
            }
          }
        }
    const char* rn[3] = {"FD", "FD clamped", "ND"};
    for (int reg = 0; reg < 3; reg++) {
      std::printf("| %g | %s |", dm41, rn[reg]);
      for (size_t ip = 0; ip < names.size(); ip++)
        std::printf(" %.0e |", rel[reg][ip]);
      std::printf("\n");
    }
    std::printf("| %g | max abs dPbar/dp (FD) |", dm41);
    for (size_t ip = 0; ip < names.size(); ip++) std::printf(" %.1e |", absmax[0][ip]);
    std::printf("\n");
    std::fflush(stdout);
  }

  }  // !timing

  // ---- (d) timing ---------------------------------------------------------------
  std::printf("\n## Timing per call (all 16 channels; 12 parameters with gradients)\n\n");
  std::printf("| case | threads | value [ms] | value + gradients [ms] |\n|---|---|---|---|\n");
  prop.set_params(nova(1.0, 0.1, false));
  prop.set_gradient_params();
  for (int thr : {1, 0}) {
    for (int c = 0; c < 4; c++) {
      AnalyticAvgOptions o;
      o.threads             = thr;
      if (c >= 2) o.tol     = 1e-8;
      const bool   isfd     = c % 2 == 0;
      const auto&  edges    = isfd ? fe : ne;
      const auto&  path     = isfd ? fd : nd;
      const BinVar var      = isfd ? BinVar::E : BinVar::LoE;
      auto         time_it  = [&](bool g) {
        std::vector<double> P, dP;
        double              best = 1e30;
        for (int rep = 0; rep < 5; rep++) {
          const double t0 = now_ms();
          if (g) prop.avg_path_analytic_grad(edges, path, false, P, dP, EMeasure::InvE, var, o);
          else P = prop.avg_path_analytic(edges, path, false, EMeasure::InvE, var, o);
          best = std::min(best, now_ms() - t0);
        }
        return best;
      };
      const double tv = time_it(false), tg = time_it(true);
      std::printf("| %s %zu bins, tol %g | %s | %.2f | %.1f |\n", isfd ? "FD" : "ND",
                  edges.size() - 1, o.tol, thr == 1 ? "1" : "all", tv, tg);
    }
  }
#ifdef _OPENMP
  std::printf("\n(all = %d OpenMP threads)\n", omp_get_max_threads());
#endif
  // GPUs: OPG_VALIDATE_DEVICES=0 (or 0,1, ...) with a CUDA build
  if (const char* env = std::getenv("OPG_VALIDATE_DEVICES")) {
    std::vector<int> devs;
    for (const char* q = env; *q;) {
      devs.push_back(std::atoi(q));
      while (*q && *q != ',') q++;
      if (*q) q++;
    }
    Propagator<M> gpu(PremModel(), devs);
    gpu.set_params(nova(1.0, 0.1, false));
    gpu.set_gradient_params();
    std::printf("\n| case | GPUs (%s) | value [ms] | value + gradients [ms] |\n|---|---|---|---|\n",
                env);
    for (int c = 0; c < 3; c++) {
      AnalyticAvgOptions o;
      if (c == 2) o.tol = 1e-8;
      const bool   isfd  = c != 1;
      const auto&  edges = isfd ? fe : ne;
      const auto&  path  = isfd ? fd : nd;
      const BinVar var   = isfd ? BinVar::E : BinVar::LoE;
      auto         time_it = [&](bool g) {
        std::vector<double> P, dP;
        double              best = 1e30;
        for (int rep = 0; rep < 5; rep++) {
          const double t0 = now_ms();
          if (g) gpu.avg_path_analytic_grad(edges, path, false, P, dP, EMeasure::InvE, var, o);
          else P = gpu.avg_path_analytic(edges, path, false, EMeasure::InvE, var, o);
          best = std::min(best, now_ms() - t0);
        }
        return best;
      };
      const double tv = time_it(false), tg = time_it(true);
      std::printf("| %s %zu bins, tol %g | %s | %.2f | %.1f |\n", isfd ? "FD" : "ND",
                  edges.size() - 1, o.tol, env, tv, tg);
    }
    // batched points on the first GPU, 6 gradients (host output)
    gpu.set_gradient_params({"th23", "th24", "th34", "d24", "dm31", "dm41"});
    std::printf("\n| batch (GPU %d) | points | value [ms] | value + 6 gradients [ms] | "
                "per point + grads [ms] |\n|---|---|---|---|---|\n", devs[0]);
    for (int c = 0; c < 2; c++) {
      const bool isfd = c == 0;
      auto       b    = gpu.analytic_batch(isfd ? fe : ne, isfd ? fd : nd, false, EMeasure::InvE,
                                           isfd ? BinVar::E : BinVar::LoE);
      for (size_t np : {size_t(100), size_t(1000), size_t(3000)}) {
        std::vector<M::Params> pts;
        for (size_t i = 0; i < np; i++)
          pts.push_back(nova(1e-3 * std::pow(1e5, double(i) / np), 0.1, false));
        std::vector<double> out, dout;
        double              tv = 1e30, tg = 1e30;
        for (int rep = 0; rep < 2; rep++) {
          double t0 = now_ms();
          gpu.avg_path_analytic_batch(b, pts, out);
          tv = std::min(tv, now_ms() - t0);
          t0 = now_ms();
          gpu.avg_path_analytic_batch(b, pts, out, &dout);
          tg = std::min(tg, now_ms() - t0);
        }
        std::printf("| %s | %zu | %.1f | %.1f | %.3f |\n", isfd ? "FD" : "ND", np, tv, tg,
                    tg / double(np));
      }
    }
  }
  return 0;
}
