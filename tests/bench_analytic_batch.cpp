// Timing of the batched analytic averages (Sterile, one segment) in the
// configuration of a NOvA-like 3+1 fit: rows = e, mu; device output.
//
//   opg_bench_analytic_batch [npts] [reps]   (env: OPG_BENCH_TOL, OPG_BENCH_CELLS)
//
// FD: 810 km, 150 bins in E over [0.1, 120] GeV; ND: 1 km, 400 bins in L/E
// over [0.005, 5] km/GeV; rho 2.84 g/cm^3, Z/A 0.5. Random parameters per
// point. Prints the best time per call over `reps` for 0, 1, 4 and 6
// gradients, and a checksum of the outputs.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "opg/propagator.h"

using namespace opg;
using M = Sterile<double>;

namespace {

  double now_ms()
  {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  std::vector<double> linspace(double a, double b, int n)
  {
    std::vector<double> e(size_t(n) + 1);
    for (int i = 0; i <= n; i++) e[size_t(i)] = a + (b - a) * i / n;
    return e;
  }

  std::vector<M::Params> points(size_t n, unsigned seed)
  {
    std::mt19937_64                        rng(seed);
    std::uniform_real_distribution<double> U(0, 1);
    std::vector<M::Params>                 pts(n);
    for (auto& p : pts) {
      p.mix.SetAngle(1, 2, 0.5872);
      p.mix.SetAngle(1, 3, 0.1485);
      p.mix.SetAngle(2, 3, 0.7 + 0.2 * U(rng));
      p.mix.SetAngle(1, 4, 0);
      p.mix.SetAngle(2, 4, std::asin(std::sqrt(0.3 * U(rng))));
      p.mix.SetAngle(3, 4, std::asin(std::sqrt(0.3 * U(rng))));
      p.mix.SetDelta(1, 3, 0);
      p.mix.SetDelta(2, 4, 2 * M_PI * U(rng));
      p.mix.SetDm(2, 7.5e-5);
      p.mix.SetDm(3, 2.4e-3 + 2e-4 * U(rng));
      p.mix.SetDm(4, std::pow(10, -2 + 4 * U(rng)));
    }
    return pts;
  }

} // namespace

int main(int argc, char** argv)
{
  const size_t npts = argc > 1 ? size_t(std::atol(argv[1])) : 1024;
  const int    reps = argc > 2 ? std::atoi(argv[2]) : 5;

  const std::vector<Segment<double>> fd{{810, 2.84, 0.5, 0}}, nd{{1, 2.84, 0.5, 0}};
  const auto                         fe = linspace(0.1, 120, 150), ne = linspace(0.005, 5, 400);
  AnalyticAvgOptions                 opt;
  opt.rows = 3;
  if (const char* t = std::getenv("OPG_BENCH_TOL")) opt.tol = std::atof(t);
  opt.cells = std::getenv("OPG_BENCH_CELLS") != nullptr;
  if (const char* c = std::getenv("OPG_BENCH_CELL_PIECES")) opt.cell_pieces = std::atoi(c);

  Propagator<M> prop(PremModel(), std::vector<int>{0});
  const auto    pts = points(npts, 12345);

  struct Case {
      const char* name;
      bool        fd, nubar;
  };
  const Case cases[] = {{"FD nu", true, false}, {"FD nubar", true, true},
                        {"ND nu", false, false}, {"ND nubar", false, true}};
  const std::vector<std::vector<std::string>> gsets = {
      {}, {"th24"}, {"th34", "th23", "d24", "dm31"},
      {"th24", "th34", "th23", "dm41", "d24", "dm31"}};

  std::printf("%zu points, tol %g, rows %u, cells %d\n\n", npts, opt.tol, opt.rows,
              int(opt.cells));
  std::printf("| handle | sub-bins | 0 grads [ms] | 1 grad | 4 grads | 6 grads | checksum |\n"
              "|---|---|---|---|---|---|---|\n");
  double tot[4] = {0, 0, 0, 0};
  for (const auto& c : cases) {
    auto b = prop.analytic_batch(c.fd ? fe : ne, c.fd ? fd : nd, c.nubar, EMeasure::InvE,
                                 c.fd ? BinVar::E : BinVar::LoE, opt, 0);
    std::printf("| %s | %zu |", c.name, b.sub.size());
    double sum = 0;
    for (size_t g = 0; g < gsets.size(); g++) {
      if (gsets[g].empty()) prop.set_gradient_params(std::vector<std::string>{});
      else prop.set_gradient_params(gsets[g]);
      const size_t nq = gsets[g].size(), nch = 16;
      double *     out = nullptr, *dout = nullptr;
      cudaMalloc(&out, npts * nch * b.nbins * sizeof(double));
      if (nq) cudaMalloc(&dout, npts * nq * nch * b.nbins * sizeof(double));
      double best = 1e30;
      for (int r = 0; r < reps + 1; r++) {
        cudaDeviceSynchronize();
        const double t0 = now_ms();
        prop.avg_path_analytic_batch(b, pts, out, dout);
        cudaDeviceSynchronize();
        if (r > 0) best = std::min(best, now_ms() - t0);
      }
      std::vector<double> h(npts * nch * b.nbins);
      cudaMemcpy(h.data(), out, h.size() * sizeof(double), cudaMemcpyDeviceToHost);
      for (double v : h) sum += v;
      if (nq) {
        h.resize(npts * nq * nch * b.nbins);
        cudaMemcpy(h.data(), dout, h.size() * sizeof(double), cudaMemcpyDeviceToHost);
        for (double v : h) sum += std::fabs(v) * 1e-6;
      }
      cudaFree(out);
      cudaFree(dout);
      tot[g] += best;
      std::printf(" %.1f |", best);
    }
    std::printf(" %.10e |\n", sum);
  }
  std::printf("| total | | %.1f | %.1f | %.1f | %.1f | |\n", tot[0], tot[1], tot[2], tot[3]);

  // weighted gradients (avg_path_analytic_batch_weighted), device weights
  std::printf("\nWeighted gradients [ms]\n\n| handle | 1 grad | 4 grads | 6 grads | checksum |\n"
              "|---|---|---|---|---|\n");
  double wt[4] = {0, 0, 0, 0};
  for (const auto& c : cases) {
    auto b = prop.analytic_batch(c.fd ? fe : ne, c.fd ? fd : nd, c.nubar, EMeasure::InvE,
                                 c.fd ? BinVar::E : BinVar::LoE, opt, 0);
    const size_t        nw = npts * 16 * b.nbins;
    std::vector<double> wh(nw);
    for (size_t i = 0; i < nw; i++) wh[i] = std::sin(0.37 * double(i));
    double* w = nullptr;
    cudaMalloc(&w, nw * sizeof(double));
    cudaMemcpy(w, wh.data(), nw * sizeof(double), cudaMemcpyHostToDevice);
    std::printf("| %s |", c.name);
    double sum = 0;
    for (size_t gi = 1; gi < gsets.size(); gi++) {
      prop.set_gradient_params(gsets[gi]);
      double* g = nullptr;
      cudaMalloc(&g, npts * gsets[gi].size() * sizeof(double));
      double best = 1e30;
      for (int r = 0; r < reps + 1; r++) {
        cudaDeviceSynchronize();
        const double t0 = now_ms();
        prop.avg_path_analytic_batch_weighted(b, pts, w, g);
        cudaDeviceSynchronize();
        if (r > 0) best = std::min(best, now_ms() - t0);
      }
      std::vector<double> h(npts * gsets[gi].size());
      cudaMemcpy(h.data(), g, h.size() * sizeof(double), cudaMemcpyDeviceToHost);
      for (double v : h) sum += std::fabs(v);
      cudaFree(g);
      wt[gi] += best;
      std::printf(" %.1f |", best);
    }
    cudaFree(w);
    std::printf(" %.10e |\n", sum);
  }
  std::printf("| total | %.1f | %.1f | %.1f | |\n", wt[1], wt[2], wt[3]);
  return 0;
}
