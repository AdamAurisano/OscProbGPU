// Cost of production-height averaging (avg/prod_height.h) against the plain
// grid on a PREM (E, cosZ) grid over cosZ in [-1, 1].
//
// Usage: opg_bench_prod_height [nE] [nC] [devices|cpu] [reps]
//   Prints, for Fast and Sterile (3+1), the time per call of calculate() and
//   calculate_prod_height() with height histograms of 1, 5 and 20 bins (one
//   per cosine row), and of the weighted gradients (reverse mode) of both.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "opg/models/all.h"
#include "opg/propagator.h"

namespace {

  std::vector<int> parse_devices(const std::string& s)
  {
    std::vector<int> d;
    if (s == "cpu") return d;
    std::stringstream ss(s);
    std::string       tok;
    while (std::getline(ss, tok, ','))
      if (!tok.empty()) d.push_back(std::stoi(tok));
    return d;
  }

  double best_ms(int reps, const std::function<void()>& f)
  {
    using clk = std::chrono::steady_clock;
    f();  // warm-up
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
      auto t0 = clk::now();
      f();
      best = std::min(best, std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    }
    return best;
  }

  /// A histogram of nb bins on [0, 60] km, falling with height.
  void histogram(int nb, std::vector<double>& e, std::vector<double>& p)
  {
    e.clear();
    p.clear();
    for (int b = 0; b <= nb; b++) e.push_back(60.0 * b / nb);
    for (int b = 0; b < nb; b++) p.push_back(std::exp(-(b + 0.5) * 60.0 / nb / 15.0));
  }

  template <class Model>
  void run(const char* label, const typename Model::Params& par, int nE, int nC,
           const std::vector<int>& dev, int reps)
  {
    std::vector<double> E(nE), C(nC);
    for (int i = 0; i < nE; i++) E[i] = std::pow(10.0, -0.5 + 2.5 * i / (nE - 1.0));
    for (int i = 0; i < nC; i++) C[i] = -1 + 2.0 * i / (nC - 1.0);
    opg::Propagator<Model> prop(opg::PremModel(), dev);
    prop.set_grid(E, C);
    prop.set_params(par);
    std::vector<std::vector<double>> ed(3), pr(3);
    histogram(1, ed[0], pr[0]);
    histogram(5, ed[1], pr[1]);
    histogram(20, ed[2], pr[2]);
    prop.set_prod_heights(ed, pr);

    const double tp = best_ms(reps, [&] {
      prop.calculate();
      prop.wait();
    });
    std::printf("%-8s %dx%d %s: plain grid %.2f ms\n", label, nE, nC,
                dev.empty() ? "cpu" : "gpu", tp);
    double th[3];
    for (int k = 0; k < 3; k++) {
      prop.set_grid_prod_heights(std::vector<int>(size_t(nC), k));
      th[k] = best_ms(reps, [&] {
        prop.calculate_prod_height();
        prop.wait();
      });
      std::printf("    heights, %2d bins: %.2f ms (x%.2f)\n", int(pr[k].size()), th[k],
                  th[k] / tp);
    }
    if constexpr (opg::Propagator<Model>::has_gradients()) {
      prop.set_gradient_params();
      std::vector<double> w(2 * Model::N * Model::N * size_t(nE) * nC, 1e-3);
      const double tg = best_ms(reps, [&] { prop.weighted_gradient(w); });
      std::printf("    weighted gradient (%zu params, %s): plain %.2f ms\n",
                  prop.n_gradient_params(), prop.adjoint_gradients() ? "reverse" : "forward",
                  tg);
      for (int k = 0; k < 3; k++) {
        prop.set_grid_prod_heights(std::vector<int>(size_t(nC), k));
        const double t = best_ms(reps, [&] { prop.weighted_gradient_prod_height(w); });
        std::printf("    heights, %2d bins: %.2f ms (x%.2f)\n", int(pr[k].size()), t, t / tg);
      }
    }
  }

} // namespace

int main(int argc, char** argv)
{
  const int  nE   = argc > 1 ? std::atoi(argv[1]) : 400;
  const int  nC   = argc > 2 ? std::atoi(argv[2]) : 400;
  const auto dev  = parse_devices(argc > 3 ? argv[3] : "0");
  const int  reps = argc > 4 ? std::atoi(argv[4]) : 5;

  opg::MixingParams<3> m3;
  opg::set_standard_3nu(m3, std::asin(std::sqrt(0.303)), std::asin(std::sqrt(0.451)),
                        std::asin(std::sqrt(0.02225)), 232 * M_PI / 180, 7.41e-5, 2.507e-3);
  opg::Fast<double>::Params pf;
  pf.mix = m3;
  run<opg::Fast<double>>("Fast", pf, nE, nC, dev, reps);

  opg::Sterile<double>::Params ps;
  opg::set_std_pars(ps.mix);
  ps.mix.SetDm(4, 0.5);
  ps.mix.SetAngle(2, 4, 0.1);
  ps.mix.SetAngle(3, 4, 0.1);
  run<opg::Sterile<double>>("Sterile", ps, nE, nC, dev, reps);
  return 0;
}
