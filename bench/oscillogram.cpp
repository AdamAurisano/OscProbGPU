// Throughput benchmark on a PREM (E, cosZ) grid.
//
// Usage: opg_bench [nE] [nC] [devices|cpu] [reps] [model]
//   devices: comma-separated CUDA ids (e.g. "0" or "0,1"), or "cpu"
//   model  : fast (default)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

  opg::MixingParams<3> nominal3()
  {
    opg::MixingParams<3> p;
    opg::set_standard_3nu(p, std::asin(std::sqrt(0.303)),
                          std::asin(std::sqrt(0.451)),
                          std::asin(std::sqrt(0.02225)), 232 * M_PI / 180,
                          7.41e-5, 2.507e-3);
    return p;
  }

  template <class Model>
  void run(typename Model::Params p, int nE, int nC,
           const std::vector<int>& dev, int reps)
  {
    using R = typename Model::Real;
    std::vector<R> E(nE), C(nC);
    for (int i = 0; i < nE; i++) E[i] = R(std::pow(10.0, -0.5 + 2.5 * i / (nE - 1.0)));
    for (int i = 0; i < nC; i++) C[i] = R(-1 + 1.0 * i / (nC - 1.0));

    using clk = std::chrono::steady_clock;
    auto t0   = clk::now();
    opg::Propagator<Model> prop(opg::PremModel(), dev);
    prop.set_grid(E, C);
    auto t1 = clk::now();

    // warm-up
    prop.set_params(p);
    prop.calculate();
    prop.wait();

    double tcalc = 1e30, tcopy = 1e30;
    for (int r = 0; r < reps; r++) {
      // perturb a parameter as a fit would
      p.mix.th[1][2] *= 1.0 + 1e-6;
      auto a = clk::now();
      prop.set_params(p);
      prop.calculate(opg::Flavor::Both);
      prop.wait();
      auto b = clk::now();
      prop.probs();
      auto c = clk::now();
      tcalc = std::min(tcalc, std::chrono::duration<double>(b - a).count());
      tcopy = std::min(tcopy, std::chrono::duration<double>(c - b).count());
    }
    double npts = 2.0 * nE * nC;
    std::printf("%-8s %-6s backend=%-8s grid=%dx%d (x2 nu/nubar)  setup=%.3fs  "
                "calc=%.4fs  D2H=%.4fs  -> %.3g points/s\n",
                Model::name, sizeof(R) == 8 ? "double" : "float",
                dev.empty() ? "cpu" : "cuda", nE, nC,
                std::chrono::duration<double>(t1 - t0).count(), tcalc, tcopy,
                npts / tcalc);
  }

} // namespace

int main(int argc, char** argv)
{
  int         nE    = argc > 1 ? std::atoi(argv[1]) : 1000;
  int         nC    = argc > 2 ? std::atoi(argv[2]) : 1000;
  std::string devs  = argc > 3 ? argv[3] : "0";
  int         reps  = argc > 4 ? std::atoi(argv[4]) : 5;
  std::string model = argc > 5 ? argv[5] : "fast";
  auto        dev   = parse_devices(devs);

  if (model == "fast" || model == "all") {
    opg::Fast<double>::Params p;
    p.mix = nominal3();
    run<opg::Fast<double>>(p, nE, nC, dev, reps);
    if (!dev.empty()) {
      opg::Fast<float>::Params pf;
      pf.mix = nominal3();
      run<opg::Fast<float>>(pf, nE, nC, dev, reps);
    }
  }
  return 0;
}
