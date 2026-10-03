// Throughput benchmark on a PREM (E, cosZ) grid.
//
// Usage: opg_bench [nE] [nC] [devices|cpu] [reps] [model]
//   devices: comma-separated CUDA ids (e.g. "0" or "0,1"), or "cpu"
//   model  : fast (default), nsi, nunm, sterile, decay, liv, snsi, binned, grad,
//            grad_nsi, grad_nunm, grad_sterile, grad_decay, grad_liv,
//            grad_snsi, grad_binned,
//            grad_points, or all

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
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

  template <class Model>
  void run_binned(typename Model::Params p, int nEb, int nCb, int ngl,
                  const std::vector<int>& dev, int reps)
  {
    std::vector<double> Ee(nEb + 1), Ce(nCb + 1);
    for (int i = 0; i <= nEb; i++) Ee[i] = std::pow(10.0, -0.5 + 2.5 * i / nEb);
    for (int i = 0; i <= nCb; i++) Ce[i] = -1 + 1.0 * i / nCb;
    opg::Propagator<Model> prop(opg::PremModel(), dev);
    prop.set_params(p);
    prop.set_bins(Ee, Ce, ngl, ngl, opg::EMeasure::Log);
    prop.calculate_binned();
    prop.binned();
    using clk = std::chrono::steady_clock;
    double t  = 1e30;
    for (int r = 0; r < reps; r++) {
      p.mix.th[1][2] *= 1.0 + 1e-6;
      auto a = clk::now();
      prop.set_params(p);
      prop.calculate_binned();
      prop.binned();
      t = std::min(t, std::chrono::duration<double>(clk::now() - a).count());
    }
    std::printf("%-8s binned backend=%-5s %dx%d bins, %d GL nodes/dir/piece (layer-adapted): "
                "%.4fs per evaluation (incl. D2H)\n",
                Model::name, dev.empty() ? "cpu" : "cuda", nEb, nCb, ngl, t);
  }

  /// Binned gradients (default parameters): binned averages only, with all
  /// binned gradients, and weighted binned gradient.
  template <class Model>
  void run_grad_binned(typename Model::Params p, int nEb, int nCb, int ngl,
                       const std::vector<int>& dev, int reps)
  {
    std::vector<double> Ee(nEb + 1), Ce(nCb + 1);
    for (int i = 0; i <= nEb; i++) Ee[i] = std::pow(10.0, -0.5 + 2.5 * i / nEb);
    for (int i = 0; i <= nCb; i++) Ce[i] = -1 + 1.0 * i / nCb;
    opg::Propagator<Model> prop(opg::PremModel(), dev);
    prop.set_params(p);
    prop.set_bins(Ee, Ce, ngl, ngl, opg::EMeasure::Log);
    prop.set_gradient_params();
    const size_t np = prop.n_gradient_params();
    std::vector<double> w(2 * Model::N * Model::N * size_t(nEb) * nCb, 1e-3);
    prop.calculate_binned(opg::Flavor::Both, true);
    prop.binned_grad();
    prop.weighted_gradient_binned(w);
    using clk = std::chrono::steady_clock;
    double tp = 1e30, tg = 1e30, tw = 1e30;
    for (int r = 0; r < reps; r++) {
      p.mix.th[1][2] *= 1.0 + 1e-6;
      prop.set_params(p);
      auto a = clk::now();
      prop.calculate_binned();
      prop.binned();
      auto b = clk::now();
      prop.calculate_binned(opg::Flavor::Both, true);
      prop.binned_grad();
      auto c = clk::now();
      prop.weighted_gradient_binned(w);
      auto d = clk::now();
      tp = std::min(tp, std::chrono::duration<double>(b - a).count());
      tg = std::min(tg, std::chrono::duration<double>(c - b).count());
      tw = std::min(tw, std::chrono::duration<double>(d - c).count());
    }
    std::printf("%-8s binned grad backend=%-5s %dx%d bins, %d GL/dir/piece, %zu params: "
                "averages %.4fs, + all gradients %.4fs (%.1fx), weighted %.4fs (%.1fx)\n",
                Model::name, dev.empty() ? "cpu" : "cuda", nEb, nCb, ngl, np, tp, tg,
                tg / tp, tw, tw / tp);
  }

  /// Event-list gradients: probabilities only, weighted sum, and one
  /// gradient vector per analysis bin.
  template <class Model>
  void run_grad_points(typename Model::Params p, size_t n, int nbins,
                       const std::vector<int>& dev, int reps)
  {
    std::vector<double>  E(n), C(n), w(Model::N * Model::N * n);
    std::vector<uint8_t> nb(n);
    std::vector<int>     bin(n);
    std::mt19937_64      rng(1);
    std::uniform_real_distribution<double> u(0, 1);
    for (size_t i = 0; i < n; i++) {
      E[i]   = std::pow(10.0, -0.5 + 2.5 * u(rng));
      C[i]   = -u(rng);
      nb[i]  = uint8_t(i & 1);
      bin[i] = int(u(rng) * nbins);
    }
    for (auto& x : w) x = u(rng);
    opg::Propagator<Model> prop(opg::PremModel(), dev);
    prop.set_params(p);
    prop.set_gradient_params();
    prop.weighted_gradient_points_binned(E, C, nb, w, bin, nbins);
    using clk = std::chrono::steady_clock;
    double tp = 1e30, tw = 1e30, tb = 1e30;
    for (int r = 0; r < reps; r++) {
      auto a = clk::now();
      prop.prob_points(E, C, nb);
      auto b = clk::now();
      prop.weighted_gradient_points(E, C, nb, w);
      auto c = clk::now();
      prop.weighted_gradient_points_binned(E, C, nb, w, bin, nbins);
      auto d = clk::now();
      tp = std::min(tp, std::chrono::duration<double>(b - a).count());
      tw = std::min(tw, std::chrono::duration<double>(c - b).count());
      tb = std::min(tb, std::chrono::duration<double>(d - c).count());
    }
    std::printf("%-8s points grad backend=%-5s %zu events, %zu params: P %.4fs, "
                "weighted %.4fs (%.1fx), per-bin weighted (%d bins) %.4fs (%.1fx)\n",
                Model::name, dev.empty() ? "cpu" : "cuda", n, prop.n_gradient_params(),
                tp, tw, tw / tp, nbins, tb, tb / tp);
  }

  /// Gradient throughput: full grid gradients and weighted gradient.
  template <class Model>
  void run_grad(typename Model::Params p, int nE, int nC, const std::vector<int>& dev,
                int reps)
  {
    using R = typename Model::Real;
    std::vector<R> E(nE), C(nC);
    for (int i = 0; i < nE; i++) E[i] = R(std::pow(10.0, -0.5 + 2.5 * i / (nE - 1.0)));
    for (int i = 0; i < nC; i++) C[i] = R(-1 + 1.0 * i / (nC - 1.0));
    opg::Propagator<Model> prop(opg::PremModel(), dev);
    prop.set_grid(E, C);
    prop.set_params(p);
    auto names = Model::param_names();
    prop.set_gradient_params(names);
    std::vector<R> w(2 * Model::N * Model::N * size_t(nE) * nC, R(1e-3));
    using clk = std::chrono::steady_clock;
    prop.calculate(opg::Flavor::Both, true);
    prop.wait();
    prop.weighted_gradient(w);
    double tfull = 1e30, tw = 1e30, tp = 1e30;
    for (int r = 0; r < reps; r++) {
      p.mix.th[1][2] *= 1.0 + 1e-6;
      prop.set_params(p);
      auto a = clk::now();
      prop.calculate();
      prop.wait();
      auto b = clk::now();
      prop.calculate(opg::Flavor::Both, true);
      prop.wait();
      auto c = clk::now();
      prop.weighted_gradient(w);
      auto d = clk::now();
      tp    = std::min(tp, std::chrono::duration<double>(b - a).count());
      tfull = std::min(tfull, std::chrono::duration<double>(c - b).count());
      tw    = std::min(tw, std::chrono::duration<double>(d - c).count());
    }
    std::printf("%-8s grad   backend=%-5s grid=%dx%d x2, %zu params (K=%d): P only %.4fs, "
                "P+full grad %.4fs (%.1fx), weighted grad %.4fs (%.1fx)\n",
                Model::name, dev.empty() ? "cpu" : "cuda", nE, nC, names.size(),
                opg::grad_traits<Model>::K, tp, tfull, tfull / tp, tw, tw / tp);
    if (!dev.empty()) {
      // weights already on the GPU (here: the device probabilities)
      prop.calculate();
      prop.wait();
      std::vector<const R*> wd;
      for (size_t k = 0; k < dev.size(); k++) wd.push_back(prop.device_probs(int(k)));
      prop.weighted_gradient_device(wd);
      double td = 1e30;
      for (int r = 0; r < reps; r++) {
        auto a = clk::now();
        prop.weighted_gradient_device(wd);
        td = std::min(td, std::chrono::duration<double>(clk::now() - a).count());
      }
      std::printf("%-8s grad   weighted grad with device-resident weights %.4fs (%.1fx)\n",
                  Model::name, td, td / tp);
    }
  }

  template <class Model> typename Model::Params nominal_params()
  {
    typename Model::Params p;
    opg::set_std_pars(p.mix);
    return p;
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
  auto        want  = [&](const char* m) { return model == m || model == "all"; };

  if (want("fast")) {
    opg::Fast<double>::Params p;
    p.mix = nominal3();
    run<opg::Fast<double>>(p, nE, nC, dev, reps);
    if (!dev.empty()) {
      opg::Fast<float>::Params pf;
      pf.mix = nominal3();
      run<opg::Fast<float>>(pf, nE, nC, dev, reps);
    }
  }
  if (want("nsi")) {
    auto p = nominal_params<opg::NSI<double>>();
    p.SetEps(0, 1, 0.1, 0.3);
    p.SetEps(0, 2, 0.1, 0);
    run<opg::NSI<double>>(p, nE, nC, dev, reps);
  }
  if (want("nunm")) {
    auto p = nominal_params<opg::NUNM<double>>();
    p.SetAlpha(1, 0, 0.02, 0.1);
    run<opg::NUNM<double>>(p, nE, nC, dev, reps);
  }
  if (want("sterile")) {
    auto p = nominal_params<opg::Sterile<double>>();
    p.mix.SetDm(4, 1.0);
    p.mix.SetAngle(2, 4, 0.1);
    run<opg::Sterile<double>>(p, nE, nC, dev, reps);
  }
  if (want("decay")) {
    auto p = nominal_params<opg::Decay<double>>();
    p.SetAlpha3(1e-4);
    run<opg::Decay<double>>(p, nE, nC, dev, reps);
  }
  if (want("liv")) {
    auto p = nominal_params<opg::LIV<double>>();
    p.SetaT(0, 1, 3, 1e-21, 0.5);
    p.SetcT(1, 2, 4, 1e-22, 0);
    run<opg::LIV<double>>(p, nE, nC, dev, reps);
  }
  if (want("snsi")) {
    auto p = nominal_params<opg::SNSI<double>>();
    p.SetLowestMass(0.05);
    p.SetEps(0, 1, 0.3, 0.4);
    run<opg::SNSI<double>>(p, nE, nC, dev, reps);
  }
  if (want("grad_liv")) {
    auto p = nominal_params<opg::LIV<double>>();
    p.SetaT(0, 1, 3, 1e-21, 0.5);
    p.SetcT(1, 2, 4, 1e-22, 0);
    run_grad<opg::LIV<double>>(p, nE, nC, dev, reps);
  }
  if (want("grad_snsi")) {
    auto p = nominal_params<opg::SNSI<double>>();
    p.SetLowestMass(0.05);
    p.SetEps(0, 1, 0.3, 0.4);
    run_grad<opg::SNSI<double>>(p, nE, nC, dev, reps);
  }
  if (want("grad")) {
    run_grad<opg::Fast<double>>(nominal_params<opg::Fast<double>>(), nE, nC, dev, reps);
  }
  if (want("grad_nsi")) {
    auto p = nominal_params<opg::NSI<double>>();
    p.SetEps(0, 1, 0.1, 0.3);
    p.SetEps(0, 2, 0.1, 0);
    run_grad<opg::NSI<double>>(p, nE, nC, dev, reps);
  }
  if (want("grad_nunm")) {
    auto p = nominal_params<opg::NUNM<double>>();
    p.SetAlpha(1, 0, 0.02, 0.1);
    run_grad<opg::NUNM<double>>(p, nE, nC, dev, reps);
  }
  if (want("grad_sterile")) {
    auto p = nominal_params<opg::Sterile<double>>();
    p.mix.SetDm(4, 1.0);
    p.mix.SetAngle(2, 4, 0.1);
    run_grad<opg::Sterile<double>>(p, nE, nC, dev, reps);
  }
  if (want("grad_decay")) {
    auto p = nominal_params<opg::Decay<double>>();
    p.SetAlpha3(1e-4);
    run_grad<opg::Decay<double>>(p, nE, nC, dev, reps);
  }
  if (want("grad_binned")) {
    run_grad_binned<opg::Fast<double>>(nominal_params<opg::Fast<double>>(), 40, 20, 8,
                                       dev, reps);
  }
  if (want("grad_points")) {
    run_grad_points<opg::Fast<double>>(nominal_params<opg::Fast<double>>(), 1000000,
                                       800, dev, reps);
  }
  if (want("binned")) {
    run_binned<opg::Fast<double>>(nominal_params<opg::Fast<double>>(), 40, 20, 8,
                                  dev, reps);
  }
  return 0;
}
