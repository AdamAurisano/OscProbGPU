// Phase 8: Gauss-Legendre bin averaging on the GPU backend.

#include <cuda_runtime.h>

#include <cstdlib>
#include <sstream>
#include <utility>

#include "../averaging.h"

namespace {
  std::vector<int> devs(const char* var, const char* def)
  {
    const char*       env = std::getenv(var);
    std::stringstream ss(env ? env : def);
    std::string       tok;
    std::vector<int>  d;
    while (std::getline(ss, tok, ','))
      if (!tok.empty()) d.push_back(std::stoi(tok));
    return d;
  }

  /// calculate_binned_batch against binned_batch (one point at a time
  /// through calculate_binned on the same GPU): per-point Earths and
  /// gradients (model and Earth parameters, several passes), values only,
  /// no Earths, and the batch split in two. All bit for bit.
  template <class Model> void check_batch(const std::vector<int>& dev)
  {
    using Params = typename Model::Params;
    opg::Propagator<Model> prop(opg::PremModel(), dev);
    // PISCES's atmospheric binning (NuTauOsc's 1 x 5 plain rule), and a layer-adapted one
    for (int rule = 0; rule < 2; rule++) {
      CAPTURE(rule);
      std::vector<double> Ee, Ce;
      for (int i = 0; i <= 40; i++) Ee.push_back(std::pow(10.0, 2.0 * i / 40));
      for (int i = 0; i <= 20; i++) Ce.push_back(-1 + 1.0 * i / 20);
      if (rule == 0) prop.set_bins(Ee, Ce, 1, 5, opg::EMeasure::Linear, opg::CosZRule::Plain);
      else prop.set_bins(Ee, Ce, 3, 4, opg::EMeasure::Log, opg::CosZRule::LayerAdapted);
      std::vector<std::string> g  = prop.default_gradient_params();
      const auto               en = prop.earth_parameter_names();
      g.push_back(en.front());  // zoa of the first layer type
      g.push_back(en.back());   // rho of the last
      const int tFirst = std::stoi(en.front().substr(4)), tLast = std::stoi(en.back().substr(4));
      prop.set_gradient_params(g);

      const int                   np = 7;
      std::vector<Params>         pts(np);
      std::vector<opg::PremModel> earths(np, prop.earth());
      const auto                  names = Model::param_names();
      for (int p = 0; p < np; p++) {
        for (size_t i = 0; i < names.size(); i++) {
          double& v = Model::template param_ref<double>(pts[size_t(p)], int(i));
          v         = v == 0 ? 1e-3 * (p + 1) : v * (1 + 0.01 * (p - 3) * (int(i % 3) - 1));
        }
        earths[size_t(p)].ScaleLayerDensity(tLast, 1 + 0.02 * (p - 3));
        earths[size_t(p)].SetLayerZoA(tFirst, 0.47 + 0.005 * p);
      }
      const size_t per = 2 * size_t(Model::N) * Model::N * 20 * 40, nq = g.size();

      std::vector<double> refP, refG;
      prop.binned_batch(pts, &earths, refP, &refG);

      auto run = [&](const std::vector<opg::PremModel>* e, bool grads,
                     const std::vector<size_t>& split) {
        double *dP = nullptr, *dG = nullptr;
        REQUIRE(cudaMalloc(&dP, np * per * sizeof(double)) == cudaSuccess);
        if (grads) REQUIRE(cudaMalloc(&dG, np * per * nq * sizeof(double)) == cudaSuccess);
        size_t p0 = 0;
        for (size_t n : split) {
          std::vector<Params>         sp(pts.begin() + long(p0), pts.begin() + long(p0 + n));
          std::vector<opg::PremModel> se;
          if (e) se.assign(e->begin() + long(p0), e->begin() + long(p0 + n));
          prop.calculate_binned_batch(sp, e ? &se : nullptr, dP + p0 * per,
                                      grads ? dG + p0 * per * nq : nullptr);
          p0 += n;
        }
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        std::pair<std::vector<double>, std::vector<double>> r;
        r.first.resize(np * per);
        cudaMemcpy(r.first.data(), dP, r.first.size() * sizeof(double), cudaMemcpyDeviceToHost);
        if (grads) {
          r.second.resize(np * per * nq);
          cudaMemcpy(r.second.data(), dG, r.second.size() * sizeof(double),
                     cudaMemcpyDeviceToHost);
        }
        cudaFree(dP);
        cudaFree(dG);
        return r;
      };
      auto ndiff = [](const std::vector<double>& a, const std::vector<double>& b) {
        size_t n = 0;
        double m = 0;
        for (size_t i = 0; i < a.size(); i++) {
          n += (a[i] != b[i]);
          m = std::max(m, std::fabs(a[i] - b[i]));
        }
        if (n) MESSAGE(n << " of " << a.size() << " differ, max " << m);
        return n;
      };
      const auto all = run(&earths, true, {size_t(np)});
      CHECK(ndiff(all.first, refP) == 0);
      CHECK(ndiff(all.second, refG) == 0);
      const auto split = run(&earths, true, {3, 4});
      CHECK(ndiff(split.first, refP) == 0);
      CHECK(ndiff(split.second, refG) == 0);
      const auto vals = run(&earths, false, {size_t(np)});
      CHECK(ndiff(vals.first, refP) == 0);
      // no Earths: every point with earth()
      std::vector<opg::PremModel> same(np, prop.earth());
      std::vector<double>         sP, sG;
      prop.binned_batch(pts, &same, sP, &sG);
      const auto none = run(nullptr, true, {size_t(np)});
      CHECK(ndiff(none.first, sP) == 0);
      CHECK(ndiff(none.second, sG) == 0);
    }
  }
} // namespace

TEST_CASE("GPU batched bin averages equal single-point ones (Fast; Deco: separate probabilities)")
{
  const auto one = devs("OPG_TEST_DEVICES", "0");
  check_batch<opg::Fast<double>>(one);
  check_batch<opg::Deco<double>>(one);
}

TEST_CASE("GPU binned averages: reduction, CPU agreement, multi-GPU")
{
  auto one   = devs("OPG_TEST_DEVICES", "0");
  auto multi = devs("OPG_TEST_MULTI_DEVICES", "0,1");

  opg::Propagator<avgtest::Fast> gpu(opg::PremModel(), one);
  avgtest::setup(gpu);
  for (auto m : {opg::EMeasure::Linear, opg::EMeasure::Log, opg::EMeasure::InvE})
    avgtest::check_reduction(gpu, m);

  // Realistic atmospheric binning: 40 log E bins x 20 cosZ bins, 5x5 nodes
  std::vector<double> Ee, Ce;
  for (int i = 0; i <= 40; i++) Ee.push_back(std::pow(10.0, -0.5 + 2.5 * i / 40));
  for (int i = 0; i <= 20; i++) Ce.push_back(-1 + 2.0 * i / 20);

  opg::Propagator<avgtest::Fast> cpu;
  avgtest::setup(cpu);
  for (auto* p : {&gpu, &cpu}) {
    p->set_bins(Ee, Ce, 5, 5, opg::EMeasure::Log);
    p->calculate_binned();
  }
  const size_t n = 2 * 9 * 40 * 20;
  double       d = 0;
  for (size_t i = 0; i < n; i++)
    d = std::max(d, std::fabs(gpu.binned()[i] - cpu.binned()[i]));
  MESSAGE("binned GPU vs CPU: " << d);
  CHECK(d < 1e-11);

  if (opg::cuda_device_count() >= int(multi.size())) {
    opg::Propagator<avgtest::Fast> mg(opg::PremModel(), multi);
    avgtest::setup(mg);
    mg.set_bins(Ee, Ce, 5, 5, opg::EMeasure::Log);
    mg.calculate_binned();
    size_t ndiff = 0;
    for (size_t i = 0; i < n; i++) ndiff += (mg.binned()[i] != gpu.binned()[i]);
    CHECK(ndiff == 0);
  }
}
