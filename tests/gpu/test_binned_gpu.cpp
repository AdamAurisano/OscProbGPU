// EarthModelBinned on the GPU backend: OscProb reference, CPU backend,
// per-event azimuths, region gradients, adjoint, batches with per-point
// binned Earths.

#include <cstdlib>
#include <sstream>
#include <type_traits>

#include <cuda_runtime.h>

#include "../binned.h"

#ifndef OPG_DISABLE_GRADIENTS
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

  std::vector<int> gpus() { return devs("OPG_TEST_DEVICES", "0"); }

  constexpr double kTolRef = 1e-10;  ///< GPU vs OscProb
  constexpr double kTolCPU = 1e-11;  ///< GPU vs CPU backend

  template <class Model>
  void check_ref(const typename Model::Params& par, const std::string& tag,
                 bool use_extra = false)
  {
    for (auto d : {gpus(), devs("OPG_TEST_MULTI_DEVICES", "0,1")}) {
      if (opg::cuda_device_count() < int(d.size())) continue;
      opg::Propagator<Model> gpu(opg::EarthModelBinned(), d), cpu{opg::EarthModelBinned()};
      for (int k = 0; k < 2; k++) {
        auto p = par;
        if constexpr (std::is_same<Model, opg::SiderealLIV<>>::value)
          p.SetAzimuth(k == 0 ? 30 : 250);
        gpu.set_params(p);
        cpu.set_params(p);
        refcmp::Result rg, re, cg, ce;
        bintest::compare_probs(gpu, tag, k, rg, re, use_extra);
        bintest::compare_probs(cpu, tag, k, cg, ce, use_extra);
        // GPU against CPU on the same grid
        const double* pg = gpu.probs();
        const double* pc = cpu.probs();
        double        m  = 0;
        for (size_t i = 0; i < 2 * Model::N * Model::N * gpu.n_cosines() * gpu.n_energies(); i++)
          m = std::max(m, std::fabs(pg[i] - pc[i]));
        MESSAGE(tag << " binned cfg " << k << " on " << d.size() << " GPU(s): grid vs OscProb "
                    << rg.max_abs << ", points " << re.max_abs << ", grid vs CPU " << m);
        CHECK(rg.max_abs < kTolRef);
        CHECK(re.max_abs < kTolRef);
        CHECK(m < kTolCPU);
      }
    }
  }

} // namespace

TEST_CASE("GPU EarthModelBinned probabilities match OscProb and the CPU backend")
{
  opg::Fast<>::Params p;
  p.mix = variants::nominal_mix<3>();
  check_ref<opg::Fast<>>(p, "fast");
  check_ref<opg::NSI<>>(variants::nsi_phases(), "nsi_phases");
  check_ref<opg::SiderealLIV<>>(variants::sidereal(), "sidereal", true);
}

TEST_CASE_TEMPLATE("GPU binned Earth: per-event azimuths equal per-propagator azimuths", M,
                   gradtest::Fast, gradtest::NSI)
{
  opg::Propagator<M> prop(opg::EarthModelBinned(), gpus());
  bintest::check_event_azimuth(prop);
}

TEST_CASE_TEMPLATE("GPU binned Earth region gradients match long-double FD", M, gradtest::Fast,
                   gradtest::NSI, gradtest::Sterile)
{
  opg::Propagator<M> prop(opg::EarthModelBinned(), gpus());
  bintest::check_region_grads(prop, 1e-9);
}

TEST_CASE_TEMPLATE("GPU binned Earth: adjoint equals forward, gradients consistent; vs CPU", M,
                   gradtest::Fast, gradtest::NSI, gradtest::Sterile)
{
  const double tol = std::is_same<M, gradtest::Sterile>::value ? 1e-10 : 1e-11;
  for (auto d : {gpus(), devs("OPG_TEST_MULTI_DEVICES", "0,1")}) {
    if (opg::cuda_device_count() < int(d.size())) continue;
    opg::Propagator<M> prop(bintest::config(1), d);
    gradtest::check_adjoint(prop, tol);
  }
  opg::Propagator<M> prop(bintest::config(1), gpus());
  gradtest::check_grid_and_weighted(prop, 1e-13);
  gradtest::check_values_unchanged(prop);

  // grid gradients against the CPU backend
  opg::Propagator<M> cpu(bintest::config(1));
  const auto         par = gradtest::param_points<M>()[0].second;
  std::vector<std::string> names = {"dm31", "th23", "zoa_7", "rho_2", "rho_7"};
  std::vector<double>      E, C;
  for (int i = 0; i < 17; i++) E.push_back(std::pow(10.0, -0.3 + 2.0 * i / 16));
  for (int i = 0; i < 13; i++) C.push_back(-1 + 2.0 * i / 12);
  double m = 0, s = 0;
  std::vector<double> G[2];
  int                 k = 0;
  for (auto* p : {&prop, &cpu}) {
    p->set_params(par);
    p->set_gradient_params(names);
    p->set_grid(E, C);
    p->calculate(opg::Flavor::Both, true);
    G[k++].assign(p->grad(), p->grad() + 2 * names.size() * M::N * M::N * E.size() * C.size());
  }
  for (size_t i = 0; i < G[0].size(); i++) {
    m = std::max(m, std::fabs(G[0][i] - G[1][i]));
    s = std::max(s, std::fabs(G[1][i]));
  }
  MESSAGE(std::string(M::name) << " binned Earth grid gradients GPU vs CPU: max diff " << m
                               << " (max |dP| " << s << ")");
  CHECK(m <= 1e-10 * s);
}

TEST_CASE("GPU calculate_binned_batch with per-point binned Earths equals binned_batch")
{
  using M = opg::Fast<>;
  opg::Propagator<M> prop(bintest::config(1), {gpus().front()});
  M::Params          p;
  p.mix = variants::nominal_mix<3>();
  prop.set_params(p);
  std::vector<double> Ee, Ce;
  for (int i = 0; i <= 12; i++) Ee.push_back(std::pow(10.0, 2.0 * i / 12));
  for (int i = 0; i <= 8; i++) Ce.push_back(-1 + 1.0 * i / 8);
  prop.set_bins(Ee, Ce, 3, 4, opg::EMeasure::Log);
  const std::vector<std::string> g = {"dm31", "th23", "zoa_7", "rho_3"};
  prop.set_gradient_params(g);
  const int                          np = 5;
  std::vector<M::Params>             pts(np, p);
  std::vector<opg::EarthModelBinned> earths(np, bintest::config(1));
  for (int k = 0; k < np; k++) {
    pts[size_t(k)].mix.dm[2] *= 1 + 0.01 * k;
    earths[size_t(k)].SetRegionZoA(7, 0.46 + 0.005 * k);
    earths[size_t(k)].ScaleRegionDensity(3, 1 - 0.01 * k);
  }
  std::vector<double> refP, refG;
  prop.binned_batch(pts, &earths, refP, &refG);
  const size_t per = refP.size() / np, nq = g.size();
  double *     dP = nullptr, *dG = nullptr;
  REQUIRE(cudaMalloc(&dP, refP.size() * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&dG, refG.size() * sizeof(double)) == cudaSuccess);
  prop.calculate_binned_batch(pts, &earths, dP, dG);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  std::vector<double> P(refP.size()), G(refG.size());
  cudaMemcpy(P.data(), dP, P.size() * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(G.data(), dG, G.size() * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(dP);
  cudaFree(dG);
  size_t nP = 0, nG = 0;
  for (size_t i = 0; i < P.size(); i++) nP += P[i] != refP[i];
  for (size_t i = 0; i < G.size(); i++) nG += G[i] != refG[i];
  MESSAGE("binned Earth batch of " << np << " points (" << per << " values, " << per * nq
                                   << " derivatives each): " << nP << " values and " << nG
                                   << " derivatives differ from binned_batch");
  CHECK(nP == 0);
  CHECK(nG == 0);
}

#endif  // OPG_DISABLE_GRADIENTS
