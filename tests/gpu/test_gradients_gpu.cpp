// Gradients on the GPU backend.

#include <cstdlib>
#include <sstream>

#include "../gradients.h"

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
} // namespace

TEST_CASE("GPU: probabilities unchanged with gradients on")
{
  opg::Propagator<gradtest::Fast> gpu(opg::PremModel(), devs("OPG_TEST_DEVICES", "0"));
  gradtest::check_values_unchanged(gpu);
}

TEST_CASE("GPU Fast gradients match long-double finite differences")
{
  opg::Propagator<gradtest::Fast> gpu(opg::PremModel(), devs("OPG_TEST_DEVICES", "0"));
  gradtest::check_against_ld(gpu, 1e-9);
}

TEST_CASE("GPU grid / event-list / weighted gradients are consistent")
{
  opg::Propagator<gradtest::Fast> gpu(opg::PremModel(), devs("OPG_TEST_DEVICES", "0"));
  gradtest::check_grid_and_weighted(gpu, 1e-13, 1e-13);
}

template <class Model> void check_gpu_vs_cpu()
{
  auto one   = devs("OPG_TEST_DEVICES", "0");
  auto multi = devs("OPG_TEST_MULTI_DEVICES", "0,1");
  auto par = gradtest::param_points<Model>()[0].second;
  std::vector<double> E, C;
  for (int i = 0; i < 97; i++) E.push_back(std::pow(10.0, -0.5 + 2.5 * i / 96));
  for (int i = 0; i < 51; i++) C.push_back(-1 + 2.0 * i / 50);
  constexpr int NN = Model::N * Model::N;
  const size_t  n  = 2 * Model::param_names().size() * NN * E.size() * C.size();
  std::vector<double> w(2 * NN * E.size() * C.size());
  for (size_t i = 0; i < w.size(); i++) w[i] = std::cos(0.37 * i);

  auto run = [&](const std::vector<int>& d, std::vector<double>& g,
                 std::vector<double>& gw) {
    opg::Propagator<Model> p(opg::PremModel(), d);
    p.set_params(par);
    p.set_gradient_params(Model::param_names());
    p.set_grid(E, C);
    p.calculate(opg::Flavor::Both, true);
    g.assign(p.grad(), p.grad() + n);
    gw = p.weighted_gradient(w);
  };
  std::vector<double> gc, gwc, g1, gw1;
  run({}, gc, gwc);
  run(one, g1, gw1);
  double d = 0, scale = 0;
  for (size_t i = 0; i < n; i++) {
    d     = std::max(d, std::fabs(g1[i] - gc[i]));
    scale = std::max(scale, std::fabs(gc[i]));
  }
  MESSAGE(std::string(Model::name) << " grid gradients GPU vs CPU: max |diff| / max |grad| = " << d / scale);
  CHECK(d / scale < 1e-12);
  // weighted sums have cancellations: compare relative to sum |w * dP|
  const size_t npt = E.size() * C.size(), np = gwc.size();
  std::vector<double> sabs(np, 0);
  for (int b = 0; b < 2; b++)
    for (size_t p = 0; p < np; p++)
      for (int ab = 0; ab < NN; ab++)
        for (size_t k = 0; k < npt; k++)
          sabs[p] += std::fabs(w[(b * NN + ab) * npt + k] *
                               gc[((b * np + p) * NN + ab) * npt + k]);
  for (size_t p = 0; p < np; p++) {
    MESSAGE("weighted p=" << p << ": GPU-CPU = " << gw1[p] - gwc[p]
                          << ", sum|w dP| = " << sabs[p]);
    CHECK(std::fabs(gw1[p] - gwc[p]) <= 1e-13 * sabs[p]);
  }

  if (opg::cuda_device_count() >= int(multi.size())) {
    std::vector<double> gm, gwm;
    run(multi, gm, gwm);
    CHECK(gm == g1);
    for (size_t p = 0; p < gw1.size(); p++)
      CHECK(std::fabs(gwm[p] - gw1[p]) <= 1e-14 * sabs[p]);
  }
}

TEST_CASE("GPU gradients agree with the CPU backend; multi-GPU identical")
{
  check_gpu_vs_cpu<gradtest::Fast>();
}

TEST_CASE_TEMPLATE("GPU G3 gradients match long-double finite differences", M,
                   gradtest::NSI, gradtest::NUNM, gradtest::Sterile)
{
  opg::Propagator<M> gpu(opg::PremModel(), devs("OPG_TEST_DEVICES", "0"));
  gradtest::check_against_ld(gpu, 1e-9);
}

TEST_CASE_TEMPLATE("GPU G3 probabilities unchanged with gradients on", M,
                   gradtest::NSI, gradtest::NUNM, gradtest::Sterile)
{
  opg::Propagator<M> gpu(opg::PremModel(), devs("OPG_TEST_DEVICES", "0"));
  gradtest::check_values_unchanged(gpu);
}

TEST_CASE_TEMPLATE("GPU G3 grid / event-list / weighted gradients are consistent", M,
                   gradtest::NSI, gradtest::NUNM, gradtest::Sterile)
{
  opg::Propagator<M> gpu(opg::PremModel(), devs("OPG_TEST_DEVICES", "0"));
  gradtest::check_grid_and_weighted(gpu, 1e-13, 1e-13);
}

TEST_CASE_TEMPLATE("GPU G3 gradients agree with the CPU backend; multi-GPU identical", M,
                   gradtest::NSI, gradtest::NUNM, gradtest::Sterile)
{
  check_gpu_vs_cpu<M>();
}
#endif
