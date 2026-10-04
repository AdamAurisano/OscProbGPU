// Gradients on the GPU backend.

#include <cstdlib>
#include <sstream>
#include <type_traits>

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

/// GPU vs CPU tolerance for grid gradients: round-off amplified by the
/// oscillation phases; the LIV test points have phases up to ~1e4 rad, where
/// both backends agree with the long-double reference to ~1e-11.
template <class Model> double gpu_cpu_tol()
{
  if constexpr (std::is_same_v<Model, opg::LIV<double>> ||
                std::is_same_v<Model, opg::SiderealLIV<double>>)
    return 2e-11;
  return 1e-12;
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
  CHECK(d / scale < gpu_cpu_tol<Model>());
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

TEST_CASE_TEMPLATE("GPU G3/G4 gradients match long-double finite differences", M,
                   gradtest::NSI, gradtest::NUNM, gradtest::Sterile,
                   gradtest::Decay, gradtest::LIV, gradtest::SNSI,
                   gradtest::Deco, gradtest::SiderealLIV, gradtest::OQS)
{
  opg::Propagator<M> gpu(opg::PremModel(), devs("OPG_TEST_DEVICES", "0"));
  gradtest::check_against_ld(gpu, 1e-9);
}

TEST_CASE_TEMPLATE("GPU G3/G4 probabilities unchanged with gradients on", M,
                   gradtest::NSI, gradtest::NUNM, gradtest::Sterile,
                   gradtest::Decay, gradtest::LIV, gradtest::SNSI,
                   gradtest::Deco, gradtest::SiderealLIV, gradtest::OQS)
{
  opg::Propagator<M> gpu(opg::PremModel(), devs("OPG_TEST_DEVICES", "0"));
  gradtest::check_values_unchanged(gpu);
}

TEST_CASE_TEMPLATE("GPU G3/G4 grid / event-list / weighted gradients are consistent", M,
                   gradtest::NSI, gradtest::NUNM, gradtest::Sterile,
                   gradtest::Decay, gradtest::LIV, gradtest::SNSI,
                   gradtest::Deco, gradtest::SiderealLIV, gradtest::OQS)
{
  opg::Propagator<M> gpu(opg::PremModel(), devs("OPG_TEST_DEVICES", "0"));
  gradtest::check_grid_and_weighted(gpu, 1e-13, 1e-13);
}

TEST_CASE_TEMPLATE("GPU G3/G4 gradients agree with the CPU backend; multi-GPU identical", M,
                   gradtest::NSI, gradtest::NUNM, gradtest::Sterile,
                   gradtest::Decay, gradtest::LIV, gradtest::SNSI,
                   gradtest::Deco, gradtest::SiderealLIV, gradtest::OQS)
{
  check_gpu_vs_cpu<M>();
}

TEST_CASE_TEMPLATE("GPU binned and avg_path gradients", M, gradtest::Fast,
                   gradtest::NSI, gradtest::NUNM, gradtest::Sterile, gradtest::Decay,
                   gradtest::LIV, gradtest::SNSI, gradtest::Deco, gradtest::SiderealLIV, gradtest::OQS)
{
  opg::Propagator<M> prop(opg::PremModel(), devs("OPG_TEST_DEVICES", "0"));
  gradtest::check_binned(prop, 1e-6, 1e-13);
}


TEST_CASE_TEMPLATE("GPU Earth Z/A gradients", M, gradtest::Fast, gradtest::NSI, gradtest::NUNM,
                   gradtest::Sterile, gradtest::Decay, gradtest::LIV, gradtest::SNSI,
                   gradtest::Deco, gradtest::SiderealLIV, gradtest::OQS)
{
  opg::Propagator<M> prop(opg::PremModel(), devs("OPG_TEST_DEVICES", "0"));
  gradtest::check_zoa(prop, 1e-9);
}


TEST_CASE_TEMPLATE("GPU reverse-mode weighted gradients equal forward mode; multi-GPU", M,
                   gradtest::Fast, gradtest::NSI, gradtest::Sterile, gradtest::LIV,
                   gradtest::SNSI, gradtest::SiderealLIV, gradtest::OQS)
{
  static_assert(opg::Propagator<M>::has_adjoint_gradients());
  for (auto d : {devs("OPG_TEST_DEVICES", "0"), devs("OPG_TEST_MULTI_DEVICES", "0,1")}) {
    if (opg::cuda_device_count() < int(d.size())) continue;
    opg::Propagator<M> prop(opg::PremModel(), d);
    gradtest::check_adjoint(prop, 1e-11);
  }
}

TEST_CASE("GPU weighted gradients from device-resident weights")
{
  // the device probabilities serve as weights already on the GPU(s)
  for (auto d : {devs("OPG_TEST_DEVICES", "0"), devs("OPG_TEST_MULTI_DEVICES", "0,1")}) {
    if (opg::cuda_device_count() < int(d.size())) continue;
    opg::Propagator<gradtest::NSI> prop(opg::PremModel(), d);
    prop.set_params(gradtest::param_points<gradtest::NSI>()[1].second);
    prop.set_gradient_params();
    std::vector<double> E, C;
    for (int i = 0; i < 53; i++) E.push_back(std::pow(10.0, -0.3 + 2.0 * i / 52));
    for (int i = 0; i < 31; i++) C.push_back(-1 + 2.0 * i / 30);
    prop.set_grid(E, C);
    prop.calculate();
    std::vector<double>       wh(prop.probs(), prop.probs() + 2 * 9 * E.size() * C.size());
    std::vector<const double*> wd;
    for (size_t k = 0; k < d.size(); k++) wd.push_back(prop.device_probs(int(k)));
    auto gh = prop.weighted_gradient(wh);
    auto gd = prop.weighted_gradient_device(wd);
    CHECK(gd == gh);
    CHECK(prop.weighted_gradient_device(wd, opg::Flavor::Neutrino) ==
          prop.weighted_gradient(wh, opg::Flavor::Neutrino));

    prop.set_bins({0.5, 1, 2, 5, 10, 20}, {-1, -0.6, -0.2, 0.3}, 3, 3);
    prop.calculate_binned();
    std::vector<double>       bh(prop.binned(), prop.binned() + 2 * 9 * 5 * 3);
    std::vector<const double*> bd;
    for (size_t k = 0; k < d.size(); k++) bd.push_back(prop.device_binned(int(k)));
    CHECK(prop.weighted_gradient_binned_device(bd) == prop.weighted_gradient_binned(bh));
    CHECK_THROWS_AS(prop.weighted_gradient_device({}), std::invalid_argument);
  }
}

#endif
