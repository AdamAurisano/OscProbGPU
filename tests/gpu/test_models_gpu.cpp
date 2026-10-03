// GPU backend: compare against the OscProb reference and the CPU backend.
//
// Devices are taken from OPG_TEST_DEVICES (comma-separated, default "0").
// Indices are relative to CUDA_VISIBLE_DEVICES (e.g. with CUDA_VISIBLE_DEVICES=6,7,
// "0" is physical GPU 6).

#include <cstdlib>
#include <sstream>

#include "doctest.h"

#include "../extras.h"
#include "../ref_compare.h"
#include "../variants.h"

#include "opg/models/all.h"

namespace {

  constexpr double kTolRef = 1e-10;  ///< GPU vs OscProb
  constexpr double kTolCPU = 1e-11;  ///< GPU vs CPU backend

  std::vector<int> test_devices(bool multi = false)
  {
    const char*      env = std::getenv(multi ? "OPG_TEST_MULTI_DEVICES"
                                             : "OPG_TEST_DEVICES");
    std::string      s   = env ? env : (multi ? "0,1" : "0");
    std::vector<int> d;
    std::stringstream ss(s);
    std::string       tok;
    while (std::getline(ss, tok, ','))
      if (!tok.empty()) d.push_back(std::stoi(tok));
    return d;
  }

  /// Max |a - b| between two backends on the PREM grid of the reference.
  template <class Model>
  double gpu_vs_cpu_grid(opg::Propagator<Model>& g, opg::Propagator<Model>& c)
  {
    auto E = npy::load<double>("grid_E_prem.npy");
    auto C = npy::load<double>("grid_cosZ.npy");
    g.set_grid(E.data, C.data);
    c.set_grid(E.data, C.data);
    g.calculate();
    c.calculate();
    const double* pg = g.probs();
    const double* pc = c.probs();
    double        m  = 0;
    for (size_t i = 0; i < 2 * Model::N * Model::N * E.size() * C.size(); i++) {
      double d = std::fabs(pg[i] - pc[i]);
      if (!(d <= m)) m = d;  // NaN-propagating max
    }
    return m;
  }

  template <class Model>
  void check_model(const typename Model::Params& p, const std::string& tag,
                   double tol_ref = kTolRef, double tol_cpu = kTolCPU)
  {
    opg::Propagator<Model> gpu(opg::PremModel(), test_devices());
    opg::Propagator<Model> cpu(opg::PremModel(), {});
    gpu.set_params(p);
    cpu.set_params(p);

    auto rt = refcmp::compare_path(gpu, tag + "_testpath.npy", refcmp::test_path());
    auto rv = refcmp::compare_path(gpu, tag + "_vacuum.npy", refcmp::vacuum_path());
    auto rp = refcmp::compare_prem(gpu, tag + "_prem.npy");
    auto re = refcmp::compare_points(gpu, tag + "_prem.npy");
    double dc = gpu_vs_cpu_grid(gpu, cpu);
    MESSAGE(tag << " [GPU]: testpath " << rt.max_abs << ", vacuum "
                << rv.max_abs << ", prem " << rp.max_abs << " ("
                << rp.n_exact << "/" << rp.n << " exact), points "
                << re.max_abs << ", GPU-CPU " << dc);
    CHECK(rt.max_abs < tol_ref);
    CHECK(rv.max_abs < tol_ref);
    CHECK(rp.max_abs < tol_ref);
    CHECK(re.max_abs < tol_ref);
    CHECK(dc < tol_cpu);
  }

} // namespace

TEST_CASE("GPU Fast matches OscProb and the CPU backend")
{
  opg::Fast<double>::Params p;
  p.mix = variants::nominal_mix<3>();
  check_model<opg::Fast<double>>(p, "fast");
  p.mix = variants::fast_io_mix();
  check_model<opg::Fast<double>>(p, "fast_io");
}

TEST_CASE("GPU NSI matches OscProb and the CPU backend")
{
  check_model<opg::NSI<>>(variants::nsi(), "nsi");
  check_model<opg::NSI<>>(variants::nsi_phases(), "nsi_phases");
}

TEST_CASE("GPU LIV matches OscProb and the CPU backend")
{
  check_model<opg::LIV<>>(variants::liv(), "liv");
  check_model<opg::LIV<>>(variants::liv_phases(), "liv_phases");
}

TEST_CASE("GPU SNSI matches OscProb and the CPU backend")
{
  check_model<opg::SNSI<>>(variants::snsi(), "snsi");
  check_model<opg::SNSI<>>(variants::snsi_io(), "snsi_io");
}

TEST_CASE("GPU Deco matches OscProb and the CPU backend")
{
  check_model<opg::Deco<>>(variants::deco(), "deco");
  check_model<opg::Deco<>>(variants::deco_power(), "deco_power");
}

TEST_CASE("GPU SiderealLIV matches OscProb and the CPU backend")
{
  check_model<opg::SiderealLIV<>>(variants::sidereal(), "sidereal");
  check_model<opg::SiderealLIV<>>(variants::sidereal_fixed(), "sidereal_fixed");
}

TEST_CASE("GPU SiderealLIV per-event azimuth and time in event lists")
{
  opg::Propagator<opg::SiderealLIV<>> prop(opg::PremModel(), test_devices());
  extratest::check_extras(prop, 1e-12);
}

TEST_CASE("GPU NUNM matches OscProb and the CPU backend")
{
  check_model<opg::NUNM<>>(variants::nunm(0), "nunm");
  check_model<opg::NUNM<>>(variants::nunm_phases(), "nunm_phases");
  check_model<opg::NUNM<>>(variants::nunm(1), "nunm_high");
}

TEST_CASE("GPU Sterile matches OscProb and the CPU backend")
{
  opg::Sterile<>::Params p;
  // Phases reach ~1e4 rad for Dm41 ~ 1 eV^2, so round-off (FMA on the GPU)
  // is amplified to ~1e-11; see the long-double CPU test.
  p.mix = variants::sterile_mix();
  check_model<opg::Sterile<>>(p, "sterile", 1e-10, 1e-10);
  p.mix = variants::sterile_phases_mix();
  check_model<opg::Sterile<>>(p, "sterile_phases", 1e-10, 1e-10);
}

TEST_CASE("GPU Decay matches OscProb and the CPU backend")
{
  check_model<opg::Decay<>>(variants::decay(), "decay", 1e-10, 1e-10);
  check_model<opg::Decay<>>(variants::decay_both(), "decay_both", 1e-10, 1e-10);
}

TEST_CASE("GPU Fast<float> is close to double precision")
{
  opg::Fast<float>::Params p;
  p.mix = variants::nominal_mix<3>();
  opg::Propagator<opg::Fast<float>> gpu(opg::PremModel(), test_devices());
  gpu.set_params(p);
  auto E  = npy::load<double>("grid_E_prem.npy");
  auto C  = npy::load<double>("grid_cosZ.npy");
  auto rf = npy::load<double>("fast_prem.npy");
  std::vector<float> Ef(E.data.begin(), E.data.end()),
      Cf(C.data.begin(), C.data.end());
  gpu.set_grid(Ef, Cf);
  gpu.calculate();
  const float* P  = gpu.probs();
  double       m  = 0;
  const size_t nE = E.size(), nC = C.size();
  for (int nb = 0; nb < 2; nb++)
    for (size_t ic = 0; ic < nC; ic++)
      for (size_t ie = 0; ie < nE; ie++)
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++)
          {
            double d = std::fabs(
                double(P[(((nb * 3 + a) * 3 + b) * nC + ic) * nE + ie]) -
                rf[((((nb * nC) + ic) * nE + ie) * 3 + a) * 3 + b]);
            if (!(d <= m)) m = d;  // NaN-propagating max
          }
  MESSAGE("Fast<float> max |dP| vs double reference: " << m);
  CHECK(m < 1e-3);  // single precision: for exploration only
}

TEST_CASE("Multi-GPU grid is identical to single GPU")
{
  auto multi = test_devices(true);
  if (opg::cuda_device_count() < int(multi.size())) {
    MESSAGE("skipping: fewer devices than OPG_TEST_MULTI_DEVICES");
    return;
  }
  opg::Fast<double>::Params p;
  p.mix = variants::nominal_mix<3>();
  opg::Propagator<opg::Fast<double>> one(opg::PremModel(), test_devices());
  opg::Propagator<opg::Fast<double>> many(opg::PremModel(), multi);
  one.set_params(p);
  many.set_params(p);
  auto E = npy::load<double>("grid_E_prem.npy");
  auto C = npy::load<double>("grid_cosZ.npy");
  one.set_grid(E.data, C.data);
  many.set_grid(E.data, C.data);
  one.calculate();
  many.calculate();
  const double* a = one.probs();
  const double* b = many.probs();
  size_t        ndiff = 0;
  for (size_t i = 0; i < 2 * 9 * E.size() * C.size(); i++) ndiff += (a[i] != b[i]);
  CHECK(ndiff == 0);

  // event list split across devices
  std::vector<double>  e, c;
  std::vector<uint8_t> nb;
  for (int i = 0; i < 1001; i++) {
    e.push_back(0.5 + 0.1 * i);
    c.push_back(-1 + 2.0 * i / 1000);
    nb.push_back(i % 2);
  }
  auto oa = one.prob_points(e, c, nb);
  auto ob = many.prob_points(e, c, nb);
  CHECK(oa == ob);
}
