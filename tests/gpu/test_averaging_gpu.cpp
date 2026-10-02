// Phase 8: Gauss-Legendre bin averaging on the GPU backend.

#include <cstdlib>
#include <sstream>

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
} // namespace

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
