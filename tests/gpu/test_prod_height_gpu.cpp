// Production-height averaging (avg/prod_height.h) on the GPU backend: the
// shared checks (brute force, fixed heights, gradients, adjoint) and the GPU
// against the CPU backend.

#include <cstdlib>
#include <sstream>

#include "../binned.h"
#include "../prod_height.h"

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

  /// GPU against CPU: event lists (values, gradients, weighted gradients in
  /// both modes) and the grid, PREM and binned Earth, one and several GPUs.
  template <class Model> void check_gpu_cpu(double tol)
  {
    constexpr int N = Model::N, NN = N * N;
    for (auto d : {gpus(), devs("OPG_TEST_MULTI_DEVICES", "0,1")}) {
      if (opg::cuda_device_count() < int(d.size())) continue;
      for (int ie = 0; ie < 2; ie++) {
        const opg::EarthModel earth = ie == 0 ? opg::EarthModel(opg::PremModel())
                                              : opg::EarthModel(bintest::config(0));
        opg::Propagator<Model> gpu(earth, d), cpu(earth);
        for (auto* p : {&gpu, &cpu}) {
          p->set_params(phtest::params<Model>());
          p->set_prod_heights(phtest::edges(), phtest::probs());
          p->set_gradient_params(p->gradient_parameter_names());
        }
        phtest::Events      ev(5, 7);
        std::vector<double> Pg, Gg, Pc, Gc;
        gpu.prob_points_grad_prod_height(ev.E, ev.C, ev.nb, ev.dist, Pg, Gg);
        cpu.prob_points_grad_prod_height(ev.E, ev.C, ev.nb, ev.dist, Pc, Gc);
        const double ep = std::max(phtest::max_abs_diff(Pg, Pc),
                                   phtest::max_abs_diff(gpu.prob_points_prod_height(
                                                            ev.E, ev.C, ev.nb, ev.dist),
                                                        Pc));
        double gs = 0;
        for (double x : Gc) gs = std::max(gs, std::fabs(x));
        const double eg = phtest::max_abs_diff(Gg, Gc) / std::max(gs, 1.0);
        // weighted, forward and reverse
        std::vector<double> w(NN * ev.E.size());
        for (size_t i = 0; i < w.size(); i++) w[i] = std::sin(0.37 * double(i));
        double ew = 0;
        for (bool adj : {false, true}) {
          gpu.set_adjoint_gradients(adj);
          cpu.set_adjoint_gradients(adj);
          auto a = gpu.weighted_gradient_points_prod_height(ev.E, ev.C, ev.nb, ev.dist, w);
          auto b = cpu.weighted_gradient_points_prod_height(ev.E, ev.C, ev.nb, ev.dist, w);
          double sc = 0;
          for (double x : b) sc = std::max(sc, std::fabs(x));
          ew = std::max(ew, phtest::max_abs_diff(a, b) / std::max(sc, 1.0));
        }
        // grid (per point distributions), values, gradients and weighted
        std::vector<double> E = {0.35, 1.1, 4.0, 15.0, 55.0}, C;
        for (int i = 0; i < 9; i++) C.push_back(-1 + 2.0 * i / 8);
        std::vector<int> dg;
        for (size_t ic = 0; ic < C.size(); ic++)
          for (size_t k = 0; k < E.size(); k++) dg.push_back(int((ic + 2 * k) % 6) - 1);
        double eG = 0, eGg = 0, eGw = 0;
        std::vector<double> wg(2 * NN * E.size() * C.size());
        for (size_t i = 0; i < wg.size(); i++) wg[i] = std::cos(0.11 * double(i));
        for (auto* p : {&gpu, &cpu}) {
          p->set_grid(E, C);
          p->set_grid_prod_heights(dg);
        }
        gpu.calculate_prod_height();
        cpu.calculate_prod_height();
        const size_t np = 2 * NN * E.size() * C.size();
        for (size_t i = 0; i < np; i++) eG = std::max(eG, std::fabs(gpu.probs()[i] - cpu.probs()[i]));
        gpu.calculate_prod_height(opg::Flavor::Both, true);
        cpu.calculate_prod_height(opg::Flavor::Both, true);
        const size_t ng = np * gpu.n_gradient_params();
        double       sg = 0;
        for (size_t i = 0; i < ng; i++) sg = std::max(sg, std::fabs(cpu.grad()[i]));
        for (size_t i = 0; i < ng; i++)
          eGg = std::max(eGg, std::fabs(gpu.grad()[i] - cpu.grad()[i]) / std::max(sg, 1.0));
        for (size_t i = 0; i < np; i++) eG = std::max(eG, std::fabs(gpu.probs()[i] - cpu.probs()[i]));
        for (bool adj : {false, true}) {
          gpu.set_adjoint_gradients(adj);
          cpu.set_adjoint_gradients(adj);
          auto   a  = gpu.weighted_gradient_prod_height(wg);
          auto   b  = cpu.weighted_gradient_prod_height(wg);
          double sc = 0;
          for (double x : b) sc = std::max(sc, std::fabs(x));
          eGw = std::max(eGw, phtest::max_abs_diff(a, b) / std::max(sc, 1.0));
        }
        MESSAGE(std::string(Model::name) << " N=" << N << std::string(ie ? " binned" : " PREM") << " on "
                << d.size() << " GPU(s) vs CPU: points P " << ep << ", dP " << eg
                << ", weighted " << ew << "; grid P " << eG << ", dP " << eGg << ", weighted "
                << eGw);
        CHECK(ep < tol);
        CHECK(eg < tol);
        CHECK(ew < tol);
        CHECK(eG < tol);
        CHECK(eGg < tol);
        CHECK(eGw < tol);
      }
    }
  }

} // namespace

TEST_CASE_TEMPLATE("GPU production heights: exact vs brute force, gradients", Model,
                   opg::Fast<double>, opg::NSI<double>, opg::NUNM<double>,
                   opg::Sterile<double>, opg::Sterile5<double>, opg::Sterile6<double>,
                   opg::LIV<double>, opg::SNSI<double>)
{
  opg::Propagator<Model> prop(opg::PremModel(), gpus());
  phtest::check_values(prop, opg::EarthModel(opg::PremModel()), 1e-11, "GPU PREM", 1e-12);
  phtest::check_values(prop, opg::EarthModel(bintest::config(0)), 1e-11, "GPU binned", 1e-12);
  if (Model::N <= 4) phtest::check_grads(prop, 1e-8, 1e-10, 1e-12);
}

TEST_CASE_TEMPLATE("GPU production heights: GPU vs CPU", Model, opg::Fast<double>,
                   opg::NSI<double>, opg::NUNM<double>, opg::Sterile<double>,
                   opg::Sterile5<double>, opg::Sterile6<double>, opg::LIV<double>,
                   opg::SNSI<double>)
{
  check_gpu_cpu<Model>(Model::N > 3 ? 2e-11 : 1e-12);
}
#endif
