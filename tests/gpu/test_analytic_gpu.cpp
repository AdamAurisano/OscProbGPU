// Analytic bin averages (avg/analytic.h) on the GPU backend: agreement with
// the host computation, single and multi-GPU.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

#include "doctest.h"

#include "opg/propagator.h"
#include "../variants.h"

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

  opg::Sterile<>::Params nova(double dm41)
  {
    opg::Sterile<>::Params p;
    p.mix = variants::sterile_phases_mix();
    p.mix.SetDm(4, dm41);
    return p;
  }

  /// max |a - b| / max(max |b|, floor) over bins [b0, b1) of out[q][bin]
  double rel_diff(const std::vector<double>& a, const std::vector<double>& b, double floor,
                  size_t nb, size_t b0, size_t b1)
  {
    REQUIRE(a.size() == b.size());
    double m = 0, s = floor;
    for (size_t i = 0; i < a.size(); i++) {
      if (i % nb < b0 || i % nb >= b1) continue;
      m = std::max(m, std::fabs(a[i] - b[i]));
      s = std::max(s, std::fabs(b[i]));
    }
    return m / s;
  }

  /// GPU vs host, all bins; with clamped = true the first bin (L/E up to
  /// 8e7 km/GeV, phases up to ~1e10 rad, where double precision limits
  /// both to ~1e-10 in P) is checked separately with looser tolerances.
  template <class Model>
  void compare(opg::Propagator<Model>& gpu, opg::Propagator<Model>& cpu,
               const std::vector<double>& edges, const std::vector<opg::Segment<double>>& path,
               opg::EMeasure m, opg::BinVar var, const std::string& label, bool clamped = false)
  {
    opg::AnalyticAvgOptions host;
    host.host       = true;
    const size_t nb = edges.size() - 1, b0 = clamped ? 1 : 0;
    for (int nubar = 0; nubar < 2; nubar++) {
      std::vector<double> Pg, dPg, Pc, dPc;
      gpu.avg_path_analytic_grad(edges, path, nubar, Pg, dPg, m, var);
      cpu.avg_path_analytic_grad(edges, path, nubar, Pc, dPc, m, var, host);
      const double ep = rel_diff(Pg, Pc, 1.0, nb, b0, nb);
      const double eg = rel_diff(dPg, dPc, 1.0, nb, b0, nb);
      MESSAGE(label << " nubar " << nubar << ": P " << ep << ", dP (rel) " << eg);
      // round-off of phases up to ~1e6 rad (Sterile, dm41 = 100 eV^2 at the FD)
      // differs between the device and host sin/cos and FMA contraction
      CHECK(ep < 1e-10);
      CHECK(eg < 1e-9);
      if (clamped) {
        const double cp = rel_diff(Pg, Pc, 1.0, nb, 0, 1), cg = rel_diff(dPg, dPc, 1.0, nb, 0, 1);
        MESSAGE(label << " clamped bin: P " << cp << ", dP (rel) " << cg);
        CHECK(cp < 1e-9);
        CHECK(cg < 1e-4);
      }
      // values without gradients equal those with
      CHECK(gpu.avg_path_analytic(edges, path, nubar, m, var) == Pg);
    }
  }
} // namespace

TEST_CASE("GPU analytic averages agree with the host computation")
{
  for (auto d : {devs("OPG_TEST_DEVICES", "0"), devs("OPG_TEST_MULTI_DEVICES", "0,1")}) {
    opg::Propagator<opg::Sterile<>> gs(opg::PremModel(), d), cs;
    for (double dm41 : {1e-3, 1.0, 100.0}) {
      gs.set_params(nova(dm41));
      cs.set_params(nova(dm41));
      gs.set_gradient_params();
      cs.set_gradient_params();
      std::vector<double> fe{1e-5};
      for (double ie = 1 / 0.3; ie > 1 / 44.7; ie -= 0.02237) fe.push_back(1 / ie);
      fe.push_back(44.7);
      fe.push_back(120);
      compare(gs, cs, fe, {{810, 2.84, 0.5, 0}}, opg::EMeasure::InvE, opg::BinVar::E,
              "Sterile FD dm41 = " + std::to_string(dm41), true);
      std::vector<double> ne;
      for (int i = 0; i <= 400; i++) ne.push_back(0.005 * std::pow(1000.0, i / 400.0));
      compare(gs, cs, ne, {{1, 2.84, 0.5, 0}}, opg::EMeasure::InvE, opg::BinVar::LoE,
              "Sterile ND dm41 = " + std::to_string(dm41));
    }
    opg::Propagator<opg::Fast<>> gf(opg::PremModel(), d), cf;
    opg::Fast<>::Params          fp;
    fp.mix = variants::nominal_mix<3>();
    gf.set_params(fp);
    cf.set_params(fp);
    gf.set_gradient_params({"th23", "dm31", "zoa_0", "rho_1"});
    cf.set_gradient_params({"th23", "dm31", "zoa_0", "rho_1"});
    compare(gf, cf, {0.5, 1.0, 2.0, 5.0, 20.0},
            {{1000, 2, 0.5, 0}, {1000, 4, 0.5, 1}, {1000, 2, 0.5, 0}}, opg::EMeasure::Linear,
            opg::BinVar::E, "Fast 3 segments");
    opg::Propagator<opg::NSI<>> gn(opg::PremModel(), d), cn;
    gn.set_params(variants::nsi_phases());
    cn.set_params(variants::nsi_phases());
    gn.set_gradient_params();
    cn.set_gradient_params();
    compare(gn, cn, {0.5, 1.0, 2.0, 5.0, 20.0}, {{1300, 2.8, 0.5, 0}}, opg::EMeasure::Log,
            opg::BinVar::E, "NSI");
  }
}

TEST_CASE("GPU batched analytic averages: per-point agreement, bit-identity across batches")
{
  const int dev = devs("OPG_TEST_DEVICES", "0").front();
  using M       = opg::Sterile<>;
  opg::Propagator<M> prop(opg::PremModel(), {dev});
  const std::vector<std::string> q{"th23", "th24", "th34", "d24", "dm31", "dm41"};
  prop.set_gradient_params(q);
  std::vector<M::Params> pts;
  for (double dm41 : {1e-3, 0.02, 0.3, 1.0, 3.0, 30.0, 100.0}) {
    auto p = nova(dm41);
    p.mix.SetAngle(2, 4, 0.1 + 0.3 * std::log10(1e4 * dm41) / 5);
    pts.push_back(p);
  }
  std::vector<double> fe{1e-5};
  for (double ie = 1 / 0.3; ie > 1 / 44.7; ie -= 0.02237) fe.push_back(1 / ie);
  fe.push_back(44.7);
  fe.push_back(120);
  std::vector<double> ne;
  for (int i = 0; i <= 400; i++) ne.push_back(0.005 * std::pow(1000.0, i / 400.0));
  const std::vector<opg::Segment<double>> fd{{810, 2.84, 0.5, 0}}, nd{{1, 2.84, 0.5, 0}};

  for (int cfg = 0; cfg < 4; cfg++) {
    const bool  isfd  = cfg < 2, nubar = cfg % 2;
    const auto& edges = isfd ? fe : ne;
    const auto& path  = isfd ? fd : nd;
    const auto  var   = isfd ? opg::BinVar::E : opg::BinVar::LoE;
    auto b = prop.analytic_batch(edges, path, nubar, opg::EMeasure::InvE, var);
    REQUIRE(b.on_device());
    const size_t nb = b.n_bins(), n = 16 * nb, np = pts.size();

    // device output
    double *dout_d = nullptr, *dd_d = nullptr;
    REQUIRE(cudaSetDevice(dev) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout_d, np * n * sizeof(double)) == cudaSuccess);
    REQUIRE(cudaMalloc(&dd_d, np * q.size() * n * sizeof(double)) == cudaSuccess);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    prop.avg_path_analytic_batch(b, pts, dout_d, dd_d, st);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
    std::vector<double> out(np * n), dout(np * q.size() * n);
    cudaMemcpy(out.data(), dout_d, out.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(dout.data(), dd_d, dout.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dout_d);
    cudaFree(dd_d);
    cudaStreamDestroy(st);

    // host output: identical
    std::vector<double> ho, hd;
    prop.avg_path_analytic_batch(b, pts, ho, &hd);
    CHECK(ho == out);
    CHECK(hd == dout);

    // per-point GPU calls (host reduction): round-off of the reduction only
    double ep = 0, eg = 0;
    for (size_t p = 0; p < np; p++) {
      prop.set_params(pts[p]);
      std::vector<double> P, dP;
      prop.avg_path_analytic_grad(edges, path, nubar, P, dP, opg::EMeasure::InvE, var);
      ep = std::max(ep, rel_diff(std::vector<double>(out.begin() + p * n, out.begin() + (p + 1) * n),
                                 P, 1.0, nb, 0, nb));
      eg = std::max(eg, rel_diff(std::vector<double>(dout.begin() + p * q.size() * n,
                                                     dout.begin() + (p + 1) * q.size() * n),
                                 dP, 1.0, nb, 0, nb));
    }
    MESSAGE("config " << cfg << ": batch vs per-point GPU: P " << ep << ", dP (rel) " << eg);
    CHECK(ep < 1e-14);
    CHECK(eg < 1e-13);

    // bit-identity: single points, reversed order, and one point per chunk
    auto one = b;
    opg::AnalyticAvgOptions tiny;
    tiny.batch_scratch_mb = 1e-6;
    auto small = prop.analytic_batch(edges, path, nubar, opg::EMeasure::InvE, var, tiny);
    std::vector<M::Params> rev(pts.rbegin(), pts.rend());
    std::vector<double>    ro, rd, so, sd;
    prop.avg_path_analytic_batch(b, rev, ro, &rd);
    prop.avg_path_analytic_batch(small, pts, so, &sd);
    CHECK(so == out);
    CHECK(sd == dout);
    bool same = true;
    for (size_t p = 0; p < np; p++) {
      std::vector<double> o1, d1;
      prop.avg_path_analytic_batch(one, {pts[p]}, o1, &d1);
      same = same && std::equal(o1.begin(), o1.end(), out.begin() + p * n) &&
             std::equal(d1.begin(), d1.end(), dout.begin() + p * q.size() * n) &&
             std::equal(o1.begin(), o1.end(), ro.begin() + (np - 1 - p) * n) &&
             std::equal(d1.begin(), d1.end(), rd.begin() + (np - 1 - p) * q.size() * n);
    }
    CHECK(same);

    // rows = {e, mu}: those channels bit-identical, the others zero
    opg::AnalyticAvgOptions r;
    r.rows  = 3;
    auto br = prop.analytic_batch(edges, path, nubar, opg::EMeasure::InvE, var, r);
    std::vector<double> rO, rD;
    prop.avg_path_analytic_batch(br, pts, rO, &rD);
    bool rok = rO.size() == out.size() && rD.size() == dout.size();
    for (size_t i = 0; rok && i < rO.size(); i++)
      rok = ((i / nb) % 16) / 4 < 2 ? rO[i] == out[i] : rO[i] == 0;
    for (size_t i = 0; rok && i < rD.size(); i++)
      rok = ((i / nb) % 16) / 4 < 2 ? rD[i] == dout[i] : rD[i] == 0;
    CHECK(rok);
  }
}
