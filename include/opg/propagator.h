///////////////////////////////////////////////////////////////////////////////
/// \file propagator.h
///
/// \brief User-facing batched oscillation calculator.
///
/// Lifecycle (modelled on CUDAProb3's Propagator):
///
///   opg::Propagator<opg::Fast<>> prop(opg::PremModel(), {0});  // GPU 0
///   prop.set_grid(energies, cosines);     // once: upload + preallocate
///   prop.set_params(params);              // per fit iteration (cheap)
///   prop.calculate(opg::Flavor::Both);    // asynchronous launch
///   const double* P = prop.probs();       // waits, copies to host lazily
///   double pmue = prop.prob(1, 0, iC, iE, /*nubar=*/false);
///
/// An empty device list selects the multi-threaded CPU backend, which runs
/// the same physics code and is used to validate the GPU results.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_PROPAGATOR_H
#define OPG_PROPAGATOR_H

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "opg/avg/gauss_legendre.h"
#include "opg/engine.h"
#include "opg/engine_cpu.h"
#include "opg/models/all.h"

namespace opg {

#ifdef OPG_HAVE_CUDA
  /// Defined in src/engine_cuda.cu, explicitly instantiated per model.
  template <class Model>
  std::unique_ptr<EngineBase<Model>> make_cuda_engine(const std::vector<int>& devices);
#endif

  /// Number of CUDA devices visible (0 if built without CUDA).
  int cuda_device_count();

  template <class Model> class Propagator {
    public:
      using R                = typename Model::Real;
      using Params           = typename Model::Params;
      using Prepared         = typename Model::Prepared;
      static constexpr int N = Model::N;

      /// devices: CUDA device ids to use; empty = CPU backend.
      /// cpu_threads: OpenMP threads for the CPU backend (0 = default).
      explicit Propagator(const PremModel& earth = PremModel(),
                          const std::vector<int>& devices = {},
                          int cpu_threads = 0)
          : fDevices(devices)
      {
        if (devices.empty()) { fEngine.reset(new CpuEngine<Model>(cpu_threads)); }
        else {
#ifdef OPG_HAVE_CUDA
          if constexpr (has_cuda_engine<Model>::value)
            fEngine = make_cuda_engine<Model>(devices);
          else
            throw std::runtime_error(
                "opg::Propagator: this model/precision is not compiled into "
                "the CUDA backend (see OPG_FOR_EACH_MODEL)");
#else
          throw std::runtime_error(
              "opg::Propagator: GPU devices requested but OscProbGPU was "
              "built without CUDA");
#endif
        }
        set_earth(earth);
      }

      bool on_gpu() const { return !fDevices.empty(); }
      const std::vector<int>& devices() const { return fDevices; }

      void set_earth(const PremModel& earth)
      {
        fEarth = earth;
        fEngine->set_earth(earth);
      }

      const PremModel& earth() const { return fEarth; }

      /// Set the (E, cosZ) grid. E in GeV.
      void set_grid(const std::vector<R>& E, const std::vector<R>& cosZ)
      {
        fNE = E.size();
        fNC = cosZ.size();
        fEngine->set_grid(E, cosZ);
      }

      /// Set the model parameters (runs Model::prepare on the host).
      void set_params(const Params& p)
      {
        fPrepared    = Model::prepare(p);
        fHaveParams  = true;
      }

      const Prepared& prepared() const { return fPrepared; }

      /// Launch the grid computation (asynchronous on GPU).
      void calculate(Flavor which = Flavor::Both)
      {
        require_params();
        fEngine->calculate(fPrepared, which);
      }

      void wait() { fEngine->wait(); }

      /// Host array probs[nubar][a][b][iC][iE]; waits for completion.
      const R* probs() { return fEngine->host_probs(); }

      /// Device array (same layout) on the given device index (GPU only).
      const R* device_probs(int device_index = 0)
      {
        return fEngine->device_probs(device_index);
      }

      /// P(a -> b) at grid point (iC, iE).
      R prob(int a, int b, size_t iC, size_t iE, bool nubar)
      {
        const R* p = probs();
        return p[(((size_t(nubar) * N + a) * N + b) * fNC + iC) * fNE + iE];
      }

      size_t n_energies() const { return fNE; }
      size_t n_cosines() const { return fNC; }

      //.......................................................................
      /// Bin-averaged probabilities on a 2D (E, cosZ) binning.
      ///
      /// E: nglE Gauss-Legendre nodes per bin, uniform in E, log E or 1/E
      /// according to `measure` (InvE is uniform in L/E at fixed baseline,
      /// the analogue of OscProb's AvgProbLoE).
      /// cosZ: uniform average with nglC Gauss-Legendre nodes; with
      /// CosZRule::LayerAdapted (default) the bin is first split at the
      /// layer-grazing cosines of the Earth model and nglC nodes are used on
      /// each piece (see CosZRule).
      void set_bins(const std::vector<double>& Eedges,
                    const std::vector<double>& Cedges, int nglE, int nglC,
                    EMeasure measure = EMeasure::Linear,
                    CosZRule rule    = CosZRule::LayerAdapted)
      {
        if (Eedges.size() < 2 || Cedges.size() < 2)
          throw std::invalid_argument("set_bins: need at least one bin");
        if (nglE < 1 || nglC < 1)
          throw std::invalid_argument("set_bins: GL orders must be >= 1");
        BinSpec<R> B;
        B.nEb  = Eedges.size() - 1;
        B.nCb  = Cedges.size() - 1;
        B.nglE = nglE;
        GaussLegendre       glE(nglE), glC(nglC);
        std::vector<double> x, w;
        for (size_t b = 0; b < B.nEb; b++) {
          double lo = Eedges[b], hi = Eedges[b + 1];
          if (!(hi > lo) || lo <= 0)
            throw std::invalid_argument("set_bins: E edges must be positive "
                                        "and increasing");
          double ulo, uhi;
          switch (measure) {
            case EMeasure::Linear: ulo = lo; uhi = hi; break;
            case EMeasure::Log: ulo = std::log(lo); uhi = std::log(hi); break;
            case EMeasure::InvE: ulo = 1 / hi; uhi = 1 / lo; break;
            default: throw std::invalid_argument("set_bins: bad measure");
          }
          glE.map(ulo, uhi, x, w);
          for (int i = 0; i < nglE; i++) {
            double e = measure == EMeasure::Linear ? x[i]
                       : measure == EMeasure::Log  ? std::exp(x[i])
                                                   : 1 / x[i];
            B.nodesE.push_back(R(e));
            B.wE.push_back(R(w[i] / (uhi - ulo)));
          }
        }

        const std::vector<double> kinks =
            rule == CosZRule::LayerAdapted ? fEarth.GetGrazingCosines()
                                           : std::vector<double>{};
        B.offC.push_back(0);
        for (size_t b = 0; b < B.nCb; b++) {
          double lo = Cedges[b], hi = Cedges[b + 1];
          if (!(hi > lo) || lo < -1 || hi > 1)
            throw std::invalid_argument("set_bins: cosZ edges must be "
                                        "increasing within [-1, 1]");
          std::vector<double> brk = {lo};
          for (double k : kinks)
            if (k > lo && k < hi) brk.push_back(k);
          brk.push_back(hi);
          for (size_t q = 0; q + 1 < brk.size(); q++) {
            double a = brk[q], c = brk[q + 1];
            if (rule == CosZRule::Plain) {
              glC.map(a, c, x, w);
              for (int j = 0; j < nglC; j++) {
                B.nodesC.push_back(R(x[j]));
                B.wC.push_back(R(w[j] / (hi - lo)));
              }
            }
            else {
              // cosZ = a + (c-a)(1 - cos(pi u))/2, u in [0, 1]
              glC.map(0, 1, x, w);
              for (int j = 0; j < nglC; j++) {
                double u   = x[j];
                double cz  = a + (c - a) * 0.5 * (1 - std::cos(M_PI * u));
                double jac = (c - a) * 0.5 * M_PI * std::sin(M_PI * u);
                B.nodesC.push_back(R(cz));
                B.wC.push_back(R(w[j] * jac / (hi - lo)));
              }
            }
          }
          B.offC.push_back(B.nodesC.size());
        }
        fNEb = B.nEb;
        fNCb = B.nCb;
        fEngine->set_bins(B);
      }

      /// Launch the bin-averaged computation (asynchronous on GPU).
      void calculate_binned(Flavor which = Flavor::Both)
      {
        require_params();
        fEngine->calculate_binned(fPrepared, which);
      }

      /// Host array avg[nubar][a][b][iCbin][iEbin]; waits for completion.
      const R* binned() { return fEngine->host_binned(); }

      const R* device_binned(int device_index = 0)
      {
        return fEngine->device_binned(device_index);
      }

      /// Bin-averaged P(a -> b) in bin (iCb, iEb).
      R binned(int a, int b, size_t iCb, size_t iEb, bool nubar)
      {
        const R* p = binned();
        return p[(((size_t(nubar) * N + a) * N + b) * fNCb + iCb) * fNEb + iEb];
      }

      size_t n_energy_bins() const { return fNEb; }
      size_t n_cosine_bins() const { return fNCb; }

      /// 1D bin averages along E for a fixed path (e.g. long baseline),
      /// using nglE Gauss-Legendre nodes per bin. Returns out[a][b][iEbin].
      std::vector<R> avg_path(const std::vector<double>& Eedges, int nglE,
                              const std::vector<Segment<R>>& path, bool nubar,
                              EMeasure measure = EMeasure::Linear)
      {
        require_params();
        if (Eedges.size() < 2 || nglE < 1)
          throw std::invalid_argument("avg_path: bad binning");
        const size_t        nb = Eedges.size() - 1;
        GaussLegendre       gl(nglE);
        std::vector<double> x, w;
        std::vector<R>      nodes, wts;
        for (size_t b = 0; b < nb; b++) {
          double lo = Eedges[b], hi = Eedges[b + 1], ulo, uhi;
          if (!(hi > lo) || lo <= 0)
            throw std::invalid_argument("avg_path: bad E edges");
          if (measure == EMeasure::Linear) { ulo = lo; uhi = hi; }
          else if (measure == EMeasure::Log) { ulo = std::log(lo); uhi = std::log(hi); }
          else { ulo = 1 / hi; uhi = 1 / lo; }
          gl.map(ulo, uhi, x, w);
          for (int i = 0; i < nglE; i++) {
            nodes.push_back(R(measure == EMeasure::Linear ? x[i]
                              : measure == EMeasure::Log  ? std::exp(x[i])
                                                          : 1 / x[i]));
            wts.push_back(R(w[i] / (uhi - ulo)));
          }
        }
        std::vector<R> pn = prob_path(nodes, path, nubar);  // [a][b][node]
        std::vector<R> out(size_t(N) * N * nb, R(0));
        for (size_t ch = 0; ch < size_t(N) * N; ch++)
          for (size_t b = 0; b < nb; b++) {
            R acc = 0;
            for (int i = 0; i < nglE; i++)
              acc += wts[b * nglE + i] * pn[ch * nodes.size() + b * nglE + i];
            out[ch * nb + b] = acc;
          }
        return out;
      }

      /// Event list through the Earth model. Returns out[a][b][i].
      std::vector<R> prob_points(const std::vector<R>& E,
                                 const std::vector<R>& cosZ,
                                 const std::vector<uint8_t>& nubar)
      {
        require_params();
        if (E.size() != cosZ.size() || E.size() != nubar.size())
          throw std::invalid_argument("prob_points: size mismatch");
        std::vector<R> out(size_t(N) * N * E.size());
        fEngine->prob_points(fPrepared, E.data(), cosZ.data(), nubar.data(),
                             E.size(), out.data());
        return out;
      }

      /// Fixed baseline through explicit segments. Returns out[a][b][iE].
      std::vector<R> prob_path(const std::vector<R>& E,
                               const std::vector<Segment<R>>& path, bool nubar)
      {
        require_params();
        std::vector<R> out(size_t(N) * N * E.size());
        fEngine->prob_path(fPrepared, E.data(), E.size(), path.data(),
                           int(path.size()), nubar, out.data());
        return out;
      }

    private:
      void require_params() const
      {
        if (!fHaveParams)
          throw std::logic_error("opg::Propagator: call set_params() first");
      }

      std::vector<int>                    fDevices;
      PremModel                           fEarth;
      std::unique_ptr<EngineBase<Model>>  fEngine;
      Prepared                            fPrepared{};
      bool                                fHaveParams = false;
      size_t                              fNE = 0, fNC = 0;
      size_t                              fNEb = 0, fNCb = 0;
  };

} // namespace opg

#endif
