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

#include <memory>
#include <stdexcept>
#include <vector>

#include "opg/engine.h"
#include "opg/engine_cpu.h"

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
          fEngine = make_cuda_engine<Model>(devices);
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

      void set_earth(const PremModel& earth) { fEngine->set_earth(earth); }

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
      std::unique_ptr<EngineBase<Model>>  fEngine;
      Prepared                            fPrepared{};
      bool                                fHaveParams = false;
      size_t                              fNE = 0, fNC = 0;
  };

} // namespace opg

#endif
