///////////////////////////////////////////////////////////////////////////////
/// \file engine_cpu.h
///
/// \brief Multi-threaded (OpenMP) CPU backend. Runs exactly the same OPG_HD
///        physics code as the CUDA backend.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_ENGINE_CPU_H
#define OPG_ENGINE_CPU_H

#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "opg/engine.h"
#include "opg/physics/propagate.h"

namespace opg {

  template <class Model> class CpuEngine : public EngineBase<Model> {
    public:
      using typename EngineBase<Model>::R;
      using typename EngineBase<Model>::Prepared;
      static constexpr int N = Model::N;

      /// nthreads <= 0 means OpenMP default (OMP_NUM_THREADS).
      explicit CpuEngine(int nthreads = 0) : fThreads(nthreads) {}

      void set_earth(const PremModel& earth) override
      {
        fTable.reset(new PremModel::HostTable<R>(earth));
      }

      void set_grid(const std::vector<R>& E, const std::vector<R>& cosZ) override
      {
        fE = E;
        fC = cosZ;
        fProbs.assign(2 * N * N * fE.size() * fC.size(), R(0));
      }

      void calculate(const Prepared& P, Flavor which) override
      {
        require_earth();
        const EarthView<R> earth = fTable->view();
        const size_t       nE = fE.size(), nC = fC.size(), npt = nE * nC;
        const size_t       stride = npt;

        for (int nb = 0; nb < 2; nb++) {
          if (!(int(which) & (1 << nb))) continue;
          R* base = fProbs.data() + nb * N * N * npt;
          const long long n = (long long)npt;
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
          for (long long k = 0; k < n; k++) {
            size_t ic = size_t(k) / nE, ie = size_t(k) % nE;
            auto   S  = evolve_prem<Model, R>(P, earth, fE[ie], fC[ic], nb == 1);
            store_probs<N, R>(S, base + k, stride);
          }
        }
      }

      void wait() override {}

      const R* host_probs() override { return fProbs.data(); }

      void prob_points(const Prepared& P, const R* E, const R* cosZ,
                       const uint8_t* nubar, size_t n, R* out) override
      {
        require_earth();
        const EarthView<R> earth = fTable->view();
        const long long    nn    = (long long)n;
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
        for (long long i = 0; i < nn; i++) {
          auto S = evolve_prem<Model, R>(P, earth, E[i], cosZ[i], nubar[i] != 0);
          store_probs<N, R>(S, out + i, n);
        }
      }

      void prob_path(const Prepared& P, const R* E, size_t nE,
                     const Segment<R>* path, int nseg, bool nubar,
                     R* out) override
      {
        const long long n = (long long)nE;
#pragma omp parallel for schedule(static) num_threads(threads())
        for (long long i = 0; i < n; i++) {
          auto S = evolve_path<Model, R>(P, path, nseg, E[i], nubar);
          store_probs<N, R>(S, out + i, nE);
        }
      }

    private:
      int threads() const
      {
#ifdef _OPENMP
        return fThreads > 0 ? fThreads : omp_get_max_threads();
#else
        return 1;
#endif
      }

      void require_earth() const
      {
        if (!fTable) throw std::logic_error("CpuEngine: earth model not set");
      }

      int                                          fThreads;
      std::unique_ptr<PremModel::HostTable<R>>     fTable;
      std::vector<R>                               fE, fC, fProbs;
  };

} // namespace opg

#endif
