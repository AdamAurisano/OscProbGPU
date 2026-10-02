///////////////////////////////////////////////////////////////////////////////
/// \file engine_cpu.h
///
/// \brief Multi-threaded (OpenMP) CPU backend. Runs exactly the same OPG_HD
///        physics code as the CUDA backend.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_ENGINE_CPU_H
#define OPG_ENGINE_CPU_H

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "opg/engine.h"
#include "opg/physics/grad.h"
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

      void set_bins(const BinSpec<R>& spec) override
      {
        fBins = spec;
        fNodeProbs.assign(2 * N * N * spec.nodesE.size() * spec.nodesC.size(), R(0));
        fBinned.assign(2 * N * N * spec.nEb * spec.nCb, R(0));
      }

      void calculate_binned(const Prepared& P, Flavor which) override
      {
        require_earth();
        const EarthView<R> earth = fTable->view();
        const auto&        B     = fBins;
        const size_t nEn = B.nodesE.size(), nCn = B.nodesC.size();
        const size_t npt = nEn * nCn;
        for (int nb = 0; nb < 2; nb++) {
          if (!(int(which) & (1 << nb))) continue;
          R* base = fNodeProbs.data() + nb * N * N * npt;
          const long long n = (long long)npt;
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
          for (long long k = 0; k < n; k++) {
            size_t ic = size_t(k) / nEn, ie = size_t(k) % nEn;
            auto   S  = evolve_prem<Model, R>(P, earth, B.nodesE[ie],
                                              B.nodesC[ic], nb == 1);
            store_probs<N, R>(S, base + k, npt);
          }
          // Weighted reduction per (channel, C bin, E bin)
          const long long nout = (long long)(N * N * B.nCb * B.nEb);
#pragma omp parallel for schedule(static) num_threads(threads())
          for (long long o = 0; o < nout; o++) {
            size_t ieb = size_t(o) % B.nEb;
            size_t icb = (size_t(o) / B.nEb) % B.nCb;
            size_t ch  = size_t(o) / (B.nEb * B.nCb);
            const R* pc = base + ch * npt;
            R        acc = 0;
            for (size_t rc = B.offC[icb]; rc < B.offC[icb + 1]; rc++) {
              R rowacc = 0;
              for (int i = 0; i < B.nglE; i++) {
                size_t ce = ieb * B.nglE + i;
                rowacc += B.wE[ce] * pc[rc * nEn + ce];
              }
              acc += B.wC[rc] * rowacc;
            }
            fBinned[(nb * N * N) * B.nCb * B.nEb + size_t(o)] = acc;
          }
        }
      }

      const R* host_binned() override { return fBinned.data(); }

      // --- gradients --------------------------------------------------------
      using typename EngineBase<Model>::Chunks;
      using GT                = grad_traits<Model>;
      static constexpr int GK = GT::K;

      void calculate_grad(const Prepared& P, const Chunks& chunks, int npar,
                          Flavor which) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          require_earth();
          const EarthView<R> earth = fTable->view();
          const size_t       nE = fE.size(), nC = fC.size(), npt = nE * nC;
          fGrad.resize(2 * size_t(npar) * N * N * npt);
          for (int nb = 0; nb < 2; nb++) {
            if (!(int(which) & (1 << nb))) continue;
            R* pbase = fProbs.data() + nb * N * N * npt;
            for (size_t c = 0; c < chunks.size(); c++) {
              const auto&     ch    = chunks[c];
              R*              gbase = fGrad.data() +
                         (size_t(nb) * npar + ch.offset) * N * N * npt;
              const long long n = (long long)npt;
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
              for (long long k = 0; k < n; k++) {
                size_t    ic = size_t(k) / nE, ie = size_t(k) % nE;
                Mat<N, R> S, dS[GK];
                evolve_prem_grad<Model, R, GK>(P, ch.P, earth, fE[ie], fC[ic],
                                               nb == 1, S, dS);
                if (c == 0) store_probs<N, R>(S, pbase + k, npt);
                store_grads<N, R, GK>(S, dS, ch.count, gbase + k, npt);
              }
            }
          }
        }
      }

      const R* host_grad() override
      {
        if constexpr (!GT::enabled) this->no_grad();
        return fGrad.data();
      }

      void weighted_grad(const Prepared& P, const Chunks& chunks, int npar,
                         Flavor which, const R* w, R* g) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          require_earth();
          const EarthView<R> earth = fTable->view();
          const size_t       nE = fE.size(), nC = fC.size(), npt = nE * nC;
          for (int p = 0; p < npar; p++) g[p] = 0;
          std::vector<R> part(nC * GK);
          for (int nb = 0; nb < 2; nb++) {
            if (!(int(which) & (1 << nb))) continue;
            const R* wbase = w + size_t(nb) * N * N * npt;
            for (const auto& ch : chunks) {
              const long long nrow = (long long)nC;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads())
              for (long long ic = 0; ic < nrow; ic++) {
                R rowacc[GK] = {};
                for (size_t ie = 0; ie < nE; ie++) {
                  Mat<N, R> S, dS[GK];
                  evolve_prem_grad<Model, R, GK>(P, ch.P, earth, fE[ie], fC[ic],
                                                 nb == 1, S, dS);
                  R acc[GK];
                  contract_grads<N, R, GK>(S, dS, wbase + ic * nE + ie, npt, acc);
                  for (int k = 0; k < GK; k++) rowacc[k] += acc[k];
                }
                for (int k = 0; k < GK; k++) part[ic * GK + k] = rowacc[k];
              }
              for (size_t ic = 0; ic < nC; ic++)  // fixed summation order
                for (int k = 0; k < ch.count; k++)
                  g[ch.offset + k] += part[ic * GK + k];
            }
          }
        }
      }

      void prob_points_grad(const Prepared& P, const Chunks& chunks, int npar,
                            const R* E, const R* cosZ, const uint8_t* nubar,
                            size_t n, R* outP, R* outG) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          require_earth();
          (void)npar;
          const EarthView<R> earth = fTable->view();
          const long long    nn    = (long long)n;
          for (size_t c = 0; c < chunks.size(); c++) {
            const auto& ch = chunks[c];
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
            for (long long i = 0; i < nn; i++) {
              Mat<N, R> S, dS[GK];
              evolve_prem_grad<Model, R, GK>(P, ch.P, earth, E[i], cosZ[i],
                                             nubar[i] != 0, S, dS);
              if (c == 0) store_probs<N, R>(S, outP + i, n);
              store_grads<N, R, GK>(S, dS, ch.count,
                                    outG + size_t(ch.offset) * N * N * n + i, n);
            }
          }
        }
      }

      void weighted_grad_points(const Prepared& P, const Chunks& chunks,
                                int npar, const R* E, const R* cosZ,
                                const uint8_t* nubar, size_t n, const R* w,
                                R* g) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          require_earth();
          const EarthView<R> earth = fTable->view();
          for (int p = 0; p < npar; p++) g[p] = 0;
          const size_t   bs = 256, nblk = (n + bs - 1) / bs;
          std::vector<R> part(nblk * GK);
          for (const auto& ch : chunks) {
            const long long nb_ = (long long)nblk;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads())
            for (long long b = 0; b < nb_; b++) {
              R blk[GK] = {};
              for (size_t i = size_t(b) * bs; i < std::min(n, size_t(b + 1) * bs); i++) {
                Mat<N, R> S, dS[GK];
                evolve_prem_grad<Model, R, GK>(P, ch.P, earth, E[i], cosZ[i],
                                               nubar[i] != 0, S, dS);
                R acc[GK];
                contract_grads<N, R, GK>(S, dS, w + i, n, acc);
                for (int k = 0; k < GK; k++) blk[k] += acc[k];
              }
              for (int k = 0; k < GK; k++) part[b * GK + k] = blk[k];
            }
            for (size_t b = 0; b < nblk; b++)
              for (int k = 0; k < ch.count; k++) g[ch.offset + k] += part[b * GK + k];
          }
        }
      }

      void prob_path_grad(const Prepared& P, const Chunks& chunks, int npar,
                          const R* E, size_t nE, const Segment<R>* path, int nseg,
                          bool nubar, R* outP, R* outG) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          (void)npar;
          const long long n = (long long)nE;
          for (size_t c = 0; c < chunks.size(); c++) {
            const auto& ch = chunks[c];
#pragma omp parallel for schedule(static) num_threads(threads())
            for (long long i = 0; i < n; i++) {
              Mat<N, R> S, dS[GK];
              evolve_path_grad<Model, R, GK>(P, ch.P, path, nseg, E[i], nubar, S, dS);
              if (c == 0) store_probs<N, R>(S, outP + i, nE);
              store_grads<N, R, GK>(S, dS, ch.count,
                                    outG + size_t(ch.offset) * N * N * nE + i, nE);
            }
          }
        }
      }

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
      BinSpec<R>                                   fBins;
      std::vector<R>                               fNodeProbs, fBinned;
      std::vector<R>                               fGrad;
  };

} // namespace opg

#endif
