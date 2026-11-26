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

#include "engine.h"
#include "physics/grad.h"
#include "physics/propagate.h"

namespace opg {

  template <class Model> class CpuEngine : public EngineBase<Model> {
    public:
      using typename EngineBase<Model>::R;
      using typename EngineBase<Model>::Prepared;
      static constexpr int N = Model::N;

      /// nthreads <= 0 means OpenMP default (OMP_NUM_THREADS).
      explicit CpuEngine(int nthreads = 0) : fThreads(nthreads) {}

      void set_earth(const EarthModel& earth) override
      {
        fTable.reset(new EarthTable<R>(earth));
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
        const EarthView<R> earth = view();
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
            store_model_probs<Model, R>(S, base + k, stride);
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
        const EarthView<R> earth = view();
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
            store_model_probs<Model, R>(S, base + k, npt);
          }
          reduce_nodes(base, N * N, fBinned.data() + (nb * N * N) * B.nCb * B.nEb);
        }
      }

      const R* host_binned() override { return fBinned.data(); }

      // --- gradients --------------------------------------------------------
      using typename EngineBase<Model>::Chunks;
      using GT                = grad_traits<Model>;
      static constexpr int GK = GT::K;

      void calculate_binned_grad(const Prepared& P, const Chunks& chunks, int npar,
                                 Flavor which) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          require_earth();
          const EarthView<R> earth = view();
          const auto&        B     = fBins;
          const size_t nEn = B.nodesE.size(), nCn = B.nodesC.size(), npt = nEn * nCn;
          const size_t nbin = B.nCb * B.nEb;
          // Keep the other half (as the CUDA engine does): computing nu then
          // nubar in two calls must leave both.
          const size_t ng = 2 * size_t(npar) * N * N * nbin;
          if (fBinnedGrad.size() != ng) fBinnedGrad.assign(ng, R(0));
          std::vector<R> node(size_t(GK) * N * N * npt);
          for (int nb = 0; nb < 2; nb++) {
            if (!(int(which) & (1 << nb))) continue;
            R* pbase = fNodeProbs.data() + nb * N * N * npt;
            for (size_t c = 0; c < chunks.size(); c++) {
              const auto&     ch = chunks[c];
              const long long n  = (long long)npt;
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
              for (long long k = 0; k < n; k++) {
                size_t    ic = size_t(k) / nEn, ie = size_t(k) % nEn;
                StateOf<Model> S, dS[GK];
                evolve_prem_grad<Model, R, GK>(P, ch.P, earth, B.nodesE[ie],
                                               B.nodesC[ic], nb == 1, S, dS);
                if (c == 0) store_model_probs<Model, R>(S, pbase + k, npt);
                store_model_grads<Model, R, GK>(S, dS, ch.count, node.data() + k, npt);
              }
              reduce_nodes(node.data(), size_t(ch.count) * N * N,
                           fBinnedGrad.data() +
                               (size_t(nb) * npar + ch.offset) * N * N * nbin);
            }
            reduce_nodes(pbase, N * N, fBinned.data() + (nb * N * N) * nbin);
          }
        }
      }

      const R* host_binned_grad() override
      {
        if constexpr (!GT::enabled) this->no_grad();
        return fBinnedGrad.data();
      }

      /// sum over bins of w_bin * (GL average of dP) = sum over nodes of
      /// (w_bin wC wE) dP: contracted per node, nothing stored.
      void weighted_grad_binned(const Prepared& P, const Chunks& chunks, int npar,
                                Flavor which, const R* w, R* g) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          require_earth();
          const EarthView<R> earth = view();
          const auto&        B     = fBins;
          const size_t nEn = B.nodesE.size(), nCn = B.nodesC.size();
          const size_t nbin = B.nCb * B.nEb;
          for (int p = 0; p < npar; p++) g[p] = 0;
          std::vector<size_t> binOfRow(nCn);
          for (size_t icb = 0; icb < B.nCb; icb++)
            for (size_t rc = B.offC[icb]; rc < B.offC[icb + 1]; rc++) binOfRow[rc] = icb;
          if constexpr (has_adjoint_v<Model>) if (this->use_adjoint(npar, fTable->max_seg)) {
            auto chunk = [&](int c) -> const auto& { return chunks[size_t(c)].P; };
            std::vector<R> part(2 * nCn * size_t(npar), R(0));
            const long long nrow = (long long)nCn;
            for (int nb = 0; nb < 2; nb++) {
              if (!(int(which) & (1 << nb))) continue;
              const R* wbase = w + size_t(nb) * N * N * nbin;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads())
              for (long long rc = 0; rc < nrow; rc++) {
                const size_t icb = binOfRow[size_t(rc)];
                R*           acc = &part[(size_t(nb) * nCn + size_t(rc)) * size_t(npar)];
                R            wn[N * N];
                for (size_t ie = 0; ie < nEn; ie++) {
                  const size_t ieb = ie / size_t(B.nglE);
                  for (int ab = 0; ab < N * N; ab++)
                    wn[ab] = wbase[(ab * B.nCb + icb) * B.nEb + ieb] * B.wC[size_t(rc)] *
                             B.wE[ie];
                  adjoint_prem<Model, R, GK>(P, chunk, npar, earth, B.nodesE[ie],
                                             B.nodesC[size_t(rc)], nb == 1, wn, 1, acc);
                }
              }
            }
            for (size_t r = 0; r < 2 * nCn; r++)  // fixed summation order
              for (int p = 0; p < npar; p++) g[p] += part[r * size_t(npar) + size_t(p)];
            return;
          }
          std::vector<R> part(nCn * GK);
          for (int nb = 0; nb < 2; nb++) {
            if (!(int(which) & (1 << nb))) continue;
            const R* wbase = w + size_t(nb) * N * N * nbin;
            for (const auto& ch : chunks) {
              const long long nrow = (long long)nCn;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads())
              for (long long rc = 0; rc < nrow; rc++) {
                const size_t icb         = binOfRow[size_t(rc)];
                R            rowacc[GK] = {};
                R            wn[N * N];
                for (size_t ie = 0; ie < nEn; ie++) {
                  const size_t ieb = ie / size_t(B.nglE);
                  for (int ab = 0; ab < N * N; ab++)
                    wn[ab] = wbase[(ab * B.nCb + icb) * B.nEb + ieb] * B.wC[size_t(rc)] *
                             B.wE[ie];
                  StateOf<Model> S, dS[GK];
                  evolve_prem_grad<Model, R, GK>(P, ch.P, earth, B.nodesE[ie],
                                                 B.nodesC[size_t(rc)], nb == 1, S, dS);
                  R acc[GK];
                  contract_model_grads<Model, R, GK>(S, dS, wn, 1, acc);
                  for (int k = 0; k < GK; k++) rowacc[k] += acc[k];
                }
                for (int k = 0; k < GK; k++) part[size_t(rc) * GK + k] = rowacc[k];
              }
              for (size_t rc = 0; rc < nCn; rc++)  // fixed summation order
                for (int k = 0; k < ch.count; k++) g[ch.offset + k] += part[rc * GK + k];
            }
          }
        }
      }

      // --- gradients (grid) --------------------------------------------------

      void calculate_grad(const Prepared& P, const Chunks& chunks, int npar,
                          Flavor which) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          require_earth();
          const EarthView<R> earth = view();
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
                StateOf<Model> S, dS[GK];
                evolve_prem_grad<Model, R, GK>(P, ch.P, earth, fE[ie], fC[ic],
                                               nb == 1, S, dS);
                if (c == 0) store_model_probs<Model, R>(S, pbase + k, npt);
                store_model_grads<Model, R, GK>(S, dS, ch.count, gbase + k, npt);
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
          const EarthView<R> earth = view();
          const size_t       nE = fE.size(), nC = fC.size(), npt = nE * nC;
          for (int p = 0; p < npar; p++) g[p] = 0;
          if constexpr (has_adjoint_v<Model>) if (this->use_adjoint(npar, fTable->max_seg)) {
            auto chunk = [&](int c) -> const auto& { return chunks[size_t(c)].P; };
            std::vector<R> part(2 * nC * size_t(npar), R(0));
            const long long nrow = (long long)nC;
            for (int nb = 0; nb < 2; nb++) {
              if (!(int(which) & (1 << nb))) continue;
              const R* wbase = w + size_t(nb) * N * N * npt;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads())
              for (long long ic = 0; ic < nrow; ic++) {
                R* acc = &part[(size_t(nb) * nC + size_t(ic)) * size_t(npar)];
                for (size_t ie = 0; ie < nE; ie++)
                  adjoint_prem<Model, R, GK>(P, chunk, npar, earth, fE[ie], fC[ic], nb == 1,
                                             wbase + ic * nE + ie, npt, acc);
              }
            }
            for (size_t r = 0; r < 2 * nC; r++)  // fixed summation order
              for (int p = 0; p < npar; p++) g[p] += part[r * size_t(npar) + size_t(p)];
            return;
          }
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
                  StateOf<Model> S, dS[GK];
                  evolve_prem_grad<Model, R, GK>(P, ch.P, earth, fE[ie], fC[ic],
                                                 nb == 1, S, dS);
                  R acc[GK];
                  contract_model_grads<Model, R, GK>(S, dS, wbase + ic * nE + ie, npt, acc);
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
                            size_t n, R* outP, R* outG, const R* extra = nullptr,
                            const R* az = nullptr) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          require_earth();
          (void)npar;
          const EarthView<R> earth = view();
          const long long    nn    = (long long)n;
          for (size_t c = 0; c < chunks.size(); c++) {
            const auto& ch = chunks[c];
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
            for (long long i = 0; i < nn; i++) {
              StateOf<Model> S, dS[GK];
              R              ex[kMaxExtra];
              evolve_prem_grad<Model, R, GK>(P, ch.P, event_earth(earth, az, size_t(i)), E[i],
                                             cosZ[i], nubar[i] != 0, S, dS,
                                             gather_extra<Model, R>(extra, size_t(i), n, ex));
              if (c == 0) store_model_probs<Model, R>(S, outP + i, n);
              store_model_grads<Model, R, GK>(S, dS, ch.count,
                                    outG + size_t(ch.offset) * N * N * n + i, n);
            }
          }
        }
      }

      void weighted_grad_points(const Prepared& P, const Chunks& chunks,
                                int npar, const R* E, const R* cosZ,
                                const uint8_t* nubar, size_t n, const R* w,
                                R* g, const R* extra = nullptr,
                                const R* az = nullptr) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          require_earth();
          const EarthView<R> earth = view();
          for (int p = 0; p < npar; p++) g[p] = 0;
          const size_t   bs = 256, nblk = (n + bs - 1) / bs;
          if constexpr (has_adjoint_v<Model>) if (this->use_adjoint(npar, fTable->max_seg)) {
            auto chunk = [&](int c) -> const auto& { return chunks[size_t(c)].P; };
            std::vector<R>  part(nblk * size_t(npar), R(0));
            const long long nb_ = (long long)nblk;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads())
            for (long long b = 0; b < nb_; b++) {
              R* acc = &part[size_t(b) * size_t(npar)];
              for (size_t i = size_t(b) * bs; i < std::min(n, size_t(b + 1) * bs); i++) {
                R ex[kMaxExtra];
                adjoint_prem<Model, R, GK>(P, chunk, npar, event_earth(earth, az, i), E[i],
                                           cosZ[i], nubar[i] != 0, w + i, n, acc,
                                           gather_extra<Model, R>(extra, i, n, ex));
              }
            }
            for (size_t b = 0; b < nblk; b++)
              for (int p = 0; p < npar; p++) g[p] += part[b * size_t(npar) + size_t(p)];
            return;
          }
          std::vector<R> part(nblk * GK);
          for (const auto& ch : chunks) {
            const long long nb_ = (long long)nblk;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads())
            for (long long b = 0; b < nb_; b++) {
              R blk[GK] = {};
              for (size_t i = size_t(b) * bs; i < std::min(n, size_t(b + 1) * bs); i++) {
                StateOf<Model> S, dS[GK];
                R              ex[kMaxExtra];
                evolve_prem_grad<Model, R, GK>(P, ch.P, event_earth(earth, az, i), E[i],
                                               cosZ[i], nubar[i] != 0, S, dS,
                                               gather_extra<Model, R>(extra, i, n, ex));
                R acc[GK];
                contract_model_grads<Model, R, GK>(S, dS, w + i, n, acc);
                for (int k = 0; k < GK; k++) blk[k] += acc[k];
              }
              for (int k = 0; k < GK; k++) part[b * GK + k] = blk[k];
            }
            for (size_t b = 0; b < nblk; b++)
              for (int k = 0; k < ch.count; k++) g[ch.offset + k] += part[b * GK + k];
          }
        }
      }

      void weighted_grad_points_binned(const Prepared& P, const Chunks& chunks,
                                       int npar, const R* E, const R* cosZ,
                                       const uint8_t* nubar, size_t n, const R* w,
                                       const int* bin, int nbins, R* G,
                                       const R* extra = nullptr, const R* az = nullptr) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          require_earth();
          const EarthView<R> earth = view();
          std::fill(G, G + size_t(nbins) * npar, R(0));
          if constexpr (has_adjoint_v<Model>) if (this->use_adjoint(npar, fTable->max_seg)) {
            auto chunk = [&](int c) -> const auto& { return chunks[size_t(c)].P; };
            std::vector<R>  contrib(n * size_t(npar), R(0));
            const long long nn = (long long)n;
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
            for (long long i = 0; i < nn; i++) {
              if (bin[i] < 0 || bin[i] >= nbins) continue;
              R ex[kMaxExtra];
              adjoint_prem<Model, R, GK>(P, chunk, npar, event_earth(earth, az, size_t(i)), E[i],
                                         cosZ[i], nubar[i] != 0, w + i, n,
                                         &contrib[size_t(i) * size_t(npar)],
                                         gather_extra<Model, R>(extra, size_t(i), n, ex));
            }
            for (size_t i = 0; i < n; i++)  // fixed summation order
              if (bin[i] >= 0 && bin[i] < nbins)
                for (int p = 0; p < npar; p++)
                  G[size_t(bin[i]) * npar + p] += contrib[i * size_t(npar) + size_t(p)];
            return;
          }
          std::vector<R> contrib(n * GK);
          for (const auto& ch : chunks) {
            const long long nn = (long long)n;
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
            for (long long i = 0; i < nn; i++) {
              if (bin[i] < 0 || bin[i] >= nbins) continue;
              StateOf<Model> S, dS[GK];
              R              ex[kMaxExtra];
              evolve_prem_grad<Model, R, GK>(P, ch.P, event_earth(earth, az, size_t(i)), E[i],
                                             cosZ[i], nubar[i] != 0, S, dS,
                                             gather_extra<Model, R>(extra, size_t(i), n, ex));
              R acc[GK];
              contract_model_grads<Model, R, GK>(S, dS, w + i, n, acc);
              for (int k = 0; k < GK; k++) contrib[size_t(i) * GK + k] = acc[k];
            }
            for (size_t i = 0; i < n; i++)  // fixed summation order
              if (bin[i] >= 0 && bin[i] < nbins)
                for (int k = 0; k < ch.count; k++)
                  G[size_t(bin[i]) * npar + ch.offset + k] += contrib[i * GK + k];
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
              StateOf<Model> S, dS[GK];
              evolve_path_grad<Model, R, GK>(P, ch.P, path, nseg, E[i], nubar, S, dS);
              if (c == 0) store_model_probs<Model, R>(S, outP + i, nE);
              store_model_grads<Model, R, GK>(S, dS, ch.count,
                                    outG + size_t(ch.offset) * N * N * nE + i, nE);
            }
          }
        }
      }

      const R* host_probs() override { return fProbs.data(); }

      // --- production-height averaging ---------------------------------------
      void set_prod_heights(const ProdHeightTable& t) override
      {
        if constexpr (!supports_prod_height_v<Model>) this->no_prod_height();
        fPH = t;
      }

      void set_grid_prod_heights(const std::vector<int>& dist, bool per_point) override
      {
        if constexpr (!supports_prod_height_v<Model>) this->no_prod_height();
        fPHDist     = dist;
        fPHPerPoint = per_point;
      }

      void calculate_prod_height(const Prepared& P, const Chunks* chunks, int npar,
                                 Flavor which) override
      {
        if constexpr (!supports_prod_height_v<Model>) { this->no_prod_height(); }
        else {
          require_earth();
          const EarthView<R>       earth = view();
          const ProdHeightView<R>  hv    = fPH.view<R>();
          const size_t             nE = fE.size(), nC = fC.size(), npt = nE * nC;
          if (chunks) {
            if constexpr (!GT::enabled) this->no_grad();
            fGrad.resize(2 * size_t(npar) * N * N * npt);
          }
          for (int nb = 0; nb < 2; nb++) {
            if (!(int(which) & (1 << nb))) continue;
            R*              base = fProbs.data() + nb * N * N * npt;
            const long long n    = (long long)npt;
            if (!chunks && !fPHPerPoint) {
              // one distribution per row: bin series once per row
              const long long nrow = (long long)nC;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads())
              for (long long ic = 0; ic < nrow; ic++) {
                const auto pre = row_series(hv, size_t(ic));
                for (size_t ie = 0; ie < nE; ie++)
                  prod_height_probs<Model, R>(P, earth, hv, fPHDist[size_t(ic)], fE[ie], fC[ic],
                                              nb == 1, base + size_t(ic) * nE + ie, npt,
                                              pre.empty() ? nullptr : pre.data());
              }
              continue;
            }
            if (!chunks) {
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
              for (long long k = 0; k < n; k++) {
                const size_t ic = size_t(k) / nE, ie = size_t(k) % nE;
                prod_height_probs<Model, R>(P, earth, hv, grid_dist(ic, ie), fE[ie], fC[ic],
                                            nb == 1, base + k, npt);
              }
              continue;
            }
            if constexpr (GT::enabled) {
              for (size_t c = 0; c < chunks->size(); c++) {
                const auto& ch    = (*chunks)[c];
                R*          gbase = fGrad.data() + (size_t(nb) * npar + ch.offset) * N * N * npt;
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
                for (long long k = 0; k < n; k++) {
                  const size_t ic = size_t(k) / nE, ie = size_t(k) % nE;
                  R            acc[GK];
                  prod_height_grad<Model, R, GK>(P, ch.P, earth, hv, grid_dist(ic, ie), fE[ie],
                                                 fC[ic], nb == 1, c == 0 ? base + k : nullptr,
                                                 gbase + k, ch.count, nullptr, npt, acc);
                }
              }
            }
          }
        }
      }

      void weighted_grad_prod_height(const Prepared& P, const Chunks& chunks, int npar,
                                     Flavor which, const R* w, R* g) override
      {
        if constexpr (!supports_prod_height_grad_v<Model>) { this->no_prod_height(); }
        else {
          require_earth();
          const EarthView<R>      earth = view();
          const ProdHeightView<R> hv    = fPH.view<R>();
          const size_t            nE = fE.size(), nC = fC.size(), npt = nE * nC;
          for (int p = 0; p < npar; p++) g[p] = 0;
          if constexpr (supports_prod_height_adjoint_v<Model>)
            if (this->use_adjoint(npar, fTable->max_seg)) {
              auto chunk = [&](int c) -> const auto& { return chunks[size_t(c)].P; };
              std::vector<R>  part(2 * nC * size_t(npar), R(0));
              const long long nrow = (long long)nC;
              for (int nb = 0; nb < 2; nb++) {
                if (!(int(which) & (1 << nb))) continue;
                const R* wbase = w + size_t(nb) * N * N * npt;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads())
                for (long long ic = 0; ic < nrow; ic++) {
                  R*         acc = &part[(size_t(nb) * nC + size_t(ic)) * size_t(npar)];
                  const auto pre = row_series(hv, size_t(ic));
                  for (size_t ie = 0; ie < nE; ie++)
                    prod_height_adjoint<Model, R, GK>(P, chunk, npar, earth, hv,
                                                      grid_dist(size_t(ic), ie), fE[ie],
                                                      fC[ic], nb == 1,
                                                      wbase + ic * nE + ie, npt, acc,
                                                      pre.empty() ? nullptr : pre.data());
                }
              }
              for (size_t r = 0; r < 2 * nC; r++)  // fixed summation order
                for (int p = 0; p < npar; p++) g[p] += part[r * size_t(npar) + size_t(p)];
              return;
            }
          std::vector<R> part(nC * GK);
          for (int nb = 0; nb < 2; nb++) {
            if (!(int(which) & (1 << nb))) continue;
            const R* wbase = w + size_t(nb) * N * N * npt;
            for (const auto& ch : chunks) {
              const long long nrow = (long long)nC;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads())
              for (long long ic = 0; ic < nrow; ic++) {
                R          rowacc[GK] = {};
                const auto pre        = row_series(hv, size_t(ic));
                for (size_t ie = 0; ie < nE; ie++) {
                  R acc[GK];
                  prod_height_grad<Model, R, GK>(P, ch.P, earth, hv, grid_dist(size_t(ic), ie),
                                                 fE[ie], fC[ic], nb == 1, nullptr, nullptr, 0,
                                                 wbase + ic * nE + ie, npt, acc,
                                                 pre.empty() ? nullptr : pre.data());
                  for (int k = 0; k < GK; k++) rowacc[k] += acc[k];
                }
                for (int k = 0; k < GK; k++) part[ic * GK + k] = rowacc[k];
              }
              for (size_t ic = 0; ic < nC; ic++)  // fixed summation order
                for (int k = 0; k < ch.count; k++) g[ch.offset + k] += part[ic * GK + k];
            }
          }
        }
      }

      void prob_points_prod_height(const Prepared& P, const Chunks* chunks, int npar,
                                   const R* E, const R* cosZ, const uint8_t* nubar,
                                   const int* dist, size_t n, R* out, R* G,
                                   const R* az = nullptr) override
      {
        if constexpr (!supports_prod_height_v<Model>) { this->no_prod_height(); }
        else {
          require_earth();
          (void)npar;
          const EarthView<R>      earth = view();
          const ProdHeightView<R> hv    = fPH.view<R>();
          const long long         nn    = (long long)n;
          if (!chunks) {
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
            for (long long i = 0; i < nn; i++)
              prod_height_probs<Model, R>(P, event_earth(earth, az, size_t(i)), hv, dist[i],
                                          E[i], cosZ[i], nubar[i] != 0, out + i, n);
            return;
          }
          if constexpr (!GT::enabled) { this->no_grad(); }
          else {
            for (size_t c = 0; c < chunks->size(); c++) {
              const auto& ch = (*chunks)[c];
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
              for (long long i = 0; i < nn; i++) {
                R acc[GK];
                prod_height_grad<Model, R, GK>(P, ch.P, event_earth(earth, az, size_t(i)), hv,
                                               dist[i], E[i], cosZ[i], nubar[i] != 0,
                                               c == 0 ? out + i : nullptr,
                                               G + size_t(ch.offset) * N * N * n + i, ch.count,
                                               nullptr, n, acc);
              }
            }
          }
        }
      }

      void weighted_grad_points_prod_height(const Prepared& P, const Chunks& chunks, int npar,
                                            const R* E, const R* cosZ, const uint8_t* nubar,
                                            const int* dist, size_t n, const R* w, R* g,
                                            const R* az = nullptr) override
      {
        if constexpr (!supports_prod_height_grad_v<Model>) { this->no_prod_height(); }
        else {
          require_earth();
          const EarthView<R>      earth = view();
          const ProdHeightView<R> hv    = fPH.view<R>();
          for (int p = 0; p < npar; p++) g[p] = 0;
          const size_t bs = 256, nblk = (n + bs - 1) / bs;
          if constexpr (supports_prod_height_adjoint_v<Model>)
            if (this->use_adjoint(npar, fTable->max_seg)) {
              auto chunk = [&](int c) -> const auto& { return chunks[size_t(c)].P; };
              std::vector<R>  part(nblk * size_t(npar), R(0));
              const long long nb_ = (long long)nblk;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads())
              for (long long b = 0; b < nb_; b++) {
                R* acc = &part[size_t(b) * size_t(npar)];
                for (size_t i = size_t(b) * bs; i < std::min(n, size_t(b + 1) * bs); i++)
                  prod_height_adjoint<Model, R, GK>(P, chunk, npar, event_earth(earth, az, i),
                                                    hv, dist[i], E[i], cosZ[i], nubar[i] != 0,
                                                    w + i, n, acc);
              }
              for (size_t b = 0; b < nblk; b++)
                for (int p = 0; p < npar; p++) g[p] += part[b * size_t(npar) + size_t(p)];
              return;
            }
          std::vector<R> part(nblk * GK);
          for (const auto& ch : chunks) {
            const long long nb_ = (long long)nblk;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads())
            for (long long b = 0; b < nb_; b++) {
              R blk[GK] = {};
              for (size_t i = size_t(b) * bs; i < std::min(n, size_t(b + 1) * bs); i++) {
                R acc[GK];
                prod_height_grad<Model, R, GK>(P, ch.P, event_earth(earth, az, i), hv, dist[i],
                                               E[i], cosZ[i], nubar[i] != 0, nullptr, nullptr, 0,
                                               w + i, n, acc);
                for (int k = 0; k < GK; k++) blk[k] += acc[k];
              }
              for (int k = 0; k < GK; k++) part[b * GK + k] = blk[k];
            }
            for (size_t b = 0; b < nblk; b++)
              for (int k = 0; k < ch.count; k++) g[ch.offset + k] += part[b * GK + k];
          }
        }
      }

      void prob_points(const Prepared& P, const R* E, const R* cosZ,
                       const uint8_t* nubar, size_t n, R* out,
                       const R* extra = nullptr, const R* az = nullptr) override
      {
        require_earth();
        const EarthView<R> earth = view();
        const long long    nn    = (long long)n;
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
        for (long long i = 0; i < nn; i++) {
          R    ex[kMaxExtra];
          auto S = evolve_prem<Model, R>(P, event_earth(earth, az, size_t(i)), E[i], cosZ[i],
                                         nubar[i] != 0,
                                         gather_extra<Model, R>(extra, size_t(i), n, ex));
          store_model_probs<Model, R>(S, out + i, n);
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
          store_model_probs<Model, R>(S, out + i, nE);
        }
      }

    private:
      /// Gauss-Legendre reduction of nch node-grid channels
      /// [nch][nCn][nEn] into bin averages out[nch][nCb][nEb].
      void reduce_nodes(const R* node, size_t nch, R* out) const
      {
        const auto&     B    = fBins;
        const size_t    nEn  = B.nodesE.size(), npt = nEn * B.nodesC.size();
        const long long nout = (long long)(nch * B.nCb * B.nEb);
#pragma omp parallel for schedule(static) num_threads(threads())
        for (long long o = 0; o < nout; o++) {
          size_t   ieb = size_t(o) % B.nEb;
          size_t   icb = (size_t(o) / B.nEb) % B.nCb;
          size_t   ch  = size_t(o) / (B.nEb * B.nCb);
          const R* pc  = node + ch * npt;
          R        acc = 0;
          for (size_t rc = B.offC[icb]; rc < B.offC[icb + 1]; rc++) {
            R rowacc = 0;
            for (int i = 0; i < B.nglE; i++) {
              size_t ce = ieb * B.nglE + i;
              rowacc += B.wE[ce] * pc[rc * nEn + ce];
            }
            acc += B.wC[rc] * rowacc;
          }
          out[size_t(o)] = acc;
        }
      }

      /// Bin series of row ic's distribution (one distribution per row), or
      /// empty (per-point distributions, or none).
      std::vector<prodh::BinSeries> row_series(const ProdHeightView<R>& hv, size_t ic) const
      {
        std::vector<prodh::BinSeries> s;
        if (fPHPerPoint || fPHDist[ic] < 0 || !(std::fabs(fC[ic]) <= 1)) return s;
        const int d = fPHDist[ic];
        s.resize(size_t(fPH.off[size_t(d) + 1] - fPH.off[size_t(d)]));
        prodh::series_table(hv, d, double(fC[ic]), s.data());
        return s;
      }

      /// Height distribution of grid point (ic, ie).
      int grid_dist(size_t ic, size_t ie) const
      {
        return fPHPerPoint ? fPHDist[ic * fE.size() + ie] : fPHDist[ic];
      }

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

      /// View of the Earth table (fGeom holds a binned model's geometry).
      EarthView<R> view() { return fTable->view(fGeom); }

      int                                          fThreads;
      std::unique_ptr<EarthTable<R>>               fTable;
      BinnedGeom                                   fGeom;
      std::vector<R>                               fE, fC, fProbs;
      BinSpec<R>                                   fBins;
      std::vector<R>                               fNodeProbs, fBinned, fBinnedGrad;
      std::vector<R>                               fGrad;
      ProdHeightTable                              fPH;
      std::vector<int>                             fPHDist;
      bool                                         fPHPerPoint = false;
  };

} // namespace opg

#endif
