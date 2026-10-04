///////////////////////////////////////////////////////////////////////////////
//
// CUDA backend.
//
// Design (after CUDAProb3, with the differences discussed in the plan):
//  * grid setup (energies, cosines, earth table) is uploaded once into
//    persistent device buffers; calculate() only launches;
//  * thread mapping: blockIdx.y = cosine row, threads over energies, so all
//    threads of a block share the same layer sequence (no divergence in the
//    segment loop);
//  * results are SoA [nubar][a][b][iC][iE] and copied to pinned host memory
//    lazily on first access;
//  * model parameters (Prepared) are passed by value as a kernel argument,
//    no global __constant__ state, so several propagators can coexist;
//  * several devices: cosine rows are dealt round-robin, as CUDAProb3's
//    CudaPropagator, and gathered on the host.
//
///////////////////////////////////////////////////////////////////////////////

#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "opg/models/all.h"
#include "opg/propagator.h"

namespace opg {

  namespace {

    //.........................................................................
    void check(cudaError_t e, const char* what)
    {
      if (e != cudaSuccess)
        throw std::runtime_error(std::string("OscProbGPU CUDA error in ") +
                                 what + ": " + cudaGetErrorString(e));
    }
#define OPG_CUDA(call) check((call), #call)

    /// Owning device buffer.
    template <class T> struct DevBuf {
        T*     ptr = nullptr;
        size_t n   = 0;
        int    dev = 0;

        DevBuf() = default;
        DevBuf(const DevBuf&) = delete;
        DevBuf& operator=(const DevBuf&) = delete;
        ~DevBuf() { release(); }

        void release()
        {
          if (ptr) {
            cudaSetDevice(dev);
            cudaFree(ptr);
          }
          ptr = nullptr;
          n   = 0;
        }
        void resize(int device, size_t count)
        {
          if (count == n && device == dev && ptr) return;
          release();
          dev = device;
          n   = count;
          if (count) OPG_CUDA(cudaMalloc(&ptr, count * sizeof(T)));
        }
        void upload(int device, const T* h, size_t count, cudaStream_t s)
        {
          resize(device, count);
          if (count)
            OPG_CUDA(cudaMemcpyAsync(ptr, h, count * sizeof(T),
                                     cudaMemcpyHostToDevice, s));
        }
    };

    /// Owning pinned host buffer.
    template <class T> struct PinnedBuf {
        T*     ptr = nullptr;
        size_t n   = 0;

        PinnedBuf() = default;
        PinnedBuf(const PinnedBuf&) = delete;
        PinnedBuf& operator=(const PinnedBuf&) = delete;
        ~PinnedBuf() { release(); }

        void release()
        {
          if (ptr) cudaFreeHost(ptr);
          ptr = nullptr;
          n   = 0;
        }
        void resize(size_t count)
        {
          if (count == n && ptr) return;
          release();
          n = count;
          if (count) OPG_CUDA(cudaMallocHost(&ptr, count * sizeof(T)));
        }
    };

    constexpr int kBlockE = 64;  ///< threads per block along energy

    //.........................................................................
    template <class Model, class R>
    __global__ void grid_kernel(const typename Model::Prepared P,
                                const EarthView<R> earth, const R* __restrict__ E,
                                int nE, const R* __restrict__ C, int nC,
                                int nb_first, R* __restrict__ out)
    {
      constexpr int N  = Model::N;
      const int     nb = nb_first + blockIdx.z;
      const int     ie = blockIdx.x * blockDim.x + threadIdx.x;
      if (ie >= nE) return;

      const size_t stride = size_t(nC) * nE;
      R*           base   = out + size_t(nb) * N * N * stride;

      for (int ic = blockIdx.y; ic < nC; ic += gridDim.y) {
        auto S = evolve_prem<Model, R>(P, earth, E[ie], C[ic], nb == 1);
        store_model_probs<Model, R>(S, base + size_t(ic) * nE + ie, stride);
      }
    }

    template <class Model, class R>
    __global__ void points_kernel(const typename Model::Prepared P,
                                  const EarthView<R> earth,
                                  const R* __restrict__ E,
                                  const R* __restrict__ C,
                                  const uint8_t* __restrict__ nubar,
                                  const R* __restrict__ extra, size_t n,
                                  size_t stride, R* __restrict__ out)
    {
      constexpr int N = Model::N;
      for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
           i += size_t(gridDim.x) * blockDim.x) {
        R    ex[kMaxExtra];
        auto S = evolve_prem<Model, R>(P, earth, E[i], C[i], nubar[i] != 0,
                                       gather_extra<Model, R>(extra, i, n, ex));
        store_model_probs<Model, R>(S, out + i, stride);
      }
    }

    template <class Model, class R>
    __global__ void path_kernel(const typename Model::Prepared P,
                                const Segment<R>* __restrict__ path, int nseg,
                                const R* __restrict__ E, size_t nE, bool nubar,
                                size_t stride, R* __restrict__ out)
    {
      constexpr int N = Model::N;
      for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < nE;
           i += size_t(gridDim.x) * blockDim.x) {
        auto S = evolve_path<Model, R>(P, path, nseg, E[i], nubar);
        store_model_probs<Model, R>(S, out + i, stride);
      }
    }

    /// Weighted Gauss-Legendre reduction of a node grid into bin averages.
    /// node: [2][nch_nb][nCn][nEn] (local C nodes), out: [2][nch_nb][nCb][nEb]
    /// (nch_nb = N*N for probabilities, npar*N*N for gradients).
    template <class R, int N>
    __global__ void reduce_bins_kernel(const R* __restrict__ node, size_t nCn,
                                       size_t nEn, const R* __restrict__ wE,
                                       const R* __restrict__ wC,
                                       const size_t* __restrict__ offC, int nglE,
                                       size_t nEb, size_t nCb, int nb_first,
                                       R* __restrict__ out, int nch_nb = N * N)
    {
      const size_t ieb = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
      if (ieb >= nEb) return;
      const int ch = blockIdx.z + nb_first * nch_nb;  // channel incl. nubar
      for (size_t icb = blockIdx.y; icb < nCb; icb += gridDim.y) {
        const R* pc  = node + size_t(ch) * nCn * nEn;
        R        acc = 0;
        for (size_t rc = offC[icb]; rc < offC[icb + 1]; rc++) {
          R rowacc = 0;
          for (int i = 0; i < nglE; i++) {
            const size_t ce = ieb * nglE + i;
            rowacc += wE[ce] * pc[rc * nEn + ce];
          }
          acc += wC[rc] * rowacc;
        }
        out[(size_t(ch) * nCb + icb) * nEb + ieb] = acc;
      }
    }

    //.........................................................................
    // Gradient kernels (instantiated only for models with gradient support)
    //.........................................................................

    /// How the dual prepared state of a gradient pass reaches the kernels:
    /// by value (kernel parameter, constant bank) when it fits next to the
    /// value state, otherwise as a pointer to a per-device copy in global
    /// memory (CUDA limits kernel arguments to 4 KB).
    template <class Model> struct GradArg {
        using PD = typename grad_traits<Model>::Prepared;
        static constexpr bool by_pointer =
            sizeof(typename Model::Prepared) + sizeof(PD) > 3500;
        using type = std::conditional_t<by_pointer, const PD*, PD>;
        __device__ static const PD& get(const PD& p) { return p; }
        __device__ static const PD& get(const PD* p) { return *p; }
    };

    /// Block-wide sum of acc[K] (blockDim.x == BS, a power of two); thread 0
    /// writes the result to out[0..K).
    template <class R, int K, int BS>
    __device__ void block_reduce_store(R (&acc)[K], R* __restrict__ out)
    {
      __shared__ R sh[BS][K];
      for (int k = 0; k < K; k++) sh[threadIdx.x][k] = acc[k];
      __syncthreads();
      for (int s = BS / 2; s > 0; s >>= 1) {
        if (int(threadIdx.x) < s)
          for (int k = 0; k < K; k++) sh[threadIdx.x][k] += sh[threadIdx.x + s][k];
        __syncthreads();
      }
      if (threadIdx.x == 0)
        for (int k = 0; k < K; k++) out[k] = sh[0][k];
    }

    template <class Model, class R, int K>
    __global__ void grid_grad_kernel(const typename Model::Prepared         P,
                                     const typename GradArg<Model>::type PDa,
                                     const EarthView<R> earth, const R* __restrict__ E,
                                     int nE, const R* __restrict__ C, int nC,
                                     int nb_first, int npar, int offset, int count,
                                     bool write_probs, R* __restrict__ probs,
                                     R* __restrict__ grad)
    {
      const auto&   PD = GradArg<Model>::get(PDa);
      constexpr int N  = Model::N;
      const int     nb = nb_first + blockIdx.z;
      const int     ie = blockIdx.x * blockDim.x + threadIdx.x;
      if (ie >= nE) return;
      const size_t stride = size_t(nC) * nE;
      for (int ic = blockIdx.y; ic < nC; ic += gridDim.y) {
        StateOf<Model> S, dS[K];
        evolve_prem_grad<Model, R, K>(P, PD, earth, E[ie], C[ic], nb == 1, S, dS);
        const size_t pt = size_t(ic) * nE + ie;
        if (write_probs) store_model_probs<Model, R>(S, probs + size_t(nb) * N * N * stride + pt, stride);
        store_model_grads<Model, R, K>(S, dS, count,
                             grad + (size_t(nb) * npar + offset) * N * N * stride + pt,
                             stride);
      }
    }

    template <class Model, class R, int K>
    __global__ void grid_wgrad_kernel(const typename Model::Prepared         P,
                                      const typename GradArg<Model>::type PDa,
                                      const EarthView<R> earth, const R* __restrict__ E,
                                      int nE, const R* __restrict__ C, int nC,
                                      int nb_first, const R* __restrict__ w,
                                      R* __restrict__ partial)
    {
      const auto&   PD = GradArg<Model>::get(PDa);
      constexpr int N  = Model::N;
      const int     nb = nb_first + blockIdx.z;
      const int     ie = blockIdx.x * blockDim.x + threadIdx.x;
      const size_t  stride = size_t(nC) * nE;
      R             acc[K] = {};
      if (ie < nE) {
        for (int ic = blockIdx.y; ic < nC; ic += gridDim.y) {
          StateOf<Model> S, dS[K];
          evolve_prem_grad<Model, R, K>(P, PD, earth, E[ie], C[ic], nb == 1, S, dS);
          R a[K];
          contract_model_grads<Model, R, K>(S, dS,
                                  w + size_t(nb) * N * N * stride + size_t(ic) * nE + ie,
                                  stride, a);
          for (int k = 0; k < K; k++) acc[k] += a[k];
        }
      }
      const size_t blk = (size_t(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      block_reduce_store<R, K, kBlockE>(acc, partial + blk * K);
    }

    /// Weighted gradient over the GL node grid of the bins, with node
    /// weights w_bin(channel, C bin of the row, E bin) * wC[row] * wE[ie]
    /// formed on the fly (w: [2][N][N][nCb_local][nEb]).
    template <class Model, class R, int K>
    __global__ void nodes_wgrad_kernel(const typename Model::Prepared P,
                                       const typename GradArg<Model>::type PDa,
                                       const EarthView<R> earth, const R* __restrict__ E,
                                       int nE, const R* __restrict__ C, int nC,
                                       const R* __restrict__ wE, const R* __restrict__ wC,
                                       const int* __restrict__ binOfRow, int nglE,
                                       int nCb, int nEb, int nb_first,
                                       const R* __restrict__ w, R* __restrict__ partial)
    {
      const auto&   PD = GradArg<Model>::get(PDa);
      constexpr int N  = Model::N;
      const int     nb = nb_first + blockIdx.z;
      const int     ie = blockIdx.x * blockDim.x + threadIdx.x;
      R             acc[K] = {};
      if (ie < nE) {
        const int ieb = ie / nglE;
        for (int ic = blockIdx.y; ic < nC; ic += gridDim.y) {
          StateOf<Model> S, dS[K];
          evolve_prem_grad<Model, R, K>(P, PD, earth, E[ie], C[ic], nb == 1, S, dS);
          const R   f   = wC[ic] * wE[ie];
          const int icb = binOfRow[ic];
          R         wn[N * N];
          for (int ab = 0; ab < N * N; ab++)
            wn[ab] = w[((size_t(nb) * N * N + ab) * nCb + icb) * nEb + ieb] * f;
          R a[K];
          contract_model_grads<Model, R, K>(S, dS, wn, 1, a);
          for (int k = 0; k < K; k++) acc[k] += a[k];
        }
      }
      const size_t blk = (size_t(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      block_reduce_store<R, K, kBlockE>(acc, partial + blk * K);
    }

    template <class Model, class R, int K>
    __global__ void points_grad_kernel(const typename Model::Prepared         P,
                                       const typename GradArg<Model>::type PDa,
                                       const EarthView<R> earth, const R* __restrict__ E,
                                       const R* __restrict__ C,
                                       const uint8_t* __restrict__ nubar,
                                       const R* __restrict__ extra, size_t n,
                                       int offset, int count, bool write_probs,
                                       R* __restrict__ probs, R* __restrict__ grad)
    {
      const auto&   PD = GradArg<Model>::get(PDa);
      constexpr int N = Model::N;
      for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
           i += size_t(gridDim.x) * blockDim.x) {
        StateOf<Model> S, dS[K];
        R ex[kMaxExtra];
        evolve_prem_grad<Model, R, K>(P, PD, earth, E[i], C[i], nubar[i] != 0, S, dS,
                                      gather_extra<Model, R>(extra, i, n, ex));
        if (write_probs) store_model_probs<Model, R>(S, probs + i, n);
        store_model_grads<Model, R, K>(S, dS, count, grad + size_t(offset) * N * N * n + i, n);
      }
    }

    template <class Model, class R, int K>
    __global__ void points_wgrad_kernel(const typename Model::Prepared         P,
                                        const typename GradArg<Model>::type PDa,
                                        const EarthView<R> earth, const R* __restrict__ E,
                                        const R* __restrict__ C,
                                        const uint8_t* __restrict__ nubar,
                                       const R* __restrict__ extra, size_t n,
                                        const R* __restrict__ w, R* __restrict__ partial)
    {
      const auto&   PD = GradArg<Model>::get(PDa);
      constexpr int N      = Model::N;
      R             acc[K] = {};
      for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
           i += size_t(gridDim.x) * blockDim.x) {
        StateOf<Model> S, dS[K];
        R ex[kMaxExtra];
        evolve_prem_grad<Model, R, K>(P, PD, earth, E[i], C[i], nubar[i] != 0, S, dS,
                                      gather_extra<Model, R>(extra, i, n, ex));
        R a[K];
        contract_model_grads<Model, R, K>(S, dS, w + i, n, a);
        for (int k = 0; k < K; k++) acc[k] += a[k];
      }
      block_reduce_store<R, K, 128>(acc, partial + size_t(blockIdx.x) * K);
    }

    /// Per-event weighted contractions contrib[i][k] (events with bin < 0
    /// are skipped).
    template <class Model, class R, int K>
    __global__ void points_contrib_kernel(const typename Model::Prepared P,
                                          const typename GradArg<Model>::type PDa,
                                          const EarthView<R> earth, const R* __restrict__ E,
                                          const R* __restrict__ C,
                                          const uint8_t* __restrict__ nubar,
                                       const R* __restrict__ extra,
                                          const int* __restrict__ bin, size_t n,
                                          const R* __restrict__ w, R* __restrict__ contrib)
    {
      const auto&   PD = GradArg<Model>::get(PDa);
      constexpr int N  = Model::N;
      for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
           i += size_t(gridDim.x) * blockDim.x) {
        if (bin[i] < 0) continue;
        StateOf<Model> S, dS[K];
        R ex[kMaxExtra];
        evolve_prem_grad<Model, R, K>(P, PD, earth, E[i], C[i], nubar[i] != 0, S, dS,
                                      gather_extra<Model, R>(extra, i, n, ex));
        R a[K];
        contract_model_grads<Model, R, K>(S, dS, w + i, n, a);
        for (int k = 0; k < K; k++) contrib[i * K + k] = a[k];
      }
    }

    /// One block per analysis bin: sum contrib over the bin's events
    /// (indices perm[start[b] .. start[b+1])) in a fixed order.
    template <class R, int K>
    __global__ void bin_sum_kernel(const R* __restrict__ contrib,
                                   const int* __restrict__ perm,
                                   const int* __restrict__ start, R* __restrict__ out)
    {
      const int b      = blockIdx.x;
      R         acc[K] = {};
      for (int j = start[b] + threadIdx.x; j < start[b + 1]; j += blockDim.x)
        for (int k = 0; k < K; k++) acc[k] += contrib[size_t(perm[j]) * K + k];
      block_reduce_store<R, K, 128>(acc, out + size_t(b) * K);
    }

    // --- reverse-mode (adjoint) weighted gradients (models with has_adjoint) --
    // Every thread accumulates all npar <= kAdjMaxPar parameters (all
    // gradient chunks at once, chunks[c] in global memory); blocks of
    // kAdjBlockT threads reduce them to partial[blk][kAdjMaxPar]. *err is
    // set if a path exceeds kAdjMaxSeg segments (checked on the host first).
    constexpr int kAdjBlockT = 64;

    template <class Model> struct AdjChunks {
        using GP = typename grad_traits<Model>::Prepared;
        const GP* p;
        __device__ const GP& operator()(int c) const { return p[c]; }
    };

    template <class Model, class R, int K>
    __global__ void grid_wadj_kernel(const typename Model::Prepared P,
                                     const typename grad_traits<Model>::Prepared* chunks,
                                     int npar, const EarthView<R> earth,
                                     const R* __restrict__ E, int nE, const R* __restrict__ C,
                                     int nC, int nb_first, const R* __restrict__ w,
                                     R* __restrict__ partial, int* err)
    {
      constexpr int            N  = Model::N;
      const AdjChunks<Model>   ch{chunks};
      const int                nb = nb_first + blockIdx.z;
      const int                ie = blockIdx.x * blockDim.x + threadIdx.x;
      const size_t             stride = size_t(nC) * nE;
      R                        acc[kAdjMaxPar] = {};
      if (ie < nE) {
        for (int ic = blockIdx.y; ic < nC; ic += gridDim.y)
          if (!adjoint_prem<Model, R, K>(P, ch, npar, earth, E[ie], C[ic], nb == 1,
                                         w + size_t(nb) * N * N * stride + size_t(ic) * nE + ie,
                                         stride, acc))
            *err = 1;
      }
      const size_t blk = (size_t(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      block_reduce_store<R, kAdjMaxPar, kAdjBlockT>(acc, partial + blk * kAdjMaxPar);
    }

    template <class Model, class R, int K>
    __global__ void nodes_wadj_kernel(const typename Model::Prepared P,
                                      const typename grad_traits<Model>::Prepared* chunks,
                                      int npar, const EarthView<R> earth,
                                      const R* __restrict__ E, int nE, const R* __restrict__ C,
                                      int nC, const R* __restrict__ wE, const R* __restrict__ wC,
                                      const int* __restrict__ binOfRow, int nglE, int nCb,
                                      int nEb, int nb_first, const R* __restrict__ w,
                                      R* __restrict__ partial, int* err)
    {
      constexpr int          N  = Model::N;
      const AdjChunks<Model> ch{chunks};
      const int              nb = nb_first + blockIdx.z;
      const int              ie = blockIdx.x * blockDim.x + threadIdx.x;
      R                      acc[kAdjMaxPar] = {};
      if (ie < nE) {
        const int ieb = ie / nglE;
        for (int ic = blockIdx.y; ic < nC; ic += gridDim.y) {
          const R   f   = wC[ic] * wE[ie];
          const int icb = binOfRow[ic];
          R         wn[N * N];
          for (int ab = 0; ab < N * N; ab++)
            wn[ab] = w[((size_t(nb) * N * N + ab) * nCb + icb) * nEb + ieb] * f;
          if (!adjoint_prem<Model, R, K>(P, ch, npar, earth, E[ie], C[ic], nb == 1, wn, 1,
                                         acc))
            *err = 1;
        }
      }
      const size_t blk = (size_t(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      block_reduce_store<R, kAdjMaxPar, kAdjBlockT>(acc, partial + blk * kAdjMaxPar);
    }

    template <class Model, class R, int K>
    __global__ void points_wadj_kernel(const typename Model::Prepared P,
                                       const typename grad_traits<Model>::Prepared* chunks,
                                       int npar, const EarthView<R> earth,
                                       const R* __restrict__ E, const R* __restrict__ C,
                                       const uint8_t* __restrict__ nubar,
                                       const R* __restrict__ extra, size_t n,
                                       const R* __restrict__ w, R* __restrict__ partial,
                                       int* err)
    {
      const AdjChunks<Model> ch{chunks};
      R                      acc[kAdjMaxPar] = {};
      for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
           i += size_t(gridDim.x) * blockDim.x) {
        R ex[kMaxExtra];
        if (!adjoint_prem<Model, R, K>(P, ch, npar, earth, E[i], C[i], nubar[i] != 0, w + i, n,
                                       acc, gather_extra<Model, R>(extra, i, n, ex)))
          *err = 1;
      }
      block_reduce_store<R, kAdjMaxPar, kAdjBlockT>(acc, partial + size_t(blockIdx.x) * kAdjMaxPar);
    }

    /// Per-event contributions contrib[i][p] (stride npar; bin < 0 skipped).
    template <class Model, class R, int K>
    __global__ void points_adj_contrib_kernel(const typename Model::Prepared P,
                                              const typename grad_traits<Model>::Prepared* chunks,
                                              int npar, const EarthView<R> earth,
                                              const R* __restrict__ E, const R* __restrict__ C,
                                              const uint8_t* __restrict__ nubar,
                                              const R* __restrict__ extra,
                                              const int* __restrict__ bin, size_t n,
                                              const R* __restrict__ w, R* __restrict__ contrib,
                                              int* err)
    {
      const AdjChunks<Model> ch{chunks};
      for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
           i += size_t(gridDim.x) * blockDim.x) {
        if (bin[i] < 0) continue;
        R acc[kAdjMaxPar] = {};
        R ex[kMaxExtra];
        if (!adjoint_prem<Model, R, K>(P, ch, npar, earth, E[i], C[i], nubar[i] != 0, w + i, n,
                                       acc, gather_extra<Model, R>(extra, i, n, ex)))
          *err = 1;
        for (int p = 0; p < npar; p++) contrib[i * npar + p] = acc[p];
      }
    }

    /// One block per analysis bin: sum contrib[i][p] (stride npar) over the
    /// bin's events in a fixed order; out[b][kAdjMaxPar].
    template <class R>
    __global__ void bin_sum_adj_kernel(const R* __restrict__ contrib, int npar,
                                       const int* __restrict__ perm,
                                       const int* __restrict__ start, R* __restrict__ out)
    {
      const int b               = blockIdx.x;
      R         acc[kAdjMaxPar] = {};
      for (int j = start[b] + threadIdx.x; j < start[b + 1]; j += blockDim.x)
        for (int p = 0; p < npar; p++) acc[p] += contrib[size_t(perm[j]) * npar + p];
      block_reduce_store<R, kAdjMaxPar, kAdjBlockT>(acc, out + size_t(b) * kAdjMaxPar);
    }

    template <class Model, class R, int K>
    __global__ void path_grad_kernel(const typename Model::Prepared         P,
                                     const typename GradArg<Model>::type PDa,
                                     const Segment<R>* __restrict__ path, int nseg,
                                     const R* __restrict__ E, size_t nE, bool nubar,
                                     int offset, int count, bool write_probs,
                                     R* __restrict__ probs, R* __restrict__ grad)
    {
      const auto&   PD = GradArg<Model>::get(PDa);
      constexpr int N = Model::N;
      for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < nE;
           i += size_t(gridDim.x) * blockDim.x) {
        StateOf<Model> S, dS[K];
        evolve_path_grad<Model, R, K>(P, PD, path, nseg, E[i], nubar, S, dS);
        if (write_probs) store_model_probs<Model, R>(S, probs + i, nE);
        store_model_grads<Model, R, K>(S, dS, count, grad + size_t(offset) * N * N * nE + i, nE);
      }
    }

    unsigned blocks_for(size_t n, unsigned bs)
    {
      size_t b = (n + bs - 1) / bs;
      return unsigned(std::min<size_t>(std::max<size_t>(b, 1), 65535u * 16));
    }

  } // namespace

  //...........................................................................
  template <class Model> class CudaEngine : public EngineBase<Model> {
    public:
      using typename EngineBase<Model>::R;
      using typename EngineBase<Model>::Prepared;
      static constexpr int N = Model::N;

      explicit CudaEngine(const std::vector<int>& devices)
      {
        int ndev = 0;
        OPG_CUDA(cudaGetDeviceCount(&ndev));
        if (ndev == 0) throw std::runtime_error("OscProbGPU: no CUDA device");
        for (int d : devices) {
          if (d < 0 || d >= ndev)
            throw std::runtime_error("OscProbGPU: invalid CUDA device id " +
                                     std::to_string(d));
          fDev.emplace_back(new Device);
          Device& D = *fDev.back();
          D.id      = d;
          OPG_CUDA(cudaSetDevice(d));
          OPG_CUDA(cudaStreamCreateWithFlags(&D.stream, cudaStreamNonBlocking));
        }
      }

      ~CudaEngine() override
      {
        for (auto& D : fDev) {
          cudaSetDevice(D->id);
          cudaStreamSynchronize(D->stream);
          cudaStreamDestroy(D->stream);
        }
      }

      void set_earth(const PremModel& earth) override
      {
        PremModel::HostTable<R> t(earth);
        for (auto& Dp : fDev) {
          Device& D = *Dp;
          OPG_CUDA(cudaSetDevice(D.id));
          D.radius.upload(D.id, t.radius.data(), t.radius.size(), D.stream);
          D.density.upload(D.id, t.density.data(), t.density.size(), D.stream);
          D.zoa.upload(D.id, t.zoa.data(), t.zoa.size(), D.stream);
          D.type.upload(D.id, t.type.data(), t.type.size(), D.stream);
          D.earth = EarthView<R>{int(t.radius.size()), t.det_layer,
                                 t.det_radius,         D.radius.ptr,
                                 D.density.ptr,        D.zoa.ptr,
                                 D.type.ptr};
          OPG_CUDA(cudaStreamSynchronize(D.stream));
        }
        fHaveEarth = true;
      }

      void set_grid(const std::vector<R>& E, const std::vector<R>& cosZ) override
      {
        fNE = E.size();
        fNC = cosZ.size();
        const size_t nd = fDev.size();
        for (size_t k = 0; k < nd; k++) {
          Device& D = *fDev[k];
          Slab&   G = D.grid;
          G.rows.clear();
          for (size_t ic = k; ic < fNC; ic += nd) G.rows.push_back(ic);
          std::vector<R> c;
          for (size_t ic : G.rows) c.push_back(cosZ[ic]);
          OPG_CUDA(cudaSetDevice(D.id));
          G.E.upload(D.id, E.data(), E.size(), D.stream);
          G.C.upload(D.id, c.data(), c.size(), D.stream);
          G.probs.resize(D.id, 2 * N * N * G.rows.size() * fNE);
          G.host.resize(G.probs.n);
          OPG_CUDA(cudaStreamSynchronize(D.stream));
        }
        if (nd > 1) fProbs.resize(2 * N * N * fNC * fNE);
        // (single device: the slab's own pinned buffer is used)
        for (size_t k = 0; k < nd && nd > 1; k++) fDev[k]->grid.host.release();
        fHostValid = false;
      }

      void calculate(const Prepared& P, Flavor which) override
      {
        if (!fHaveEarth) throw std::logic_error("CudaEngine: earth not set");
        const int lo = (int(which) & 1) ? 0 : 1;
        const int hi = (int(which) & 2) ? 1 : 0;
        if (hi < lo) return;
        for (auto& Dp : fDev) {
          Device& D = *Dp;
          OPG_CUDA(cudaSetDevice(D.id));
          launch_grid(D, D.grid, fNE, P, lo, hi);
        }
        fHostValid = false;
      }

      void wait() override
      {
        for (auto& D : fDev) {
          OPG_CUDA(cudaSetDevice(D->id));
          OPG_CUDA(cudaStreamSynchronize(D->stream));
        }
      }

      const R* host_probs() override
      {
        return gather(&Device::grid, fNC, fNE, fProbs, fHostValid);
      }

      const R* device_probs(int k) override
      {
        if (k < 0 || k >= int(fDev.size())) return nullptr;
        return fDev[k]->grid.probs.ptr;
      }

      //.......................................................................
      void set_bins(const BinSpec<R>& B) override
      {
        fBins           = B;
        const size_t nd = fDev.size();
        for (size_t k = 0; k < nd; k++) {
          Device& D = *fDev[k];
          // C bins dealt round-robin; all nodes of a bin on the same device
          D.binned.rows.clear();
          D.nodes.rows.clear();
          std::vector<R>      c, wc;
          std::vector<size_t> off = {0};
          for (size_t icb = k; icb < B.nCb; icb += nd) {
            D.binned.rows.push_back(icb);
            for (size_t rc = B.offC[icb]; rc < B.offC[icb + 1]; rc++) {
              D.nodes.rows.push_back(rc);
              c.push_back(B.nodesC[rc]);
              wc.push_back(B.wC[rc]);
            }
            off.push_back(c.size());
          }
          OPG_CUDA(cudaSetDevice(D.id));
          D.nodes.E.upload(D.id, B.nodesE.data(), B.nodesE.size(), D.stream);
          D.nodes.C.upload(D.id, c.data(), c.size(), D.stream);
          D.nodes.probs.resize(D.id, 2 * N * N * c.size() * B.nodesE.size());
          D.wE.upload(D.id, B.wE.data(), B.wE.size(), D.stream);
          D.wC.upload(D.id, wc.data(), wc.size(), D.stream);
          D.offC.upload(D.id, off.data(), off.size(), D.stream);
          D.binned.probs.resize(D.id, 2 * N * N * D.binned.rows.size() * B.nEb);
          D.binned.host.resize(D.binned.probs.n);
          OPG_CUDA(cudaStreamSynchronize(D.stream));
        }
        fBinnedGradValid = false;
        if (nd > 1) fBinned.resize(2 * N * N * B.nCb * B.nEb);
        for (size_t k = 0; k < nd && nd > 1; k++) fDev[k]->binned.host.release();
        fBinnedValid = false;
      }

      void calculate_binned(const Prepared& P, Flavor which) override
      {
        if (!fHaveEarth) throw std::logic_error("CudaEngine: earth not set");
        const int lo = (int(which) & 1) ? 0 : 1;
        const int hi = (int(which) & 2) ? 1 : 0;
        if (hi < lo) return;
        const size_t nEn = fBins.nodesE.size();
        for (auto& Dp : fDev) {
          Device& D = *Dp;
          if (D.binned.rows.empty()) continue;
          OPG_CUDA(cudaSetDevice(D.id));
          launch_grid(D, D.nodes, nEn, P, lo, hi);
          const size_t ncb = D.binned.rows.size();
          dim3 block(64);
          dim3 grid(unsigned((fBins.nEb + 63) / 64),
                    unsigned(std::min<size_t>(ncb, 65535)),
                    unsigned((hi - lo + 1) * N * N));
          reduce_bins_kernel<R, N><<<grid, block, 0, D.stream>>>(
              D.nodes.probs.ptr, D.nodes.C.n, nEn, D.wE.ptr, D.wC.ptr,
              D.offC.ptr, fBins.nglE, fBins.nEb, ncb, lo, D.binned.probs.ptr);
          OPG_CUDA(cudaGetLastError());
        }
        fBinnedValid = false;
      }

      const R* host_binned() override
      {
        return gather(&Device::binned, fBins.nCb, fBins.nEb, fBinned,
                      fBinnedValid);
      }

      const R* device_binned(int k) override
      {
        if (k < 0 || k >= int(fDev.size())) return nullptr;
        return fDev[k]->binned.probs.ptr;
      }

      void prob_points(const Prepared& P, const R* E, const R* cosZ,
                       const uint8_t* nubar, size_t n, R* out,
                       const R* extra = nullptr) override
      {
        if (!fHaveEarth) throw std::logic_error("CudaEngine: earth not set");
        // Contiguous chunks per device.
        const size_t nd = fDev.size(), chunk = (n + nd - 1) / nd;
        std::vector<size_t> off(nd), cnt(nd);
        for (size_t k = 0; k < nd; k++) {
          off[k] = std::min(n, k * chunk);
          cnt[k] = std::min(n, off[k] + chunk) - off[k];
          Device& D = *fDev[k];
          if (!cnt[k]) continue;
          OPG_CUDA(cudaSetDevice(D.id));
          D.pE.upload(D.id, E + off[k], cnt[k], D.stream);
          D.pC.upload(D.id, cosZ + off[k], cnt[k], D.stream);
          D.pNb.upload(D.id, nubar + off[k], cnt[k], D.stream);
          D.pOut.resize(D.id, N * N * cnt[k]);
          points_kernel<Model, R><<<blocks_for(cnt[k], 128), 128, 0, D.stream>>>(
              P, D.earth, D.pE.ptr, D.pC.ptr, D.pNb.ptr, xptr(D, extra, off[k], cnt[k], n),
              cnt[k], cnt[k], D.pOut.ptr);
          OPG_CUDA(cudaGetLastError());
        }
        gather_chunks(off, cnt, n, out);
      }

      void prob_path(const Prepared& P, const R* E, size_t nE,
                     const Segment<R>* path, int nseg, bool nubar,
                     R* out) override
      {
        const size_t nd = fDev.size(), chunk = (nE + nd - 1) / nd;
        std::vector<size_t> off(nd), cnt(nd);
        for (size_t k = 0; k < nd; k++) {
          off[k] = std::min(nE, k * chunk);
          cnt[k] = std::min(nE, off[k] + chunk) - off[k];
          Device& D = *fDev[k];
          if (!cnt[k]) continue;
          OPG_CUDA(cudaSetDevice(D.id));
          D.pE.upload(D.id, E + off[k], cnt[k], D.stream);
          D.path.upload(D.id, path, size_t(nseg), D.stream);
          D.pOut.resize(D.id, N * N * cnt[k]);
          path_kernel<Model, R><<<blocks_for(cnt[k], 128), 128, 0, D.stream>>>(
              P, D.path.ptr, nseg, D.pE.ptr, cnt[k], nubar, cnt[k], D.pOut.ptr);
          OPG_CUDA(cudaGetLastError());
        }
        gather_chunks(off, cnt, nE, out);
      }

      //.......................................................................
      // Gradients
      //.......................................................................
      using typename EngineBase<Model>::Chunks;
      using GT                = grad_traits<Model>;
      static constexpr int GK = GT::K;
      /// Probabilities from the probability-only kernels (see separate_probs).
      static constexpr bool kSepP = separate_probs<Model>::value;

      void calculate_grad(const Prepared& P, const Chunks& chunks, int npar,
                          Flavor which) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          if (!fHaveEarth) throw std::logic_error("CudaEngine: earth not set");
          check_param_size();
          upload_chunks(chunks);
          const int lo = (int(which) & 1) ? 0 : 1;
          const int hi = (int(which) & 2) ? 1 : 0;
          if (hi < lo) return;
          fNpar = npar;
          for (auto& Dp : fDev) {
            Device& D = *Dp;
            Slab&   G = D.grid;
            D.gslab.rows = G.rows;
            if (G.rows.empty() || fNE == 0) continue;
            OPG_CUDA(cudaSetDevice(D.id));
            D.gslab.probs.resize(D.id, 2 * size_t(npar) * N * N * G.rows.size() * fNE);
            if (fDev.size() == 1) D.gslab.host.resize(D.gslab.probs.n);
            const size_t nrow = G.C.n;
            dim3 block(kBlockE);
            dim3 grid(unsigned((fNE + kBlockE - 1) / kBlockE),
                      unsigned(std::min<size_t>(nrow, 65535)), unsigned(hi - lo + 1));
            if constexpr (kSepP) launch_grid(D, G, fNE, P, lo, hi);
            for (size_t c = 0; c < chunks.size(); c++) {
              grid_grad_kernel<Model, R, GK><<<grid, block, 0, D.stream>>>(
                  P, pd_arg(D, chunks, c), D.earth, G.E.ptr, int(fNE), G.C.ptr, int(nrow), lo,
                  npar, chunks[c].offset, chunks[c].count, c == 0 && !kSepP, G.probs.ptr,
                  D.gslab.probs.ptr);
              OPG_CUDA(cudaGetLastError());
            }
          }
          if (fDev.size() > 1) fGradFull.resize(2 * size_t(npar) * N * N * fNC * fNE);
          fHostValid = false;
          fGradValid = false;
        }
      }

      const R* host_grad() override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          return gather(&Device::gslab, fNC, fNE, fGradFull, fGradValid,
                        2 * size_t(fNpar) * N * N);
        }
      }

      const R* device_grad(int k) override
      {
        if (k < 0 || k >= int(fDev.size())) return nullptr;
        return fDev[k]->gslab.probs.ptr;
      }

      void weighted_grad(const Prepared& P, const Chunks& chunks, int npar,
                         Flavor which, const R* w, R* g) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          if (!fHaveEarth) throw std::logic_error("CudaEngine: earth not set");
          check_param_size();
          upload_chunks(chunks);
          for (int p = 0; p < npar; p++) g[p] = 0;
          const int lo = (int(which) & 1) ? 0 : 1;
          const int hi = (int(which) & 2) ? 1 : 0;
          if (hi < lo) return;
          const size_t nd = fDev.size(), npt = fNC * fNE;
          // weights for the rows of each device, [2][N][N][nrow][nE], staged
          // in pinned memory and uploaded asynchronously (devices overlap)
          for (size_t k = 0; k < nd; k++) {
            Device& D = *fDev[k];
            Slab&   G = D.grid;
            if (G.rows.empty()) continue;
            OPG_CUDA(cudaSetDevice(D.id));
            const size_t nr = G.rows.size();
            D.wtsHost.resize(2 * N * N * nr * fNE);
            if (nd == 1)
              std::copy(w, w + 2 * N * N * npt, D.wtsHost.ptr);
            else
              for (size_t ch = 0; ch < size_t(2 * N * N); ch++)
                for (size_t r = 0; r < nr; r++)
                  std::copy(w + (ch * fNC + G.rows[r]) * fNE,
                            w + (ch * fNC + G.rows[r] + 1) * fNE,
                            D.wtsHost.ptr + (ch * nr + r) * fNE);
            D.wts.upload(D.id, D.wtsHost.ptr, D.wtsHost.n, D.stream);
          }
          std::vector<const R*> wp(nd);
          for (size_t k = 0; k < nd; k++) wp[k] = fDev[k]->wts.ptr;
          run_grid_wgrad(P, chunks, lo, hi, wp, g);
        }
      }

      void weighted_grad_device(const Prepared& P, const Chunks& chunks, int npar,
                                Flavor which, const std::vector<const R*>& w,
                                R* g) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          if (!fHaveEarth) throw std::logic_error("CudaEngine: earth not set");
          if (w.size() != fDev.size())
            throw std::invalid_argument("weighted_gradient_device: need one weight "
                                        "pointer per device");
          check_param_size();
          upload_chunks(chunks);
          for (int p = 0; p < npar; p++) g[p] = 0;
          const int lo = (int(which) & 1) ? 0 : 1;
          const int hi = (int(which) & 2) ? 1 : 0;
          if (hi < lo) return;
          run_grid_wgrad(P, chunks, lo, hi, w, g);
        }
      }

      /// Weighted grid gradient with weights w[k] (device k, layout of
      /// device_probs(k)).
      void run_grid_wgrad(const Prepared& P, const Chunks& chunks, int lo, int hi,
                          const std::vector<const R*>& w, R* g)
      {
        if constexpr (GT::enabled) {
          const size_t nd   = fDev.size();
          const int    npar = chunks.empty() ? 0 : chunks.back().offset + chunks.back().count;
          if constexpr (has_adjoint_v<Model>)
            if (use_adj(npar)) {
              upload_adj_chunks(chunks);
              for (size_t k = 0; k < nd; k++) {
                Device& D = *fDev[k];
                Slab&   G = D.grid;
                if (G.rows.empty() || fNE == 0) continue;
                OPG_CUDA(cudaSetDevice(D.id));
                const size_t nrow = G.C.n;
                dim3 grid(unsigned((fNE + kAdjBlockT - 1) / kAdjBlockT),
                          unsigned(std::min<size_t>(nrow, 4096)), unsigned(hi - lo + 1));
                const size_t nblk = size_t(grid.x) * grid.y * grid.z;
                D.partial.resize(D.id, nblk * kAdjMaxPar);
                grid_wadj_kernel<Model, R, GK><<<grid, dim3(kAdjBlockT), 0, D.stream>>>(
                    P, D.pd.ptr, npar, D.earth, G.E.ptr, int(fNE), G.C.ptr, int(nrow), lo,
                    w[k], D.partial.ptr, D.adjErr.ptr);
                OPG_CUDA(cudaGetLastError());
              }
              sum_adj_partials(npar, g);
              return;
            }
          for (size_t c = 0; c < chunks.size(); c++) {
            std::vector<dim3> grids(nd);
            for (size_t k = 0; k < nd; k++) {
              Device& D = *fDev[k];
              Slab&   G = D.grid;
              if (G.rows.empty() || fNE == 0) continue;
              OPG_CUDA(cudaSetDevice(D.id));
              const size_t nrow = G.C.n;
              grids[k] = dim3(unsigned((fNE + kBlockE - 1) / kBlockE),
                              unsigned(std::min<size_t>(nrow, 4096)),
                              unsigned(hi - lo + 1));
              const size_t nblk = size_t(grids[k].x) * grids[k].y * grids[k].z;
              D.partial.resize(D.id, nblk * GK);
              grid_wgrad_kernel<Model, R, GK><<<grids[k], dim3(kBlockE), 0, D.stream>>>(
                  P, pd_arg(D, chunks, c), D.earth, G.E.ptr, int(fNE), G.C.ptr, int(nrow),
                  lo, w[k], D.partial.ptr);
              OPG_CUDA(cudaGetLastError());
            }
            sum_partials(chunks[c], g);
          }
        }
      }

      void prob_points_grad(const Prepared& P, const Chunks& chunks, int npar,
                            const R* E, const R* cosZ, const uint8_t* nubar,
                            size_t n, R* outP, R* outG,
                            const R* extra = nullptr) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          if (!fHaveEarth) throw std::logic_error("CudaEngine: earth not set");
          check_param_size();
          upload_chunks(chunks);
          const size_t nd = fDev.size(), chunk = (n + nd - 1) / nd;
          std::vector<size_t> off(nd), cnt(nd);
          for (size_t k = 0; k < nd; k++) {
            off[k] = std::min(n, k * chunk);
            cnt[k] = std::min(n, off[k] + chunk) - off[k];
            Device& D = *fDev[k];
            if (!cnt[k]) continue;
            OPG_CUDA(cudaSetDevice(D.id));
            D.pE.upload(D.id, E + off[k], cnt[k], D.stream);
            D.pC.upload(D.id, cosZ + off[k], cnt[k], D.stream);
            D.pNb.upload(D.id, nubar + off[k], cnt[k], D.stream);
            xptr(D, extra, off[k], cnt[k], n);
            D.pOut.resize(D.id, N * N * cnt[k]);
            D.pGrad.resize(D.id, size_t(npar) * N * N * cnt[k]);
            if constexpr (kSepP) {
              points_kernel<Model, R><<<blocks_for(cnt[k], 128), 128, 0, D.stream>>>(
                  P, D.earth, D.pE.ptr, D.pC.ptr, D.pNb.ptr, D.pX.ptr, cnt[k], cnt[k],
                  D.pOut.ptr);
              OPG_CUDA(cudaGetLastError());
            }
            for (size_t c = 0; c < chunks.size(); c++) {
              points_grad_kernel<Model, R, GK>
                  <<<blocks_for(cnt[k], 128), 128, 0, D.stream>>>(
                      P, pd_arg(D, chunks, c), D.earth, D.pE.ptr, D.pC.ptr, D.pNb.ptr,
                      D.pX.ptr, cnt[k], chunks[c].offset, chunks[c].count, c == 0 && !kSepP,
                      D.pOut.ptr,
                      D.pGrad.ptr);
              OPG_CUDA(cudaGetLastError());
            }
          }
          gather_chunks(off, cnt, n, outP, &Device::pOut, N * N);
          gather_chunks(off, cnt, n, outG, &Device::pGrad, size_t(npar) * N * N);
        }
      }

      void weighted_grad_points(const Prepared& P, const Chunks& chunks, int npar,
                                const R* E, const R* cosZ, const uint8_t* nubar,
                                size_t n, const R* w, R* g,
                                const R* extra = nullptr) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          if (!fHaveEarth) throw std::logic_error("CudaEngine: earth not set");
          check_param_size();
          upload_chunks(chunks);
          for (int p = 0; p < npar; p++) g[p] = 0;
          const size_t nd = fDev.size(), chunk = (n + nd - 1) / nd;
          std::vector<size_t> off(nd), cnt(nd);
          for (size_t k = 0; k < nd; k++) {
            off[k] = std::min(n, k * chunk);
            cnt[k] = std::min(n, off[k] + chunk) - off[k];
            Device& D = *fDev[k];
            if (!cnt[k]) continue;
            OPG_CUDA(cudaSetDevice(D.id));
            D.pE.upload(D.id, E + off[k], cnt[k], D.stream);
            D.pC.upload(D.id, cosZ + off[k], cnt[k], D.stream);
            D.pNb.upload(D.id, nubar + off[k], cnt[k], D.stream);
            xptr(D, extra, off[k], cnt[k], n);
            std::vector<R> loc(N * N * cnt[k]);
            for (int ab = 0; ab < N * N; ab++)
              std::copy(w + ab * n + off[k], w + ab * n + off[k] + cnt[k],
                        loc.begin() + ab * cnt[k]);
            D.wts.upload(D.id, loc.data(), loc.size(), D.stream);
            OPG_CUDA(cudaStreamSynchronize(D.stream));
          }
          if constexpr (has_adjoint_v<Model>)
            if (use_adj(npar)) {
              upload_adj_chunks(chunks);
              for (size_t k = 0; k < nd; k++) {
                Device& D = *fDev[k];
                if (!cnt[k]) continue;
                OPG_CUDA(cudaSetDevice(D.id));
                const unsigned nblk = std::min(blocks_for(cnt[k], kAdjBlockT), 4096u);
                D.partial.resize(D.id, size_t(nblk) * kAdjMaxPar);
                points_wadj_kernel<Model, R, GK><<<nblk, kAdjBlockT, 0, D.stream>>>(
                    P, D.pd.ptr, npar, D.earth, D.pE.ptr, D.pC.ptr, D.pNb.ptr, D.pX.ptr,
                    cnt[k], D.wts.ptr, D.partial.ptr, D.adjErr.ptr);
                OPG_CUDA(cudaGetLastError());
              }
              sum_adj_partials(npar, g);
              return;
            }
          for (size_t c = 0; c < chunks.size(); c++) {
            for (size_t k = 0; k < nd; k++) {
              Device& D = *fDev[k];
              if (!cnt[k]) continue;
              OPG_CUDA(cudaSetDevice(D.id));
              const unsigned nblk = std::min(blocks_for(cnt[k], 128), 4096u);
              D.partial.resize(D.id, size_t(nblk) * GK);
              points_wgrad_kernel<Model, R, GK><<<nblk, 128, 0, D.stream>>>(
                  P, pd_arg(D, chunks, c), D.earth, D.pE.ptr, D.pC.ptr, D.pNb.ptr,
                  D.pX.ptr, cnt[k], D.wts.ptr, D.partial.ptr);
              OPG_CUDA(cudaGetLastError());
            }
            sum_partials(chunks[c], g);
          }
        }
      }

      void prob_path_grad(const Prepared& P, const Chunks& chunks, int npar,
                          const R* E, size_t nE, const Segment<R>* path, int nseg,
                          bool nubar, R* outP, R* outG) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          check_param_size();
          upload_chunks(chunks);
          const size_t nd = fDev.size(), chunk = (nE + nd - 1) / nd;
          std::vector<size_t> off(nd), cnt(nd);
          for (size_t k = 0; k < nd; k++) {
            off[k] = std::min(nE, k * chunk);
            cnt[k] = std::min(nE, off[k] + chunk) - off[k];
            Device& D = *fDev[k];
            if (!cnt[k]) continue;
            OPG_CUDA(cudaSetDevice(D.id));
            D.pE.upload(D.id, E + off[k], cnt[k], D.stream);
            D.path.upload(D.id, path, size_t(nseg), D.stream);
            D.pOut.resize(D.id, N * N * cnt[k]);
            D.pGrad.resize(D.id, size_t(npar) * N * N * cnt[k]);
            if constexpr (kSepP) {
              path_kernel<Model, R><<<blocks_for(cnt[k], 128), 128, 0, D.stream>>>(
                  P, D.path.ptr, nseg, D.pE.ptr, cnt[k], nubar, cnt[k], D.pOut.ptr);
              OPG_CUDA(cudaGetLastError());
            }
            for (size_t c = 0; c < chunks.size(); c++) {
              path_grad_kernel<Model, R, GK>
                  <<<blocks_for(cnt[k], 128), 128, 0, D.stream>>>(
                      P, pd_arg(D, chunks, c), D.path.ptr, nseg, D.pE.ptr, cnt[k], nubar,
                      chunks[c].offset, chunks[c].count, c == 0 && !kSepP, D.pOut.ptr,
                      D.pGrad.ptr);
              OPG_CUDA(cudaGetLastError());
            }
          }
          gather_chunks(off, cnt, nE, outP, &Device::pOut, N * N);
          gather_chunks(off, cnt, nE, outG, &Device::pGrad, size_t(npar) * N * N);
        }
      }

      void calculate_binned_grad(const Prepared& P, const Chunks& chunks, int npar,
                                 Flavor which) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          if (!fHaveEarth) throw std::logic_error("CudaEngine: earth not set");
          check_param_size();
          upload_chunks(chunks);
          const int lo = (int(which) & 1) ? 0 : 1;
          const int hi = (int(which) & 2) ? 1 : 0;
          if (hi < lo) return;
          fNpar            = npar;
          const size_t nEn = fBins.nodesE.size();
          for (auto& Dp : fDev) {
            Device& D = *Dp;
            Slab&   G = D.nodes;
            D.bgslab.rows = D.binned.rows;
            if (D.binned.rows.empty()) continue;
            OPG_CUDA(cudaSetDevice(D.id));
            const size_t nrow = G.C.n, ncb = D.binned.rows.size();
            D.ngrad.resize(D.id, 2 * size_t(npar) * N * N * nrow * nEn);
            D.bgslab.probs.resize(D.id, 2 * size_t(npar) * N * N * ncb * fBins.nEb);
            if (fDev.size() == 1) D.bgslab.host.resize(D.bgslab.probs.n);
            dim3 block(kBlockE);
            dim3 grid(unsigned((nEn + kBlockE - 1) / kBlockE),
                      unsigned(std::min<size_t>(nrow, 65535)), unsigned(hi - lo + 1));
            if constexpr (kSepP) launch_grid(D, G, nEn, P, lo, hi);
            for (size_t c = 0; c < chunks.size(); c++) {
              grid_grad_kernel<Model, R, GK><<<grid, block, 0, D.stream>>>(
                  P, pd_arg(D, chunks, c), D.earth, G.E.ptr, int(nEn), G.C.ptr, int(nrow),
                  lo, npar, chunks[c].offset, chunks[c].count, c == 0 && !kSepP, G.probs.ptr,
                  D.ngrad.ptr);
              OPG_CUDA(cudaGetLastError());
            }
            dim3 rblock(64);
            dim3 rgrid(unsigned((fBins.nEb + 63) / 64), unsigned(std::min<size_t>(ncb, 65535)),
                       unsigned((hi - lo + 1) * N * N));
            reduce_bins_kernel<R, N><<<rgrid, rblock, 0, D.stream>>>(
                G.probs.ptr, nrow, nEn, D.wE.ptr, D.wC.ptr, D.offC.ptr, fBins.nglE,
                fBins.nEb, ncb, lo, D.binned.probs.ptr);
            OPG_CUDA(cudaGetLastError());
            rgrid.z = unsigned((hi - lo + 1) * npar * N * N);
            reduce_bins_kernel<R, N><<<rgrid, rblock, 0, D.stream>>>(
                D.ngrad.ptr, nrow, nEn, D.wE.ptr, D.wC.ptr, D.offC.ptr, fBins.nglE,
                fBins.nEb, ncb, lo, D.bgslab.probs.ptr, npar * N * N);
            OPG_CUDA(cudaGetLastError());
          }
          if (fDev.size() > 1)
            fBinnedGradFull.resize(2 * size_t(npar) * N * N * fBins.nCb * fBins.nEb);
          fBinnedValid     = false;
          fBinnedGradValid = false;
        }
      }

      const R* host_binned_grad() override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          return gather(&Device::bgslab, fBins.nCb, fBins.nEb, fBinnedGradFull,
                        fBinnedGradValid, 2 * size_t(fNpar) * N * N);
        }
      }

      const R* device_binned_grad(int k) override
      {
        if (k < 0 || k >= int(fDev.size())) return nullptr;
        return fDev[k]->bgslab.probs.ptr;
      }

      void weighted_grad_binned(const Prepared& P, const Chunks& chunks, int npar,
                                Flavor which, const R* w, R* g) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          if (!fHaveEarth) throw std::logic_error("CudaEngine: earth not set");
          check_param_size();
          upload_chunks(chunks);
          for (int p = 0; p < npar; p++) g[p] = 0;
          const int lo = (int(which) & 1) ? 0 : 1;
          const int hi = (int(which) & 2) ? 1 : 0;
          if (hi < lo) return;
          const size_t nd = fDev.size(), nEn = fBins.nodesE.size();
          const size_t nCb = fBins.nCb, nEb = fBins.nEb;
          // bin weights of each device's C bins, [2][N][N][ncb][nEb]
          for (size_t k = 0; k < nd; k++) {
            Device& D = *fDev[k];
            if (D.binned.rows.empty()) continue;
            OPG_CUDA(cudaSetDevice(D.id));
            const size_t ncb = D.binned.rows.size();
            std::vector<R> wl(2 * N * N * ncb * nEb);
            for (size_t ch = 0; ch < size_t(2 * N * N); ch++)
              for (size_t r = 0; r < ncb; r++)
                std::copy(w + (ch * nCb + D.binned.rows[r]) * nEb,
                          w + (ch * nCb + D.binned.rows[r] + 1) * nEb,
                          wl.begin() + (ch * ncb + r) * nEb);
            D.wts.upload(D.id, wl.data(), wl.size(), D.stream);
            OPG_CUDA(cudaStreamSynchronize(D.stream));
          }
          std::vector<const R*> wp(nd);
          for (size_t k = 0; k < nd; k++) wp[k] = fDev[k]->wts.ptr;
          run_binned_wgrad(P, chunks, lo, hi, wp, g);
        }
      }

      void weighted_grad_binned_device(const Prepared& P, const Chunks& chunks,
                                       int npar, Flavor which,
                                       const std::vector<const R*>& w, R* g) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          if (!fHaveEarth) throw std::logic_error("CudaEngine: earth not set");
          if (w.size() != fDev.size())
            throw std::invalid_argument("weighted_gradient_binned_device: need one "
                                        "weight pointer per device");
          check_param_size();
          upload_chunks(chunks);
          for (int p = 0; p < npar; p++) g[p] = 0;
          const int lo = (int(which) & 1) ? 0 : 1;
          const int hi = (int(which) & 2) ? 1 : 0;
          if (hi < lo) return;
          run_binned_wgrad(P, chunks, lo, hi, w, g);
        }
      }

      /// Weighted binned gradient with bin weights w[k] (device k, layout of
      /// device_binned(k)).
      void run_binned_wgrad(const Prepared& P, const Chunks& chunks, int lo, int hi,
                            const std::vector<const R*>& w, R* g)
      {
        if constexpr (GT::enabled) {
          const size_t nd = fDev.size(), nEn = fBins.nodesE.size(), nEb = fBins.nEb;
          // local C bin of each local node row
          for (size_t k = 0; k < nd; k++) {
            Device& D = *fDev[k];
            if (D.binned.rows.empty()) continue;
            OPG_CUDA(cudaSetDevice(D.id));
            std::vector<int> bor;
            for (size_t r = 0; r < D.binned.rows.size(); r++) {
              const size_t icb = D.binned.rows[r];
              for (size_t rc = fBins.offC[icb]; rc < fBins.offC[icb + 1]; rc++)
                bor.push_back(int(r));
            }
            D.binOfRow.upload(D.id, bor.data(), bor.size(), D.stream);
          }
          const int npar = chunks.empty() ? 0 : chunks.back().offset + chunks.back().count;
          if constexpr (has_adjoint_v<Model>)
            if (use_adj(npar)) {
              upload_adj_chunks(chunks);
              for (size_t k = 0; k < nd; k++) {
                Device& D = *fDev[k];
                if (D.binned.rows.empty() || nEn == 0) continue;
                OPG_CUDA(cudaSetDevice(D.id));
                const size_t nrow = D.nodes.C.n;
                dim3 grid(unsigned((nEn + kAdjBlockT - 1) / kAdjBlockT),
                          unsigned(std::min<size_t>(nrow, 4096)), unsigned(hi - lo + 1));
                const size_t nblk = size_t(grid.x) * grid.y * grid.z;
                D.partial.resize(D.id, nblk * kAdjMaxPar);
                nodes_wadj_kernel<Model, R, GK><<<grid, dim3(kAdjBlockT), 0, D.stream>>>(
                    P, D.pd.ptr, npar, D.earth, D.nodes.E.ptr, int(nEn), D.nodes.C.ptr,
                    int(nrow), D.wE.ptr, D.wC.ptr, D.binOfRow.ptr, fBins.nglE,
                    int(D.binned.rows.size()), int(nEb), lo, w[k], D.partial.ptr,
                    D.adjErr.ptr);
                OPG_CUDA(cudaGetLastError());
              }
              sum_adj_partials(npar, g);
              return;
            }
          for (size_t c = 0; c < chunks.size(); c++) {
            for (size_t k = 0; k < nd; k++) {
              Device& D = *fDev[k];
              if (D.binned.rows.empty() || nEn == 0) continue;
              OPG_CUDA(cudaSetDevice(D.id));
              const size_t nrow = D.nodes.C.n;
              dim3 grid(unsigned((nEn + kBlockE - 1) / kBlockE),
                        unsigned(std::min<size_t>(nrow, 4096)), unsigned(hi - lo + 1));
              const size_t nblk = size_t(grid.x) * grid.y * grid.z;
              D.partial.resize(D.id, nblk * GK);
              nodes_wgrad_kernel<Model, R, GK><<<grid, dim3(kBlockE), 0, D.stream>>>(
                  P, pd_arg(D, chunks, c), D.earth, D.nodes.E.ptr, int(nEn),
                  D.nodes.C.ptr, int(nrow), D.wE.ptr, D.wC.ptr, D.binOfRow.ptr,
                  fBins.nglE, int(D.binned.rows.size()), int(nEb), lo, w[k],
                  D.partial.ptr);
              OPG_CUDA(cudaGetLastError());
            }
            sum_partials(chunks[c], g);
          }
        }
      }

      void weighted_grad_points_binned(const Prepared& P, const Chunks& chunks,
                                       int npar, const R* E, const R* cosZ,
                                       const uint8_t* nubar, size_t n, const R* w,
                                       const int* bin, int nbins, R* G,
                                       const R* extra = nullptr) override
      {
        if constexpr (!GT::enabled) { this->no_grad(); }
        else {
          if (!fHaveEarth) throw std::logic_error("CudaEngine: earth not set");
          check_param_size();
          upload_chunks(chunks);
          std::fill(G, G + size_t(nbins) * npar, R(0));
          if (nbins <= 0) return;
          const size_t nd = fDev.size(), chunk = (n + nd - 1) / nd;
          std::vector<size_t> off(nd), cnt(nd);
          for (size_t k = 0; k < nd; k++) {
            off[k] = std::min(n, k * chunk);
            cnt[k] = std::min(n, off[k] + chunk) - off[k];
            Device& D = *fDev[k];
            if (!cnt[k]) continue;
            OPG_CUDA(cudaSetDevice(D.id));
            D.pE.upload(D.id, E + off[k], cnt[k], D.stream);
            D.pC.upload(D.id, cosZ + off[k], cnt[k], D.stream);
            D.pNb.upload(D.id, nubar + off[k], cnt[k], D.stream);
            xptr(D, extra, off[k], cnt[k], n);
            std::vector<R> loc(N * N * cnt[k]);
            for (int ab = 0; ab < N * N; ab++)
              std::copy(w + ab * n + off[k], w + ab * n + off[k] + cnt[k],
                        loc.begin() + ab * cnt[k]);
            D.wts.upload(D.id, loc.data(), loc.size(), D.stream);
            // local bins (out of range -> -1) and events grouped by bin
            std::vector<int> lb(cnt[k]), start(size_t(nbins) + 1, 0), perm;
            for (size_t i = 0; i < cnt[k]; i++) {
              const int b = bin[off[k] + i];
              lb[i]       = (b >= 0 && b < nbins) ? b : -1;
              if (lb[i] >= 0) start[size_t(lb[i]) + 1]++;
            }
            for (int b = 0; b < nbins; b++) start[b + 1] += start[b];
            perm.resize(size_t(start[nbins]));
            std::vector<int> fill(start.begin(), start.end() - 1);
            for (size_t i = 0; i < cnt[k]; i++)
              if (lb[i] >= 0) perm[size_t(fill[lb[i]]++)] = int(i);
            D.pBin.upload(D.id, lb.data(), lb.size(), D.stream);
            D.binStart.upload(D.id, start.data(), start.size(), D.stream);
            D.binPerm.upload(D.id, perm.data(), perm.size(), D.stream);
            D.contrib.resize(D.id, cnt[k] * GK);
            OPG_CUDA(cudaStreamSynchronize(D.stream));
          }
          if constexpr (has_adjoint_v<Model>)
            if (use_adj(npar)) {
              upload_adj_chunks(chunks);
              for (size_t k = 0; k < nd; k++) {
                Device& D = *fDev[k];
                if (!cnt[k]) continue;
                OPG_CUDA(cudaSetDevice(D.id));
                D.contrib.resize(D.id, cnt[k] * size_t(npar));
                points_adj_contrib_kernel<Model, R, GK>
                    <<<blocks_for(cnt[k], kAdjBlockT), kAdjBlockT, 0, D.stream>>>(
                        P, D.pd.ptr, npar, D.earth, D.pE.ptr, D.pC.ptr, D.pNb.ptr, D.pX.ptr,
                        D.pBin.ptr, cnt[k], D.wts.ptr, D.contrib.ptr, D.adjErr.ptr);
                OPG_CUDA(cudaGetLastError());
                D.partial.resize(D.id, size_t(nbins) * kAdjMaxPar);
                if (D.binPerm.n)
                  bin_sum_adj_kernel<R><<<unsigned(nbins), kAdjBlockT, 0, D.stream>>>(
                      D.contrib.ptr, npar, D.binPerm.ptr, D.binStart.ptr, D.partial.ptr);
                else
                  OPG_CUDA(cudaMemsetAsync(D.partial.ptr, 0, D.partial.n * sizeof(R),
                                           D.stream));
                OPG_CUDA(cudaGetLastError());
              }
              for (size_t k = 0; k < nd; k++) {  // fixed order
                Device& D = *fDev[k];
                if (!cnt[k]) continue;
                OPG_CUDA(cudaSetDevice(D.id));
                D.partialHost.resize(D.partial.n);
                OPG_CUDA(cudaMemcpyAsync(D.partialHost.data(), D.partial.ptr,
                                         D.partial.n * sizeof(R), cudaMemcpyDeviceToHost,
                                         D.stream));
                OPG_CUDA(cudaStreamSynchronize(D.stream));
                for (int b = 0; b < nbins; b++)
                  for (int p = 0; p < npar; p++)
                    G[size_t(b) * npar + p] += D.partialHost[size_t(b) * kAdjMaxPar + p];
              }
              check_adj_err();
              return;
            }
          for (size_t c = 0; c < chunks.size(); c++) {
            for (size_t k = 0; k < nd; k++) {
              Device& D = *fDev[k];
              if (!cnt[k]) continue;
              OPG_CUDA(cudaSetDevice(D.id));
              points_contrib_kernel<Model, R, GK>
                  <<<blocks_for(cnt[k], 128), 128, 0, D.stream>>>(
                      P, pd_arg(D, chunks, c), D.earth, D.pE.ptr, D.pC.ptr, D.pNb.ptr,
                      D.pX.ptr, D.pBin.ptr, cnt[k], D.wts.ptr, D.contrib.ptr);
              OPG_CUDA(cudaGetLastError());
              D.partial.resize(D.id, size_t(nbins) * GK);
              if (D.binPerm.n)
                bin_sum_kernel<R, GK><<<unsigned(nbins), 128, 0, D.stream>>>(
                    D.contrib.ptr, D.binPerm.ptr, D.binStart.ptr, D.partial.ptr);
              else
                OPG_CUDA(cudaMemsetAsync(D.partial.ptr, 0, D.partial.n * sizeof(R),
                                         D.stream));
              OPG_CUDA(cudaGetLastError());
            }
            // add device partials in a fixed order
            const auto& ch = chunks[c];
            for (size_t k = 0; k < nd; k++) {
              Device& D = *fDev[k];
              if (!cnt[k]) continue;
              OPG_CUDA(cudaSetDevice(D.id));
              D.partialHost.resize(D.partial.n);
              OPG_CUDA(cudaMemcpyAsync(D.partialHost.data(), D.partial.ptr,
                                       D.partial.n * sizeof(R), cudaMemcpyDeviceToHost,
                                       D.stream));
              OPG_CUDA(cudaStreamSynchronize(D.stream));
              for (int b = 0; b < nbins; b++)
                for (int kk = 0; kk < ch.count; kk++)
                  G[size_t(b) * npar + ch.offset + kk] += D.partialHost[size_t(b) * GK + kk];
            }
          }
        }
      }

    private:
      /// A (E x cosZ-rows) grid resident on one device.
      struct Slab {
          DevBuf<R>           E, C, probs;
          PinnedBuf<R>        host;
          std::vector<size_t> rows;  ///< global row (cosine or bin) indices
      };

      struct Device {
          int                id     = 0;
          cudaStream_t       stream = nullptr;
          DevBuf<R>          radius, density, zoa;
          DevBuf<int>        type;
          EarthView<R>       earth{};
          Slab               grid;      ///< plain (E, cosZ) grid
          Slab               nodes;     ///< GL node grid (rows = C nodes)
          Slab               binned;    ///< reduced bins (rows = C bins)
          Slab               gslab;     ///< grid gradients (rows as grid)
          Slab               bgslab;    ///< binned gradients (rows as binned)
          DevBuf<R>          ngrad;     ///< node-grid gradients (binned mode)
          DevBuf<int>        binOfRow;  ///< local C bin of each node row
          DevBuf<int>        pBin, binStart, binPerm;  ///< per-bin event lists
          DevBuf<R>          contrib;   ///< per-event weighted contractions
          DevBuf<R>          pX;        ///< per-event extra inputs [x][cnt]
          DevBuf<int>        adjErr;    ///< adjoint path overflow flag
          DevBuf<R>          wts;       ///< weights for weighted gradients
          DevBuf<typename grad_traits<Model>::Prepared> pd;  ///< dual states (GradArg)
          PinnedBuf<R>       wtsHost;   ///< pinned staging for wts
          DevBuf<R>          partial;   ///< per-block partial sums
          std::vector<R>     partialHost;
          DevBuf<R>          pGrad;     ///< one-shot gradient output
          DevBuf<R>          wE, wC;    ///< GL weights (local C bins)
          DevBuf<size_t>     offC;      ///< node offsets of local C bins
          // one-shot buffers
          DevBuf<R>          pE, pC, pOut;
          DevBuf<uint8_t>    pNb;
          DevBuf<Segment<R>> path;
          std::vector<R>     pHost;
      };

      /// Launch the grid kernel on a slab.
      void launch_grid(Device& D, Slab& G, size_t nE, const Prepared& P,
                       int lo, int hi)
      {
        if (G.rows.empty() || nE == 0) return;
        const size_t nrow = G.C.n;
        dim3 block(kBlockE);
        dim3 grid(unsigned((nE + kBlockE - 1) / kBlockE),
                  unsigned(std::min<size_t>(nrow, 65535)), unsigned(hi - lo + 1));
        grid_kernel<Model, R><<<grid, block, 0, D.stream>>>(
            P, D.earth, G.E.ptr, int(nE), G.C.ptr, int(nrow), lo, G.probs.ptr);
        OPG_CUDA(cudaGetLastError());
      }

      /// Copy each device's slab (selected by member pointer) to the host.
      /// With one device the slab's pinned buffer is the result. With several,
      /// rows were dealt round-robin (row r of device k is global row
      /// k + r*nd), so each channel is one strided 2D copy straight into the
      /// pinned full array full[2*N*N][nrows][ncols].
      const R* gather(Slab Device::*which, size_t nrows, size_t ncols,
                      PinnedBuf<R>& full, bool& valid,
                      size_t nchan = size_t(2 * N * N))
      {
        const size_t nd = fDev.size();
        if (valid) return nd == 1 ? (fDev[0].get()->*which).host.ptr : full.ptr;
        for (size_t k = 0; k < nd; k++) {
          Device& D = *fDev[k];
          Slab&   G = D.*which;
          if (G.probs.n == 0) continue;
          OPG_CUDA(cudaSetDevice(D.id));
          if (nd == 1) {
            OPG_CUDA(cudaMemcpyAsync(G.host.ptr, G.probs.ptr, G.probs.n * sizeof(R),
                                     cudaMemcpyDeviceToHost, D.stream));
            continue;
          }
          const size_t nr = G.rows.size();
          for (size_t ch = 0; ch < nchan; ch++)
            OPG_CUDA(cudaMemcpy2DAsync(
                full.ptr + (ch * nrows + k) * ncols, nd * ncols * sizeof(R),
                G.probs.ptr + ch * nr * ncols, ncols * sizeof(R),
                ncols * sizeof(R), nr, cudaMemcpyDeviceToHost, D.stream));
        }
        for (auto& Dp : fDev) {
          OPG_CUDA(cudaSetDevice(Dp->id));
          OPG_CUDA(cudaStreamSynchronize(Dp->stream));
        }
        valid = true;
        return nd == 1 ? (fDev[0].get()->*which).host.ptr : full.ptr;
      }

      /// Copy per-device [ch][i] chunks (buffer selected by member pointer)
      /// into out[ch][i] (i < n).
      void gather_chunks(const std::vector<size_t>& off,
                         const std::vector<size_t>& cnt, size_t n, R* out,
                         DevBuf<R> Device::*buf = &Device::pOut,
                         size_t nchan = size_t(N * N))
      {
        for (size_t k = 0; k < fDev.size(); k++) {
          Device& D = *fDev[k];
          if (!cnt[k]) continue;
          OPG_CUDA(cudaSetDevice(D.id));
          DevBuf<R>& B = D.*buf;
          D.pHost.resize(B.n);
          OPG_CUDA(cudaMemcpyAsync(D.pHost.data(), B.ptr, B.n * sizeof(R),
                                   cudaMemcpyDeviceToHost, D.stream));
          OPG_CUDA(cudaStreamSynchronize(D.stream));
          for (size_t ch = 0; ch < nchan; ch++)
            std::copy(D.pHost.begin() + ch * cnt[k],
                      D.pHost.begin() + (ch + 1) * cnt[k], out + ch * n + off[k]);
        }
      }

      /// Add per-block partial sums of one gradient chunk from all devices
      /// to g, in a fixed order (device, block).
      void sum_partials(const GradChunk<Model>& ch, R* g)
      {
        for (auto& Dp : fDev) {
          Device& D = *Dp;
          if (D.partial.n == 0) continue;
          OPG_CUDA(cudaSetDevice(D.id));
          D.partialHost.resize(D.partial.n);
          OPG_CUDA(cudaMemcpyAsync(D.partialHost.data(), D.partial.ptr,
                                   D.partial.n * sizeof(R), cudaMemcpyDeviceToHost,
                                   D.stream));
        }
        for (auto& Dp : fDev) {
          Device& D = *Dp;
          if (D.partial.n == 0) continue;
          OPG_CUDA(cudaSetDevice(D.id));
          OPG_CUDA(cudaStreamSynchronize(D.stream));
          const size_t nblk = D.partial.n / grad_traits<Model>::K;
          for (size_t b = 0; b < nblk; b++)
            for (int k = 0; k < ch.count; k++)
              g[ch.offset + k] += D.partialHost[b * grad_traits<Model>::K + k];
          D.partial.release();  // sized per launch
        }
      }

      /// Upload the extra inputs of events off .. off+cnt (global layout
      /// extra[x * n + i]) to D.pX as [x][cnt]; returns the device pointer,
      /// or nullptr (and D.pX empty) without extras.
      const R* xptr(Device& D, const R* extra, size_t off, size_t cnt, size_t n)
      {
        constexpr int NX = n_extra_v<Model>;
        if (NX == 0 || !extra) {
          D.pX.release();
          return nullptr;
        }
        std::vector<R> loc(size_t(NX) * cnt);
        for (int x = 0; x < NX; x++)
          std::copy(extra + size_t(x) * n + off, extra + size_t(x) * n + off + cnt,
                    loc.begin() + size_t(x) * cnt);
        D.pX.upload(D.id, loc.data(), loc.size(), D.stream);
        OPG_CUDA(cudaStreamSynchronize(D.stream));
        return D.pX.ptr;
      }

      /// The value state is a kernel argument (CUDA limit: 4 KB in total);
      /// the dual state too unless GradArg passes it by pointer.
      static void check_param_size()
      {
        static_assert(sizeof(Prepared) < 3500,
                      "Prepared state too large to pass as kernel argument");
      }

      /// Copy the dual states of all gradient passes to every device (only
      /// when they are passed by pointer). The copies are ordered on each
      /// device's stream, after any kernels still using the previous ones.
      /// Whether weighted modes with npar parameters use the adjoint path.
      bool use_adj(int npar) const
      {
        return !fDev.empty() && this->use_adjoint(npar, fDev[0]->earth);
      }

      /// Upload all gradient chunks to D.pd and clear D.adjErr (adjoint path).
      void upload_adj_chunks(const Chunks& chunks)
      {
        fPDHost.resize(chunks.size());
        for (size_t c = 0; c < chunks.size(); c++) fPDHost[c] = chunks[c].P;
        for (auto& Dp : fDev) {
          OPG_CUDA(cudaSetDevice(Dp->id));
          Dp->pd.upload(Dp->id, fPDHost.data(), fPDHost.size(), Dp->stream);
          Dp->adjErr.resize(Dp->id, 1);
          OPG_CUDA(cudaMemsetAsync(Dp->adjErr.ptr, 0, sizeof(int), Dp->stream));
        }
      }

      /// g[p] += partials (blocks of kAdjMaxPar) of all devices, in a fixed
      /// order; throws if a kernel flagged a path too long for the adjoint.
      void sum_adj_partials(int npar, R* g)
      {
        for (auto& Dp : fDev) {
          Device& D = *Dp;
          if (D.partial.n == 0) continue;
          OPG_CUDA(cudaSetDevice(D.id));
          D.partialHost.resize(D.partial.n);
          OPG_CUDA(cudaMemcpyAsync(D.partialHost.data(), D.partial.ptr,
                                   D.partial.n * sizeof(R), cudaMemcpyDeviceToHost, D.stream));
        }
        for (auto& Dp : fDev) {
          Device& D = *Dp;
          if (D.partial.n == 0) continue;
          OPG_CUDA(cudaSetDevice(D.id));
          OPG_CUDA(cudaStreamSynchronize(D.stream));
          const size_t nblk = D.partial.n / kAdjMaxPar;
          for (size_t b = 0; b < nblk; b++)
            for (int p = 0; p < npar; p++) g[p] += D.partialHost[b * kAdjMaxPar + p];
          D.partial.release();
        }
        check_adj_err();
      }

      void check_adj_err()
      {
        for (auto& Dp : fDev) {
          int err = 0;
          OPG_CUDA(cudaSetDevice(Dp->id));
          if (Dp->adjErr.n)
            OPG_CUDA(cudaMemcpy(&err, Dp->adjErr.ptr, sizeof(int), cudaMemcpyDeviceToHost));
          if (err)
            throw std::runtime_error("CudaEngine: path longer than kAdjMaxSeg segments in "
                                     "the adjoint gradient (use set_adjoint_gradients(false))");
        }
      }

      void upload_chunks(const Chunks& chunks)
      {
        if constexpr (GradArg<Model>::by_pointer) {
          fPDHost.resize(chunks.size());
          for (size_t c = 0; c < chunks.size(); c++) fPDHost[c] = chunks[c].P;
          for (auto& Dp : fDev) {
            OPG_CUDA(cudaSetDevice(Dp->id));
            Dp->pd.upload(Dp->id, fPDHost.data(), fPDHost.size(), Dp->stream);
          }
        }
      }

      /// Kernel argument for gradient pass c on device D (see GradArg).
      static typename GradArg<Model>::type pd_arg(Device& D, const Chunks& chunks,
                                                  size_t c)
      {
        if constexpr (GradArg<Model>::by_pointer) {
          (void)chunks;
          return D.pd.ptr + c;
        }
        else {
          (void)D;
          return chunks[c].P;
        }
      }

      std::vector<std::unique_ptr<Device>> fDev;
      PinnedBuf<R>                         fProbs, fBinned;  ///< multi-GPU results
      PinnedBuf<R>                         fGradFull;        ///< multi-GPU gradients
      PinnedBuf<R>                         fBinnedGradFull;  ///< multi-GPU binned grads
      bool                                 fBinnedGradValid = false;
      std::vector<typename grad_traits<Model>::Prepared> fPDHost;  ///< staging (GradArg)
      int                                  fNpar      = 0;
      bool                                 fGradValid = false;
      BinSpec<R>                           fBins;
      size_t                               fNE = 0, fNC = 0;
      bool                                 fHaveEarth   = false;
      bool                                 fHostValid   = false;
      bool                                 fBinnedValid = false;
  };

  //...........................................................................
  template <class Model>
  std::unique_ptr<EngineBase<Model>> make_cuda_engine(const std::vector<int>& devices)
  {
    return std::unique_ptr<EngineBase<Model>>(new CudaEngine<Model>(devices));
  }

#define OPG_INSTANTIATE(M)                                                    \
  template std::unique_ptr<EngineBase<M>> make_cuda_engine<M>(               \
      const std::vector<int>&);
  OPG_FOR_EACH_MODEL(OPG_INSTANTIATE)
#undef OPG_INSTANTIATE

} // namespace opg
