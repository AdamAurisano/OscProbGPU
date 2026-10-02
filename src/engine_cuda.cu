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
        store_probs<N, R>(S, base + size_t(ic) * nE + ie, stride);
      }
    }

    template <class Model, class R>
    __global__ void points_kernel(const typename Model::Prepared P,
                                  const EarthView<R> earth,
                                  const R* __restrict__ E,
                                  const R* __restrict__ C,
                                  const uint8_t* __restrict__ nubar, size_t n,
                                  size_t stride, R* __restrict__ out)
    {
      constexpr int N = Model::N;
      for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
           i += size_t(gridDim.x) * blockDim.x) {
        auto S = evolve_prem<Model, R>(P, earth, E[i], C[i], nubar[i] != 0);
        store_probs<N, R>(S, out + i, stride);
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
        store_probs<N, R>(S, out + i, stride);
      }
    }

    /// Weighted Gauss-Legendre reduction of a node grid into bin averages.
    /// node: [2][N][N][nCn][nEn] (local C nodes), out: [2][N][N][nCb][nEb].
    template <class R, int N>
    __global__ void reduce_bins_kernel(const R* __restrict__ node, size_t nCn,
                                       size_t nEn, const R* __restrict__ wE,
                                       const R* __restrict__ wC,
                                       const size_t* __restrict__ offC, int nglE,
                                       size_t nEb, size_t nCb, int nb_first,
                                       R* __restrict__ out)
    {
      const size_t ieb = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
      if (ieb >= nEb) return;
      const int ch = blockIdx.z + nb_first * N * N;  // channel incl. nubar
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
        if (nd > 1) fProbs.assign(2 * N * N * fNC * fNE, R(0));
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
        if (nd > 1) fBinned.assign(2 * N * N * B.nCb * B.nEb, R(0));
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
                       const uint8_t* nubar, size_t n, R* out) override
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
              P, D.earth, D.pE.ptr, D.pC.ptr, D.pNb.ptr, cnt[k], cnt[k],
              D.pOut.ptr);
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

      /// Copy each device's slab (selected by member pointer) to the host and
      /// gather its rows into full[2*N*N][nrows][ncols].
      const R* gather(Slab Device::*which, size_t nrows, size_t ncols,
                      std::vector<R>& full, bool& valid)
      {
        if (valid) return fDev.size() == 1 ? (fDev[0].get()->*which).host.ptr
                                           : full.data();
        for (auto& Dp : fDev) {
          Device& D = *Dp;
          Slab&   G = D.*which;
          if (G.probs.n == 0) continue;
          OPG_CUDA(cudaSetDevice(D.id));
          OPG_CUDA(cudaMemcpyAsync(G.host.ptr, G.probs.ptr, G.probs.n * sizeof(R),
                                   cudaMemcpyDeviceToHost, D.stream));
        }
        for (auto& Dp : fDev) {
          Device& D = *Dp;
          Slab&   G = D.*which;
          OPG_CUDA(cudaSetDevice(D.id));
          OPG_CUDA(cudaStreamSynchronize(D.stream));
          if (fDev.size() == 1) continue;  // pinned buffer is the result
          const size_t nr = G.rows.size();
          for (size_t ch = 0; ch < size_t(2 * N * N); ch++)
            for (size_t r = 0; r < nr; r++)
              std::copy(G.host.ptr + (ch * nr + r) * ncols,
                        G.host.ptr + (ch * nr + r + 1) * ncols,
                        full.begin() + (ch * nrows + G.rows[r]) * ncols);
        }
        valid = true;
        return fDev.size() == 1 ? (fDev[0].get()->*which).host.ptr : full.data();
      }

      /// Copy per-device [a][b][i] chunks into out[a][b][i] (i < n).
      void gather_chunks(const std::vector<size_t>& off,
                         const std::vector<size_t>& cnt, size_t n, R* out)
      {
        for (size_t k = 0; k < fDev.size(); k++) {
          Device& D = *fDev[k];
          if (!cnt[k]) continue;
          OPG_CUDA(cudaSetDevice(D.id));
          D.pHost.resize(D.pOut.n);
          OPG_CUDA(cudaMemcpyAsync(D.pHost.data(), D.pOut.ptr,
                                   D.pOut.n * sizeof(R),
                                   cudaMemcpyDeviceToHost, D.stream));
        }
        for (size_t k = 0; k < fDev.size(); k++) {
          Device& D = *fDev[k];
          if (!cnt[k]) continue;
          OPG_CUDA(cudaSetDevice(D.id));
          OPG_CUDA(cudaStreamSynchronize(D.stream));
          for (int ch = 0; ch < N * N; ch++)
            std::copy(D.pHost.begin() + ch * cnt[k],
                      D.pHost.begin() + (ch + 1) * cnt[k],
                      out + ch * n + off[k]);
        }
      }

      std::vector<std::unique_ptr<Device>> fDev;
      std::vector<R>                       fProbs, fBinned;
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
