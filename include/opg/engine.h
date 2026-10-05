///////////////////////////////////////////////////////////////////////////////
/// \file engine.h
///
/// \brief Backend interface used by opg::Propagator.
///
/// Result layouts (all structure-of-arrays, so that consecutive threads
/// write consecutive addresses):
///   grid   : probs[nu][a][b][iC][iE]   nu = 0 (neutrino) / 1 (antineutrino)
///   points : out[a][b][i]
///   path   : out[a][b][iE]
///   binned : avg[nu][a][b][iCbin][iEbin]
/// where P(a -> b) and a, b are flavour indices (0=e, 1=mu, 2=tau, 3=s).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_ENGINE_H
#define OPG_ENGINE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "opg/earth/prem.h"
#include "opg/physics/adjoint.h"
#include "opg/physics/grad.h"

namespace opg {

  /// Which neutrino types to compute.
  enum class Flavor : int { Neutrino = 1, Antineutrino = 2, Both = 3 };

  /// Measure used to average over energy bins.
  enum class EMeasure : int {
    Linear = 0,  ///< uniform in E
    Log    = 1,  ///< uniform in log E
    InvE   = 2   ///< uniform in 1/E (i.e. in L/E at fixed L)
  };

  /// How the cosZ direction of a bin is integrated.
  enum class CosZRule : int {
    /// Plain Gauss-Legendre in cosZ on each bin.
    Plain = 0,
    /// Split each bin at the cosines where trajectories graze an Earth layer
    /// boundary (where P(cosZ) has square-root kinks) and use Gauss-Legendre
    /// in u with cosZ = a + (b-a)(1 - cos(pi u))/2 on each piece, which
    /// restores fast convergence. Default.
    LayerAdapted = 1
  };

  /// Quadrature node grid for bin averaging (built by Propagator).
  /// Nodes of E bin b are nodesE[b*nglE .. (b+1)*nglE); cosZ bin c owns
  /// nodesC[offC[c] .. offC[c+1]) (a variable number with LayerAdapted).
  /// Weights are normalised to sum to 1 within each bin.
  template <class R> struct BinSpec {
      std::vector<R>      nodesE, wE, nodesC, wC;
      std::vector<size_t> offC;  ///< size nCb + 1
      size_t              nEb = 0, nCb = 0;
      int                 nglE = 0;
  };

  /// Variable of the bin edges given to the analytic averages.
  enum class BinVar : int {
    E   = 0,  ///< edges in E (GeV)
    LoE = 1   ///< edges in L/E (km/GeV), L = total path length
  };

  /// Options of the analytic bin averages (avg/analytic.h).
  struct AnalyticAvgOptions {
      /// Target for the first-order error (sets the sub-bin width r from
      /// the matter phase of the path: r = sqrt(tol / (0.005 sum V L)), V
      /// at Z/A = 1); observed errors are about 0.5-1 tol. The cost grows
      /// as 1/sqrt(tol).
      double tol = 1e-6;
      /// Largest relative sub-bin width r = u_hi / u_lo - 1.
      double max_width = 0.1;
      /// > 0: this many sub-bins per bin, uniform in 1/E (tol, max_width
      /// unused).
      int nsub = 0;
      /// Fade-out of fully fast pairs, in |mu_n - mu_m| h (see analytic.h).
      double fast_begin = 1e6;
      double fast_end   = 1e7;
      /// OpenMP threads of the host computation (0: default).
      int threads = 0;
      /// Compute on the host even with a GPU backend.
      bool host = false;
      /// Batched averages: per-(point, sub-bin) device scratch per chunk of
      /// points (MB).
      double batch_scratch_mb = 1024;
  };

  /// One sub-bin of the analytic averages: uniform in u = 1/E (GeV^-1) on
  /// [u0 - h, u0 + h] with weight weight * (1 + beta (u - u0) / h), added to
  /// bin `bin`.
  struct AnalyticSubBin {
      int    bin;
      double u0, h, weight, beta;
  };

  /// One gradient pass: the prepared state in dual numbers, seeded with
  /// parameters offset .. offset+count-1 (count <= grad_traits<Model>::K).
  template <class Model> struct GradChunk {
      typename grad_traits<Model>::Prepared P;
      int                                   offset = 0;
      int                                   count  = 0;
  };

  template <class Model> class EngineBase {
    protected:
      [[noreturn]] static void no_grad()
      {
        throw std::logic_error(
            "OscProbGPU: gradients are not available for this model/backend "
            "(or were disabled at build time)");
      }

    public:
      using R                = typename Model::Real;
      using Prepared         = typename Model::Prepared;
      static constexpr int N = Model::N;

      virtual ~EngineBase() = default;

      /// Weighted gradient modes use the reverse-mode (adjoint) path where
      /// the model provides it (on by default; off: forward mode, e.g. for
      /// cross-checks).
      void set_adjoint(bool on) { fAdjoint = on; }
      bool adjoint() const { return fAdjoint; }

    protected:
      bool fAdjoint = true;

      /// Whether a weighted mode with npar parameters through the Earth
      /// model e uses the adjoint path (model support, the switch, and the
      /// bounds of adjoint_prem on parameters and path segments).
      bool use_adjoint(int npar, const EarthView<R>& e) const
      {
        if constexpr (has_adjoint_v<Model>)
          return fAdjoint && npar <= kAdjMaxPar && e.nlayers + e.det_layer <= kAdjMaxSeg;
        else {
          (void)npar, (void)e;
          return false;
        }
      }

    public:

      virtual void set_earth(const PremModel& earth) = 0;

      // --- grid mode ---------------------------------------------------------
      virtual void set_grid(const std::vector<R>& E,
                            const std::vector<R>& cosZ)       = 0;
      virtual void calculate(const Prepared& P, Flavor which) = 0;
      virtual void wait()                                     = 0;
      /// Host pointer to probs[2][N][N][nC][nE] (copies from device if needed)
      virtual const R* host_probs() = 0;
      /// Device pointer for GPU consumers (nullptr for CPU backends).
      virtual const R* device_probs(int /*device_index*/) { return nullptr; }

      // --- bin-averaged grid mode ----------------------------------------------
      virtual void set_bins(const BinSpec<R>& spec)                  = 0;
      virtual void calculate_binned(const Prepared& P, Flavor which) = 0;
      /// Host pointer to avg[2][N][N][nCbins][nEbins]
      virtual const R* host_binned() = 0;
      virtual const R* device_binned(int /*device_index*/) { return nullptr; }

      // --- gradients --------------------------------------------------------
      // Layouts: grid grad[nu][p][a][b][iC][iE]; points grad[p][a][b][i];
      // path grad[p][a][b][iE]. Weighted modes return
      //   g[p] = sum over points and channels of w * dP/dp,
      // with w in the layout of the corresponding probabilities, summed in
      // a fixed order (deterministic for a given device configuration).
      using Chunks = std::vector<GradChunk<Model>>;

      /// Analytic averages on the device (avg/analytic.h): per-sub-bin
      /// results sp[i][a][b] and, with chunks, sg[i][p][a][b], for the
      /// vacuum term A = dH/du and its derivatives dA[p]. Returns false if
      /// the backend has none (the caller then computes on the host).
      virtual bool analytic_subbins(const Prepared&, const Chunks*, int /*npar*/,
                                    const Mat<N, R>& /*A*/, const Mat<N, R>* /*dA*/,
                                    const AnalyticSubBin*, size_t /*nsub*/,
                                    const Segment<R>*, int /*nseg*/, bool /*nubar*/,
                                    double /*fast_begin*/, double /*fast_end*/, R* /*sp*/,
                                    R* /*sg*/)
      {
        return false;
      }

      /// Batched analytic averages over parameter points
      /// (Propagator::analytic_batch): device state of one batch handle on
      /// device `device_index` for these sub-bins and path, with at most
      /// scratch_bytes of per-(point, sub-bin) scratch; nullptr if the
      /// backend has no device implementation.
      virtual std::shared_ptr<void> analytic_batch_create(int /*device_index*/,
                                                          const std::vector<AnalyticSubBin>&,
                                                          size_t /*nbins*/,
                                                          const std::vector<Segment<R>>&,
                                                          size_t /*scratch_bytes*/)
      {
        return nullptr;
      }
      /// Run npts points of a batch: prepared states P[p], vacuum terms
      /// A[p], gradient passes G[p * nchunk + c] and dA[p * npar + q] (none
      /// without gradients). Writes out[p][a][b][bin] and, with gradients,
      /// dout[p][q][a][b][bin], to device memory of the batch's device on
      /// `stream` (device_out, asynchronous) or to host memory (blocking).
      virtual void analytic_batch_run(void* /*state*/, bool /*nubar*/, double /*fast_begin*/,
                                      double /*fast_end*/, size_t /*npts*/, const Prepared*,
                                      const Mat<N, R>*, const GradChunk<Model>*, int /*nchunk*/,
                                      const Mat<N, R>*, int /*npar*/, R* /*out*/, R* /*dout*/,
                                      void* /*stream*/, bool /*device_out*/)
      {
        throw std::logic_error("OscProbGPU: no device implementation of batched analytic "
                               "averages for this backend/model");
      }

      /// Grid probabilities and gradients (probs as in calculate()).
      virtual void calculate_grad(const Prepared&, const Chunks&, int /*npar*/,
                                  Flavor)
      {
        no_grad();
      }
      virtual const R* host_grad()
      {
        no_grad();
        return nullptr;
      }
      virtual const R* device_grad(int) { return nullptr; }
      virtual void     weighted_grad(const Prepared&, const Chunks&, int, Flavor,
                                     const R* /*w*/, R* /*g*/)
      {
        no_grad();
      }
      /// Weighted modes with the weights already on the GPU: w[k] is device
      /// memory on the k-th device, in the layout of device_probs(k) (grid)
      /// or device_binned(k) (binned). GPU backend only.
      virtual void weighted_grad_device(const Prepared&, const Chunks&, int, Flavor,
                                        const std::vector<const R*>&, R*)
      {
        throw std::logic_error("weighted_gradient_device: needs the CUDA backend");
      }
      virtual void weighted_grad_binned_device(const Prepared&, const Chunks&, int,
                                               Flavor, const std::vector<const R*>&, R*)
      {
        throw std::logic_error("weighted_gradient_binned_device: needs the CUDA "
                               "backend");
      }
      // Event-list modes take optional per-event extra inputs (models with
      // n_extra > 0, e.g. azimuth and sidereal time): extra[x * n + i], or
      // nullptr for the model's defaults.
      virtual void prob_points_grad(const Prepared&, const Chunks&, int,
                                    const R*, const R*, const uint8_t*, size_t,
                                    R* /*P*/, R* /*G*/, const R* /*extra*/ = nullptr)
      {
        no_grad();
      }
      virtual void weighted_grad_points(const Prepared&, const Chunks&, int,
                                        const R*, const R*, const uint8_t*,
                                        size_t, const R* /*w*/, R* /*g*/,
                                        const R* /*extra*/ = nullptr)
      {
        no_grad();
      }
      /// Per-analysis-bin weighted gradient over an event list:
      /// G[b][p] = sum over events i with bin[i] == b (bins < 0 ignored) of
      /// sum_ab w[ab][i] dP_ab(i)/dp, summed in a fixed order.
      virtual void weighted_grad_points_binned(const Prepared&, const Chunks&, int,
                                               const R*, const R*, const uint8_t*,
                                               size_t, const R* /*w*/,
                                               const int* /*bin*/, int /*nbins*/,
                                               R* /*G*/, const R* /*extra*/ = nullptr)
      {
        no_grad();
      }
      virtual void prob_path_grad(const Prepared&, const Chunks&, int, const R*,
                                  size_t, const Segment<R>*, int, bool,
                                  R* /*P*/, R* /*G*/)
      {
        no_grad();
      }

      /// Bin-averaged probabilities (as calculate_binned()) and gradients
      /// binned_grad[nu][p][a][b][iCb][iEb] (same Gauss-Legendre weights).
      virtual void calculate_binned_grad(const Prepared&, const Chunks&, int, Flavor)
      {
        no_grad();
      }
      virtual const R* host_binned_grad()
      {
        no_grad();
        return nullptr;
      }
      virtual const R* device_binned_grad(int) { return nullptr; }
      /// g[p] = sum w * d(avg P)/dp, w in the layout of host_binned().
      virtual void weighted_grad_binned(const Prepared&, const Chunks&, int, Flavor,
                                        const R* /*w*/, R* /*g*/)
      {
        no_grad();
      }

      // --- one-shot modes ---------------------------------------------------
      /// Event list: out[a][b][i], i < n
      virtual void prob_points(const Prepared& P, const R* E, const R* cosZ,
                               const uint8_t* nubar, size_t n, R* out,
                               const R* extra = nullptr) = 0;
      /// Fixed path: out[a][b][iE]
      virtual void prob_path(const Prepared& P, const R* E, size_t nE,
                             const Segment<R>* path, int nseg, bool nubar,
                             R* out) = 0;
  };

} // namespace opg

#endif
