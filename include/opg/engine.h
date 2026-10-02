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
/// where P(a -> b) and a, b are flavour indices (0=e, 1=mu, 2=tau, 3=s).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_ENGINE_H
#define OPG_ENGINE_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "opg/earth/prem.h"

namespace opg {

  /// Which neutrino types to compute.
  enum class Flavor : int { Neutrino = 1, Antineutrino = 2, Both = 3 };

  /// Measure used to average over energy bins.
  enum class EMeasure : int {
    Linear = 0,  ///< uniform in E
    Log    = 1,  ///< uniform in log E
    InvE   = 2   ///< uniform in 1/E (i.e. in L/E at fixed L)
  };

  template <class Model> class EngineBase {
    public:
      using R                = typename Model::Real;
      using Prepared         = typename Model::Prepared;
      static constexpr int N = Model::N;

      virtual ~EngineBase() = default;

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

      // --- one-shot modes ---------------------------------------------------
      /// Event list: out[a][b][i], i < n
      virtual void prob_points(const Prepared& P, const R* E, const R* cosZ,
                               const uint8_t* nubar, size_t n, R* out) = 0;
      /// Fixed path: out[a][b][iE]
      virtual void prob_path(const Prepared& P, const R* E, size_t nE,
                             const Segment<R>* path, int nseg, bool nubar,
                             R* out) = 0;
  };

} // namespace opg

#endif
