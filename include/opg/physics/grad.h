///////////////////////////////////////////////////////////////////////////////
/// \file grad.h
///
/// \brief Gradient support detection and evolution with derivatives.
///
/// A model supports gradients if it defines `grad_chunk` (number of
/// derivative directions per pass, K), templates ParamsT<S> / PreparedT<S>
/// on the scalar type, and provides param_names(), param_ref<S>(),
/// cast<S>(), prepare_generic<S>(), and the K-templated initial_grad,
/// step_grad and finalize_grad. Gradient code can be compiled out with
/// OPG_DISABLE_GRADIENTS (CMake: OPG_ENABLE_GRADIENTS=OFF).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_PHYSICS_GRAD_H
#define OPG_PHYSICS_GRAD_H

#include <type_traits>

#include "opg/core/dual.h"
#include "opg/physics/eigen_grad.h"
#include "opg/physics/propagate.h"

namespace opg {

  namespace detail {
    struct NoGradPrepared {};
  } // namespace detail

  template <class Model, class = void> struct grad_traits {
      static constexpr bool enabled = false;
      static constexpr int  K       = 1;
      using Prepared                = detail::NoGradPrepared;
  };

#ifndef OPG_DISABLE_GRADIENTS
  template <class Model>
  struct grad_traits<Model, std::void_t<decltype(Model::grad_chunk)>> {
      static constexpr bool enabled = true;
      static constexpr int  K       = Model::grad_chunk;
      using Dual_                   = Dual<typename Model::Real, K>;
      using Prepared = typename Model::template PreparedT<Dual_>;
  };
#endif

  /// Values and derivatives (columns of S and dS_k) through a PREM path.
  /// The value part S is computed exactly as by evolve_prem.
  template <class Model, class R, int K>
  OPG_HD inline void evolve_prem_grad(
      const typename Model::Prepared&                       P,
      const typename Model::template PreparedT<Dual<R, K>>& PD,
      const EarthView<R>& earth, R E, R cosZ, bool nubar,
      Mat<Model::N, R>& S, Mat<Model::N, R> (&dS)[K])
  {
    S = Model::initial(P, nubar);
    Model::template initial_grad<K>(PD, nubar, dS);
    for_each_segment(earth, cosZ, [&](const Segment<R>& s) {
      Model::template step_grad<K>(P, PD, E, nubar, s, S, dS);
    });
    Model::template finalize_grad<K>(P, PD, nubar, S, dS);
    Model::finalize(P, nubar, S);
  }

  /// Values and derivatives through an explicit list of segments.
  template <class Model, class R, int K>
  OPG_HD inline void evolve_path_grad(
      const typename Model::Prepared&                       P,
      const typename Model::template PreparedT<Dual<R, K>>& PD,
      const Segment<R>* path, int nseg, R E, bool nubar, Mat<Model::N, R>& S,
      Mat<Model::N, R> (&dS)[K])
  {
    S = Model::initial(P, nubar);
    Model::template initial_grad<K>(PD, nubar, dS);
    for (int k = 0; k < nseg; k++)
      Model::template step_grad<K>(P, PD, E, nubar, path[k], S, dS);
    Model::template finalize_grad<K>(P, PD, nubar, S, dS);
    Model::finalize(P, nubar, S);
  }

} // namespace opg

#endif
