///////////////////////////////////////////////////////////////////////////////
/// \file propagate.h
///
/// \brief Generic per-point propagation shared by every backend.
///
/// The evolution of all N initial flavours is tracked at once in the
/// columns of S (S(:, a) is the state that started as flavour a), which is
/// what OscProb's ProbMatrix does with its vector of states. The operation
/// order inside each column is kept identical to PMNS_Base::PropagatePath.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_PHYSICS_PROPAGATE_H
#define OPG_PHYSICS_PROPAGATE_H

#include <cmath>

#include <type_traits>

#include "opg/core/constants.h"
#include "opg/core/matrix.h"
#include "opg/earth/prem.h"

namespace opg {

  /// S <- V diag(exp(-i lam L)) V^dagger S, for an eigensystem (V, lam) of
  /// a hermitian Hamiltonian and a segment of length L in eV^-1.
  /// Port of PMNS_Base::PropagatePath applied to each column of S.
  template <int N, class R>
  OPG_HD OPG_INLINE void apply_eigen_step(const Mat<N, R>& V, const R lam[N],
                                          R LengthIneV, Mat<N, R>& S)
  {
    Complex<R> phases[N];
    OPG_UNROLL
    for (int i = 0; i < N; i++) {
      R arg     = lam[i] * LengthIneV;
      phases[i] = Complex<R>(std::cos(arg), -std::sin(arg));
    }

    OPG_UNROLL
    for (int a = 0; a < N; a++) {
      Complex<R> buffer[N];
      OPG_UNROLL
      for (int i = 0; i < N; i++) {
        buffer[i] = Complex<R>(0, 0);
        OPG_UNROLL
        for (int j = 0; j < N; j++) buffer[i] += conj(V(j, i)) * S(j, a);
        buffer[i] *= phases[i];
      }
      OPG_UNROLL
      for (int i = 0; i < N; i++) {
        S(i, a) = Complex<R>(0, 0);
        OPG_UNROLL
        for (int j = 0; j < N; j++) S(i, a) += V(i, j) * buffer[j];
      }
    }
  }

  /// S <- U S for a general evolution operator U.
  template <int N, class R>
  OPG_HD OPG_INLINE void apply_operator(const Mat<N, R>& U, Mat<N, R>& S)
  {
    OPG_UNROLL
    for (int a = 0; a < N; a++) {
      Complex<R> buffer[N];
      OPG_UNROLL
      for (int i = 0; i < N; i++) {
        buffer[i] = Complex<R>(0, 0);
        OPG_UNROLL
        for (int j = 0; j < N; j++) buffer[i] += U(i, j) * S(j, a);
      }
      OPG_UNROLL
      for (int i = 0; i < N; i++) S(i, a) = buffer[i];
    }
  }

  /// Segment length in eV^-1 (km * kKm2eV), as OscProb.
  template <class R> OPG_HD OPG_INLINE R length_in_eV(R km)
  {
    return R(constants::kKm2eV) * km;
  }

  /// Write P(a -> b) = |S(b, a)|^2 to out[(a*N + b) * stride].
  template <int N, class R>
  OPG_HD OPG_INLINE void store_probs(const Mat<N, R>& S, R* out, size_t stride)
  {
    OPG_UNROLL
    for (int a = 0; a < N; a++)
      OPG_UNROLL
    for (int b = 0; b < N; b++) out[(a * N + b) * stride] = norm(S(b, a));
  }

  // Every model provides
  //   initial(P, nubar)        -> starting state (identity, or e.g. alpha^dagger)
  //   step(P, E, nubar, seg, S) -> S <- U_seg S
  //   finalize(P, nubar, S)    -> e.g. S <- alpha S
  // and the probability is P(a -> b) = |S(b, a)|^2. A model may instead use
  // its own state type (`using State = ...`, e.g. density matrices) and then
  // provides store_probs(S, out, stride) writing P(a -> b) to
  // out[(a*N + b) * stride].

  namespace detail {
    template <class Model, class = void> struct state_of {
        using type = Mat<Model::N, typename Model::Real>;
        static constexpr bool custom = false;
    };
    template <class Model>
    struct state_of<Model, std::void_t<typename Model::State>> {
        using type = typename Model::State;
        static constexpr bool custom = true;
    };
  } // namespace detail

  /// State evolved by a model (amplitude matrix unless the model defines one).
  template <class Model> using StateOf = typename detail::state_of<Model>::type;

  namespace detail {
    template <class Model, class = void> struct path_aware : std::false_type {};
    template <class Model>
    struct path_aware<Model, std::void_t<decltype(Model::initial(
                                 std::declval<const typename Model::Prepared&>(), bool(),
                                 typename Model::Real()))>> : std::true_type {};
  } // namespace detail

  /// Number of extra per-event inputs a model takes in event lists
  /// (Model::n_extra, e.g. azimuth and sidereal time; default 0).
  namespace detail {
    template <class Model, class = void> struct extra_count {
        static constexpr int value = 0;
    };
    template <class Model>
    struct extra_count<Model, std::void_t<decltype(Model::n_extra)>> {
        static constexpr int value = Model::n_extra;
    };
  } // namespace detail
  template <class Model> constexpr int n_extra_v = detail::extra_count<Model>::value;

  /// Initial state for a path through the Earth with direction cosZ (models
  /// whose Hamiltonian depends on the direction, e.g. sidereal LIV, provide
  /// initial(P, nubar, cosZ)); with per-event extra inputs ex (n_extra
  /// values, or nullptr) models that take them provide
  /// initial(P, nubar, cosZ, ex).
  template <class Model, class R>
  OPG_HD OPG_INLINE StateOf<Model> initial_state(const typename Model::Prepared& P,
                                                 bool nubar, R cosZ,
                                                 const R* ex = nullptr)
  {
    if constexpr (n_extra_v<Model> > 0) {
      if (ex) return Model::initial(P, nubar, cosZ, ex);
    }
    if constexpr (detail::path_aware<Model>::value)
      return Model::initial(P, nubar, cosZ);
    else
      return Model::initial(P, nubar);
  }

  /// Maximum number of extra per-event inputs.
  constexpr int kMaxExtra = 4;

  /// Extra inputs of event i, read from extra[x * n + i] into ex; returns
  /// ex, or nullptr if the model takes none or extra is nullptr.
  template <class Model, class R>
  OPG_HD OPG_INLINE const R* gather_extra(const R* extra, size_t i, size_t n,
                                          R (&ex)[kMaxExtra])
  {
    static_assert(n_extra_v<Model> <= kMaxExtra, "too many extra inputs");
    if constexpr (n_extra_v<Model> > 0) {
      if (!extra) return nullptr;
      OPG_UNROLL
      for (int x = 0; x < n_extra_v<Model>; x++) ex[x] = extra[size_t(x) * n + i];
      return ex;
    }
    else {
      (void)extra, (void)i, (void)n, (void)ex;
      return nullptr;
    }
  }

  /// Write P(a -> b) for a model state to out[(a*N + b) * stride].
  template <class Model, class R>
  OPG_HD OPG_INLINE void store_model_probs(const StateOf<Model>& S, R* out, size_t stride)
  {
    if constexpr (detail::state_of<Model>::custom)
      Model::store_probs(S, out, stride);
    else
      store_probs<Model::N, R>(S, out, stride);
  }

  /// Evolution matrix through a PREM path for direction cosZ.
  template <class Model, class R>
  OPG_HD inline StateOf<Model>
  evolve_prem(const typename Model::Prepared& P, const EarthView<R>& earth,
              R E, R cosZ, bool nubar, const R* ex = nullptr)
  {
    auto S = initial_state<Model, R>(P, nubar, cosZ, ex);
    for_each_segment(earth, cosZ, [&](const Segment<R>& s) {
      Model::step(P, E, nubar, s, S);
    });
    Model::finalize(P, nubar, S);
    return S;
  }

  /// Evolution matrix through an explicit list of segments.
  template <class Model, class R>
  OPG_HD inline StateOf<Model>
  evolve_path(const typename Model::Prepared& P, const Segment<R>* path,
              int nseg, R E, bool nubar)
  {
    auto S = Model::initial(P, nubar);
    for (int k = 0; k < nseg; k++) Model::step(P, E, nubar, path[k], S);
    Model::finalize(P, nubar, S);
    return S;
  }

} // namespace opg

#endif
