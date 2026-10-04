///////////////////////////////////////////////////////////////////////////////
/// \file adjoint.h
///
/// \brief Reverse-mode (adjoint) weighted gradients through a PREM path.
///
/// For models with Model::has_adjoint, the weighted gradient
///   g_p = sum_ab w_ab dP(a -> b)/dp
/// of one point is computed for all selected parameters at once: a forward
/// pass stores checkpoints of the state (every kAdjBlock segments), and the
/// backward pass recomputes each block of states, propagates the cotangents
/// with Model::adj_step (which also returns the segment cotangent G) and
/// contracts G with the derivative of the segment's model quantities for
/// every gradient chunk (Model::adj_contract_step, cheap). The initial state
/// and the final rotation contribute through adj_contract_initial and
/// adj_contract_final. The cost does not grow with the number of parameters
/// except for those contractions.
///
/// Model interface (in addition to step/initial):
///   has_adjoint, AdjSeg, adj_final, adj_step, adj_contract_step<K>,
///   adj_contract_initial<K>, adj_contract_final<K>  (see models/oqs.h).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_PHYSICS_ADJOINT_H
#define OPG_PHYSICS_ADJOINT_H

#include <type_traits>

#include "opg/earth/prem.h"
#include "opg/physics/grad.h"
#include "opg/physics/propagate.h"

namespace opg {

  /// Segments per checkpoint block, maximum number of checkpoints (so paths
  /// of up to kAdjBlock * kAdjMaxCkpt segments) and maximum number of
  /// gradient parameters of the adjoint path. Engines fall back to forward
  /// mode beyond these.
  constexpr int kAdjBlock    = 16;
  constexpr int kAdjMaxCkpt  = 16;
  constexpr int kAdjMaxSeg   = kAdjBlock * kAdjMaxCkpt;
  constexpr int kAdjMaxPar   = 64;

  namespace detail {
    template <class Model, class = void> struct has_adjoint : std::false_type {};
    template <class Model>
    struct has_adjoint<Model, std::void_t<decltype(Model::has_adjoint)>>
        : std::bool_constant<Model::has_adjoint> {};
  } // namespace detail

  /// Whether a model provides the adjoint interface (and gradients are on).
  template <class Model>
  constexpr bool has_adjoint_v = detail::has_adjoint<Model>::value && grad_traits<Model>::enabled;

  /// acc[p] += sum_ab w[(a*N + b)*stride] dP(a -> b)/dp for p < npar, for one
  /// point through the Earth model. chunk(c) returns the GradPrepared<Model, K>
  /// that differentiates parameters c*K .. c*K+K-1. Returns false (acc
  /// untouched) if the path has more than kAdjMaxSeg segments.
  template <class Model, class R, int K, class Chunk>
  OPG_HD inline bool adjoint_prem(const typename Model::Prepared& P, const Chunk& chunk,
                                  int npar,
                                  const EarthView<R>& earth, R E, R cosZ, bool nubar,
                                  const R* w, size_t stride, R* acc, const R* ex = nullptr)
  {
    using State     = StateOf<Model>;
    const int nchunk = (npar + K - 1) / K;

    // forward pass with checkpoints
    State ckpt[kAdjMaxCkpt];
    State S    = initial_state<Model, R>(P, nubar, cosZ, ex);
    int   nseg = 0;
    bool  ok   = true;
    for_each_segment(earth, cosZ, [&](const Segment<R>& s) {
      if (nseg % kAdjBlock == 0) {
        if (nseg / kAdjBlock >= kAdjMaxCkpt) ok = false;
        else ckpt[nseg / kAdjBlock] = S;
      }
      if (ok) Model::step(P, E, nubar, s, S);
      nseg++;
    });
    if (!ok) return false;

    // final rotation: cotangents and direct contributions
    State lam;
    Model::adj_final(P, nubar, S, w, stride, lam);
    for (int c = 0; c < nchunk; c++) {
      R a[K] = {};
      Model::template adj_contract_final<K>(chunk(c).P, nubar, S, w, stride, a);
      for (int k = 0; k < K && c * K + k < npar; k++) acc[c * K + k] += a[k];
    }

    // backward pass, block by block
    const int nblk = (nseg + kAdjBlock - 1) / kAdjBlock;
    for (int b = nblk - 1; b >= 0; b--) {
      const int    first = b * kAdjBlock;
      const int    len   = nseg - first < kAdjBlock ? nseg - first : kAdjBlock;
      State        buf[kAdjBlock];
      Segment<R>   seg[kAdjBlock];
      int          i = 0;
      State        T = ckpt[b];
      for_each_segment(earth, cosZ, [&](const Segment<R>& s) {
        const int j = i++ - first;
        if (j < 0 || j >= len) return;
        buf[j] = T;
        seg[j] = s;
        if (j + 1 < len) Model::step(P, E, nubar, s, T);
      });
      for (int j = len - 1; j >= 0; j--) {
        typename Model::AdjSeg G;
        Model::adj_step(P, E, nubar, seg[j], buf[j], lam, G);
        for (int c = 0; c < nchunk; c++) {
          R a[K] = {};
          Model::template adj_contract_step<K>(chunk(c).P, E, nubar,
                                               seed_segment(chunk(c), seg[j]), G, a);
          for (int k = 0; k < K && c * K + k < npar; k++) acc[c * K + k] += a[k];
        }
      }
    }

    // initial state
    for (int c = 0; c < nchunk; c++) {
      R a[K] = {};
      Model::template adj_contract_initial<K>(chunk(c).P, nubar, lam, a);
      for (int k = 0; k < K && c * K + k < npar; k++) acc[c * K + k] += a[k];
    }
    return true;
  }

} // namespace opg

#endif
