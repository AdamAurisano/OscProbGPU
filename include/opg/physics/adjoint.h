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
///
/// Models whose hamiltonian is, at fixed energy, affine in the segment's
/// density and density * Z/A, H = H0 + rho H1 + rho zoa H2 (adj_affine,
/// e.g. Fast, NSI, LIV), also provide adj_hbar(G) (the cotangent
/// of H), adj_affine_segment(s, rho_eff) and adj_hamiltonian<S>(P, S0, E,
/// nubar, rho, zoa, H). The segment cotangents are then summed into
/// A0 = sum Hb, A1 = sum rho Hb, A2 = sum rho zoa Hb, and the model
/// parameters are contracted once per point with the dual hamiltonian at
/// (rho, zoa) = (0, 0), (1, 0), (1, 1) instead of once per segment; the
/// Earth parameters (zoa_<t>, rho_<t>) take rho Re<Hb, H2> and
/// rho Re<Hb, H1 + zoa H2> per segment.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_PHYSICS_ADJOINT_H
#define OPG_PHYSICS_ADJOINT_H

#include <type_traits>

#include "opg/earth/prem.h"
#include "opg/physics/eigen_grad.h"
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
  constexpr int kAdjMaxPar   = 96;

  namespace detail {
    template <class Model, class = void> struct has_adjoint : std::false_type {};
    template <class Model>
    struct has_adjoint<Model, std::void_t<decltype(Model::has_adjoint)>>
        : std::bool_constant<Model::has_adjoint> {};
    template <class Model, class = void> struct adj_affine : std::false_type {};
    template <class Model>
    struct adj_affine<Model, std::void_t<decltype(Model::adj_affine)>>
        : std::bool_constant<Model::adj_affine> {};
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

    // affine models: value H1, H2 and the cotangent sums
    constexpr bool kAffine = detail::adj_affine<Model>::value;
    constexpr int  NM      = Model::N;
    Mat<NM, R>     A0 = Mat<NM, R>::zero(), A1 = A0, A2 = A0, H1 = A0, H2 = A0;
    if constexpr (kAffine) {
      Mat<NM, R> H00, H10, H11;
      Model::template adj_hamiltonian<R>(P, ckpt[0], E, nubar, R(0), R(0), H00);
      Model::template adj_hamiltonian<R>(P, ckpt[0], E, nubar, R(1), R(0), H10);
      Model::template adj_hamiltonian<R>(P, ckpt[0], E, nubar, R(1), R(1), H11);
      for (int i = 0; i < NM; i++)
        for (int j = i; j < NM; j++) {
          H1(i, j) = H10(i, j) - H00(i, j);
          H2(i, j) = H11(i, j) - H10(i, j);
        }
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
        if constexpr (kAffine) {
          R rho;
          if (Model::adj_affine_segment(seg[j], rho)) {
            const Mat<NM, R>& Hb = Model::adj_hbar(G);
            const R           z  = seg[j].zoa;
            for (int i = 0; i < NM; i++)
              for (int l = 0; l < NM; l++) {
                A0(i, l) += Hb(i, l);
                A1(i, l) += rho * Hb(i, l);
                A2(i, l) += (rho * z) * Hb(i, l);
              }
            const R h1 = hbar_dot<NM, R>(Hb, H1), h2 = hbar_dot<NM, R>(Hb, H2);
            const R sz = rho * h2, sr = rho * (h1 + z * h2);
            for (int c = 0; c < nchunk; c++)
              for (int k = 0; k < K && c * K + k < npar; k++) {
                if (chunk(c).zoa_type[k] == seg[j].layer) acc[c * K + k] += sz;
                if (chunk(c).rho_type[k] == seg[j].layer) acc[c * K + k] += sr;
              }
            continue;
          }
        }
        for (int c = 0; c < nchunk; c++) {
          R a[K] = {};
          Model::template adj_contract_step<K>(chunk(c).P, E, nubar,
                                               seed_segment(chunk(c), seg[j]), G, a);
          for (int k = 0; k < K && c * K + k < npar; k++) acc[c * K + k] += a[k];
        }
      }
    }

    // affine models: model parameters, once per point
    if constexpr (kAffine) {
      using D = Dual<R, K>;
      for (int c = 0; c < nchunk; c++) {
        Mat<NM, D> D00, D10, D11;
        Model::template adj_hamiltonian<D>(chunk(c).P, ckpt[0], E, nubar, D(0), D(0), D00);
        Model::template adj_hamiltonian<D>(chunk(c).P, ckpt[0], E, nubar, D(1), D(0), D10);
        Model::template adj_hamiltonian<D>(chunk(c).P, ckpt[0], E, nubar, D(1), D(1), D11);
        for (int i = 0; i < NM; i++)
          for (int l = i; l < NM; l++) {
            const Complex<D> d1 = D10(i, l) - D00(i, l), d2 = D11(i, l) - D10(i, l);
            D10(i, l)           = d1;
            D11(i, l)           = d2;
          }
        R a[K] = {};
        contract_hbar<NM, R, K>(A0, D00, a);
        contract_hbar<NM, R, K>(A1, D10, a);
        contract_hbar<NM, R, K>(A2, D11, a);
        for (int k = 0; k < K && c * K + k < npar; k++) acc[c * K + k] += a[k];
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
