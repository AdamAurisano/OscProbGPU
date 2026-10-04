///////////////////////////////////////////////////////////////////////////////
/// \file hermitian3.h
///
/// \brief Shared machinery for 3-flavour models whose Hamiltonian is
///        hermitian and which inherit PMNS_Fast::SolveHam in OscProb
///        (PMNS_Fast, PMNS_NSI, PMNS_NUNM).
///
/// In matter, the Hamiltonian built by the model's hamiltonian() is
/// diagonalised with the ported Kopp zheevh3. For densities below 1e-6
/// g/cm^3, OscProb uses the analytic vacuum eigensystem
/// (PMNS_Fast::SetVacuumEigensystem), which we precompute in prepare().
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_HERMITIAN3_H
#define OPG_MODELS_HERMITIAN3_H

#include <cmath>
#include <complex>
#include <type_traits>

#include "opg/core/constants.h"
#include "opg/linalg/kopp/zheevh3.h"
#include "opg/physics/eigen_grad.h"
#include "opg/physics/mixing.h"
#include "opg/physics/propagate.h"

namespace opg {

  /// Parts of the prepared state common to all hermitian 3-flavour models,
  /// with scalar type S (Real, or Dual<Real, K> for derivatives).
  template <class S> struct Hermitian3Common {
      Mat<3, S> Hms;      ///< U diag(dm) U^dagger (upper triangle), eV^2
      Mat<3, S> Uvac[2];  ///< vacuum eigenvectors for nu / nubar
      S         dm[3];    ///< mass splittings dm_j1 (dm[0] = 0)
      S         vfac;     ///< kK2*sqrt(2)*G_F, matter potential prefactor
  };

  /// Port of PMNS_Fast::SetVacuumEigensystem, generic scalar type. With
  /// S = double the operations are those of OscProb with std::complex.
  template <class S>
  void prepare_vacuum3_generic(const MixingParamsT<3, S>& p, Mat<3, S> Uvac[2])
  {
    using std::cos;
    using std::sin;
    using C = Complex<S>;
    for (int nb = 0; nb < 2; nb++) {
      // exp(+-i delta), with delta -> -delta for antineutrinos
      const S dlt   = nb ? -p.dcp[0][2] : p.dcp[0][2];
      const S mdlt  = -dlt;
      const C eip   = C(cos(dlt), sin(dlt));    // exp(idelta)
      const C eim   = C(cos(mdlt), sin(mdlt));  // exp(-idelta)

      const S s12 = sin(p.th[0][1]);
      const S s23 = sin(p.th[1][2]);
      const S s13 = sin(p.th[0][2]);
      const S c12 = cos(p.th[0][1]);
      const S c23 = cos(p.th[1][2]);
      const S c13 = cos(p.th[0][2]);

      Mat<3, S>& E = Uvac[nb];
      E(0, 0) = C(c12 * c13, S(0));
      E(0, 1) = C(s12 * c13, S(0));
      E(0, 2) = s13 * eim;

      E(1, 0) = -s12 * c23 - c12 * s23 * s13 * eip;
      E(1, 1) = c12 * c23 - s12 * s23 * s13 * eip;
      E(1, 2) = C(s23 * c13, S(0));

      E(2, 0) = s12 * s23 - c12 * c23 * s13 * eip;
      E(2, 1) = -c12 * s23 - s12 * c23 * s13 * eip;
      E(2, 2) = C(c23 * c13, S(0));
    }
  }

  template <class S>
  void prepare_hermitian3_generic(const MixingParamsT<3, S>& p,
                                  Hermitian3Common<S>&       c)
  {
    c.Hms = build_hms_generic<3, S>(p);
    prepare_vacuum3_generic<S>(p, c.Uvac);
    c.dm[0] = S(0);
    c.dm[1] = p.dm[1];
    c.dm[2] = p.dm[2];
    c.vfac  = S(constants::matter_prefactor());
  }

  /// From double-precision parameters (used by models without gradients).
  template <class R>
  void prepare_hermitian3(const MixingParams<3>& p, Hermitian3Common<R>& c)
  {
    prepare_hermitian3_generic<R>(cast_mixing<R>(p), c);
  }

  /// Diagonalise a hermitian 3x3 matrix (upper triangle) with zheevh3.
  ///
  /// Hamiltonians in eV are O(1e-12). Kopp's zheevh3 decides whether to
  /// trust Cardano's analytic solution with an absolute error estimate made
  /// for O(1) matrices, so OscProb, which calls it on the raw Hamiltonian,
  /// always falls back to the slower iterative QL algorithm (and in single
  /// precision the cubic invariants would underflow). We therefore rescale
  /// H to O(1) first; near-degenerate cases still fall back to QL. This is
  /// ~2x faster on GPUs and agrees with OscProb to round-off (~1e-13 in P).
  /// Define OPG_OSCPROB_BITWISE (CMake option of the same name) to call
  /// zheevh3 on the unscaled matrix in double precision, which reproduces
  /// OscProb bit-for-bit on the CPU.
  template <class R>
  OPG_HD OPG_INLINE void diagonalize3(Mat<3, R>& H, Mat<3, R>& V, R lam[3])
  {
#ifdef OPG_OSCPROB_BITWISE
    if constexpr (sizeof(R) < sizeof(double)) {
#else
    if constexpr (true) {
#endif
      R sc = 0;
      OPG_UNROLL
      for (int i = 0; i < 3; i++)
        OPG_UNROLL
      for (int j = i; j < 3; j++) {
        using std::fmax;
        sc = fmax(sc, abs(H(i, j)));
      }
      if (sc > 0) {
        R inv = R(1) / sc;
        OPG_UNROLL
        for (int i = 0; i < 3; i++)
          OPG_UNROLL
        for (int j = i; j < 3; j++) H(i, j) *= inv;
      }
      kopp::zheevh3(H, V, lam);
      OPG_UNROLL
      for (int i = 0; i < 3; i++) lam[i] *= sc;
    }
    else {
      kopp::zheevh3(H, V, lam);
    }
  }

  /// Whether a model uses PMNS_Fast's analytic vacuum eigensystem for
  /// densities below 1e-6 g/cm^3 (models whose Hamiltonian has
  /// matter-independent extra terms, e.g. LIV, set
  /// `static constexpr bool vacuum_shortcut = false`).
  template <class Model, class = void> struct uses_vacuum_shortcut {
      static constexpr bool value = true;
  };
  template <class Model>
  struct uses_vacuum_shortcut<Model, std::void_t<decltype(Model::vacuum_shortcut)>> {
      static constexpr bool value = Model::vacuum_shortcut;
  };

  /// Value eigensystem of one segment (vacuum shortcut or zheevh3).
  template <class Model, class R>
  OPG_HD OPG_INLINE void hermitian3_eigen(const typename Model::Prepared& P,
                                          R E, bool nubar, const Segment<R>& s,
                                          Mat<3, R>& V, R lam[3])
  {
    // PMNS_Fast::SolveHam: do vacuum oscillation in low density
    if (uses_vacuum_shortcut<Model>::value && s.density < R(1.0e-6)) {
      V      = P.common.Uvac[nubar ? 1 : 0];
      lam[0] = 0;
      lam[1] = P.common.dm[1] / (2 * R(constants::kGeV2eV) * E);
      lam[2] = P.common.dm[2] / (2 * R(constants::kGeV2eV) * E);
    }
    else {
      Mat<3, R> H;
      Model::hamiltonian(P, E, nubar, s, H);
      diagonalize3(H, V, lam);
    }
  }

  /// One segment step for a hermitian 3-flavour model. Model must provide
  ///   OPG_HD static void hamiltonian(const Prepared&, R E, bool nubar,
  ///                                  const Segment<R>&, Mat<3,R>& H)
  /// filling at least the diagonal and upper triangle of H, and Prepared
  /// must have a member `common` of type Hermitian3Common<R>.
  template <class Model, class R>
  OPG_HD OPG_INLINE void hermitian3_step(const typename Model::Prepared& P,
                                         R E, bool nubar, const Segment<R>& s,
                                         Mat<3, R>& S)
  {
    Mat<3, R> V;
    R         lam[3];
    hermitian3_eigen<Model, R>(P, E, nubar, s, V, lam);
    apply_eigen_step<3, R>(V, lam, length_in_eV(s.length), S);
  }

  /// Segment step with derivatives. PD is the prepared state with dual
  /// numbers (value parts equal to P); the value eigensystem is that of
  /// hermitian3_step, and the derivative of H comes from evaluating the
  /// model's hamiltonian() in dual arithmetic (Z/A of the segment possibly
  /// seeded too). In vacuum the same holds with zero density, since there
  /// H = Hms / 2E.
  template <class Model, class R, int K>
  OPG_HD OPG_INLINE void hermitian3_step_grad(
      const typename Model::Prepared&                         P,
      const typename Model::template PreparedT<Dual<R, K>>& PD, R E, bool nubar,
      const SegmentZ<R, Dual<R, K>>& sz, Mat<3, R>& S, Mat<3, R> (&dS)[K])
  {
    const Segment<R> s{sz.length, sz.density.v, sz.zoa.v, sz.layer};
    Mat<3, R>        V;
    R                lam[3];
    hermitian3_eigen<Model, R>(P, E, nubar, s, V, lam);

    Mat<3, Dual<R, K>>      HD;
    SegmentZ<R, Dual<R, K>> sd = sz;
    if (uses_vacuum_shortcut<Model>::value && s.density < R(1.0e-6)) sd.density = 0;
    Model::hamiltonian(PD, E, nubar, sd, HD);

    eigen_step_grad<3, R, K>(V, lam, length_in_eV(s.length), HD, S, dS);
  }

  // --- reverse mode (see physics/adjoint.h) ----------------------------------

  /// Reverse of hermitian3_step: Sb <- U^dag Sb, Hb = cotangent of H.
  template <class Model, class R>
  OPG_HD OPG_INLINE void hermitian3_adj_step(const typename Model::Prepared& P, R E,
                                             bool nubar, const Segment<R>& s,
                                             const Mat<3, R>& S, Mat<3, R>& Sb, Mat<3, R>& Hb)
  {
    Mat<3, R> V;
    R         lam[3];
    hermitian3_eigen<Model, R>(P, E, nubar, s, V, lam);
    eigen_step_adj<3, R>(V, lam, length_in_eV(s.length), S, Sb, Hb);
  }

  /// acc[k] += Re tr(Hb^dag dH_k) with dH_k from the dual hamiltonian (as
  /// hermitian3_step_grad, vacuum shortcut included).
  template <class Model, class R, int K>
  OPG_HD OPG_INLINE void hermitian3_adj_contract(
      const typename Model::template PreparedT<Dual<R, K>>& PD, R E, bool nubar,
      const SegmentZ<R, Dual<R, K>>& sz, const Mat<3, R>& Hb, R (&acc)[K])
  {
    SegmentZ<R, Dual<R, K>> sd = sz;
    if (uses_vacuum_shortcut<Model>::value && sz.density.v < R(1.0e-6)) sd.density = 0;
    Mat<3, Dual<R, K>> HD;
    Model::hamiltonian(PD, E, nubar, sd, HD);
    contract_hbar<3, R, K>(Hb, HD, acc);
  }

} // namespace opg

/// Reverse-mode members (physics/adjoint.h) of a hermitian 3-flavour model
/// whose state is the amplitude matrix (identity initial state, no final
/// step); MODEL is the model class name, AFFINE whether its hamiltonian is
/// affine in density and density * Z/A (see physics/adjoint.h).
#define OPG_HERMITIAN3_ADJOINT(MODEL, AFFINE)                                             \
  static constexpr bool has_adjoint = true;                                              \
  static constexpr bool adj_affine  = AFFINE;                                            \
  using AdjSeg                      = Mat<3, R>;                                         \
  OPG_HD OPG_INLINE static const Mat<3, R>& adj_hbar(const Mat<3, R>& G) { return G; }   \
  OPG_HD OPG_INLINE static bool adj_affine_segment(const Segment<R>& s, R& rho)          \
  {                                                                                      \
    rho = uses_vacuum_shortcut<MODEL>::value && s.density < R(1.0e-6) ? R(0) : s.density; \
    return true;                                                                         \
  }                                                                                      \
  template <class S>                                                                     \
  OPG_HD OPG_INLINE static void adj_hamiltonian(const PreparedT<S>& P, const Mat<3, R>&, \
                                                R E, bool nubar, S rho, S zoa,           \
                                                Mat<3, S>& H)                            \
  {                                                                                      \
    const SegmentZ<R, S> s{R(0), rho, zoa, -1};                                          \
    hamiltonian(P, E, nubar, s, H);                                                      \
  }                                                                                      \
  OPG_HD OPG_INLINE static void adj_final(const Prepared&, bool, const Mat<3, R>& S,     \
                                          const R* w, size_t stride, Mat<3, R>& Sb)      \
  {                                                                                      \
    amplitude_adj_final<3, R>(S, w, stride, Sb);                                         \
  }                                                                                      \
  OPG_HD OPG_INLINE static void adj_step(const Prepared& P, R E, bool nubar,             \
                                         const Segment<R>& s, const Mat<3, R>& S,        \
                                         Mat<3, R>& Sb, Mat<3, R>& Hb)                   \
  {                                                                                      \
    hermitian3_adj_step<MODEL, R>(P, E, nubar, s, S, Sb, Hb);                            \
  }                                                                                      \
  template <int K>                                                                       \
  OPG_HD OPG_INLINE static void adj_contract_step(                                       \
      const PreparedT<Dual<R, K>>& PD, R E, bool nubar,                                  \
      const SegmentZ<R, Dual<R, K>>& sz, const Mat<3, R>& Hb, R (&acc)[K])                \
  {                                                                                      \
    hermitian3_adj_contract<MODEL, R, K>(PD, E, nubar, sz, Hb, acc);                     \
  }                                                                                      \
  template <int K>                                                                       \
  OPG_HD OPG_INLINE static void adj_contract_initial(const PreparedT<Dual<R, K>>&, bool, \
                                                     const Mat<3, R>&, R (&)[K])         \
  {                                                                                      \
  }                                                                                      \
  template <int K>                                                                       \
  OPG_HD OPG_INLINE static void adj_contract_final(const PreparedT<Dual<R, K>>&, bool,   \
                                                   const Mat<3, R>&, const R*, size_t,   \
                                                   R (&)[K])                             \
  {                                                                                      \
  }

#endif
