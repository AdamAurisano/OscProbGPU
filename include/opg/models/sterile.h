///////////////////////////////////////////////////////////////////////////////
/// \file sterile.h
///
/// \brief 3+1 oscillations with one sterile neutrino (OscProb::PMNS_Sterile
///        restricted to N = 4).
///
/// Flavour indices: 0 = e, 1 = mu, 2 = tau, 3 = s. The 4x4 hermitian
/// Hamiltonian is diagonalised with a complex Jacobi solver instead of
/// Eigen's SelfAdjointEigenSolver, so CPU results agree with OscProb to
/// round-off rather than bit-for-bit.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_STERILE_H
#define OPG_MODELS_STERILE_H

#include <string>
#include <vector>

#include "opg/core/constants.h"
#include "opg/core/dual.h"
#include "opg/linalg/jacobi_herm.h"
#include "opg/physics/eigen_grad.h"
#include "opg/physics/mixing.h"
#include "opg/physics/propagate.h"

namespace opg {

  /// Sterile parameters with scalar type S.
  template <class S> struct SterileParams {
      MixingParamsT<4, S> mix;
  };

  /// Sterile prepared state with scalar type S.
  template <class S> struct SterilePrepared {
      Mat<4, S> Hms;   ///< U diag(dm) U^dagger (upper triangle), eV^2
      S         vfac;  ///< kK2*sqrt(2)*G_F
  };

  template <class R = double> struct Sterile {
      using Real                        = R;
      static constexpr int         N    = 4;
      static constexpr const char* name = "Sterile";

      /// Derivative directions per gradient pass (see opg::grad_traits).
#ifdef OPG_STERILE_GRAD_CHUNK
      static constexpr int grad_chunk = OPG_STERILE_GRAD_CHUNK;
#else
      static constexpr int grad_chunk = 1;  // tuned on V100 (K = 2 spills ~5 KB, 1.35x slower)
#endif

      template <class S> using ParamsT   = SterileParams<S>;
      using Params                       = SterileParams<double>;
      template <class S> using PreparedT = SterilePrepared<S>;
      using Prepared                     = SterilePrepared<R>;

      /// Names of the differentiable parameters (MixingRegistry<4>):
      /// th12, th13, th23, th14, th24, th34, d13, d14, d24, dm21, dm31, dm41.
      static std::vector<std::string> param_names()
      {
        return MixingRegistry<4>::names();
      }
      template <class S> static S& param_ref(ParamsT<S>& p, int idx)
      {
        return MixingRegistry<4>::ref(p.mix, idx);
      }
      template <class S> static ParamsT<S> cast(const Params& p)
      {
        return ParamsT<S>{cast_mixing<S>(p.mix)};
      }

      template <class S> static PreparedT<S> prepare_generic(const ParamsT<S>& p)
      {
        PreparedT<S> out;
        out.Hms  = build_hms_generic<4, S>(p.mix);
        out.vfac = S(constants::matter_prefactor());
        return out;
      }

      /// Hms is built in double precision for any R.
      static Prepared prepare(const Params& p)
      {
        Prepared out;
        out.Hms  = build_hms<4, R>(p.mix);
        out.vfac = R(constants::matter_prefactor());
        return out;
      }

      /// Port of PMNS_Sterile::UpdateHam, written for the upper triangle
      /// (OscProb fills the lower triangle with the conjugate), for any
      /// scalar type S of the prepared state and segment type Seg.
      template <class S, class Seg>
      OPG_HD OPG_INLINE static void hamiltonian(const PreparedT<S>& P, R E,
                                                bool nubar, const Seg& s,
                                                Mat<4, S>& H)
      {
        const auto rho = s.density;
        const auto zoa = s.zoa;
        const S lv  = S(2 * R(constants::kGeV2eV) * E);  // 2E in eV

        // Electron matter potential
        S kr2GNe = P.vfac;
        kr2GNe *= rho;
        kr2GNe *= zoa;
        // Neutron matter potential
        S kr2GNn = P.vfac;
        kr2GNn *= rho;
        kr2GNn *= (R(1) - zoa);
        kr2GNn /= R(2);

        OPG_UNROLL
        for (int i = 0; i < 4; i++) {
          H(i, i) = P.Hms(i, i) / lv;
          OPG_UNROLL
          for (int j = i + 1; j < 4; j++) {
            if (!nubar)
              H(i, j) = P.Hms(i, j) / lv;
            else
              H(i, j) = conj(P.Hms(i, j)) / lv;
          }
          // Subtract NC coherent forward scattering from sterile neutrinos.
          if (i > 2) {
            if (!nubar)
              H(i, i).re += kr2GNn;
            else
              H(i, i).re -= kr2GNn;
          }
        }
        // Add nue CC coherent forward scattering.
        if (!nubar)
          H(0, 0).re += kr2GNe;
        else
          H(0, 0).re -= kr2GNe;
      }

      OPG_HD OPG_INLINE static Mat<4, R> initial(const Prepared&, bool)
      {
        return Mat<4, R>::identity();
      }

      OPG_HD OPG_INLINE static void step(const Prepared& P, R E, bool nubar,
                                         const Segment<R>& s, Mat<4, R>& S)
      {
        Mat<4, R> H, V;
        R         lam[4];
        hamiltonian(P, E, nubar, s, H);
        jacobi_hermitian<4, R>(H, V, lam);
        apply_eigen_step<4, R>(V, lam, length_in_eV(s.length), S);
      }

      OPG_HD OPG_INLINE static void finalize(const Prepared&, bool, Mat<4, R>&) {}

      // --- gradients ---------------------------------------------------------
      template <int K>
      OPG_HD OPG_INLINE static void initial_grad(const PreparedT<Dual<R, K>>&,
                                                 bool, Mat<4, R> (&dS)[K])
      {
        OPG_UNROLL
        for (int k = 0; k < K; k++) dS[k] = Mat<4, R>::zero();
      }

      /// Value eigensystem from Jacobi (as step()), derivative of H from
      /// the dual hamiltonian (Daleckii-Krein, opg::eigen_step_grad).
      template <int K>
      OPG_HD OPG_INLINE static void step_grad(const Prepared&               P,
                                              const PreparedT<Dual<R, K>>& PD,
                                              R E, bool nubar,
                                              const SegmentZ<R, Dual<R, K>>& sz,
                                              Mat<4, R>& S, Mat<4, R> (&dS)[K])
      {
        const Segment<R> s{sz.length, sz.density.v, sz.zoa.v, sz.layer};
        Mat<4, R>        H, V;
        R                lam[4];
        hamiltonian(P, E, nubar, s, H);
        jacobi_hermitian<4, R>(H, V, lam);

        Mat<4, Dual<R, K>> HD;
        hamiltonian(PD, E, nubar, sz, HD);

        eigen_step_grad<4, R, K>(V, lam, length_in_eV(s.length), HD, S, dS);
      }

      template <int K>
      OPG_HD OPG_INLINE static void finalize_grad(const Prepared&,
                                                  const PreparedT<Dual<R, K>>&,
                                                  bool, Mat<4, R>&,
                                                  Mat<4, R> (&)[K])
      {
      }

      // --- reverse mode (physics/adjoint.h) ------------------------------------
      static constexpr bool has_adjoint = true;
      // Not adj_affine: the summed cotangents lose about a digit at
      // dm41 ~ 1 eV^2 for only ~12% speed.
      using AdjSeg                      = Mat<4, R>;
      OPG_HD OPG_INLINE static void adj_final(const Prepared&, bool, const Mat<4, R>& S,
                                              const R* w, size_t stride, Mat<4, R>& Sb)
      {
        amplitude_adj_final<4, R>(S, w, stride, Sb);
      }
      OPG_HD OPG_INLINE static void adj_step(const Prepared& P, R E, bool nubar,
                                             const Segment<R>& s, const Mat<4, R>& S,
                                             Mat<4, R>& Sb, Mat<4, R>& Hb)
      {
        Mat<4, R> H, V;
        R         lam[4];
        hamiltonian(P, E, nubar, s, H);
        jacobi_hermitian<4, R>(H, V, lam);
        eigen_step_adj<4, R>(V, lam, length_in_eV(s.length), S, Sb, Hb);
      }
      template <int K>
      OPG_HD OPG_INLINE static void adj_contract_step(const PreparedT<Dual<R, K>>& PD, R E,
                                                      bool nubar,
                                                      const SegmentZ<R, Dual<R, K>>& sz,
                                                      const Mat<4, R>& Hb, R (&acc)[K])
      {
        Mat<4, Dual<R, K>> HD;
        hamiltonian(PD, E, nubar, sz, HD);
        contract_hbar<4, R, K>(Hb, HD, acc);
      }
      template <int K>
      OPG_HD OPG_INLINE static void adj_contract_initial(const PreparedT<Dual<R, K>>&, bool,
                                                         const Mat<4, R>&, R (&)[K])
      {
      }
      template <int K>
      OPG_HD OPG_INLINE static void adj_contract_final(const PreparedT<Dual<R, K>>&, bool,
                                                       const Mat<4, R>&, const R*, size_t,
                                                       R (&)[K])
      {
      }
  };

} // namespace opg

#endif
