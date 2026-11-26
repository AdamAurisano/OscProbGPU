///////////////////////////////////////////////////////////////////////////////
/// \file sterile.h
///
/// \brief 3+N oscillations with sterile neutrinos (OscProb::PMNS_Sterile).
///
/// Sterile<R, NF> has NF flavours (OscProb's PMNS_Sterile(NF)); the default
/// NF = 4 is 3+1. The library instantiates NF = 4, 5 and 6 (aliases
/// Sterile5, Sterile6) in the CUDA engine and Python; any NF >= 2 works on
/// the CPU backend.
///
/// Flavour indices: 0 = e, 1 = mu, 2 = tau, 3.. = sterile. The NF x NF
/// hermitian Hamiltonian is diagonalised with a complex Jacobi solver
/// instead of Eigen's SelfAdjointEigenSolver, so CPU results agree with
/// OscProb to round-off rather than bit-for-bit.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_STERILE_H
#define OPG_MODELS_STERILE_H

#include <string>
#include <vector>

#include "../core/constants.h"
#include "../core/dual.h"
#include "../linalg/jacobi_herm.h"
#include "../physics/eigen_grad.h"
#include "../physics/mixing.h"
#include "../physics/propagate.h"

namespace opg {

  /// Sterile parameters with scalar type S.
  template <class S, int N = 4> struct SterileParams {
      MixingParamsT<N, S> mix;
  };

  /// Sterile prepared state with scalar type S.
  template <class S, int N = 4> struct SterilePrepared {
      Mat<N, S> Hms;   ///< U diag(dm) U^dagger (upper triangle), eV^2
      S         vfac;  ///< kK2*sqrt(2)*G_F
  };

  template <class R = double, int NF = 4> struct Sterile {
      static_assert(NF >= 2, "Sterile: at least 2 flavours");
      using Real                        = R;
      static constexpr int         N    = NF;
      static constexpr const char* name = NF == 4 ? "Sterile" : NF == 5 ? "Sterile5"
                                        : NF == 6 ? "Sterile6" : "SterileN";
      /// Hermitian and affine in 1/E: analytic bin averages (avg/analytic.h).
      static constexpr bool analytic_avg = true;
      /// The matter terms of hamiltonian() do not depend on the model
      /// parameters: dH/dp = u dA/dp (A = dH/du) for all of them.
      static constexpr bool analytic_static_matter = true;

      /// Derivative directions per gradient pass (see opg::grad_traits).
#ifdef OPG_STERILE_GRAD_CHUNK
      static constexpr int grad_chunk = OPG_STERILE_GRAD_CHUNK;
#else
      static constexpr int grad_chunk = 1;  // tuned on V100 (K = 2 spills ~5 KB, 1.35x slower)
#endif

      template <class S> using ParamsT   = SterileParams<S, N>;
      using Params                       = SterileParams<double, N>;
      template <class S> using PreparedT = SterilePrepared<S, N>;
      using Prepared                     = SterilePrepared<R, N>;

      /// Names of the differentiable parameters (MixingRegistry<N>): the
      /// angles th_ij (i < j, ordered by j then i), the phases d_ij
      /// (i + 1 < j, same order), then dm_j1. For N = 4: th12, th13, th23,
      /// th14, th24, th34, d13, d14, d24, dm21, dm31, dm41; N = 5 adds
      /// th15, th25, th35, th45, d15, d25, d35 and dm51 (each in its group).
      static std::vector<std::string> param_names()
      {
        return MixingRegistry<N>::names();
      }
      template <class S> static S& param_ref(ParamsT<S>& p, int idx)
      {
        return MixingRegistry<N>::ref(p.mix, idx);
      }
      template <class S> static ParamsT<S> cast(const Params& p)
      {
        return ParamsT<S>{cast_mixing<S>(p.mix)};
      }

      template <class S> static PreparedT<S> prepare_generic(const ParamsT<S>& p)
      {
        PreparedT<S> out;
        out.Hms  = build_hms_generic<N, S>(p.mix);
        out.vfac = S(constants::matter_prefactor());
        return out;
      }

      /// Hms is built in double precision for any R.
      static Prepared prepare(const Params& p)
      {
        Prepared out;
        out.Hms  = build_hms<N, R>(p.mix);
        out.vfac = R(constants::matter_prefactor());
        return out;
      }

      /// Port of PMNS_Sterile::UpdateHam, written for the upper triangle
      /// (OscProb fills the lower triangle with the conjugate), for any
      /// scalar type S of the prepared state and segment type Seg.
      template <class S, class Seg>
      OPG_HD OPG_INLINE static void hamiltonian(const PreparedT<S>& P, R E,
                                                bool nubar, const Seg& s,
                                                Mat<N, S>& H)
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
        for (int i = 0; i < N; i++) {
          H(i, i) = P.Hms(i, i) / lv;
          OPG_UNROLL
          for (int j = i + 1; j < N; j++) {
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

      OPG_HD OPG_INLINE static Mat<N, R> initial(const Prepared&, bool)
      {
        return Mat<N, R>::identity();
      }

      OPG_HD OPG_INLINE static void step(const Prepared& P, R E, bool nubar,
                                         const Segment<R>& s, Mat<N, R>& S)
      {
        Mat<N, R> H, V;
        R         lam[N];
        hamiltonian(P, E, nubar, s, H);
        jacobi_hermitian<N, R>(H, V, lam);
        apply_eigen_step<N, R>(V, lam, length_in_eV(s.length), S);
      }

      OPG_HD OPG_INLINE static void finalize(const Prepared&, bool, Mat<N, R>&) {}

      // --- gradients ---------------------------------------------------------
      template <int K>
      OPG_HD OPG_INLINE static void initial_grad(const PreparedT<Dual<R, K>>&,
                                                 bool, Mat<N, R> (&dS)[K])
      {
        OPG_UNROLL
        for (int k = 0; k < K; k++) dS[k] = Mat<N, R>::zero();
      }

      /// Value eigensystem from Jacobi (as step()), derivative of H from
      /// the dual hamiltonian (Daleckii-Krein, opg::eigen_step_grad).
      template <int K>
      OPG_HD OPG_INLINE static void step_grad(const Prepared&               P,
                                              const PreparedT<Dual<R, K>>& PD,
                                              R E, bool nubar,
                                              const SegmentZ<R, Dual<R, K>>& sz,
                                              Mat<N, R>& S, Mat<N, R> (&dS)[K])
      {
        const Segment<R> s{sz.length, sz.density.v, sz.zoa.v, sz.layer};
        Mat<N, R>        H, V;
        R                lam[N];
        hamiltonian(P, E, nubar, s, H);
        jacobi_hermitian<N, R>(H, V, lam);

        Mat<N, Dual<R, K>> HD;
        hamiltonian(PD, E, nubar, sz, HD);

        eigen_step_grad<N, R, K>(V, lam, length_in_eV(s.length), HD, S, dS);
      }

      template <int K>
      OPG_HD OPG_INLINE static void finalize_grad(const Prepared&,
                                                  const PreparedT<Dual<R, K>>&,
                                                  bool, Mat<N, R>&,
                                                  Mat<N, R> (&)[K])
      {
      }

      // --- reverse mode (physics/adjoint.h) ------------------------------------
      static constexpr bool has_adjoint = true;
      // Not adj_affine: the summed cotangents lose about a digit at
      // dm41 ~ 1 eV^2 for only ~12% speed.
      using AdjSeg                      = Mat<N, R>;
      OPG_HD OPG_INLINE static void adj_final(const Prepared&, bool, const Mat<N, R>& S,
                                              const R* w, size_t stride, Mat<N, R>& Sb)
      {
        amplitude_adj_final<N, R>(S, w, stride, Sb);
      }
      OPG_HD OPG_INLINE static void adj_step(const Prepared& P, R E, bool nubar,
                                             const Segment<R>& s, const Mat<N, R>& S,
                                             Mat<N, R>& Sb, Mat<N, R>& Hb)
      {
        Mat<N, R> H, V;
        R         lam[N];
        hamiltonian(P, E, nubar, s, H);
        jacobi_hermitian<N, R>(H, V, lam);
        eigen_step_adj<N, R>(V, lam, length_in_eV(s.length), S, Sb, Hb);
      }
      template <int K>
      OPG_HD OPG_INLINE static void adj_contract_step(const PreparedT<Dual<R, K>>& PD, R E,
                                                      bool nubar,
                                                      const SegmentZ<R, Dual<R, K>>& sz,
                                                      const Mat<N, R>& Hb, R (&acc)[K])
      {
        Mat<N, Dual<R, K>> HD;
        hamiltonian(PD, E, nubar, sz, HD);
        contract_hbar<N, R, K>(Hb, HD, acc);
      }
      template <int K>
      OPG_HD OPG_INLINE static void adj_contract_initial(const PreparedT<Dual<R, K>>&, bool,
                                                         const Mat<N, R>&, R (&)[K])
      {
      }

      // --- production-height averaging (avg/prod_height.h) --------------------
      static constexpr bool has_seg_eigen = true;
      OPG_HD OPG_INLINE static void seg_eigen(const Prepared& P, R E, bool nubar,
                                              const Segment<R>& s, Mat<N, R>& V, R lam[N])
      {
        Mat<N, R> H;
        hamiltonian(P, E, nubar, s, H);
        jacobi_hermitian<N, R>(H, V, lam);
      }
      template <int K>
      OPG_HD OPG_INLINE static void seg_hamiltonian_dual(const PreparedT<Dual<R, K>>& PD, R E,
                                                         bool nubar,
                                                         const SegmentZ<R, Dual<R, K>>& sz,
                                                         Mat<N, Dual<R, K>>& HD)
      {
        hamiltonian(PD, E, nubar, sz, HD);
      }
      OPG_HD OPG_INLINE static void adj_final_cot(const Prepared&, bool, const Mat<N, R>&,
                                                  const Mat<N, R>& Fb, Mat<N, R>& Sb)
      {
        Sb = Fb;
      }
      template <int K>
      OPG_HD OPG_INLINE static void adj_contract_final_cot(const PreparedT<Dual<R, K>>&, bool,
                                                           const Mat<N, R>&, const Mat<N, R>&,
                                                           R (&)[K])
      {
      }
      template <int K>
      OPG_HD OPG_INLINE static void adj_contract_final(const PreparedT<Dual<R, K>>&, bool,
                                                       const Mat<N, R>&, const R*, size_t,
                                                       R (&)[K])
      {
      }
  };

  /// 3+2 and 3+3 (OscProb PMNS_Sterile(5), PMNS_Sterile(6)).
  template <class R = double> using Sterile5 = Sterile<R, 5>;
  template <class R = double> using Sterile6 = Sterile<R, 6>;

} // namespace opg

#endif
