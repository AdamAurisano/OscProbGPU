///////////////////////////////////////////////////////////////////////////////
/// \file decay.h
///
/// \brief 3-flavour oscillations with invisible neutrino decay
///        (OscProb::PMNS_Decay).
///
/// The effective Hamiltonian is non-hermitian,
///   H = (U diag(dm) U^dag - i U diag(alpha) U^dag) / 2E + V_cc,
/// with alpha_j = m_j / tau_j in eV^2 (alpha_1 = 0). Each segment applies
/// the evolution operator exp(-i H L), computed with the scaling and
/// squaring Pade algorithm (as Eigen's MatrixExponential used by OscProb).
/// Derivatives differentiate the same algorithm in dual arithmetic.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_DECAY_H
#define OPG_MODELS_DECAY_H

#include <stdexcept>
#include <string>
#include <vector>

#include "opg/core/constants.h"
#include "opg/core/dual.h"
#include "opg/linalg/expm.h"
#include "opg/physics/mixing.h"
#include "opg/physics/propagate.h"

namespace opg {

  /// Decay parameters with scalar type S.
  template <class S> struct DecayParams {
      MixingParamsT<3, S> mix;
      S                   alpha[3] = {};  ///< alpha_j in eV^2 (j = 2, 3)

      /// As PMNS_Decay::SetAlpha2 / SetAlpha3 (must be >= 0).
      void SetAlpha2(double a)
      {
        if (a < 0) throw std::invalid_argument("Decay: alpha2 < 0");
        alpha[1] = S(a);
      }
      void SetAlpha3(double a)
      {
        if (a < 0) throw std::invalid_argument("Decay: alpha3 < 0");
        alpha[2] = S(a);
      }
  };

  /// Decay prepared state with scalar type S.
  template <class S> struct DecayPrepared {
      Mat<3, S> Heff[2];  ///< Hms - i Hd (full matrix) for nu / nubar
      S         vfac;     ///< kK2*sqrt(2)*G_F
  };

  template <class R = double> struct Decay {
      using Real                        = R;
      static constexpr int         N    = 3;
      static constexpr const char* name = "Decay";

      /// Derivative directions per gradient pass (see opg::grad_traits).
#ifdef OPG_DECAY_GRAD_CHUNK
      static constexpr int grad_chunk = OPG_DECAY_GRAD_CHUNK;
#else
      static constexpr int grad_chunk = 2;  // tuned on V100 (1.3x faster than K = 1, 3)
#endif

      template <class S> using ParamsT   = DecayParams<S>;
      using Params                       = DecayParams<double>;
      template <class S> using PreparedT = DecayPrepared<S>;
      using Prepared                     = DecayPrepared<R>;

      /// Differentiable parameters: the mixing ones (MixingRegistry<3>),
      /// alpha2 and alpha3. alpha >= 0 physically; the derivatives at
      /// alpha = 0 are those of the analytic continuation (i.e. one-sided
      /// derivatives from alpha > 0).
      static std::vector<std::string> param_names()
      {
        auto n = MixingRegistry<3>::names();
        n.push_back("alpha2");
        n.push_back("alpha3");
        return n;
      }
      template <class S> static S& param_ref(ParamsT<S>& p, int idx)
      {
        const int nmix = MixingRegistry<3>::count();
        if (idx < nmix) return MixingRegistry<3>::ref(p.mix, idx);
        if (idx == nmix) return p.alpha[1];
        if (idx == nmix + 1) return p.alpha[2];
        throw std::out_of_range("Decay: bad parameter index");
      }
      template <class S> static ParamsT<S> cast(const Params& p)
      {
        ParamsT<S> q;
        q.mix = cast_mixing<S>(p.mix);
        for (int j = 0; j < 3; j++) q.alpha[j] = S(p.alpha[j]);
        return q;
      }

      /// Port of PMNS_Decay::BuildHms for each of nu and nubar. With
      /// S = double the arithmetic is OscProb's (std::complex).
      template <class S> static PreparedT<S> prepare_generic(const ParamsT<S>& p)
      {
        using C = Complex<S>;
        PreparedT<S> out;
        const S dm[3] = {S(0), p.mix.dm[1], p.mix.dm[2]};
        const S al[3] = {p.alpha[0], p.alpha[1], p.alpha[2]};
        for (int nb = 0; nb < 2; nb++) {
          C Hms[3][3], Hd[3][3];
          rotate_diagonal_generic<3, S>(dm, p.mix.th, p.mix.dcp, Hms);
          rotate_diagonal_generic<3, S>(al, p.mix.th, p.mix.dcp, Hd);

          // Antineutrinos: delta -> -delta; fill the lower triangle because
          // the final Hamiltonian is not hermitian.
          for (int i = 0; i < 3; i++) {
            for (int j = i + 1; j < 3; j++) {
              if (nb) {
                Hms[i][j] = conj(Hms[i][j]);
                Hd[i][j]  = conj(Hd[i][j]);
              }
              Hms[j][i] = conj(Hms[i][j]);
              Hd[j][i]  = conj(Hd[i][j]);
            }
          }

          const C numi(S(0), S(1));
          for (int j = 0; j < 3; j++)
            for (int i = 0; i < 3; i++) out.Heff[nb](i, j) = Hms[i][j] - numi * Hd[i][j];
        }
        out.vfac = S(constants::matter_prefactor());
        return out;
      }

      /// The values are prepared in double precision for any R.
      static Prepared prepare(const Params& p)
      {
        auto     d = prepare_generic<double>(p);
        Prepared out;
        for (int nb = 0; nb < 2; nb++)
          for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
              out.Heff[nb](i, j) =
                  Complex<R>(R(d.Heff[nb](i, j).re), R(d.Heff[nb](i, j).im));
        out.vfac = R(d.vfac);
        return out;
      }

      /// -i H L for one segment (port of PMNS_Decay::UpdateHam), scalar
      /// type S of the prepared state.
      template <class S>
      OPG_HD OPG_INLINE static Mat<3, S> exponent(const PreparedT<S>& P, R E,
                                                  bool nubar, const Segment<R>& s)
      {
        const S lv = S(2 * R(constants::kGeV2eV) * E);  // 2E in eV

        S kr2GNe = P.vfac;
        kr2GNe *= s.density * s.zoa;  // Matter potential in eV

        const Mat<3, S>& Hf = P.Heff[nubar ? 1 : 0];
        Mat<3, S>        H;
        OPG_UNROLL
        for (int i = 0; i < 3; i++)
          OPG_UNROLL
        for (int j = 0; j < 3; j++) H(i, j) = Hf(i, j) / lv;
        if (!nubar)
          H(0, 0).re += kr2GNe;
        else
          H(0, 0).re -= kr2GNe;

        // Evolution operator exp(-i H L)
        const Complex<S> mil(S(0), S(-length_in_eV(s.length)));
        OPG_UNROLL
        for (int i = 0; i < 3; i++)
          OPG_UNROLL
        for (int j = 0; j < 3; j++) H(i, j) *= mil;
        return H;
      }

      OPG_HD OPG_INLINE static Mat<3, R> initial(const Prepared&, bool)
      {
        return Mat<3, R>::identity();
      }

      /// Port of PMNS_Decay::UpdateHam + PropagatePath.
      OPG_HD OPG_INLINE static void step(const Prepared& P, R E, bool nubar,
                                         const Segment<R>& s, Mat<3, R>& S)
      {
        Mat<3, R> U = expm<3, R>(exponent(P, E, nubar, s));
        apply_operator<3, R>(U, S);
      }

      OPG_HD OPG_INLINE static void finalize(const Prepared&, bool, Mat<3, R>&) {}

      // --- gradients ---------------------------------------------------------
      template <int K>
      OPG_HD OPG_INLINE static void initial_grad(const PreparedT<Dual<R, K>>&,
                                                 bool, Mat<3, R> (&dS)[K])
      {
        OPG_UNROLL
        for (int k = 0; k < K; k++) dS[k] = Mat<3, R>::zero();
      }

      /// dS_k <- U dS_k + dU_k S, S <- U S. U is computed exactly as in
      /// step() (so values are unchanged); dU_k are the derivative parts of
      /// expm evaluated in dual arithmetic.
      template <int K>
      OPG_HD OPG_INLINE static void step_grad(const Prepared&               P,
                                              const PreparedT<Dual<R, K>>& PD,
                                              R E, bool nubar,
                                              const Segment<R>& s, Mat<3, R>& S,
                                              Mat<3, R> (&dS)[K])
      {
        using D = Dual<R, K>;
        const Mat<3, D> UD = expm<3, D>(exponent(PD, E, nubar, s));
        const Mat<3, R> U  = expm<3, R>(exponent(P, E, nubar, s));
        OPG_UNROLL
        for (int k = 0; k < K; k++) {
          apply_operator<3, R>(U, dS[k]);
          OPG_UNROLL
          for (int a = 0; a < 3; a++)
            OPG_UNROLL
          for (int i = 0; i < 3; i++) {
            Complex<R> acc(0, 0);
            OPG_UNROLL
            for (int j = 0; j < 3; j++)
              acc += Complex<R>(UD(i, j).re.d[k], UD(i, j).im.d[k]) * S(j, a);
            dS[k](i, a) += acc;
          }
        }
        apply_operator<3, R>(U, S);
      }

      template <int K>
      OPG_HD OPG_INLINE static void finalize_grad(const Prepared&,
                                                  const PreparedT<Dual<R, K>>&,
                                                  bool, Mat<3, R>&,
                                                  Mat<3, R> (&)[K])
      {
      }
  };

} // namespace opg

#endif
