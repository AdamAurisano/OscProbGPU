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
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_DECAY_H
#define OPG_MODELS_DECAY_H

#include <complex>
#include <stdexcept>

#include "opg/core/constants.h"
#include "opg/linalg/expm.h"
#include "opg/physics/mixing.h"
#include "opg/physics/propagate.h"

namespace opg {

  template <class R = double> struct Decay {
      using Real                        = R;
      static constexpr int         N    = 3;
      static constexpr const char* name = "Decay";

      struct Params {
          MixingParams<3> mix;
          double          alpha[3] = {0, 0, 0};  ///< alpha_j in eV^2 (j = 2, 3)

          /// As PMNS_Decay::SetAlpha2 / SetAlpha3 (must be >= 0).
          void SetAlpha2(double a)
          {
            if (a < 0) throw std::invalid_argument("Decay: alpha2 < 0");
            alpha[1] = a;
          }
          void SetAlpha3(double a)
          {
            if (a < 0) throw std::invalid_argument("Decay: alpha3 < 0");
            alpha[2] = a;
          }
      };

      struct Prepared {
          Mat<3, R> Heff[2];  ///< Hms - i Hd (full matrix) for nu / nubar
          R         vfac;     ///< kK2*sqrt(2)*G_F
      };

      /// Port of PMNS_Decay::BuildHms for each of nu and nubar.
      static Prepared prepare(const Params& p)
      {
        using cplx = std::complex<double>;
        Prepared out;
        double   dm[3] = {0, p.mix.dm[1], p.mix.dm[2]};
        double   al[3] = {p.alpha[0], p.alpha[1], p.alpha[2]};
        for (int nb = 0; nb < 2; nb++) {
          cplx Hms[3][3], Hd[3][3];
          rotate_diagonal<3>(dm, p.mix.th, p.mix.dcp, Hms);
          rotate_diagonal<3>(al, p.mix.th, p.mix.dcp, Hd);

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

          const cplx numi(0.0, 1.0);
          for (int j = 0; j < 3; j++)
            for (int i = 0; i < 3; i++) {
              cplx h = Hms[i][j] - numi * Hd[i][j];
              out.Heff[nb](i, j) = Complex<R>(R(h.real()), R(h.imag()));
            }
        }
        out.vfac = R(constants::matter_prefactor());
        return out;
      }

      OPG_HD OPG_INLINE static Mat<3, R> initial(const Prepared&, bool)
      {
        return Mat<3, R>::identity();
      }

      /// Port of PMNS_Decay::UpdateHam + PropagatePath.
      OPG_HD OPG_INLINE static void step(const Prepared& P, R E, bool nubar,
                                         const Segment<R>& s, Mat<3, R>& S)
      {
        R lv = 2 * R(constants::kGeV2eV) * E;  // 2E in eV

        R kr2GNe = P.vfac;
        kr2GNe *= s.density * s.zoa;  // Matter potential in eV

        const Mat<3, R>& Hf = P.Heff[nubar ? 1 : 0];
        Mat<3, R>        H;
        OPG_UNROLL
        for (int i = 0; i < 3; i++)
          OPG_UNROLL
        for (int j = 0; j < 3; j++) H(i, j) = Hf(i, j) / lv;
        if (!nubar)
          H(0, 0).re += kr2GNe;
        else
          H(0, 0).re -= kr2GNe;

        // Evolution operator exp(-i H L)
        const Complex<R> mil(0, -length_in_eV(s.length));
        OPG_UNROLL
        for (int i = 0; i < 3; i++)
          OPG_UNROLL
        for (int j = 0; j < 3; j++) H(i, j) *= mil;

        Mat<3, R> U = expm<3, R>(H);
        apply_operator<3, R>(U, S);
      }

      OPG_HD OPG_INLINE static void finalize(const Prepared&, bool, Mat<3, R>&) {}
  };

} // namespace opg

#endif
