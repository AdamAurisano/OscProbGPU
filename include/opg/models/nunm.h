///////////////////////////////////////////////////////////////////////////////
/// \file nunm.h
///
/// \brief 3-flavour oscillations with non-unitary neutrino mixing
///        (OscProb::PMNS_NUNM).
///
/// The mixing matrix is N = alpha U with alpha lower-triangular. Initial
/// flavour states are rotated by alpha^dagger, propagated with
///   H = Hms/2E + alpha^dag V alpha             (scale 0, low scale)
///   H = Hms/2E + alpha^-1 V (alpha^dag)^-1     (scale 1, high scale)
/// where V = diag(Vcc - Vnc, -Vnc, -Vnc), and the final state is rotated
/// by alpha. In the high-scale scenario the rows of alpha are normalised
/// (once, in prepare(); see the note in the README about OscProb applying
/// this inside PropagatePath).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_NUNM_H
#define OPG_MODELS_NUNM_H

#include <complex>
#include <stdexcept>

#include "opg/models/hermitian3.h"

namespace opg {

  template <class R = double> struct NUNM {
      using Real                        = R;
      static constexpr int         N    = 3;
      static constexpr const char* name = "NUNM";

      struct Params {
          MixingParams<3> mix;
          int             scale = 0;  ///< 0 = low scale, 1 = high scale
          /// alpha matrix; lower triangle with diagonal 1 + alpha_ii
          std::complex<double> alpha[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
          double               fracVnc     = 1.0;  ///< NC potential fraction

          /// As PMNS_NUNM::SetAlpha(i, j, val, phase), 0-based, i >= j.
          void SetAlpha(int i, int j, double val, double phase)
          {
            if (i < j) std::swap(i, j);
            if (j < 0 || i > 2) throw std::invalid_argument("NUNM::SetAlpha");
            std::complex<double> h = val;
            if (i == j)
              h = 1. + val;
            else
              h *= std::complex<double>(std::cos(phase), std::sin(phase));
            alpha[i][j] = h;
          }
          void SetFracVnc(double f) { fracVnc = f; }
      };

      struct Prepared {
          Hermitian3Common<R> common;
          Mat<3, R>           alpha;   ///< (normalised if scale 1)
          Mat<3, R>           alphaD;  ///< alpha^dagger
          Mat<3, R>           L, Rm;   ///< potential sandwich L V Rm
          R                   fracVnc;
      };

      static Prepared prepare(const Params& p)
      {
        using cplx = std::complex<double>;
        Prepared out;
        prepare_hermitian3<R>(p.mix, out.common);

        cplx a[3][3];
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) a[i][j] = p.alpha[i][j];

        if (p.scale == 1) {
          // Normalise the mixing matrix in the high-scale scenario to ensure
          // completeness (PMNS_NUNM::PropagatePath).
          cplx X[3][3];
          for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
              X[i][j] = 0;
              for (int k = 0; k < 3; k++) X[i][j] += a[i][k] * conj(a[j][k]);
            }
          for (int i = 0; i < 3; i++)
            for (int j = 0; j < i + 1; j++)
              a[i][j] *= 1 / std::sqrt(X[i][i].real());
        }

        cplx ad[3][3];
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) ad[i][j] = conj(a[j][i]);

        cplx L[3][3], Rm[3][3];
        if (p.scale == 0) {
          copy3(ad, L);
          copy3(a, Rm);
        }
        else if (p.scale == 1) {
          inverse3(a, L);
          inverse3(ad, Rm);
        }
        else
          throw std::invalid_argument("NUNM: scale must be 0 or 1");

        to_mat(a, out.alpha);
        to_mat(ad, out.alphaD);
        to_mat(L, out.L);
        to_mat(Rm, out.Rm);
        out.fracVnc = R(p.fracVnc);
        return out;
      }

      /// Port of PMNS_NUNM::UpdateHam (upper triangle + diagonal).
      OPG_HD OPG_INLINE static void hamiltonian(const Prepared& P, R E,
                                                bool nubar, const Segment<R>& s,
                                                Mat<3, R>& H)
      {
        R rho = s.density;
        R zoa = s.zoa;
        R lv  = 2 * R(constants::kGeV2eV) * E;  // 2*E in eV

        R kr2GNe = P.common.vfac * rho * zoa;  // Electron matter potential
        R kr2GNn = P.common.vfac * rho * (1 - zoa) / 2 *
                   P.fracVnc;  // Neutron matter potential

        R V[3];
        OPG_UNROLL
        for (int i = 0; i < 3; i++) V[i] = nubar ? kr2GNn : -kr2GNn;
        if (!nubar)
          V[0] += kr2GNe;
        else
          V[0] -= kr2GNe;

        const Mat<3, R>& Hms = P.common.Hms;
        OPG_UNROLL
        for (int i = 0; i < 3; i++) {
          OPG_UNROLL
          for (int j = i; j < 3; j++) {
            Complex<R> h = !nubar ? Hms(i, j) / lv : conj(Hms(i, j)) / lv;
            Complex<R> w(0, 0);
            OPG_UNROLL
            for (int k = 0; k < 3; k++) w += P.L(i, k) * V[k] * P.Rm(k, j);
            H(i, j) = h + w;
          }
        }
      }

      OPG_HD OPG_INLINE static Mat<3, R> initial(const Prepared& P, bool)
      {
        // Columns are alpha^dagger e_a (PMNS_NUNM::ApplyAlphaDagger)
        return P.alphaD;
      }

      OPG_HD OPG_INLINE static void step(const Prepared& P, R E, bool nubar,
                                         const Segment<R>& s, Mat<3, R>& S)
      {
        hermitian3_step<NUNM, R>(P, E, nubar, s, S);
      }

      OPG_HD OPG_INLINE static void finalize(const Prepared& P, bool,
                                             Mat<3, R>& S)
      {
        apply_operator<3, R>(P.alpha, S);  // PMNS_NUNM::ApplyAlpha
      }

    private:
      using cplx = std::complex<double>;

      static void copy3(const cplx a[3][3], cplx b[3][3])
      {
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) b[i][j] = a[i][j];
      }

      static void to_mat(const cplx a[3][3], Mat<3, R>& m)
      {
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
            m(i, j) = Complex<R>(R(a[i][j].real()), R(a[i][j].imag()));
      }

      /// 3x3 inverse by cofactors (as Eigen does for fixed 3x3).
      static void inverse3(const cplx m[3][3], cplx inv[3][3])
      {
        cplx c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
        cplx c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
        cplx c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
        cplx det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
        if (std::abs(det) == 0) throw std::runtime_error("NUNM: singular alpha");
        cplx id = 1.0 / det;
        inv[0][0] = c00 * id;
        inv[1][0] = c01 * id;
        inv[2][0] = c02 * id;
        inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id;
        inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id;
        inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id;
        inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id;
        inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id;
        inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id;
      }
  };

} // namespace opg

#endif
