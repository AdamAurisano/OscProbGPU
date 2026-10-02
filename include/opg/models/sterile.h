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

#include "opg/core/constants.h"
#include "opg/linalg/jacobi_herm.h"
#include "opg/physics/mixing.h"
#include "opg/physics/propagate.h"

namespace opg {

  template <class R = double> struct Sterile {
      using Real                        = R;
      static constexpr int         N    = 4;
      static constexpr const char* name = "Sterile";

      struct Params {
          MixingParams<4> mix;
      };

      struct Prepared {
          Mat<4, R> Hms;   ///< U diag(dm) U^dagger (upper triangle), eV^2
          R         vfac;  ///< kK2*sqrt(2)*G_F
      };

      static Prepared prepare(const Params& p)
      {
        Prepared out;
        out.Hms  = build_hms<4, R>(p.mix);
        out.vfac = R(constants::matter_prefactor());
        return out;
      }

      /// Port of PMNS_Sterile::UpdateHam, written for the upper triangle
      /// (OscProb fills the lower triangle with the conjugate).
      OPG_HD OPG_INLINE static void hamiltonian(const Prepared& P, R E,
                                                bool nubar, const Segment<R>& s,
                                                Mat<4, R>& H)
      {
        R rho = s.density;
        R zoa = s.zoa;
        R lv  = 2 * R(constants::kGeV2eV) * E;  // 2E in eV

        R kr2GNe = P.vfac * rho * zoa;            // Electron matter potential
        R kr2GNn = P.vfac * rho * (1 - zoa) / 2;  // Neutron matter potential

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
  };

} // namespace opg

#endif
