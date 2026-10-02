///////////////////////////////////////////////////////////////////////////////
/// \file nsi.h
///
/// \brief 3-flavour oscillations with vector non-standard interactions
///        (OscProb::PMNS_NSI).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_NSI_H
#define OPG_MODELS_NSI_H

#include <complex>
#include <stdexcept>

#include "opg/models/hermitian3.h"

namespace opg {

  template <class R = double> struct NSI {
      using Real                        = R;
      static constexpr int         N    = 3;
      static constexpr const char* name = "NSI";

      struct Params {
          MixingParams<3>      mix;
          std::complex<double> eps[3][3] = {};  ///< upper triangle used
          double               coup[3]   = {1, 0, 0};  ///< e, u, d couplings

          /// As PMNS_NSI::SetEps(flvi, flvj, val, phase), 0-based flavours,
          /// flvi <= flvj. The phase is ignored on the diagonal.
          void SetEps(int i, int j, double val, double phase)
          {
            if (i > j) std::swap(i, j);
            if (i < 0 || j > 2) throw std::invalid_argument("NSI::SetEps");
            std::complex<double> h = val;
            if (i != j) h *= std::complex<double>(std::cos(phase), std::sin(phase));
            eps[i][j] = h;
          }
          /// As PMNS_NSI::SetFermCoup(e, u, d).
          void SetFermCoup(double e, double u, double d)
          {
            coup[0] = e;
            coup[1] = u;
            coup[2] = d;
          }
      };

      struct Prepared {
          Hermitian3Common<R> common;
          Complex<R>          eps[3][3];
          R                   coup[3];
      };

      static Prepared prepare(const Params& p)
      {
        Prepared out;
        prepare_hermitian3<R>(p.mix, out.common);
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
            out.eps[i][j] =
                Complex<R>(R(p.eps[i][j].real()), R(p.eps[i][j].imag()));
        for (int k = 0; k < 3; k++) out.coup[k] = R(p.coup[k]);
        return out;
      }

      /// PMNS_NSI::GetZoACoup
      OPG_HD OPG_INLINE static R zoa_coup(const Prepared& P, R zoa)
      {
        return P.coup[0] * zoa              // electrons: Z
               + P.coup[1] * (1 + zoa)      // u-quarks:  A + Z
               + P.coup[2] * (2 - zoa);     // d-quarks: 2A - Z
      }

      /// Port of PMNS_NSI::UpdateHam (upper triangle + diagonal).
      OPG_HD OPG_INLINE static void hamiltonian(const Prepared& P, R E,
                                                bool nubar, const Segment<R>& s,
                                                Mat<3, R>& H)
      {
        R lv = 2 * R(constants::kGeV2eV) * E;  // 2*E in eV

        R kr2GNe   = P.common.vfac * s.density;
        R kr2GNnsi = kr2GNe;

        kr2GNe *= s.zoa;                 // Std matter potential in eV
        kr2GNnsi *= zoa_coup(P, s.zoa);  // NSI matter potential in eV

        const Mat<3, R>& Hms = P.common.Hms;
        OPG_UNROLL
        for (int i = 0; i < 3; i++) {
          OPG_UNROLL
          for (int j = i; j < 3; j++) {
            if (!nubar)
              H(i, j) = Hms(i, j) / lv + kr2GNnsi * P.eps[i][j];
            else
              H(i, j) = conj(Hms(i, j) / lv - kr2GNnsi * P.eps[i][j]);
          }
        }
        if (!nubar)
          H(0, 0).re += kr2GNe;
        else
          H(0, 0).re -= kr2GNe;
      }

      OPG_HD OPG_INLINE static Mat<3, R> initial(const Prepared&, bool)
      {
        return Mat<3, R>::identity();
      }

      OPG_HD OPG_INLINE static void step(const Prepared& P, R E, bool nubar,
                                         const Segment<R>& s, Mat<3, R>& S)
      {
        hermitian3_step<NSI, R>(P, E, nubar, s, S);
      }

      OPG_HD OPG_INLINE static void finalize(const Prepared&, bool, Mat<3, R>&) {}
  };

} // namespace opg

#endif
