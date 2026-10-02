///////////////////////////////////////////////////////////////////////////////
/// \file fast.h
///
/// \brief Standard 3-flavour oscillations in matter (OscProb::PMNS_Fast).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_FAST_H
#define OPG_MODELS_FAST_H

#include "opg/models/hermitian3.h"

namespace opg {

  template <class R = double> struct Fast {
      using Real                   = R;
      static constexpr int  N      = 3;
      static constexpr const char* name = "Fast";

      struct Params {
          MixingParams<3> mix;
      };

      struct Prepared {
          Hermitian3Common<R> common;
      };

      static Prepared prepare(const Params& p)
      {
        Prepared out;
        prepare_hermitian3<R>(p.mix, out.common);
        return out;
      }

      /// Port of PMNS_Fast::UpdateHam (upper triangle + diagonal).
      OPG_HD OPG_INLINE static void hamiltonian(const Prepared& P, R E,
                                                bool nubar, const Segment<R>& s,
                                                Mat<3, R>& H)
      {
        R lv = 2 * R(constants::kGeV2eV) * E;  // 2E in eV

        R kr2GNe = P.common.vfac;
        kr2GNe *= s.density * s.zoa;  // Matter potential in eV

        const Mat<3, R>& Hms = P.common.Hms;
        OPG_UNROLL
        for (int i = 0; i < 3; i++) {
          H(i, i) = Hms(i, i) / lv;
          OPG_UNROLL
          for (int j = i + 1; j < 3; j++) {
            if (!nubar)
              H(i, j) = Hms(i, j) / lv;
            else
              H(i, j) = conj(Hms(i, j)) / lv;
          }
        }
        if (!nubar)
          H(0, 0).re += kr2GNe;
        else
          H(0, 0).re -= kr2GNe;
      }

      OPG_HD OPG_INLINE static void step(const Prepared& P, R E, bool nubar,
                                         const Segment<R>& s, Mat<3, R>& S)
      {
        hermitian3_step<Fast, R>(P, E, nubar, s, S);
      }

      OPG_HD OPG_INLINE static void finalize(const Prepared&, bool,
                                             Mat<3, R>&)
      {
      }
  };

} // namespace opg

#endif
