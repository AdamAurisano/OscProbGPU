///////////////////////////////////////////////////////////////////////////////
/// \file fast.h
///
/// \brief Standard 3-flavour oscillations in matter (OscProb::PMNS_Fast).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_FAST_H
#define OPG_MODELS_FAST_H

#include <string>
#include <vector>

#include "opg/models/hermitian3.h"

namespace opg {

  /// Fast parameters with scalar type S (independent of the device precision).
  template <class S> struct FastParams {
      MixingParamsT<3, S> mix;
  };

  /// Fast prepared state with scalar type S.
  template <class S> struct FastPrepared {
      Hermitian3Common<S> common;
  };

  template <class R = double> struct Fast {
      using Real                        = R;
      static constexpr int         N    = 3;
      static constexpr const char* name = "Fast";

      /// Derivative directions per gradient pass (see opg::grad_traits).
#ifdef OPG_FAST_GRAD_CHUNK
      static constexpr int grad_chunk = OPG_FAST_GRAD_CHUNK;
#else
      static constexpr int grad_chunk = 2;  // tuned on V100 (2 and 3 equal, 2 spills less)
#endif

      template <class S> using ParamsT   = FastParams<S>;
      using Params                       = FastParams<double>;
      template <class S> using PreparedT = FastPrepared<S>;
      using Prepared                     = FastPrepared<R>;

      /// Names of the differentiable parameters (MixingRegistry<3>):
      /// th12, th13, th23, d13, dm21, dm31.
      static std::vector<std::string> param_names()
      {
        return MixingRegistry<3>::names();
      }
      template <class S> static S& param_ref(ParamsT<S>& p, int idx)
      {
        return MixingRegistry<3>::ref(p.mix, idx);
      }
      template <class S> static ParamsT<S> cast(const Params& p)
      {
        return ParamsT<S>{cast_mixing<S>(p.mix)};
      }

      template <class S> static PreparedT<S> prepare_generic(const ParamsT<S>& p)
      {
        PreparedT<S> out;
        prepare_hermitian3_generic<S>(p.mix, out.common);
        return out;
      }

      static Prepared prepare(const Params& p)
      {
        return prepare_generic<R>(cast<R>(p));
      }

      /// Port of PMNS_Fast::UpdateHam (upper triangle + diagonal), for any
      /// scalar type S of the prepared state and segment type Seg (whose Z/A
      /// may be dual).
      template <class S, class Seg>
      OPG_HD OPG_INLINE static void hamiltonian(const PreparedT<S>& P, R E,
                                                bool nubar, const Seg& s,
                                                Mat<3, S>& H)
      {
        const S lv = S(2 * R(constants::kGeV2eV) * E);  // 2E in eV

        S kr2GNe = P.common.vfac;
        kr2GNe *= s.density * s.zoa;  // Matter potential in eV

        const Mat<3, S>& Hms = P.common.Hms;
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

      OPG_HD OPG_INLINE static Mat<3, R> initial(const Prepared&, bool)
      {
        return Mat<3, R>::identity();
      }

      OPG_HD OPG_INLINE static void step(const Prepared& P, R E, bool nubar,
                                         const Segment<R>& s, Mat<3, R>& S)
      {
        hermitian3_step<Fast, R>(P, E, nubar, s, S);
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

      template <int K>
      OPG_HD OPG_INLINE static void step_grad(const Prepared&               P,
                                              const PreparedT<Dual<R, K>>& PD,
                                              R E, bool nubar,
                                              const SegmentZ<R, Dual<R, K>>& s,
                                              Mat<3, R>& S, Mat<3, R> (&dS)[K])
      {
        hermitian3_step_grad<Fast, R, K>(P, PD, E, nubar, s, S, dS);
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
