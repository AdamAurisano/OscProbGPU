///////////////////////////////////////////////////////////////////////////////
/// \file liv.h
///
/// \brief 3-flavour oscillations with Lorentz invariance violation as
///        modelled by the SME (OscProb::PMNS_LIV).
///
/// H = Hms/2E + V_cc + sum_{d=3..8} E^(d-3) [ (+-) aT^(d) for odd d,
///                                           -cT^(d) for even d ]
/// (cT^(4) with the factor 4/3 of OscProb), aT in GeV^(4-d), cT in
/// GeV^(4-d); aT changes sign for antineutrinos (CPT-odd). The LIV terms do
/// not vanish in vacuum, so (as OscProb) the Hamiltonian is always
/// diagonalised.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_LIV_H
#define OPG_MODELS_LIV_H

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "opg/models/hermitian3.h"

namespace opg {

  /// LIV parameters with scalar type S. Coefficients of dimension d are at
  /// index (d-3)/2 (aT: d = 3, 5, 7; cT: d = 4, 6, 8), stored as a (signed)
  /// magnitude and a phase (upper triangle; phases ignored on the diagonal).
  template <class S> struct LIVParams {
      MixingParamsT<3, S> mix;
      S                   aT[3][3][3]    = {};
      S                   aT_ph[3][3][3] = {};
      S                   cT[3][3][3]    = {};
      S                   cT_ph[3][3][3] = {};

      /// As PMNS_LIV::SetaT(flvi, flvj, dim, val, phase), dim = 3, 5, 7.
      void SetaT(int i, int j, int dim, double val, double phase)
      {
        set(aT, aT_ph, i, j, dim, 3, val, phase, "SetaT");
      }
      /// As PMNS_LIV::SetcT(flvi, flvj, dim, val, phase), dim = 4, 6, 8.
      void SetcT(int i, int j, int dim, double val, double phase)
      {
        set(cT, cT_ph, i, j, dim, 4, val, phase, "SetcT");
      }

    private:
      static void set(S (&v)[3][3][3], S (&ph)[3][3][3], int i, int j, int dim,
                      int dmin, double val, double phase, const char* what)
      {
        if (i > j) std::swap(i, j);
        if (i < 0 || j > 2)
          throw std::invalid_argument(std::string("LIV::") + what + ": bad flavours");
        if (dim < dmin || dim > dmin + 4 || (dim - dmin) % 2)
          throw std::invalid_argument(std::string("LIV::") + what + ": bad dimension");
        const int k = (dim - 3) / 2;
        v[i][j][k]  = S(val);
        ph[i][j][k] = S(i != j ? phase : 0.0);
      }
  };

  /// LIV prepared state with scalar type S.
  template <class S> struct LIVPrepared {
      Hermitian3Common<S> common;
      Complex<S>          aT[3][3][3];  ///< upper triangle
      Complex<S>          cT[3][3][3];
  };

  template <class R = double> struct LIV {
      using Real                        = R;
      static constexpr int         N    = 3;
      static constexpr const char* name = "LIV";

      /// LIV terms are present in vacuum: always diagonalise.
      static constexpr bool vacuum_shortcut = false;

      /// Derivative directions per gradient pass (see opg::grad_traits).
#ifdef OPG_LIV_GRAD_CHUNK
      static constexpr int grad_chunk = OPG_LIV_GRAD_CHUNK;
#else
      static constexpr int grad_chunk = 2;
#endif

      template <class S> using ParamsT   = LIVParams<S>;
      using Params                       = LIVParams<double>;
      template <class S> using PreparedT = LIVPrepared<S>;
      using Prepared                     = LIVPrepared<R>;

      /// Differentiable parameters: mixing (MixingRegistry<3>), then for each
      /// dimension d = 3..8 the magnitudes <c><d>_<ab> (a <= b) and phases
      /// ph_<c><d>_<ab> (a < b), with <c> = aT for odd d and cT for even d.
      static std::vector<std::string> param_names()
      {
        auto n = MixingRegistry<3>::names();
        for (int dim = 3; dim <= 8; dim++) {
          const std::string c = (dim % 2 ? "aT" : "cT") + std::to_string(dim);
          for (int i = 0; i < 3; i++)
            for (int j = i; j < 3; j++) n.push_back(c + "_" + pair_name(i, j));
          for (int i = 0; i < 3; i++)
            for (int j = i + 1; j < 3; j++) n.push_back("ph_" + c + "_" + pair_name(i, j));
        }
        return n;
      }

      /// Default gradient selection: mixing, aT3 and cT4 (the minimal SME);
      /// higher dimensions can be selected by name.
      static std::vector<std::string> default_param_names()
      {
        auto n = param_names();
        n.resize(MixingRegistry<3>::count() + 2 * 9);
        return n;
      }

      template <class S> static S& param_ref(ParamsT<S>& p, int idx)
      {
        const int nmix = MixingRegistry<3>::count();
        if (idx < nmix) return MixingRegistry<3>::ref(p.mix, idx);
        int k = nmix;
        for (int dim = 3; dim <= 8; dim++) {
          const int c  = (dim - 3) / 2;
          auto&     v  = dim % 2 ? p.aT : p.cT;
          auto&     ph = dim % 2 ? p.aT_ph : p.cT_ph;
          for (int i = 0; i < 3; i++)
            for (int j = i; j < 3; j++)
              if (k++ == idx) return v[i][j][c];
          for (int i = 0; i < 3; i++)
            for (int j = i + 1; j < 3; j++)
              if (k++ == idx) return ph[i][j][c];
        }
        throw std::out_of_range("LIV: bad parameter index");
      }

      template <class S> static ParamsT<S> cast(const Params& p)
      {
        ParamsT<S> q;
        q.mix = cast_mixing<S>(p.mix);
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
            for (int c = 0; c < 3; c++) {
              q.aT[i][j][c]    = S(p.aT[i][j][c]);
              q.aT_ph[i][j][c] = S(p.aT_ph[i][j][c]);
              q.cT[i][j][c]    = S(p.cT[i][j][c]);
              q.cT_ph[i][j][c] = S(p.cT_ph[i][j][c]);
            }
        return q;
      }

      template <class S> static PreparedT<S> prepare_generic(const ParamsT<S>& p)
      {
        using std::cos;
        using std::sin;
        using C = Complex<S>;
        PreparedT<S> out;
        prepare_hermitian3_generic<S>(p.mix, out.common);
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
            for (int c = 0; c < 3; c++) {
              // PMNS_LIV::SetaT/SetcT: h = val; h *= exp(i phase) off the
              // diagonal (the same complex product as std::complex)
              C a(p.aT[i][j][c], S(0)), b(p.cT[i][j][c], S(0));
              if (i < j) {
                a *= C(cos(p.aT_ph[i][j][c]), sin(p.aT_ph[i][j][c]));
                b *= C(cos(p.cT_ph[i][j][c]), sin(p.cT_ph[i][j][c]));
              }
              out.aT[i][j][c] = i <= j ? a : C(S(0), S(0));
              out.cT[i][j][c] = i <= j ? b : C(S(0), S(0));
            }
        return out;
      }

      static Prepared prepare(const Params& p)
      {
        return prepare_generic<R>(cast<R>(p));
      }

      /// Port of PMNS_LIV::UpdateHam (upper triangle + diagonal), for any
      /// scalar type S of the prepared state and segment type Seg.
      template <class S, class Seg>
      OPG_HD OPG_INLINE static void hamiltonian(const PreparedT<S>& P, R E,
                                                bool nubar, const Seg& s,
                                                Mat<3, S>& H)
      {
        using C    = Complex<S>;
        const S lv = S(2 * R(constants::kGeV2eV) * E);  // 2*E in eV

        // Set the vacuum Hamiltonian
        const Mat<3, S>& Hms = P.common.Hms;
        OPG_UNROLL
        for (int i = 0; i < 3; i++)
          OPG_UNROLL
        for (int j = i; j < 3; j++) H(i, j) = Hms(i, j) / lv;

        // Add matter potential (kK2 * sqrt2 * Gf * rho * zoa, in this order)
        S kr2GNe = P.common.vfac;
        kr2GNe *= s.density;
        kr2GNe *= s.zoa;
        if (!nubar)
          H(0, 0).re += kr2GNe;
        else
          H(0, 0).re -= kr2GNe;

        // Add LIV terms for all dimensions
        R energy_pow = R(constants::kGeV2eV);
        OPG_UNROLL
        for (int dim = 3; dim < 9; dim++) {
          if (dim > 3) energy_pow *= E;
          const int c = (dim - 3) / 2;
          OPG_UNROLL
          for (int i = 0; i < 3; i++) {
            OPG_UNROLL
            for (int j = i; j < 3; j++) {
              C t(S(energy_pow), S(0));
              if (dim % 2 == 1)  // aT is CPT-odd
                t *= nubar ? -P.aT[i][j][c] : P.aT[i][j][c];
              else if (dim == 4)  // cT: 4/3 for backward compatibility
                t *= S(R(-4) / R(3)) * P.cT[i][j][c];
              else
                t *= -P.cT[i][j][c];
              H(i, j) += t;
            }
          }
        }

        // Conjugate Hamiltonian for antineutrinos
        if (nubar) {
          OPG_UNROLL
          for (int i = 0; i < 3; i++)
            OPG_UNROLL
          for (int j = i + 1; j < 3; j++) H(i, j) = conj(H(i, j));
        }
      }

      OPG_HD OPG_INLINE static Mat<3, R> initial(const Prepared&, bool)
      {
        return Mat<3, R>::identity();
      }

      OPG_HD OPG_INLINE static void step(const Prepared& P, R E, bool nubar,
                                         const Segment<R>& s, Mat<3, R>& S)
      {
        hermitian3_step<LIV, R>(P, E, nubar, s, S);
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
        hermitian3_step_grad<LIV, R, K>(P, PD, E, nubar, s, S, dS);
      }

      template <int K>
      OPG_HD OPG_INLINE static void finalize_grad(const Prepared&,
                                                  const PreparedT<Dual<R, K>>&,
                                                  bool, Mat<3, R>&,
                                                  Mat<3, R> (&)[K])
      {
      }

    private:
      static std::string pair_name(int i, int j)
      {
        static const char* fl[3] = {"e", "mu", "tau"};
        return std::string(fl[i]) + fl[j];
      }
  };

} // namespace opg

#endif
