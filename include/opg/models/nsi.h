///////////////////////////////////////////////////////////////////////////////
/// \file nsi.h
///
/// \brief 3-flavour oscillations with vector non-standard interactions
///        (OscProb::PMNS_NSI).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_NSI_H
#define OPG_MODELS_NSI_H

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "opg/models/hermitian3.h"

namespace opg {

  /// NSI parameters with scalar type S. The couplings eps_ij (upper
  /// triangle) are stored as a (signed) magnitude and a phase, as passed to
  /// PMNS_NSI::SetEps; the phase is ignored on the diagonal.
  template <class S> struct NSIParams {
      MixingParamsT<3, S> mix;
      S                   eps[3][3]    = {};  ///< magnitudes (upper triangle)
      S                   eps_ph[3][3] = {};  ///< phases (i < j)
      S                   coup[3] = {S(1), S(0), S(0)};  ///< e, u, d couplings

      /// As PMNS_NSI::SetEps(flvi, flvj, val, phase), 0-based flavours.
      void SetEps(int i, int j, double val, double phase)
      {
        if (i > j) std::swap(i, j);
        if (i < 0 || j > 2) throw std::invalid_argument("NSI::SetEps");
        eps[i][j]    = S(val);
        eps_ph[i][j] = S(i != j ? phase : 0.0);
      }
      /// As PMNS_NSI::SetFermCoup(e, u, d).
      void SetFermCoup(double e, double u, double d)
      {
        coup[0] = S(e);
        coup[1] = S(u);
        coup[2] = S(d);
      }
  };

  /// NSI prepared state with scalar type S.
  template <class S> struct NSIPrepared {
      Hermitian3Common<S> common;
      Complex<S>          eps[3][3];
      S                   coup[3];
  };

  template <class R = double> struct NSI {
      using Real                        = R;
      static constexpr int         N    = 3;
      static constexpr const char* name = "NSI";

      /// Derivative directions per gradient pass (see opg::grad_traits).
#ifdef OPG_NSI_GRAD_CHUNK
      static constexpr int grad_chunk = OPG_NSI_GRAD_CHUNK;
#else
      static constexpr int grad_chunk = 2;  // tuned on V100 (1.2-1.3x faster than K = 1 and 3)
#endif

      template <class S> using ParamsT   = NSIParams<S>;
      using Params                       = NSIParams<double>;
      template <class S> using PreparedT = NSIPrepared<S>;
      using Prepared                     = NSIPrepared<R>;

      /// Differentiable parameters: the mixing ones (MixingRegistry<3>),
      /// eps_<ab> magnitudes, ph_<ab> phases (a != b) and coup_<f>.
      static std::vector<std::string> param_names()
      {
        auto n = MixingRegistry<3>::names();
        for (int i = 0; i < 3; i++)
          for (int j = i; j < 3; j++) n.push_back("eps_" + pair_name(i, j));
        for (int i = 0; i < 3; i++)
          for (int j = i + 1; j < 3; j++) n.push_back("ph_" + pair_name(i, j));
        for (const char* f : {"e", "u", "d"}) n.push_back(std::string("coup_") + f);
        return n;
      }
      /// Parameters differentiated by default (Propagator::
      /// set_gradient_params() without arguments): all but the fermion
      /// couplings coup_<f>, which can still be selected by name.
      static std::vector<std::string> default_param_names()
      {
        auto n = param_names();
        n.resize(n.size() - 3);
        return n;
      }

      template <class S> static S& param_ref(ParamsT<S>& p, int idx)
      {
        const int nmix = MixingRegistry<3>::count();
        if (idx < nmix) return MixingRegistry<3>::ref(p.mix, idx);
        int k = nmix;
        for (int i = 0; i < 3; i++)
          for (int j = i; j < 3; j++)
            if (k++ == idx) return p.eps[i][j];
        for (int i = 0; i < 3; i++)
          for (int j = i + 1; j < 3; j++)
            if (k++ == idx) return p.eps_ph[i][j];
        for (int f = 0; f < 3; f++)
          if (k++ == idx) return p.coup[f];
        throw std::out_of_range("NSI: bad parameter index");
      }
      template <class S> static ParamsT<S> cast(const Params& p)
      {
        ParamsT<S> q;
        q.mix = cast_mixing<S>(p.mix);
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) {
            q.eps[i][j]    = S(p.eps[i][j]);
            q.eps_ph[i][j] = S(p.eps_ph[i][j]);
          }
        for (int f = 0; f < 3; f++) q.coup[f] = S(p.coup[f]);
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
          for (int j = 0; j < 3; j++) {
            // PMNS_NSI::SetEps: h = val; h *= exp(i phase) off the diagonal
            // (the same complex product as std::complex)
            C h(p.eps[i][j], S(0));
            if (i < j) h *= C(cos(p.eps_ph[i][j]), sin(p.eps_ph[i][j]));
            out.eps[i][j] = i <= j ? h : C(S(0), S(0));
          }
        for (int f = 0; f < 3; f++) out.coup[f] = p.coup[f];
        return out;
      }

      static Prepared prepare(const Params& p)
      {
        return prepare_generic<R>(cast<R>(p));
      }

      /// PMNS_NSI::GetZoACoup
      template <class S, class Z>
      OPG_HD OPG_INLINE static auto zoa_coup(const PreparedT<S>& P, Z zoa)
      {
        return P.coup[0] * zoa              // electrons: Z
               + P.coup[1] * (R(1) + zoa)   // u-quarks:  A + Z
               + P.coup[2] * (R(2) - zoa);  // d-quarks: 2A - Z
      }

      /// Port of PMNS_NSI::UpdateHam (upper triangle + diagonal), for any
      /// scalar type S of the prepared state and segment type Seg.
      template <class S, class Seg>
      OPG_HD OPG_INLINE static void hamiltonian(const PreparedT<S>& P, R E,
                                                bool nubar, const Seg& s,
                                                Mat<3, S>& H)
      {
        const S lv = S(2 * R(constants::kGeV2eV) * E);  // 2*E in eV

        S kr2GNe = P.common.vfac;
        kr2GNe *= s.density;
        S kr2GNnsi = kr2GNe;

        kr2GNe *= s.zoa;                 // Std matter potential in eV
        kr2GNnsi *= zoa_coup(P, s.zoa);  // NSI matter potential in eV

        const Mat<3, S>& Hms = P.common.Hms;
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
        hermitian3_step_grad<NSI, R, K>(P, PD, E, nubar, s, S, dS);
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

    public:
      OPG_HERMITIAN3_ADJOINT(NSI, true)
  };

} // namespace opg

#endif
