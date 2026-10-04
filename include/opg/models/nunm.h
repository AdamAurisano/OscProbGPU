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

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "opg/models/hermitian3.h"

namespace opg {

  /// NUNM parameters with scalar type S. The lower triangle of alpha is
  /// stored as passed to PMNS_NUNM::SetAlpha: a (signed) magnitude, with
  /// alpha_ii = 1 + value on the diagonal, and a phase (i > j).
  template <class S> struct NUNMParams {
      MixingParamsT<3, S> mix;
      int                 scale          = 0;   ///< 0 = low, 1 = high scale
      S                   alpha[3][3]    = {};  ///< values (lower triangle)
      S                   alpha_ph[3][3] = {};  ///< phases (i > j)
      S                   fracVnc        = S(1);  ///< NC potential fraction

      /// As PMNS_NUNM::SetAlpha(i, j, val, phase), 0-based, i >= j.
      void SetAlpha(int i, int j, double val, double phase)
      {
        if (i < j) std::swap(i, j);
        if (j < 0 || i > 2) throw std::invalid_argument("NUNM::SetAlpha");
        alpha[i][j]    = S(val);
        alpha_ph[i][j] = S(i != j ? phase : 0.0);
      }
      void SetFracVnc(double f) { fracVnc = S(f); }
  };

  /// NUNM prepared state with scalar type S.
  template <class S> struct NUNMPrepared {
      Hermitian3Common<S> common;
      Mat<3, S>           alpha;   ///< (normalised if scale 1)
      Mat<3, S>           alphaD;  ///< alpha^dagger
      Mat<3, S>           L, Rm;   ///< potential sandwich L V Rm
      S                   fracVnc;
  };

  template <class R = double> struct NUNM {
      using Real                        = R;
      static constexpr int         N    = 3;
      static constexpr const char* name = "NUNM";

      /// Derivative directions per gradient pass (see opg::grad_traits).
#ifdef OPG_NUNM_GRAD_CHUNK
      static constexpr int grad_chunk = OPG_NUNM_GRAD_CHUNK;
#else
      static constexpr int grad_chunk = 2;  // tuned on V100 (1.2-1.3x faster than K = 1 and 3)
#endif

      template <class S> using ParamsT   = NUNMParams<S>;
      using Params                       = NUNMParams<double>;
      template <class S> using PreparedT = NUNMPrepared<S>;
      using Prepared                     = NUNMPrepared<R>;

      /// Differentiable parameters: the mixing ones (MixingRegistry<3>),
      /// alpha_<ab> values (a >= b), ph_<ab> phases (a > b) and frac_vnc.
      static std::vector<std::string> param_names()
      {
        auto n = MixingRegistry<3>::names();
        for (int j = 0; j < 3; j++)
          for (int i = j; i < 3; i++) n.push_back("alpha_" + pair_name(i, j));
        for (int j = 0; j < 3; j++)
          for (int i = j + 1; i < 3; i++) n.push_back("ph_" + pair_name(i, j));
        n.push_back("frac_vnc");
        return n;
      }
      template <class S> static S& param_ref(ParamsT<S>& p, int idx)
      {
        const int nmix = MixingRegistry<3>::count();
        if (idx < nmix) return MixingRegistry<3>::ref(p.mix, idx);
        int k = nmix;
        for (int j = 0; j < 3; j++)
          for (int i = j; i < 3; i++)
            if (k++ == idx) return p.alpha[i][j];
        for (int j = 0; j < 3; j++)
          for (int i = j + 1; i < 3; i++)
            if (k++ == idx) return p.alpha_ph[i][j];
        if (k == idx) return p.fracVnc;
        throw std::out_of_range("NUNM: bad parameter index");
      }
      template <class S> static ParamsT<S> cast(const Params& p)
      {
        ParamsT<S> q;
        q.mix   = cast_mixing<S>(p.mix);
        q.scale = p.scale;
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) {
            q.alpha[i][j]    = S(p.alpha[i][j]);
            q.alpha_ph[i][j] = S(p.alpha_ph[i][j]);
          }
        q.fracVnc = S(p.fracVnc);
        return q;
      }

      /// With S = double, the arithmetic is that of OscProb with
      /// std::complex, except for the complex division in the high-scale
      /// inverse (round-off level).
      template <class S> static PreparedT<S> prepare_generic(const ParamsT<S>& p)
      {
        using std::cos;
        using std::sin;
        using std::sqrt;
        using C = Complex<S>;
        if (p.scale != 0 && p.scale != 1)
          throw std::invalid_argument("NUNM: scale must be 0 or 1");

        PreparedT<S> out;
        prepare_hermitian3_generic<S>(p.mix, out.common);

        // PMNS_NUNM::SetAlpha
        Mat<3, S> a;
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) {
            C h(S(0), S(0));
            if (i == j)
              h = C(S(1) + p.alpha[i][j], S(0));
            else if (i > j) {
              h = C(p.alpha[i][j], S(0));
              h *= C(cos(p.alpha_ph[i][j]), sin(p.alpha_ph[i][j]));
            }
            a(i, j) = h;
          }

        if (p.scale == 1) {
          // Normalise the mixing matrix in the high-scale scenario to ensure
          // completeness (PMNS_NUNM::PropagatePath).
          S xii[3];
          for (int i = 0; i < 3; i++) {
            C x(S(0), S(0));
            for (int k = 0; k < 3; k++) x += a(i, k) * conj(a(i, k));
            xii[i] = x.re;
          }
          for (int i = 0; i < 3; i++) {
            const S f = S(1) / sqrt(xii[i]);
            for (int j = 0; j < i + 1; j++) a(i, j) *= f;
          }
        }

        Mat<3, S> ad;
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) ad(i, j) = conj(a(j, i));

        out.alpha  = a;
        out.alphaD = ad;
        if (p.scale == 0) {
          out.L  = ad;
          out.Rm = a;
        }
        else {
          out.L  = inverse3(a);
          out.Rm = inverse3(ad);
        }
        out.fracVnc = p.fracVnc;
        return out;
      }

      static Prepared prepare(const Params& p)
      {
        return prepare_generic<R>(cast<R>(p));
      }

      /// Port of PMNS_NUNM::UpdateHam (upper triangle + diagonal), for any
      /// scalar type S of the prepared state and segment type Seg.
      template <class S, class Seg>
      OPG_HD OPG_INLINE static void hamiltonian(const PreparedT<S>& P, R E,
                                                bool nubar, const Seg& s,
                                                Mat<3, S>& H)
      {
        const auto rho = s.density;
        const auto zoa = s.zoa;
        const S lv  = S(2 * R(constants::kGeV2eV) * E);  // 2*E in eV

        // Electron matter potential
        S kr2GNe = P.common.vfac;
        kr2GNe *= rho;
        kr2GNe *= zoa;
        // Neutron matter potential
        S kr2GNn = P.common.vfac;
        kr2GNn *= rho;
        kr2GNn *= (R(1) - zoa);
        kr2GNn /= R(2);
        kr2GNn *= P.fracVnc;

        S V[3];
        OPG_UNROLL
        for (int i = 0; i < 3; i++) V[i] = nubar ? kr2GNn : -kr2GNn;
        if (!nubar)
          V[0] += kr2GNe;
        else
          V[0] -= kr2GNe;

        const Mat<3, S>& Hms = P.common.Hms;
        OPG_UNROLL
        for (int i = 0; i < 3; i++) {
          OPG_UNROLL
          for (int j = i; j < 3; j++) {
            Complex<S> h = !nubar ? Hms(i, j) / lv : conj(Hms(i, j)) / lv;
            Complex<S> w(S(0), S(0));
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

      // --- gradients ---------------------------------------------------------
      /// dS_k = d(alpha^dagger)/dp_k
      template <int K>
      OPG_HD OPG_INLINE static void initial_grad(const PreparedT<Dual<R, K>>& PD,
                                                 bool, Mat<3, R> (&dS)[K])
      {
        OPG_UNROLL
        for (int k = 0; k < K; k++) dS[k] = derivative<K>(PD.alphaD, k);
      }

      template <int K>
      OPG_HD OPG_INLINE static void step_grad(const Prepared&               P,
                                              const PreparedT<Dual<R, K>>& PD,
                                              R E, bool nubar,
                                              const SegmentZ<R, Dual<R, K>>& s,
                                              Mat<3, R>& S, Mat<3, R> (&dS)[K])
      {
        hermitian3_step_grad<NUNM, R, K>(P, PD, E, nubar, s, S, dS);
      }

      /// dS_k <- alpha dS_k + dalpha_k S (S before finalize()).
      template <int K>
      OPG_HD OPG_INLINE static void finalize_grad(const Prepared&               P,
                                                  const PreparedT<Dual<R, K>>& PD,
                                                  bool, Mat<3, R>& S,
                                                  Mat<3, R> (&dS)[K])
      {
        OPG_UNROLL
        for (int k = 0; k < K; k++) {
          apply_operator<3, R>(P.alpha, dS[k]);
          Mat<3, R> t = S;
          apply_operator<3, R>(derivative<K>(PD.alpha, k), t);
          OPG_UNROLL
          for (int i = 0; i < 3; i++)
            OPG_UNROLL
          for (int a = 0; a < 3; a++) dS[k](i, a) += t(i, a);
        }
      }

    private:
      template <int K>
      OPG_HD OPG_INLINE static Mat<3, R> derivative(const Mat<3, Dual<R, K>>& M,
                                                    int k)
      {
        Mat<3, R> d;
        OPG_UNROLL
        for (int i = 0; i < 3; i++)
          OPG_UNROLL
        for (int j = 0; j < 3; j++)
          d(i, j) = Complex<R>(M(i, j).re.d[k], M(i, j).im.d[k]);
        return d;
      }

      static std::string pair_name(int i, int j)
      {
        static const char* fl[3] = {"e", "mu", "tau"};
        return std::string(fl[i]) + fl[j];
      }

      /// 3x3 inverse by cofactors (as Eigen does for fixed 3x3).
      template <class S> static Mat<3, S> inverse3(const Mat<3, S>& m)
      {
        using C = Complex<S>;
        C c00 = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);
        C c01 = m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2);
        C c02 = m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0);
        C det = m(0, 0) * c00 + m(0, 1) * c01 + m(0, 2) * c02;
        if (det.re == S(0) && det.im == S(0))
          throw std::runtime_error("NUNM: singular alpha");
        C id = C(S(1), S(0)) / det;
        Mat<3, S> inv;
        inv(0, 0) = c00 * id;
        inv(1, 0) = c01 * id;
        inv(2, 0) = c02 * id;
        inv(0, 1) = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * id;
        inv(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * id;
        inv(2, 1) = (m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1)) * id;
        inv(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * id;
        inv(1, 2) = (m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2)) * id;
        inv(2, 2) = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) * id;
        return inv;
      }
  };

} // namespace opg

#endif
