///////////////////////////////////////////////////////////////////////////////
/// \file snsi.h
///
/// \brief 3-flavour oscillations with scalar non-standard interactions
///        (OscProb::PMNS_SNSI).
///
/// Scalar NSI modify the neutrino mass matrix: with M = U diag(m) U^dag,
///   H = (M + n_f eps)(M + n_f eps)^dag / 2E + V_cc,
/// where n_f = 1e6 kK2 rho (coupling-weighted Z/A) in eV MeV^2 and eps is in
/// MeV^-2. The absolute masses follow from the lightest mass (SetLowestMass)
/// and the splittings. eps and the fermion couplings are set as for NSI.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_SNSI_H
#define OPG_MODELS_SNSI_H

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "opg/models/nsi.h"

namespace opg {

  /// SNSI parameters: NSI parameters plus the lightest neutrino mass (eV).
  template <class S> struct SNSIParams : NSIParams<S> {
      S mlight = S(0);
      /// As PMNS_SNSI::SetLowestMass(m), m in eV.
      void SetLowestMass(double m) { mlight = S(m); }
  };

  /// SNSI prepared state: as NSI, with common.Hms the mass matrix
  /// U diag(m) U^dag (eV) instead of the mass-squared matrix.
  template <class S> struct SNSIPrepared : NSIPrepared<S> {
      S         kk2e6;  ///< 1e6 * kK2 (scalar NSI prefactor)
      S         m1;     ///< mass of state 1 (eV)
      Mat<3, S> B;      ///< M - m1 I (upper triangle)
  };

  template <class R = double> struct SNSI {
      using Real                        = R;
      static constexpr int         N    = 3;
      static constexpr const char* name = "SNSI";
      using Base                        = NSI<R>;

      /// Derivative directions per gradient pass (see opg::grad_traits).
#ifdef OPG_SNSI_GRAD_CHUNK
      static constexpr int grad_chunk = OPG_SNSI_GRAD_CHUNK;
#else
      static constexpr int grad_chunk = 2;
#endif

      template <class S> using ParamsT   = SNSIParams<S>;
      using Params                       = SNSIParams<double>;
      template <class S> using PreparedT = SNSIPrepared<S>;
      using Prepared                     = SNSIPrepared<R>;

      /// NSI's parameters followed by mlight (lightest mass, eV).
      static std::vector<std::string> param_names()
      {
        auto n = Base::param_names();
        n.push_back("mlight");
        return n;
      }
      /// Default gradient selection: as NSI (without the fermion couplings
      /// and the lightest mass, which can be selected by name).
      static std::vector<std::string> default_param_names()
      {
        return Base::default_param_names();
      }
      template <class S> static S& param_ref(ParamsT<S>& p, int idx)
      {
        const int nb = int(Base::param_names().size());
        if (idx < nb) return Base::template param_ref<S>(p, idx);
        if (idx == nb) return p.mlight;
        throw std::out_of_range("SNSI: bad parameter index");
      }
      template <class S> static ParamsT<S> cast(const Params& p)
      {
        ParamsT<S> q;
        static_cast<NSIParams<S>&>(q) = Base::template cast<S>(p);
        q.mlight                      = S(p.mlight);
        return q;
      }

      /// Absolute masses m_j (eV). Values: PMNS_SNSI::BuildHms arithmetic
      /// (bit-identical), the lightest mass made the smallest one (e.g. for
      /// inverted ordering). Derivatives: from m_j^2 = M^2 + dm_j - dm_l (l
      /// the lightest state), which has no cancellation; the lightest mass is
      /// |M|, differentiated from M >= 0 (one-sided at M = 0).
      template <class S> static void masses(const ParamsT<S>& p, S m[3])
      {
        using std::fabs;
        using std::sqrt;
        using V             = value_type_t<S>;
        const V M           = value_of(p.mlight);
        const V dm[3]       = {V(0), value_of(p.mix.dm[1]), value_of(p.mix.dm[2])};
        V       m1          = M;
        for (int i = 1; i < 3; i++)
          if (dm[i] + m1 * m1 < M * M) m1 = sqrt(fabs(M * M - dm[i]));
        V mv[3] = {m1, V(0), V(0)};
        for (int j = 1; j < 3; j++) mv[j] = sqrt(fabs(dm[j] + m1 * m1));
        if constexpr (std::is_same_v<V, long double>) {
          // long double serves reference calculations: use the exact masses
          // m_j = sqrt(M^2 + dm_j - dm_l) (no cancellation for the lightest
          // state) rather than reproducing OscProb's rounding
          int l = 0;
          for (int j = 1; j < 3; j++)
            if (dm[j] < dm[l]) l = j;
          for (int j = 0; j < 3; j++)
            mv[j] = j == l ? fabs(M) : sqrt(M * M + dm[j] - dm[l]);
        }
        for (int j = 0; j < 3; j++) m[j] = S(mv[j]);
        if constexpr (dual_size<S>::value > 0) {
          int l = 0;
          for (int j = 1; j < 3; j++)
            if (dm[j] < dm[l]) l = j;
          for (int k = 0; k < dual_size<S>::value; k++) {
            const V dM     = p.mlight.d[k];
            const V ddm[3] = {V(0), p.mix.dm[1].d[k], p.mix.dm[2].d[k]};
            for (int j = 0; j < 3; j++)
              m[j].d[k] = (j == l || mv[j] == V(0))
                              ? (M >= V(0) ? dM : -dM)
                              : (M * dM + (ddm[j] - ddm[l]) / V(2)) / mv[j];
          }
        }
      }

      /// Port of PMNS_SNSI::BuildHms: M = U diag(m) U^dag (see masses()).
      template <class S> static PreparedT<S> prepare_generic(const ParamsT<S>& p)
      {
        PreparedT<S> out;
        static_cast<NSIPrepared<S>&>(out) = Base::template prepare_generic<S>(p);

        S m[3];
        masses<S>(p, m);
        S diag[3];
        diag[0] = S(0);
        for (int j = 1; j < 3; j++) diag[j] = m[j] - m[0];
        Complex<S> H[3][3];
        rotate_diagonal_generic<3, S>(diag, p.mix.th, p.mix.dcp, H);
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) out.B(i, j) = H[i][j];
        for (int i = 0; i < 3; i++) H[i][i].re += m[0];
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) out.common.Hms(i, j) = H[i][j];
        out.kk2e6 = S(1e6 * constants::kK2());
        out.m1    = m[0];
        return out;
      }

      static Prepared prepare(const Params& p)
      {
        return prepare_generic<R>(cast<R>(p));
      }

      /// Port of PMNS_SNSI::UpdateHam (upper triangle + diagonal), for any
      /// scalar type S of the prepared state and segment type Seg.
      template <class S, class Seg>
      OPG_HD OPG_INLINE static void hamiltonian(const PreparedT<S>& P, R E,
                                                bool nubar, const Seg& s,
                                                Mat<3, S>& H)
      {
        using C = Complex<S>;
        using std::sqrt;
        const S sqrtlv = S(sqrt(2 * R(constants::kGeV2eV) * E));  // sqrt(2E) in eV^1/2

        // Effective density of fermions in eV * MeV^2
        S nsiCoup = P.kk2e6;
        nsiCoup *= s.density;
        nsiCoup *= Base::zoa_coup(P, s.zoa);

#ifdef OPG_OSCPROB_BITWISE
        // (M + n eps)^2 / 2E exactly as OscProb (HermitianSquare)
        C A[3][3];
        OPG_UNROLL
        for (int i = 0; i < 3; i++)
          OPG_UNROLL
        for (int j = i; j < 3; j++) {
          A[i][j] = (P.common.Hms(i, j) + nsiCoup * P.eps[i][j]) / sqrtlv;
          if (nubar) A[i][j] = conj(A[i][j]);
        }
        const S c1 = S(0);
#else
        // With M = m1 I + B and A = (B + n eps) / sqrt(2E):
        //   (M + n eps)^2 / 2E = m1^2/2E I + 2 m1/sqrt(2E) A + A^2.
        // The first term is a common phase and is dropped, which avoids the
        // cancellation of m1^2 (up to ~30x the splittings) in the square.
        C A[3][3];
        OPG_UNROLL
        for (int i = 0; i < 3; i++)
          OPG_UNROLL
        for (int j = i; j < 3; j++) {
          A[i][j] = (P.B(i, j) + nsiCoup * P.eps[i][j]) / sqrtlv;
          if (nubar) A[i][j] = conj(A[i][j]);
        }
        const S c1 = S(2) * P.m1 / sqrtlv;
#endif
        // Square (HermitianSquare) and add c1 A
        C T[3][3];
        OPG_UNROLL
        for (int i = 0; i < 3; i++)
          OPG_UNROLL
        for (int j = i; j < 3; j++) {
          T[i][j] = A[i][j];
          if (i < j) T[j][i] = conj(A[i][j]);
        }
        OPG_UNROLL
        for (int i = 0; i < 3; i++)
          OPG_UNROLL
        for (int j = i; j < 3; j++) {
          C acc(S(0), S(0));
          OPG_UNROLL
          for (int k = 0; k < 3; k++) acc += T[i][k] * T[k][j];
#ifndef OPG_OSCPROB_BITWISE
          acc += c1 * A[i][j];
#endif
          H(i, j) = acc;
        }
        (void)c1;

        // Add matter potential in eV (kK2 * sqrt2 * Gf * rho * zoa)
        S kr2GNe = P.common.vfac;
        kr2GNe *= s.density;
        kr2GNe *= s.zoa;
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
        hermitian3_step<SNSI, R>(P, E, nubar, s, S);
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
        hermitian3_step_grad<SNSI, R, K>(P, PD, E, nubar, s, S, dS);
      }

      template <int K>
      OPG_HD OPG_INLINE static void finalize_grad(const Prepared&,
                                                  const PreparedT<Dual<R, K>>&,
                                                  bool, Mat<3, R>&,
                                                  Mat<3, R> (&)[K])
      {
      }

      OPG_HERMITIAN3_ADJOINT(SNSI, false)
  };

} // namespace opg

#endif
