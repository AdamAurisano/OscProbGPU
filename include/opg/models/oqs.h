///////////////////////////////////////////////////////////////////////////////
/// \file oqs.h
///
/// \brief 3-flavour oscillations in matter as an open quantum system
///        (OscProb::PMNS_OQS).
///
/// The density matrix of each initial flavour is kept in the vacuum mass
/// basis as a real 9-vector R in the Gell-Mann (SU(3)) representation. On a
/// segment of length L
///   R_{1..8} <- exp((H_GM + D E^n) L) R_{1..8},
/// with H_GM the 8x8 operator of the Hamiltonian in the vacuum mass basis
/// (vacuum terms dm/2E plus the matter potential) and D the dissipator built
/// from the decoherence vectors |a_i| (i = 1..8) and the cosines of the
/// angles between them. The real 8x8 exponential uses the scaling and
/// squaring Pade algorithm of Eigen's MatrixExponential (opg::expm).
/// P(a -> b) = |rho_a(b, b)| after rotating back to the flavour basis.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_OQS_H
#define OPG_MODELS_OQS_H

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "opg/linalg/expm.h"
#include "opg/models/hermitian3.h"

namespace opg {

  /// OQS parameters with scalar type S (PMNS_OQS conventions; |a_i| in GeV).
  template <class S> struct OQSParamsT {
      MixingParamsT<3, S> mix;
      S                   a[9]     = {};    ///< |a_i|, i = 1..8 (a[0] unused)
      S                   ang[9][9] = {};   ///< angle between a_i and a_j (i < j)
      S                   power    = S(0);  ///< D ~ (E/GeV)^power

      /// As PMNS_OQS::SetDecoElement(i, val), i = 1..8 (|val| is used).
      void SetDecoElement(int i, double val)
      {
        if (i < 1 || i > 8) throw std::invalid_argument("OQS::SetDecoElement: i in [1, 8]");
        a[i] = S(std::fabs(val));
      }
      /// As PMNS_OQS::SetDecoAngle(i, j, th), radians.
      void SetDecoAngle(int i, int j, double th)
      {
        if (i < 1 || i > 8 || j < 1 || j > 8 || i == j)
          throw std::invalid_argument("OQS::SetDecoAngle: i != j in [1, 8]");
        if (i > j) std::swap(i, j);
        ang[i][j] = S(th);
      }
      /// As PMNS_OQS::SetPower(n).
      void SetPower(double n) { power = S(n); }
  };
  using OQSParams = OQSParamsT<double>;

  /// Prepared state: vacuum eigensystem, matter prefactor and the 8x8
  /// dissipator (in eV, OscProb's -kGeV2eV factor included).
  template <class S> struct OQSPrepared {
      Hermitian3Common<S> common;
      S                   D[8][8];
      S                   power;
  };

  /// Gell-Mann vectors of the three initial flavours (vacuum mass basis);
  /// after finalize, R[a][b] (b < 3) holds P(a -> b).
  template <class R> struct OQSState {
      R R9[3][9];
  };

  template <class R = double> struct OQS {
      using Real                        = R;
      static constexpr int         N    = 3;
      static constexpr const char* name = "OQS";

#ifdef OPG_OQS_GRAD_CHUNK
      static constexpr int grad_chunk = OPG_OQS_GRAD_CHUNK;
#else
      static constexpr int grad_chunk = 1;
#endif
      /// Probabilities in gradient mode from the probability-only GPU
      /// kernels (opg::separate_probs).
      static constexpr bool separate_probs = true;

      template <class S> using ParamsT   = OQSParamsT<S>;
      using Params                       = OQSParams;
      template <class S> using PreparedT = OQSPrepared<S>;
      using Prepared                     = OQSPrepared<R>;
      using State                        = OQSState<R>;

      /// mixing, a1..a8, ang<i><j> (1 <= i < j <= 8), power
      static std::vector<std::string> param_names()
      {
        auto n = MixingRegistry<3>::names();
        for (int i = 1; i <= 8; i++) n.push_back("a" + std::to_string(i));
        for (int i = 1; i <= 8; i++)
          for (int j = i + 1; j <= 8; j++)
            n.push_back("ang" + std::to_string(i) + std::to_string(j));
        n.push_back("power");
        return n;
      }
      /// Default gradient selection: mixing and a1..a8 (the angles and the
      /// power are opt-in).
      static std::vector<std::string> default_param_names()
      {
        auto n = param_names();
        n.resize(MixingRegistry<3>::count() + 8);
        return n;
      }
      template <class S> static S& param_ref(ParamsT<S>& p, int idx)
      {
        const int nmix = MixingRegistry<3>::count();
        if (idx < nmix) return MixingRegistry<3>::ref(p.mix, idx);
        idx -= nmix;
        if (idx < 8) return p.a[idx + 1];
        idx -= 8;
        for (int i = 1; i <= 8; i++)
          for (int j = i + 1; j <= 8; j++)
            if (idx-- == 0) return p.ang[i][j];
        if (idx == 0) return p.power;
        throw std::out_of_range("OQS: bad parameter index");
      }
      template <class S> static ParamsT<S> cast(const Params& p)
      {
        ParamsT<S> q;
        q.mix = cast_mixing<S>(p.mix);
        for (int i = 0; i < 9; i++) {
          q.a[i] = S(p.a[i]);
          for (int j = 0; j < 9; j++) q.ang[i][j] = S(p.ang[i][j]);
        }
        q.power = S(p.power);
        return q;
      }

      /// Port of PMNS_OQS::BuildDissipator (generic scalar type).
      template <class S> static void dissipator(const ParamsT<S>& p, S (&D8)[8][8])
      {
        using std::cos;
        const double r3 = std::sqrt(3.0);
        S            aa[9][9];
        for (int i = 1; i < 9; i++)
          for (int j = i; j < 9; j++) {
            aa[i][j] = p.a[i] * p.a[j];
            if (i == 8) aa[i][j] *= r3;
            if (j == 8) aa[i][j] *= r3;
            if (i < j) aa[i][j] *= cos(p.ang[i][j]);
          }
        const S sum12   = aa[1][1] + aa[2][2];
        const S sum45   = aa[4][4] + aa[5][5];
        const S sum67   = aa[6][6] + aa[7][7];
        const S gamma21 = aa[3][3];
        const S gamma31 = (aa[3][3] + aa[8][8] + 2.0 * aa[3][8]) / 4.0;
        const S gamma32 = (aa[3][3] + aa[8][8] - 2.0 * aa[3][8]) / 4.0;

        S D[9][9];
        D[1][1] = gamma21 + aa[2][2] + (sum45 + sum67) / 4.0;
        D[2][2] = gamma21 + aa[1][1] + (sum45 + sum67) / 4.0;
        D[3][3] = sum12 + (sum45 + sum67) / 4.0;

        D[4][4] = gamma31 + aa[5][5] + (sum12 + sum67) / 4.0;
        D[5][5] = gamma31 + aa[4][4] + (sum12 + sum67) / 4.0;
        D[6][6] = gamma32 + aa[7][7] + (sum12 + sum45) / 4.0;
        D[7][7] = gamma32 + aa[6][6] + (sum12 + sum45) / 4.0;

        D[8][8] = (sum45 + sum67) * 3.0 / 4.0;
        D[3][8] = (sum45 - sum67) * r3 / 4.0;

        D[1][2] = -aa[1][2];
        D[1][3] = -aa[1][3];
        D[2][3] = -aa[2][3];
        D[4][5] = -aa[4][5];
        D[6][7] = -aa[6][7];

        D[1][8] = (aa[4][6] + aa[5][7]) * r3 / 2.0;
        D[2][8] = (aa[5][6] - aa[4][7]) * r3 / 2.0;

        D[4][6] = -(aa[4][6] - 2.0 * aa[1][8] - 3.0 * aa[5][7]) / 4.0;
        D[4][7] = -(aa[4][7] + 2.0 * aa[2][8] + 3.0 * aa[5][6]) / 4.0;
        D[5][6] = -(aa[5][6] - 2.0 * aa[2][8] + 3.0 * aa[4][7]) / 4.0;
        D[5][7] = -(aa[5][7] - 2.0 * aa[1][8] - 3.0 * aa[4][6]) / 4.0;

        D[1][4] = -(aa[1][4] - 3.0 * (aa[2][5] - aa[3][6]) + aa[6][8]) / 4.0;
        D[1][5] = -(aa[1][5] + 3.0 * (aa[2][4] + aa[3][7]) + aa[7][8]) / 4.0;
        D[1][6] = -(aa[1][6] + 3.0 * (aa[2][7] - aa[3][4]) + aa[4][8]) / 4.0;
        D[1][7] = -(aa[1][7] - 3.0 * (aa[2][6] + aa[3][5]) + aa[5][8]) / 4.0;
        D[2][4] = -(aa[2][4] + 3.0 * (aa[1][5] - aa[3][7]) - aa[7][8]) / 4.0;
        D[2][5] = -(aa[2][5] - 3.0 * (aa[1][4] - aa[3][6]) + aa[6][8]) / 4.0;
        D[2][6] = -(aa[2][6] - 3.0 * (aa[1][7] + aa[3][5]) + aa[5][8]) / 4.0;
        D[2][7] = -(aa[2][7] + 3.0 * (aa[1][6] + aa[3][4]) - aa[4][8]) / 4.0;
        D[3][4] = -(aa[3][4] - 3.0 * (aa[1][6] - aa[2][7]) + aa[4][8]) / 4.0;
        D[3][5] = -(aa[3][5] - 3.0 * (aa[1][7] + aa[2][6]) + aa[5][8]) / 4.0;
        D[3][6] = -(aa[3][6] + 3.0 * (aa[1][4] + aa[2][5]) - aa[6][8]) / 4.0;
        D[3][7] = -(aa[3][7] + 3.0 * (aa[1][5] - aa[2][4]) - aa[7][8]) / 4.0;

        D[4][8] = -(aa[1][6] - aa[2][7] + aa[3][4] + aa[4][8]) * r3 / 4.0;
        D[5][8] = -(aa[1][7] + aa[2][6] + aa[3][5] + aa[5][8]) * r3 / 4.0;
        D[6][8] = -(aa[1][4] + aa[2][5] - aa[3][6] + aa[6][8]) * r3 / 4.0;
        D[7][8] = -(aa[1][5] - aa[2][4] - aa[3][7] + aa[7][8]) * r3 / 4.0;

        for (int j = 1; j < 9; j++)
          for (int k = j; k < 9; k++) {
            D8[j - 1][k - 1] = -D[j][k] * constants::kGeV2eV;
            D8[k - 1][j - 1] = D8[j - 1][k - 1];
          }
      }

      template <class S> static PreparedT<S> prepare_generic(const ParamsT<S>& p)
      {
        PreparedT<S> out;
        prepare_hermitian3_generic<S>(p.mix, out.common);
        dissipator<S>(p, out.D);
        out.power = p.power;
        return out;
      }

      static Prepared prepare(const Params& p) { return prepare_generic<R>(cast<R>(p)); }

      /// E^n for a generic n (dual: d/dn E^n = E^n ln E).
      template <class S> OPG_HD OPG_INLINE static S energy_pow(R E, const S& n)
      {
        using std::log;
        using std::pow;
        if constexpr (dual_size<S>::value > 0) {
          S       out(pow(E, n.v));
          const R f = out.v * log(E);
          for (int k = 0; k < dual_size<S>::value; k++) out.d[k] = f * n.d[k];
          return out;
        }
        else
          return pow(E, n);
      }

      /// Port of PMNS_OQS::BuildM times the segment length: the exponent
      /// (H_GM + D E^n) L, for any scalar type S and segment type Seg.
      template <class S, class Seg>
      OPG_HD OPG_INLINE static RMat<8, S> exponent(const PreparedT<S>& P, R E, bool nubar,
                                                   const Seg& s)
      {
        using C = Complex<S>;
        // BuildHVMB: conj(UM[0][i]) Ve UM[0][j] with UM = conj(Uvac)
        S kr2GNe = P.common.vfac;
        kr2GNe *= s.density * s.zoa;
        const S    Ve = nubar ? -kr2GNe : kr2GNe;
        const auto& U = P.common.Uvac[nubar ? 1 : 0];
        C           H[3][3];
        OPG_UNROLL
        for (int i = 0; i < 3; i++)
          OPG_UNROLL
        for (int j = i; j < 3; j++) {
          const C ui = U(0, i);
          H[i][j]    = C(ui.re * Ve, ui.im * Ve) * conj(U(0, j));
        }
        const R lv = 2 * R(constants::kGeV2eV) * E;
        H[1][1] += C(P.common.dm[1] / lv, S(0));
        H[2][2] += C(P.common.dm[2] / lv, S(0));

        // get_GMOP (upper triangle, 1-based Gell-Mann indices)
        using std::sqrt;
        const double r3 = sqrt(3.0);
        S            B[9][9];
        OPG_UNROLL
        for (int i = 0; i < 9; i++)
          OPG_UNROLL
        for (int j = 0; j < 9; j++) B[i][j] = S(0);
        B[1][2] = H[0][0].re - H[1][1].re;
        B[1][3] = 2.0 * H[0][1].im;
        B[1][4] = -H[1][2].im;
        B[1][5] = -H[1][2].re;
        B[1][6] = -H[0][2].im;
        B[1][7] = -H[0][2].re;

        B[2][3] = 2.0 * H[0][1].re;
        B[2][4] = H[1][2].re;
        B[2][5] = -H[1][2].im;
        B[2][6] = -H[0][2].re;
        B[2][7] = H[0][2].im;

        B[3][4] = -H[0][2].im;
        B[3][5] = -H[0][2].re;
        B[3][6] = H[1][2].im;
        B[3][7] = H[1][2].re;

        B[4][5] = H[0][0].re - H[2][2].re;
        B[4][6] = -H[0][1].im;
        B[4][7] = H[0][1].re;
        B[4][8] = r3 * H[0][2].im;

        B[5][6] = -H[0][1].re;
        B[5][7] = -H[0][1].im;
        B[5][8] = r3 * H[0][2].re;

        B[6][7] = H[1][1].re - H[2][2].re;
        B[6][8] = r3 * H[1][2].im;

        B[7][8] = r3 * H[1][2].re;
        OPG_UNROLL
        for (int i = 1; i < 9; i++)
          OPG_UNROLL
        for (int j = i + 1; j < 9; j++) B[j][i] = -B[i][j];

        // M = (H_GM + D E^n) L
        const S    Ec = energy_pow<S>(E, P.power);
        const R    L  = length_in_eV(s.length);
        RMat<8, S> M;
        OPG_UNROLL
        for (int k = 0; k < 8; k++)
          OPG_UNROLL
        for (int j = 0; j < 8; j++) {
          M(k, j) = B[k + 1][j + 1] + P.D[k][j] * Ec;
          M(k, j) *= L;
        }
        return M;
      }

      /// get_GM of U^dag |a><a| U in the mass basis (PMNS_OQS::RotateState(true)
      /// after ResetToFlavour(a)), U = Uvac.
      template <class S>
      OPG_HD OPG_INLINE static void initial_gm(const Mat<3, S>& U, int a, S (&v)[9])
      {
        using C = Complex<S>;
        C rho[3][3];  // upper triangle: Uvac(a, i) conj(Uvac(a, j))
        OPG_UNROLL
        for (int i = 0; i < 3; i++)
          OPG_UNROLL
        for (int j = i; j < 3; j++) rho[i][j] = U(a, i) * conj(U(a, j));
        using std::sqrt;
        const double s6 = sqrt(6.0), s12 = sqrt(12.0);
        v[0] = (rho[0][0].re + rho[1][1].re + rho[2][2].re) / s6;
        v[3] = (rho[0][0].re - rho[1][1].re) / 2.0;
        v[8] = (rho[0][0].re + rho[1][1].re - 2.0 * rho[2][2].re) / s12;
        v[1] = rho[0][1].re;
        v[4] = rho[0][2].re;
        v[6] = rho[1][2].re;
        v[2] = -rho[0][1].im;
        v[5] = -rho[0][2].im;
        v[7] = -rho[1][2].im;
      }

      /// rho(b, b) in the flavour basis from a Gell-Mann vector v (get_SU3,
      /// then PMNS_OQS::RotateState(false)), U = Uvac.
      template <class S>
      OPG_HD OPG_INLINE static Complex<S> flavour_diag(const Mat<3, S>& U, const S (&v)[9],
                                                      int b)
      {
        using C           = Complex<S>;
        using std::sqrt;
        const double s23  = sqrt(2 / 3.);
        const double r3   = sqrt(3.0);
        C            A[3][3];
        A[0][0] = C(v[0] * s23 + v[3] + v[8] / r3, S(0));
        A[1][1] = C(v[0] * s23 - v[3] + v[8] / r3, S(0));
        A[2][2] = C(v[0] * s23 - 2.0 * v[8] / r3, S(0));
        A[1][0] = C(v[1], v[2]);
        A[2][0] = C(v[4], v[5]);
        A[2][1] = C(v[6], v[7]);
        A[0][1] = conj(A[1][0]);
        A[0][2] = conj(A[2][0]);
        A[1][2] = conj(A[2][1]);
        // buffer(k, b) = sum_l A(k, l) conj(UM(b, l)) = sum_l A(k, l) U(b, l)
        C buf[3];
        OPG_UNROLL
        for (int k = 0; k < 3; k++) {
          C s(S(0), S(0));
          OPG_UNROLL
          for (int l = 0; l < 3; l++) s += A[k][l] * U(b, l);
          buf[k] = s;
        }
        // rho(b, b) = sum_k UM(b, k) buffer(k, b), UM = conj(U)
        C r(S(0), S(0));
        OPG_UNROLL
        for (int k = 0; k < 3; k++) r += conj(U(b, k)) * buf[k];
        return r;
      }

      OPG_HD OPG_INLINE static State initial(const Prepared& P, bool nubar)
      {
        State st;
        OPG_UNROLL
        for (int a = 0; a < 3; a++) initial_gm<R>(P.common.Uvac[nubar ? 1 : 0], a, st.R9[a]);
        return st;
      }

      /// R_{1..8} <- X R_{1..8} (PMNS_OQS::PropagatePath).
      OPG_HD OPG_INLINE static void apply(const RMat<8, R>& X, R (&v)[9])
      {
        R t[8];
        OPG_UNROLL
        for (int i = 0; i < 8; i++) {
          R s = 0;
          OPG_UNROLL
          for (int j = 0; j < 8; j++) s += X(i, j) * v[j + 1];
          t[i] = s;
        }
        OPG_UNROLL
        for (int i = 0; i < 8; i++) v[i + 1] = t[i];
      }

      OPG_HD OPG_INLINE static void step(const Prepared& P, R E, bool nubar,
                                         const Segment<R>& s, State& st)
      {
        const RMat<8, R> X = expm(exponent(P, E, nubar, s));
        OPG_UNROLL
        for (int a = 0; a < 3; a++) apply(X, st.R9[a]);
      }

      /// P(a -> b) = |rho_a(b, b)|, stored in R9[a][b].
      OPG_HD OPG_INLINE static void finalize(const Prepared& P, bool nubar, State& st)
      {
        const Mat<3, R>& U = P.common.Uvac[nubar ? 1 : 0];
        OPG_UNROLL
        for (int a = 0; a < 3; a++) {
          R p[3];
          OPG_UNROLL
          for (int b = 0; b < 3; b++) p[b] = abs(flavour_diag<R>(U, st.R9[a], b));
          OPG_UNROLL
          for (int b = 0; b < 3; b++) st.R9[a][b] = p[b];
        }
      }

      template <class Rr>
      OPG_HD OPG_INLINE static void store_probs(const State& st, Rr* out, size_t stride)
      {
        OPG_UNROLL
        for (int a = 0; a < 3; a++)
          OPG_UNROLL
        for (int b = 0; b < 3; b++) out[(a * 3 + b) * stride] = st.R9[a][b];
      }

      // --- gradients ---------------------------------------------------------
      /// Derivatives of the initial Gell-Mann vectors (they depend on the
      /// mixing through Uvac).
      template <int K>
      OPG_HD OPG_INLINE static void initial_grad(const PreparedT<Dual<R, K>>& PD, bool nubar,
                                                 State (&dS)[K])
      {
        using D = Dual<R, K>;
        OPG_UNROLL
        for (int a = 0; a < 3; a++) {
          D v[9];
          initial_gm<D>(PD.common.Uvac[nubar ? 1 : 0], a, v);
          OPG_UNROLL
          for (int k = 0; k < K; k++)
            OPG_UNROLL
          for (int i = 0; i < 9; i++) dS[k].R9[a][i] = v[i].d[k];
        }
      }

      /// Segment step with derivatives: X = exp(M) in dual arithmetic (the
      /// derivative parts are the exact derivatives of the Pade
      /// approximant), dR' = dX R + X dR. Values: as step().
      template <int K>
      OPG_HD OPG_INLINE static void step_grad(const Prepared&, const PreparedT<Dual<R, K>>& PD,
                                              R E, bool nubar,
                                              const SegmentZ<R, Dual<R, K>>& sz, State& S,
                                              State (&dS)[K])
      {
        using D                 = Dual<R, K>;
        const RMat<8, D> XD     = expm(exponent(PD, E, nubar, sz));
        RMat<8, R>       X;
        OPG_UNROLL
        for (int i = 0; i < 8; i++)
          OPG_UNROLL
        for (int j = 0; j < 8; j++) X(i, j) = XD(i, j).v;
        OPG_UNROLL
        for (int a = 0; a < 3; a++) {
          OPG_UNROLL
          for (int k = 0; k < K; k++) {
            R t[8];
            OPG_UNROLL
            for (int i = 0; i < 8; i++) {
              R s = 0;
              OPG_UNROLL
              for (int j = 0; j < 8; j++)
                s += XD(i, j).d[k] * S.R9[a][j + 1] + X(i, j) * dS[k].R9[a][j + 1];
              t[i] = s;
            }
            OPG_UNROLL
            for (int i = 0; i < 8; i++) dS[k].R9[a][i + 1] = t[i];
          }
          apply(X, S.R9[a]);
        }
      }

      /// dP(a -> b) = d|rho_a(b, b)|, stored in dS[k].R9[a][b].
      template <int K>
      OPG_HD OPG_INLINE static void finalize_grad(const Prepared&,
                                                  const PreparedT<Dual<R, K>>& PD, bool nubar,
                                                  State& S, State (&dS)[K])
      {
        using D              = Dual<R, K>;
        const Mat<3, D>& U   = PD.common.Uvac[nubar ? 1 : 0];
        OPG_UNROLL
        for (int a = 0; a < 3; a++) {
          D v[9];
          OPG_UNROLL
          for (int i = 0; i < 9; i++) {
            v[i].v = S.R9[a][i];
            OPG_UNROLL
            for (int k = 0; k < K; k++) v[i].d[k] = dS[k].R9[a][i];
          }
          R dp[3][K];
          OPG_UNROLL
          for (int b = 0; b < 3; b++) {
            const Complex<D> r = flavour_diag<D>(U, v, b);
            const R          m = std::hypot(r.re.v, r.im.v);
            OPG_UNROLL
            for (int k = 0; k < K; k++)
              dp[b][k] = m > R(0) ? (r.re.v * r.re.d[k] + r.im.v * r.im.d[k]) / m : R(0);
          }
          OPG_UNROLL
          for (int b = 0; b < 3; b++)
            OPG_UNROLL
          for (int k = 0; k < K; k++) dS[k].R9[a][b] = dp[b][k];
        }
      }

      template <int K>
      OPG_HD OPG_INLINE static void store_grads(const State&, const State (&dS)[K], int count,
                                                R* out, size_t stride)
      {
        OPG_UNROLL
        for (int k = 0; k < K; k++) {
          if (k >= count) break;
          OPG_UNROLL
          for (int a = 0; a < 3; a++)
            OPG_UNROLL
          for (int b = 0; b < 3; b++)
            out[((size_t(k) * 3 + a) * 3 + b) * stride] = dS[k].R9[a][b];
        }
      }

      template <int K>
      OPG_HD OPG_INLINE static void contract_grads(const State&, const State (&dS)[K],
                                                   const R* w, size_t stride, R (&acc)[K])
      {
        OPG_UNROLL
        for (int k = 0; k < K; k++) acc[k] = 0;
        OPG_UNROLL
        for (int a = 0; a < 3; a++)
          OPG_UNROLL
        for (int b = 0; b < 3; b++) {
          const R wab = w[(a * 3 + b) * stride];
          if (wab == R(0)) continue;
          OPG_UNROLL
          for (int k = 0; k < K; k++) acc[k] += wab * dS[k].R9[a][b];
        }
      }
  };

} // namespace opg

#endif
