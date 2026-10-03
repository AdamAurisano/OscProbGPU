///////////////////////////////////////////////////////////////////////////////
/// \file deco.h
///
/// \brief 3-flavour oscillations in matter with decoherence
///        (OscProb::PMNS_Deco).
///
/// Density matrices rho_a (one per initial flavour a) are propagated. On each
/// segment rho is rotated to the effective mass basis of PMNS_Fast's
/// Hamiltonian, its off-diagonal elements are multiplied by
///   exp(-Gamma_ij E^n L) exp(-i (lam_i - lam_j) L),
/// and it is rotated back. The Gamma_ij of the mass states are matched to the
/// eigenvalues by rank (as OscProb). P(a -> b) = |rho_a(b, b)|.
/// Gamma_32 follows from Gamma_21, Gamma_31 and the decoherence angle.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_DECO_H
#define OPG_MODELS_DECO_H

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "opg/models/fast.h"

namespace opg {

  /// Deco parameters with scalar type S (PMNS_Deco conventions; Gamma in
  /// GeV). dtheta is a tangent parameter of the decoherence angle (value 0):
  /// cos(theta) enters as gamma[0] - sin(theta) dtheta, so that derivatives
  /// are with respect to theta while gamma[0] keeps OscProb's value.
  template <class S> struct DecoParamsT {
      MixingParamsT<3, S> mix;
      S                   gamma[3] = {S(1), S(0), S(0)};  ///< {cos(theta), G21, G31}
      S                   power    = S(0);  ///< Gamma ~ (E/GeV)^power
      S                   dtheta   = S(0);

      /// As PMNS_Deco::SetGamma(j, val), j = 2 or 3 (|val| is used).
      void SetGamma(int j, double val)
      {
        if (j != 2 && j != 3) throw std::invalid_argument("Deco::SetGamma: j must be 2 or 3");
        gamma[j - 1] = S(std::fabs(val));
      }
      /// As PMNS_Deco::SetDecoAngle(th).
      void SetDecoAngle(double th) { gamma[0] = S(std::cos(th)); }
      /// As PMNS_Deco::SetPower(n).
      void SetPower(double n) { power = S(n); }
      /// As PMNS_Deco::SetGamma32(g): sets Gamma_31 (and, if g is not
      /// reachable with the current Gamma_21 and angle, cos(theta)).
      void SetGamma32(double gamma32)
      {
        double g[3] = {value_of(gamma[0]), value_of(gamma[1]), value_of(gamma[2])};
        gamma32     = std::fabs(gamma32);
        double min32 = 0.25 * g[1];
        if (g[0] >= 0)
          min32 *= 1 - std::pow(g[0], 2);
        else
          min32 *= 1 + 3 * std::pow(g[0], 2);
        if (gamma32 < min32) {
          if (g[0] >= 0 || 4 * gamma32 / g[1] < 1) {
            g[0] = std::sqrt(1 - 4 * gamma32 / g[1]);
            g[2] = 0.25 * g[1] * (1 + 3 * std::pow(g[0], 2));
          }
          else {
            g[0] = -std::sqrt((4 * gamma32 / g[1] - 1) / 3);
            g[2] = 0.25 * g[1] * (1 - std::pow(g[0], 2));
          }
        }
        else {
          double arg = g[1] * (4 * gamma32 - g[1] * (1 - std::pow(g[0], 2)));
          if (arg < 0) {
            arg  = 0;
            g[0] = std::sqrt(1 - 4 * gamma32 / g[1]);
          }
          g[2] = gamma32 + g[0] * (g[0] * g[1] + std::sqrt(arg));
        }
        for (int k = 0; k < 3; k++) gamma[k] = S(g[k]);
      }
      /// As PMNS_Deco::GetGamma(i, j), mass indices 1..3 (values).
      double GetGamma(int i, int j) const
      {
        if (i < j) std::swap(i, j);
        return value_of(gamma_ij(i, j));
      }
      /// Gamma_ij for i > j (generic scalar; Gamma_32 from the others).
      S gamma_ij(int i, int j) const
      {
        using std::sin;
        using std::sqrt;
        if (j == 1) return gamma[i - 1];
        // cos(theta), with the tangent parameter (exact value: gamma[0])
        S c = gamma[0];
        if constexpr (dual_size<S>::value > 0) {
          const double sn = std::sqrt(std::fmax(0.0, 1 - c.v * c.v));
          for (int k = 0; k < dual_size<S>::value; k++) c.d[k] -= sn * dtheta.d[k];
        }
        else if (dtheta != S(0)) {
          using std::fabs;
          c -= sqrt(fabs(S(1) - c * c)) * dtheta;
        }
        const S arg = gamma[1] * (S(4) * gamma[2] - gamma[1] * (S(1) - c * c));
        if (arg < S(0)) return gamma[1] - S(3) * gamma[2];
        // Where the argument vanishes (e.g. Gamma_21 = 0) sqrt(arg) is not
        // differentiable; its derivative is taken as 0 there, which keeps all
        // gradients finite (and is exact at Gamma = 0 for cos(theta) = +-1).
        S r = S(0);
        if (arg > S(0)) r = sqrt(arg);
        return gamma[2] + gamma[1] * (c * c) - c * r;
      }
  };
  using DecoParams = DecoParamsT<double>;

  /// Deco prepared state: Fast's, plus the Gamma table of the mass states.
  template <class R> struct DecoPrepared : FastPrepared<R> {
      R gam[3][3];  ///< Gamma_ij (0-based mass indices, symmetric), GeV
      R power;
  };

  /// Density matrices for the three initial flavours.
  template <class R> struct DecoState {
      Mat<3, R> rho[3];
  };

  template <class R = double> struct Deco {
      using Real                        = R;
      static constexpr int         N    = 3;
      static constexpr const char* name = "Deco";

      /// Derivative directions per gradient pass (see opg::grad_traits).
#ifdef OPG_DECO_GRAD_CHUNK
      static constexpr int grad_chunk = OPG_DECO_GRAD_CHUNK;
#else
      static constexpr int grad_chunk = 2;
#endif

      template <class S> using ParamsT   = DecoParamsT<S>;
      using Params                       = DecoParams;
      template <class S> using PreparedT = DecoPrepared<S>;
      using Prepared                     = DecoPrepared<R>;
      using State                        = DecoState<R>;

      /// mixing (MixingRegistry<3>), gamma21, gamma31, deco_angle, deco_power
      static std::vector<std::string> param_names()
      {
        auto n = MixingRegistry<3>::names();
        for (const char* x : {"gamma21", "gamma31", "deco_angle", "deco_power"}) n.push_back(x);
        return n;
      }
      /// Default gradient selection: all but deco_power (a model choice).
      static std::vector<std::string> default_param_names()
      {
        auto n = param_names();
        n.pop_back();
        return n;
      }
      template <class S> static S& param_ref(ParamsT<S>& p, int idx)
      {
        const int nmix = MixingRegistry<3>::count();
        if (idx < nmix) return MixingRegistry<3>::ref(p.mix, idx);
        switch (idx - nmix) {
          case 0: return p.gamma[1];
          case 1: return p.gamma[2];
          case 2: return p.dtheta;
          case 3: return p.power;
        }
        throw std::out_of_range("Deco: bad parameter index");
      }
      template <class S> static ParamsT<S> cast(const Params& p)
      {
        ParamsT<S> q;
        q.mix = cast_mixing<S>(p.mix);
        for (int k = 0; k < 3; k++) q.gamma[k] = S(p.gamma[k]);
        q.power  = S(p.power);
        q.dtheta = S(p.dtheta);
        return q;
      }

      template <class S> static PreparedT<S> prepare_generic(const ParamsT<S>& p)
      {
        PreparedT<S> out;
        static_cast<FastPrepared<S>&>(out) =
            Fast<R>::template prepare_generic<S>(FastParams<S>{p.mix});
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
            out.gam[i][j] =
                i == j ? S(0) : p.gamma_ij(std::max(i, j) + 1, std::min(i, j) + 1);
        out.power = p.power;
        return out;
      }

      static Prepared prepare(const Params& p)
      {
        return prepare_generic<R>(cast<R>(p));
      }

      /// PMNS_Fast's Hamiltonian.
      template <class S, class Seg>
      OPG_HD OPG_INLINE static void hamiltonian(const FastPrepared<S>& P, R E, bool nubar,
                                                const Seg& s, Mat<3, S>& H)
      {
        Fast<R>::hamiltonian(P, E, nubar, s, H);
      }

      OPG_HD OPG_INLINE static State initial(const Prepared&, bool)
      {
        State st;
        OPG_UNROLL
        for (int a = 0; a < 3; a++) {
          st.rho[a]       = Mat<3, R>::zero();
          st.rho[a](a, a) = Complex<R>(1, 0);
        }
        return st;
      }

      /// Indices of x sorted in ascending order (OscProb's sort3).
      OPG_HD OPG_INLINE static void sort3(const R x[3], int out[3])
      {
        out[0] = out[1] = out[2] = 0;
        if (x[0] < x[1] && x[0] < x[2]) {
          if (x[1] < x[2]) {
            out[1] = 1;
            out[2] = 2;
          }
          else {
            out[1] = 2;
            out[2] = 1;
          }
        }
        else if (x[1] < x[2]) {
          out[0] = 1;
          if (x[0] < x[2])
            out[2] = 2;
          else
            out[1] = 2;
        }
        else {
          out[0] = 2;
          if (x[0] < x[1])
            out[2] = 1;
          else
            out[1] = 1;
        }
      }

      /// E^n for a generic n (dual: d/dn E^n = E^n ln E).
      template <class S> OPG_HD OPG_INLINE static S energy_pow(R E, const S& n)
      {
        using std::log;
        using std::pow;
        if constexpr (dual_size<S>::value > 0) {
          S out(pow(E, n.v));
          const R f = out.v * log(E);
          for (int k = 0; k < dual_size<S>::value; k++) out.d[k] = f * n.d[k];
          return out;
        }
        else
          return pow(E, n);
      }

      /// Port of PMNS_Deco::PropagatePath for density matrices rho[0..na),
      /// given the eigensystem (V, lam) of the segment's Hamiltonian and the
      /// rank matching idx. Generic scalar type S (real or dual).
      template <class S, int NA>
      OPG_HD OPG_INLINE static void deco_step(const Mat<3, S>& V, const S lam[3],
                                              const S (&gam)[3][3], const S& power,
                                              const int idx[3], R E, R lengthIneV,
                                              Mat<3, S> (&rho)[NA])
      {
        using C = Complex<S>;
        using std::exp;
        const S energyCorr = energy_pow<S>(E, power);

        C damp[3][3];
        OPG_UNROLL
        for (int j = 0; j < 3; j++)
          OPG_UNROLL
        for (int i = 0; i < j; i++) {
          S gamma_ij = gam[idx[i]][idx[j]] * energyCorr;
          gamma_ij *= S(R(constants::kGeV2eV) * lengthIneV);
          const S arg = (lam[i] - lam[j]) * lengthIneV;
          S       sn, cs;
          if constexpr (dual_size<S>::value > 0) {
            using std::cos;
            using std::sin;
            sn = sin(arg);
            cs = cos(arg);
          }
          else
            sin_cos(arg, sn, cs);
          const S e = exp(-gamma_ij);
          damp[i][j] = C(e * cs, e * -sn);
        }

        OPG_UNROLL
        for (int a = 0; a < NA; a++) {
          Mat<3, S>& r = rho[a];
          // r = V^dag r V (PMNS_DensityMatrix::RotateState(true))
          Mat<3, S> buf;
          OPG_UNROLL
          for (int i = 0; i < 3; i++)
            OPG_UNROLL
          for (int j = 0; j < 3; j++) {
            C acc(S(0), S(0));
            OPG_UNROLL
            for (int k = 0; k < 3; k++) acc += r(i, k) * V(k, j);
            buf(i, j) = acc;
          }
          OPG_UNROLL
          for (int i = 0; i < 3; i++)
            OPG_UNROLL
          for (int j = i; j < 3; j++) {
            C acc(S(0), S(0));
            OPG_UNROLL
            for (int k = 0; k < 3; k++) acc += conj(V(k, i)) * buf(k, j);
            r(i, j) = acc;
            if (j > i) r(j, i) = conj(acc);
          }
          // decoherence and phases
          OPG_UNROLL
          for (int j = 0; j < 3; j++)
            OPG_UNROLL
          for (int i = 0; i < j; i++) {
            r(i, j) *= damp[i][j];
            r(j, i) = conj(r(i, j));
          }
          // r = V r V^dag (RotateState(false))
          OPG_UNROLL
          for (int i = 0; i < 3; i++)
            OPG_UNROLL
          for (int j = 0; j < 3; j++) {
            C acc(S(0), S(0));
            OPG_UNROLL
            for (int k = 0; k < 3; k++) acc += r(i, k) * conj(V(j, k));
            buf(i, j) = acc;
          }
          OPG_UNROLL
          for (int i = 0; i < 3; i++)
            OPG_UNROLL
          for (int j = i; j < 3; j++) {
            C acc(S(0), S(0));
            OPG_UNROLL
            for (int k = 0; k < 3; k++) acc += V(i, k) * buf(k, j);
            r(i, j) = acc;
            if (j > i) r(j, i) = conj(acc);
          }
        }
      }

      /// Rank matching of the Gamma indices (eigenvalues vs splittings).
      OPG_HD OPG_INLINE static void match(const Prepared& P, const R lam[3], int idx[3])
      {
        const R dm[3] = {R(0), P.common.dm[1], P.common.dm[2]};
        int     dm_idx[3], ev_idx[3];
        sort3(dm, dm_idx);
        sort3(lam, ev_idx);
        OPG_UNROLL
        for (int i = 0; i < 3; i++) idx[ev_idx[i]] = dm_idx[i];
      }

      /// Port of PMNS_Deco::PropagatePath for the three density matrices.
      OPG_HD OPG_INLINE static void step(const Prepared& P, R E, bool nubar,
                                         const Segment<R>& s, State& st)
      {
        Mat<3, R> V;
        R         lam[3];
        hermitian3_eigen<Deco, R>(P, E, nubar, s, V, lam);
        int idx[3];
        match(P, lam, idx);
        deco_step<R, 3>(V, lam, P.gam, P.power, idx, E, length_in_eV(s.length), st.rho);
      }

      // --- gradients ---------------------------------------------------------
      template <int K>
      OPG_HD OPG_INLINE static void initial_grad(const PreparedT<Dual<R, K>>&, bool,
                                                 State (&dS)[K])
      {
        OPG_UNROLL
        for (int k = 0; k < K; k++)
          OPG_UNROLL
        for (int a = 0; a < 3; a++) dS[k].rho[a] = Mat<3, R>::zero();
      }

      /// The segment step in dual arithmetic. The eigenvalue and eigenvector
      /// derivatives follow from first-order perturbation theory of the dual
      /// Hamiltonian (non-degenerate eigenvalues; gauge v_i^dag dv_i = 0, the
      /// step is invariant under eigenvector phases). Values: as step().
      template <int K>
      OPG_HD OPG_INLINE static void step_grad(const Prepared&               P,
                                              const PreparedT<Dual<R, K>>& PD,
                                              R E, bool nubar,
                                              const SegmentZ<R, Dual<R, K>>& sz,
                                              State& S, State (&dS)[K])
      {
        using D  = Dual<R, K>;
        using C  = Complex<R>;
        using CD = Complex<D>;
        const Segment<R> s{sz.length, sz.density, sz.zoa.v, sz.layer};
        Mat<3, R>        V;
        R                lam[3];
        hermitian3_eigen<Deco, R>(P, E, nubar, s, V, lam);
        int idx[3];
        match(P, lam, idx);

        Mat<3, D>               HD;
        SegmentZ<R, Dual<R, K>> sd = sz;
        if (s.density < R(1.0e-6)) sd.density = 0;  // vacuum: H = Hms / 2E
        hamiltonian(PD, E, nubar, sd, HD);

        // eigensystem with derivatives
        Mat<3, D> VD;
        D         lamD[3];
        OPG_UNROLL
        for (int i = 0; i < 3; i++) {
          lamD[i] = D(lam[i]);
          OPG_UNROLL
          for (int j = 0; j < 3; j++) VD(i, j) = CD(D(V(i, j).re), D(V(i, j).im));
        }
        OPG_UNROLL
        for (int k = 0; k < K; k++) {
          Mat<3, R> dH;  // hermitian, from the upper triangle
          OPG_UNROLL
          for (int i = 0; i < 3; i++)
            OPG_UNROLL
          for (int j = i; j < 3; j++) {
            dH(i, j) = C(HD(i, j).re.d[k], i == j ? R(0) : HD(i, j).im.d[k]);
            if (j > i) dH(j, i) = conj(dH(i, j));
          }
          const Mat<3, R> M = matmul(adjoint(V), matmul(dH, V));
          OPG_UNROLL
          for (int i = 0; i < 3; i++) lamD[i].d[k] = M(i, i).re;
          OPG_UNROLL
          for (int r = 0; r < 3; r++)
            OPG_UNROLL
          for (int i = 0; i < 3; i++) {
            C dv(0, 0);  // (dV)(r, i) = sum_l V(r, l) M(l, i) / (lam_i - lam_l)
            OPG_UNROLL
            for (int l = 0; l < 3; l++)
              if (l != i) dv += V(r, l) * M(l, i) / (lam[i] - lam[l]);
            VD(r, i).re.d[k] = dv.re;
            VD(r, i).im.d[k] = dv.im;
          }
        }

        // density matrices with derivatives
        Mat<3, D> rho[3];
        OPG_UNROLL
        for (int a = 0; a < 3; a++)
          OPG_UNROLL
        for (int i = 0; i < 3; i++)
          OPG_UNROLL
        for (int j = 0; j < 3; j++) {
          D re(S.rho[a](i, j).re), im(S.rho[a](i, j).im);
          OPG_UNROLL
          for (int k = 0; k < K; k++) {
            re.d[k] = dS[k].rho[a](i, j).re;
            im.d[k] = dS[k].rho[a](i, j).im;
          }
          rho[a](i, j) = CD(re, im);
        }
        deco_step<D, 3>(VD, lamD, PD.gam, PD.power, idx, E, length_in_eV(s.length), rho);
        OPG_UNROLL
        for (int k = 0; k < K; k++)
          OPG_UNROLL
        for (int a = 0; a < 3; a++)
          OPG_UNROLL
        for (int i = 0; i < 3; i++)
          OPG_UNROLL
        for (int j = 0; j < 3; j++)
          dS[k].rho[a](i, j) = C(rho[a](i, j).re.d[k], rho[a](i, j).im.d[k]);

        // values exactly as step()
        deco_step<R, 3>(V, lam, P.gam, P.power, idx, E, length_in_eV(s.length), S.rho);
      }

      template <int K>
      OPG_HD OPG_INLINE static void finalize_grad(const Prepared&,
                                                  const PreparedT<Dual<R, K>>&, bool,
                                                  State&, State (&)[K])
      {
      }

      /// dP(a -> b)/dp_k = d|rho_a(b, b)| for k < count, written to
      /// out[((k * N + a) * N + b) * stride].
      template <int K>
      OPG_HD OPG_INLINE static void store_grads(const State& S, const State (&dS)[K],
                                                int count, R* out, size_t stride)
      {
        OPG_UNROLL
        for (int k = 0; k < K; k++) {
          if (k >= count) break;
          OPG_UNROLL
          for (int a = 0; a < 3; a++)
            OPG_UNROLL
          for (int b = 0; b < 3; b++)
            out[((size_t(k) * 3 + a) * 3 + b) * stride] = dabs(S.rho[a](b, b), dS[k].rho[a](b, b));
        }
      }

      template <int K>
      OPG_HD OPG_INLINE static void contract_grads(const State& S, const State (&dS)[K],
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
          for (int k = 0; k < K; k++) acc[k] += wab * dabs(S.rho[a](b, b), dS[k].rho[a](b, b));
        }
      }

      /// d|z| = Re(conj(z) dz) / |z| (0 at z = 0).
      OPG_HD OPG_INLINE static R dabs(const Complex<R>& z, const Complex<R>& dz)
      {
        const R m = abs(z);
        return m > R(0) ? (z.re * dz.re + z.im * dz.im) / m : R(0);
      }

      OPG_HD OPG_INLINE static void finalize(const Prepared&, bool, State&) {}

      /// P(a -> b) = |rho_a(b, b)| (PMNS_DensityMatrix::P).
      template <class Rr>
      OPG_HD OPG_INLINE static void store_probs(const State& st, Rr* out, size_t stride)
      {
        OPG_UNROLL
        for (int a = 0; a < 3; a++)
          OPG_UNROLL
        for (int b = 0; b < 3; b++) out[(a * 3 + b) * stride] = abs(st.rho[a](b, b));
      }
  };

} // namespace opg

#endif
