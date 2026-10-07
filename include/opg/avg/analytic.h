///////////////////////////////////////////////////////////////////////////////
/// \file analytic.h
///
/// \brief Analytic bin averages over 1/E (L/E) for fixed paths, after
///        OscProb's PMNS_Maltoni (arXiv:2308.00037), with exact parameter
///        derivatives.
///
/// For H(u) = A u + B(rho), u = 1/E (models whose vacuum Hamiltonian is
/// proportional to 1/E and whose matter terms do not depend on E), the
/// evolution operator near u0 is, to first order in d = u - u0,
///
///   S(u0 + d) = S0 exp(-i K d),   K = sum_seg T^dag K_seg T,
///   K_seg = int_0^L exp(iHs) A exp(-iHs) ds,
///
/// with T the evolution before the segment. With K = W diag(mu) W^dag, the
/// average of P(a -> b) over d in [-h, h] with weight 1 + beta d / h is
///
///   Pbar_ab = sum_nm B_bn conj(B_bm) conj(W_an) W_am F(mu_n - mu_m),
///   B = S0 W,  F(x) = j0(xh) - i beta j1(xh),
///
/// which is exact in vacuum. Bins are split into sub-bins of geometric
/// widths (u_hi/u_lo = 1 + r, r fixed by the binning, the path and the
/// options, never by the oscillation parameters, so averages are smooth in
/// the parameters). The first-order error is ~ 0.004 r^2 sum(V L), with V
/// the matter potential at Z/A = 1 (NOvA FD: sum(V L) = 0.88), i.e. about
/// 0.7 tol with the default rule r = sqrt(tol / (0.005 sum(V L))).
///
/// Pairs with |mu_n - mu_m| h beyond `fast_begin` are faded out smoothly
/// (C2 in log|x|) up to `fast_end`, where they are dropped: their exact
/// contribution is below 1/fast_begin, and their phases can no longer be
/// resolved in double precision.
///
/// Derivatives (forward mode, the model's dual prepared state) use only
/// divided differences of smooth functions (Daleckii-Krein style), so they
/// stay finite and accurate at degenerate eigenvalues of H or K.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_AVG_ANALYTIC_H
#define OPG_AVG_ANALYTIC_H

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "opg/core/constants.h"
#include "opg/core/dual.h"
#include "opg/core/matrix.h"
#include "opg/earth/prem.h"
#include "opg/engine.h"
#include "opg/linalg/jacobi_herm.h"
#include "opg/physics/eigen_grad.h"
#include "opg/physics/grad.h"
#include "opg/physics/propagate.h"

namespace opg {

  /// Whether the analytic averages support a model: its hamiltonian() must
  /// be hermitian, affine in 1/E with E-independent matter terms, and the
  /// state the plain amplitude matrix (Model::analytic_avg = true).
  template <class Model, class = void> struct has_analytic_avg : std::false_type {};
  template <class Model>
  struct has_analytic_avg<Model, std::void_t<decltype(Model::analytic_avg)>>
      : std::bool_constant<Model::analytic_avg> {};

  /// Whether dH/dp = u dA/dp for every model parameter p
  /// (Model::analytic_static_matter); not for the Earth parameters.
  template <class Model, class = void> struct has_static_matter : std::false_type {};
  template <class Model>
  struct has_static_matter<Model, std::void_t<decltype(Model::analytic_static_matter)>>
      : std::bool_constant<Model::analytic_static_matter> {};

  namespace analytic {

    /// 1 / (k! (n + k + 1)): coefficient of (iz)^k in the series of J_n.
    template <class R> constexpr R jn_coef(int n, int k)
    {
      R f = 1;
      for (int i = 2; i <= k; i++) f *= R(i);
      return R(1) / (f * R(n + k + 1));
    }
    template <class R, int NMAX, int K> struct JnCoef {
        static constexpr R value = jn_coef<R>(NMAX, K);
    };
    /// sum_{j >= 0} (-z2)^j c_{K + 2j}, c_k = jn_coef(NMAX, k), k <= 37
    /// (Horner, constant coefficients).
    template <class R, int NMAX, int K> OPG_HD OPG_INLINE R jn_series(R z2)
    {
      if constexpr (K > 37) return R(0);
      else return JnCoef<R, NMAX, K>::value - z2 * jn_series<R, NMAX, K + 2>(z2);
    }
    /// 1 / n as a constant.
    template <class R, int n> struct Inv {
        static constexpr R value = R(1) / R(n);
    };
    template <class R, int NMAX, int n> OPG_HD OPG_INLINE void jn_down(const Complex<R>& e, R z, Complex<R>* J)
    {
      if constexpr (n > 0) {
        // J_{n-1} = (e - iz J_n) / n
        const Complex<R>& j = J[n];
        J[n - 1] = Complex<R>(e.re + z * j.im, e.im - z * j.re) * Inv<R, n>::value;
        jn_down<R, NMAX, n - 1>(e, z, J);
      }
    }

    /// J_n(z) = int_0^1 t^n exp(izt) dt for n = 0..NMAX.
    template <int NMAX, class R> OPG_HD inline void jn_all(R z, Complex<R>* J)
    {
      static_assert(NMAX >= 0 && NMAX <= 15, "jn_all: NMAX");
      const Complex<R> e = expi(z);
      if (std::fabs(z) <= R(4)) {
        // Series for J_NMAX, sum_k (iz)^k / (k! (NMAX + k + 1)), truncated
        // at k = 37 (4^38 / 38! < 1e-21; even k real, odd k imaginary),
        // then the recurrence downwards (stable for |z| <= 4: errors grow by
        // |z| / n per step, at most 4^8 / 8!).
        const R z2 = z * z;
        J[NMAX]    = Complex<R>(jn_series<R, NMAX, 0>(z2), z * jn_series<R, NMAX, 1>(z2));
        jn_down<R, NMAX, NMAX>(e, z, J);
      }
      else {
        // J_0 = (e - 1) / (iz), J_n = (e - n J_{n-1}) / (iz); a / (iz) = -i a / z
        const R iv = R(1) / z;
        Complex<R> a(e.re - R(1), e.im);
        J[0] = Complex<R>(a.im * iv, -a.re * iv);
        OPG_UNROLL
        for (int n = 1; n <= NMAX; n++) {
          a    = Complex<R>(e.re - R(n) * J[n - 1].re, e.im - R(n) * J[n - 1].im);
          J[n] = Complex<R>(a.im * iv, -a.re * iv);
        }
      }
    }

    template <class R> OPG_HD inline R sinc(R w)
    {
      return std::fabs(w) < R(1e-4) ? R(1) - w * w / R(6) : std::sin(w) / w;
    }

    /// g(x) = int_0^L exp(ixs) ds and its divided differences.
    template <class R> struct SegFn {
        R L;
        OPG_HD Complex<R> g(R x) const
        {
          const R w = x * L / 2;
          R       s, c;
          sin_cos(w, s, c);
          return Complex<R>(c, s) * (L * sinc(w));
        }
        /// (g(x) - g(y)) / (x - y), g'(x) at x = y.
        OPG_HD Complex<R> dd(R x, R y) const
        {
          const R d = x - y;
          if (std::fabs(d) * L > R(0.1)) return (g(x) - g(y)) / d;
          return dd_near(x, y);
        }
        /// dd() with gx = g(x), gy = g(y) given.
        OPG_HD Complex<R> dd(R x, R y, const Complex<R>& gx, const Complex<R>& gy) const
        {
          const R d = x - y;
          if (std::fabs(d) * L > R(0.1)) return (gx - gy) / d;
          return dd_near(x, y);
        }
        OPG_HD Complex<R> dd_near(R x, R y) const
        {
          const R    d = x - y;
          const R    m = (x + y) / 2, lt = L * d / 2;
          Complex<R> J[8];
          const Complex<R> I(0, 1);
          if (d == R(0)) {
            jn_all<1>(m * L, J);
            return I * J[1] * (L * L);
          }
          // sum over odd n of L^{n+1} i^n J_n(mL) t^{n-1} / n!, t = d / 2
          jn_all<7>(m * L, J);
          const R    t2 = lt * lt;
          Complex<R> acc =
              J[1] - J[3] * (t2 / 6) + J[5] * (t2 * t2 / 120) - J[7] * (t2 * t2 * t2 / 5040);
          return I * acc * (L * L);
        }
    };

    /// F(x) = <(1 + beta (d - c)/h) exp(-ixd)> over d in [c - h, c + h]
    /// (= exp(-ixc) F_0(x), F_0 the centred average), faded out for |x h| in
    /// [z1, z2], and its divided differences. The shift c places a piece of
    /// a cell whose expansion point is d = 0.
    template <class R> struct AvgFn {
        R h, beta, z1, z2;
        R c = 0;

        OPG_HD static void j01(R z, R& j0, R& j1)
        {
          if (std::fabs(z) < R(0.1)) {
            const R z2 = z * z;
            j0 = R(1) - z2 / 6 * (R(1) - z2 / 20 * (R(1) - z2 / 42 * (R(1) - z2 / 72)));
            j1 = z / 3 * (R(1) - z2 / 10 * (R(1) - z2 / 28 * (R(1) - z2 / 54)));
          }
          else {
            R s, c;
            sin_cos(z, s, c);
            j0 = s / z;
            j1 = (s - z * c) / (z * z);
          }
        }
        /// Fade factor chi(s), s = |z|, and dchi/ds.
        OPG_HD void fade(R s, R& chi, R& dchi) const
        {
          dchi = 0;
          if (s <= z1) { chi = 1; return; }
          if (s >= z2) { chi = 0; return; }
          const R lr = std::log(z2 / z1), t = std::log(s / z1) / lr;
          chi        = R(1) - t * t * t * (R(10) - R(15) * t + R(6) * t * t);
          dchi       = -R(30) * t * t * (R(1) - t) * (R(1) - t) / (s * lr);
        }
        OPG_HD Complex<R> F(R x) const
        {
          if (c != R(0)) return F0(x) * expi(-x * c);
          return F0(x);
        }
        /// (F(x) - F(y)) / (x - y), F'(x) at x = y.
        OPG_HD Complex<R> dd(R x, R y) const
        {
          if (c != R(0)) return dd_shift(x, y, F0(y));
          return dd0(x, y);
        }
        /// dd() with Fx = F(x), Fy = F(y) given.
        OPG_HD Complex<R> dd(R x, R y, const Complex<R>& Fx, const Complex<R>& Fy) const
        {
          if (c != R(0)) return dd_shift(x, y, Fy * expi(y * c));
          return dd0(x, y, Fx, Fy);
        }
        /// Product rule, F = E F_0 with E(x) = exp(-ixc): F[x, y] =
        /// E(x) F_0[x, y] + E[x, y] F_0(y), E[x, y] = -ic exp(-ic(x + y)/2)
        /// sinc(c (x - y)/2).
        OPG_HD Complex<R> dd_shift(R x, R y, const Complex<R>& F0y) const
        {
          const Complex<R> Ex = expi(-x * c);
          const Complex<R> Ed = expi(-c * (x + y) / 2) * Complex<R>(0, -c * sinc(c * (x - y) / 2));
          return Ex * dd0(x, y) + Ed * F0y;
        }
        /// The centred average F_0.
        OPG_HD Complex<R> F0(R x) const
        {
          const R z = x * h;
          R       chi, dchi, j0, j1;
          fade(std::fabs(z), chi, dchi);
          if (chi == R(0)) return Complex<R>(0, 0);
          j01(z, j0, j1);
          return Complex<R>(j0, -beta * j1) * chi;
        }
        /// dF_0/dx (used inside the fade region only).
        OPG_HD Complex<R> dF(R x) const
        {
          const R z = x * h;
          R       chi, dchi, j0, j1;
          fade(std::fabs(z), chi, dchi);
          if (chi == R(0)) return Complex<R>(0, 0);
          j01(z, j0, j1);
          const Complex<R> f(j0, -beta * j1);
          const Complex<R> df(-j1, -beta * (j0 - 2 * j1 / z));
          return (df * chi + f * (dchi * (z < 0 ? R(-1) : R(1)))) * h;
        }
        /// (F_0(x) - F_0(y)) / (x - y), F_0'(x) at x = y.
        OPG_HD Complex<R> dd0(R x, R y) const
        {
          const R d = x - y;
          if (std::fabs(d) * h > R(0.1)) return (F0(x) - F0(y)) / d;
          return dd_near(x, y);
        }
        /// dd0() with Fx = F_0(x), Fy = F_0(y) given.
        OPG_HD Complex<R> dd0(R x, R y, const Complex<R>& Fx, const Complex<R>& Fy) const
        {
          const R d = x - y;
          if (std::fabs(d) * h > R(0.1)) return (Fx - Fy) / d;
          return dd_near(x, y);
        }
        OPG_HD Complex<R> dd_near(R x, R y) const
        {
          const R d = x - y;
          const R m = (x + y) / 2;
          if (std::fabs(m * h) + R(0.1) >= z1) return dF(m);  // |F| < 1/z1 here
          // F^(n)(m) = h^n (-i)^n [E_n + beta E_{n+1}](mh),
          // E_n(z) = (conj J_n(z) + (-1)^n J_n(z)) / 2; odd n: -i Im J_n,
          // even n: Re J_n.
          Complex<R> J[9];
          if (d == R(0)) {
            jn_all<2>(m * h, J);
            return G(J, 1) * h;
          }
          jn_all<8>(m * h, J);
          const R t2 = (h * d / 2) * (h * d / 2);
          return (G(J, 1) + G(J, 3) * (t2 / 6) + G(J, 5) * (t2 * t2 / 120) +
                  G(J, 7) * (t2 * t2 * t2 / 5040)) *
                 h;
        }
        /// (-i)^n [E_n + beta E_{n+1}] for odd n.
        OPG_HD Complex<R> G(const Complex<R>* J, int n) const
        {
          const R sg = (n % 4 == 1) ? R(-1) : R(1);  // (-i)^{n+1}
          return Complex<R>(sg * J[n].im, sg * beta * J[n + 1].re);
        }
    };

    /// Full hermitian matrix from the upper triangle (values).
    template <int N, class R> OPG_HD inline Mat<N, R> full(Mat<N, R> H)
    {
      hermitize_from_upper(H);
      return H;
    }

    /// Derivative direction k of a dual matrix (upper triangle), completed
    /// to a hermitian matrix.
    template <int N, class R, int K>
    OPG_HD inline Mat<N, R> dual_dir(const Mat<N, Dual<R, K>>& HD, int k)
    {
      Mat<N, R> d;
      for (int i = 0; i < N; i++) {
        d(i, i) = Complex<R>(HD(i, i).re.d[k], 0);
        for (int j = i + 1; j < N; j++) {
          d(i, j) = Complex<R>(HD(i, j).re.d[k], HD(i, j).im.d[k]);
          d(j, i) = conj(d(i, j));
        }
      }
      return d;
    }

    template <int N, class R> OPG_HD inline Mat<N, R> mul_ah(const Mat<N, R>& A, const Mat<N, R>& B)
    {
      return matmul(adjoint(A), B);  // A^dag B
    }
    template <int N, class R>
    OPG_HD inline Mat<N, R> sandwich(const Mat<N, R>& V, const Mat<N, R>& X)  // V^dag X V
    {
      return matmul(adjoint(V), matmul(X, V));
    }
    template <int N, class R>
    OPG_HD inline Mat<N, R> unsandwich(const Mat<N, R>& V, const Mat<N, R>& X)  // V X V^dag
    {
      return matmul(V, matmul(X, adjoint(V)));
    }
    template <int N, class R> OPG_HD inline void add_to(Mat<N, R>& A, const Mat<N, R>& B)
    {
      for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) A(i, j) += B(i, j);
    }
    template <int N, class R> OPG_HD inline Mat<N, R> herm_part(const Mat<N, R>& A)
    {
      Mat<N, R> H;
      for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) H(i, j) = (A(i, j) + conj(A(j, i))) * R(0.5);
      return H;
    }

    /// Per-segment data of the value pass, reused by the derivatives.
    template <int N, class R> struct SegData {
        Mat<N, R>  V, At, T, Sseg, Kseg;
        R          lam[N], L;
        /// G1[i][l][j] = g[lam_i - lam_j, lam_l - lam_j]; the other table,
        /// g[lam_i - lam_l, lam_i - lam_j], is -conj G1[l][j][i] (g(-x) =
        /// conj g(x)).
        Complex<R> gm[N][N], Gam[N][N], G1[N][N][N];
    };

    /// Value-pass results of a sub-bin reused by the derivatives: K =
    /// W diag(mu) W^dag, B = S0 W, rho_a = W^dag e_a e_a^dag W, M_a = rho_a o F.
    template <int N, class R> struct KState {
        Mat<N, R>  W, B, M[N], rho[N];
        R          mu[N];
        /// F1[n][l][m] = F[mu_n - mu_m, mu_l - mu_m] (with gradients); the
        /// other table is -conj F1[l][m][n] (F(-x) = conj F(x)).
        Complex<R> F1[N][N][N];
    };

    /// Average over one sub-bin: out[a*N + b]. seg: scratch of size nseg
    /// (with want_grad, also filled with what subbin_grad needs).
    template <class Model>
    OPG_HD void subbin(const typename Model::Prepared& P, const Mat<Model::N, typename Model::Real>& A,
                const Segment<typename Model::Real>* path, int nseg, bool nubar,
                typename Model::Real u0, const AvgFn<typename Model::Real>& fn,
                typename Model::Real* out, bool want_grad,
                SegData<Model::N, typename Model::Real>* seg,
                KState<Model::N, typename Model::Real>& ks)
    {
      using R         = typename Model::Real;
      constexpr int N = Model::N;
      const R       E0 = R(1) / u0;

      Mat<N, R> S = Mat<N, R>::identity(), K = Mat<N, R>::zero();
      for (int s = 0; s < nseg; s++) {
        SegData<N, R>& d = seg[s];
        Mat<N, R>      H;
        Model::hamiltonian(P, E0, nubar, path[s], H);
        jacobi_hermitian<N, R>(H, d.V, d.lam);
        d.L  = length_in_eV(path[s].length);
        d.At = sandwich(d.V, A);
        const SegFn<R> g{d.L};
        Mat<N, R>      Kt;
        for (int i = 0; i < N; i++)
          for (int j = 0; j < N; j++) {
            d.gm[i][j] = g.g(d.lam[i] - d.lam[j]);
            Kt(i, j)   = d.At(i, j) * d.gm[i][j];
          }
        d.Kseg = unsandwich(d.V, Kt);
        d.T    = S;
        Mat<N, R> Ph = Mat<N, R>::identity();
        apply_eigen_step<N, R>(d.V, d.lam, d.L, Ph);
        d.Sseg = Ph;
        if (s == 0) {  // S = 1
          K = d.Kseg;
          S = Ph;
        }
        else {
          add_to(K, matmul(adjoint(S), matmul(d.Kseg, S)));
          S = matmul(Ph, S);
        }
        if (want_grad) {
          Complex<R> f[N];
          dk_gamma<N, R>(d.lam, d.L, d.Gam, f);
          for (int j = 0; j < N; j++)  // G1 is symmetric in i, l
            for (int i = 0; i < N; i++)
              for (int l = 0; l <= i; l++)
                d.G1[i][l][j] = d.G1[l][i][j] =
                    g.dd(d.lam[i] - d.lam[j], d.lam[l] - d.lam[j]);
        }
      }
      jacobi_hermitian<N, R>(herm_part(K), ks.W, ks.mu);
      ks.B = matmul(S, ks.W);

      Complex<R> F[N][N];
      for (int n = 0; n < N; n++)
        for (int m = 0; m < N; m++) F[n][m] = fn.F(ks.mu[n] - ks.mu[m]);

      for (int a = 0; a < N; a++) {
        for (int n = 0; n < N; n++)
          for (int m = 0; m < N; m++) {
            ks.rho[a](n, m) = conj(ks.W(a, n)) * ks.W(a, m);
            ks.M[a](n, m)   = ks.rho[a](n, m) * F[n][m];
          }
        for (int b = 0; b < N; b++) {
          R acc = 0;
          for (int n = 0; n < N; n++) {
            Complex<R> t(0, 0);
            for (int m = 0; m < N; m++) t += ks.M[a](n, m) * conj(ks.B(b, m));
            acc += (ks.B(b, n) * t).re;
          }
          out[a * N + b] = acc;
        }
      }
      if (want_grad)
        for (int m = 0; m < N; m++)  // F1 is symmetric in n, l
          for (int n = 0; n < N; n++)
            for (int l = 0; l <= n; l++)
              ks.F1[n][l][m] = ks.F1[l][n][m] =
                  fn.dd(ks.mu[n] - ks.mu[m], ks.mu[l] - ks.mu[m]);
    }

    /// Derivatives of a sub-bin average after subbin(..., want_grad = true)
    /// for one gradient pass G (parameters offset .. offset + count - 1):
    /// dout[(p*N + a)*N + b], dA[p] being dA/dp.
    template <class Model, int KD>
    OPG_HD void subbin_grad(const GradPrepared<Model, KD>& G, int offset, int count,
                            const Mat<Model::N, typename Model::Real>* dA,
                            const Segment<typename Model::Real>* path, int nseg, bool nubar,
                            typename Model::Real u0,
                            const SegData<Model::N, typename Model::Real>* seg,
                            const KState<Model::N, typename Model::Real>& ks,
                            typename Model::Real* dout)
    {
      using R         = typename Model::Real;
      constexpr int N = Model::N;
      using D         = Dual<R, KD>;
      const R E0      = R(1) / u0;
      Mat<N, R> dT[KD], dK[KD];  // dS before the segment, dK so far
      for (int k = 0; k < KD; k++) dT[k] = dK[k] = Mat<N, R>::zero();
      for (int s = 0; s < nseg; s++) {
        const SegData<N, R>& d = seg[s];
        Mat<N, D>            HD;
        Model::hamiltonian(G.P, E0, nubar, seed_segment(G, path[s]), HD);
        for (int k = 0; k < count; k++) {
          // In the eigenbasis of H: dS_seg = Gamma o dH, and
          // dK_seg = dA o g + sum_l (dH_il A_lj g[l_i - l_j, l_l - l_j] -
          //                        A_il dH_lj g[l_i - l_l, l_i - l_j]).
          const Mat<N, R> dHt = sandwich(d.V, dual_dir<N, R, KD>(HD, k));
          const Mat<N, R> dAt = sandwich(d.V, dA[offset + k]);
          Mat<N, R>       dSt, dKt;
          for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
              dSt(i, j)      = d.Gam[i][j] * dHt(i, j);
              Complex<R> acc = dAt(i, j) * d.gm[i][j];
              for (int l = 0; l < N; l++)
                acc += dHt(i, l) * d.At(l, j) * d.G1[i][l][j] +
                       d.At(i, l) * dHt(l, j) * conj(d.G1[l][j][i]);
              dKt(i, j) = acc;
            }
          const Mat<N, R> dSseg = unsandwich(d.V, dSt);
          const Mat<N, R> dKseg = unsandwich(d.V, dKt);
          if (s == 0) {  // T = 1, dT = 0
            dK[k] = dKseg;
            dT[k] = dSseg;
            continue;
          }
          // K += T^dag K_seg T, T <- S_seg T
          const Mat<N, R> KT = matmul(d.Kseg, d.T);
          add_to(dK[k], matmul(adjoint(dT[k]), KT));
          add_to(dK[k], matmul(adjoint(d.T), matmul(dKseg, d.T)));
          add_to(dK[k], matmul(adjoint(KT), dT[k]));
          Mat<N, R> nT = matmul(dSseg, d.T);
          add_to(nT, matmul(d.Sseg, dT[k]));
          dT[k] = nT;
        }
      }
      for (int k = 0; k < count; k++) {
        // In the eigenbasis of K: dM_a = sum_l (dK_nl rho_lm F[..]_nlm -
        // rho_nl dK_lm F[mu_n - mu_l, mu_n - mu_m]); dPbar_ab = 2 Re (dB M_a B^dag)_bb +
        // (B dM_a B^dag)_bb with dB = dS0 W.
        const Mat<N, R> dKt = sandwich(ks.W, herm_part(dK[k]));
        const Mat<N, R> dB  = matmul(dT[k], ks.W);
        R*              o   = dout + size_t(offset + k) * N * N;
        for (int a = 0; a < N; a++) {
          Mat<N, R> dM;
          for (int n = 0; n < N; n++)
            for (int m = 0; m < N; m++) {
              Complex<R> acc(0, 0);
              for (int l = 0; l < N; l++)
                acc += dKt(n, l) * ks.rho[a](l, m) * ks.F1[n][l][m] +
                       ks.rho[a](n, l) * dKt(l, m) * conj(ks.F1[l][m][n]);
              dM(n, m) = acc;
            }
          for (int b = 0; b < N; b++) {
            R acc = 0;
            for (int n = 0; n < N; n++) {
              Complex<R> t(0, 0), u(0, 0);
              for (int m = 0; m < N; m++) {
                t += ks.M[a](n, m) * conj(ks.B(b, m));
                u += dM(n, m) * conj(ks.B(b, m));
              }
              acc += 2 * (dB(b, n) * t).re + (ks.B(b, n) * u).re;
            }
            o[a * N + b] = acc;
          }
        }
      }
    }

    /// Parameters per block of cell_fused's contraction (registers).
    constexpr int FUSED_MAXP = 8;

    /// Initial flavour a selected by the mask `rows` (0: all).
    OPG_HD inline bool row_on(unsigned rows, int a) { return rows == 0 || ((rows >> a) & 1u); }

    /// One-segment sub-bin state of the fused path (eigenbasis of H: H =
    /// V diag(lam) V^dag; K = V Kt V^dag, Kt = X diag(mu) X^dag; W = V X,
    /// B = V diag(e^{-i lam L}) X; F[n][m] = F(mu_n - mu_m)).
    template <int N, class R> struct FusedState {
        Mat<N, R>  V, At, X, W, B;
        R          lam[N], mu[N], L, u0;
        Complex<R> gm[N][N], F[N][N];
    };

    /// The state at u0 (all but F, see fused_F).
    template <class Model>
    OPG_HD inline void fused_state(const typename Model::Prepared& P,
                                   const Mat<Model::N, typename Model::Real>& A,
                                   const Segment<typename Model::Real>& seg, bool nubar,
                                   typename Model::Real u0,
                                   FusedState<Model::N, typename Model::Real>& s)
    {
      using R         = typename Model::Real;
      constexpr int N = Model::N;
      s.u0            = u0;
      s.L             = length_in_eV(seg.length);
      const SegFn<R> g{s.L};

      Mat<N, R> H;
      Model::hamiltonian(P, R(1) / u0, nubar, seg, H);
      jacobi_hermitian<N, R>(H, s.V, s.lam);
      s.At = sandwich(s.V, A);

      Mat<N, R> Kt;
      OPG_UNROLL
      for (int i = 0; i < N; i++) {
        s.gm[i][i] = Complex<R>(s.L, 0);
        Kt(i, i)   = s.At(i, i) * s.L;
        OPG_UNROLL
        for (int j = i + 1; j < N; j++) {
          s.gm[i][j] = g.g(s.lam[i] - s.lam[j]);
          s.gm[j][i] = conj(s.gm[i][j]);
          Kt(i, j)   = s.At(i, j) * s.gm[i][j];
        }
      }
      jacobi_hermitian<N, R>(Kt, s.X, s.mu);

      s.W = matmul(s.V, s.X);
      {
        Mat<N, R> PX;
        OPG_UNROLL
        for (int i = 0; i < N; i++) {
          R sn, cs;
          sin_cos(s.lam[i] * s.L, sn, cs);
          const Complex<R> ph(cs, -sn);
          OPG_UNROLL
          for (int n = 0; n < N; n++) PX(i, n) = ph * s.X(i, n);
        }
        s.B = matmul(s.V, PX);
      }
    }

    /// F[n][m] = F(mu_n - mu_m) of the averaging function of a piece.
    template <int N, class R> OPG_HD inline void fused_F(FusedState<N, R>& s, const AvgFn<R>& fn)
    {
      OPG_UNROLL
      for (int n = 0; n < N; n++) {
        s.F[n][n] = fn.F(R(0));
        OPG_UNROLL
        for (int m = n + 1; m < N; m++) {
          s.F[n][m] = fn.F(s.mu[n] - s.mu[m]);
          s.F[m][n] = conj(s.F[n][m]);
        }
      }
    }

    /// Pbar_ab for the initial flavours in `rows` (others zero).
    template <int N, class R>
    OPG_HD inline void fused_values(const FusedState<N, R>& s, unsigned rows, R* out)
    {
      for (int a = 0; a < N; a++) {
        if (!row_on(rows, a)) {
          for (int b = 0; b < N; b++) out[a * N + b] = R(0);
          continue;
        }
        for (int b = 0; b < N; b++) {
          Complex<R> cy[N];  // cy = conj(y)
          OPG_UNROLL
          for (int m = 0; m < N; m++) cy[m] = conj(s.B(b, m)) * s.W(a, m);
          R acc = 0;
          OPG_UNROLL
          for (int n = 0; n < N; n++) {
            Complex<R> t(0, 0);
            OPG_UNROLL
            for (int m = 0; m < N; m++) t += s.F[n][m] * cy[m];
            acc += (conj(cy[n]) * t).re;
          }
          out[a * N + b] = acc;
        }
      }
    }

    /// Parameter-independent derivative tables: Gamma (dS) and G1[i][l][j] =
    /// g[lam_i - lam_j, lam_l - lam_j] in the eigenbasis of H, F1[n][l][m] =
    /// F[mu_n - mu_m, mu_l - mu_m] in that of K (both symmetric in the first
    /// two indices).
    template <int N, class R> struct FusedTables {
        Complex<R> Gam[N][N], G1[N][N][N], F1[N][N][N];
    };

    /// The tables of the eigenbasis of H (per expansion point).
    template <int N, class R>
    OPG_HD inline void fused_tables_H(const FusedState<N, R>& s, FusedTables<N, R>& t)
    {
      const SegFn<R> g{s.L};
      Complex<R>     f[N];
      dk_gamma<N, R>(s.lam, s.L, t.Gam, f);
      OPG_UNROLL
      for (int j = 0; j < N; j++)
        OPG_UNROLL
      for (int i = 0; i < N; i++)
        OPG_UNROLL
      for (int l = 0; l <= i; l++)
        t.G1[i][l][j] = t.G1[l][i][j] =
            g.dd(s.lam[i] - s.lam[j], s.lam[l] - s.lam[j], s.gm[i][j], s.gm[l][j]);
    }

    /// F1 of a piece (after fused_F with the same fn).
    template <int N, class R>
    OPG_HD inline void fused_tables_F1(const FusedState<N, R>& s, const AvgFn<R>& fn,
                                       FusedTables<N, R>& t)
    {
      OPG_UNROLL
      for (int m = 0; m < N; m++)
        OPG_UNROLL
      for (int n = 0; n < N; n++)
        OPG_UNROLL
      for (int l = 0; l <= n; l++)
        t.F1[n][l][m] = t.F1[l][n][m] =
            fn.dd(s.mu[n] - s.mu[m], s.mu[l] - s.mu[m], s.F[n][m], s.F[l][m]);
    }

    /// Derivative of one parameter (dAt = V^dag dA V, dHt = V^dag dH V) as
    /// dKk = X^dag dKt X (N*N reals: diagonal, then Re, Im of n < l) and
    /// the rows b < N-1 of dB = V dSt X.
    template <int N, class R>
    OPG_HD inline void fused_dparam(const FusedState<N, R>& s, const FusedTables<N, R>& t,
                                    const Mat<N, R>& dAt, const Mat<N, R>& dHt, R* dk,
                                    Complex<R> (*db)[N])
    {
      // dKt (hermitian: upper triangle, then mirrored), dSt = Gamma o dHt
      Mat<N, R> dKt, dSt;
      OPG_UNROLL
      for (int i = 0; i < N; i++) {
        OPG_UNROLL
        for (int j = 0; j < N; j++) dSt(i, j) = t.Gam[i][j] * dHt(i, j);
        OPG_UNROLL
        for (int j = i; j < N; j++) {
          Complex<R> acc = dAt(i, j) * s.gm[i][j];
          OPG_UNROLL
          for (int l = 0; l < N; l++)
            acc += dHt(i, l) * s.At(l, j) * t.G1[i][l][j] +
                   s.At(i, l) * dHt(l, j) * conj(t.G1[l][j][i]);
          if (j == i) acc.im = 0;
          dKt(i, j) = acc;
          if (j != i) dKt(j, i) = conj(acc);
        }
      }
      const Mat<N, R> dKk = sandwich(s.X, dKt);
      const Mat<N, R> dB  = matmul(s.V, matmul(dSt, s.X));
      int             k   = 0;
      OPG_UNROLL
      for (int n = 0; n < N; n++) dk[k++] = dKk(n, n).re;
      OPG_UNROLL
      for (int n = 0; n < N; n++)
        OPG_UNROLL
      for (int l = n + 1; l < N; l++) {
        dk[k++] = dKk(n, l).re;
        dk[k++] = dKk(n, l).im;
      }
      OPG_UNROLL
      for (int b = 0; b < N - 1; b++)
        OPG_UNROLL
      for (int n = 0; n < N; n++) db[b][n] = dB(b, n);
    }

    /// Channel (a, b) tables, as reals, scaled by w: rs . dk = 2 w Re
    /// sum_{n<=l} dKk_nl Rs_nl and z . (Re, Im dB_bn)_n = 2 w Re sum_n dB_bn
    /// Z_n, with y_m = B_bm conj(W_am), Y_n = sum_m F_nm conj(y_m), Z_n =
    /// conj(W_an) Y_n, Q_nl = sum_m F1[n][l][m] conj(y_m), R_nl = B_bn
    /// conj(W_al) Q_nl, Rs_nn = R_nn, Rs_nl = R_nl + conj(R_ln) (n < l).
    /// With add, the tables are accumulated.
    template <int N, class R>
    OPG_HD inline void fused_channel(const FusedState<N, R>& s, const FusedTables<N, R>& t,
                                     int a, int b, R w, bool add, R* rs, R* z)
    {
      Complex<R> cy[N];
      OPG_UNROLL
      for (int m = 0; m < N; m++) cy[m] = conj(s.B(b, m)) * s.W(a, m);
      OPG_UNROLL
      for (int n = 0; n < N; n++) {
        Complex<R> y(0, 0);
        OPG_UNROLL
        for (int m = 0; m < N; m++) y += s.F[n][m] * cy[m];
        const Complex<R> zz = conj(s.W(a, n)) * y;
        const R          zr = 2 * w * zz.re, zi = -2 * w * zz.im;
        z[2 * n]            = add ? z[2 * n] + zr : zr;
        z[2 * n + 1]        = add ? z[2 * n + 1] + zi : zi;
      }
      int kd = 0, ko = N;
      OPG_UNROLL
      for (int n = 0; n < N; n++)
        OPG_UNROLL
      for (int l = n; l < N; l++) {
        Complex<R> q(0, 0);
        OPG_UNROLL
        for (int m = 0; m < N; m++) q += t.F1[n][l][m] * cy[m];
        Complex<R> r = s.B(b, n) * conj(s.W(a, l)) * q;
        if (l != n) {
          r += conj(s.B(b, l) * conj(s.W(a, n)) * q);
          const R rr = 2 * w * r.re, ri = -2 * w * r.im;
          rs[ko]     = add ? rs[ko] + rr : rr;
          rs[ko + 1] = add ? rs[ko + 1] + ri : ri;
          ko += 2;
        }
        else {
          const R rr = 2 * w * r.re;
          rs[kd]     = add ? rs[kd] + rr : rr;
          kd++;
        }
      }
    }

    /// V^dag dA V and V^dag dH V of parameter k of a chunk (dA: the
    /// chunk's dA; HD its dual Hamiltonian unless st: dH = u dA for the
    /// model parameters of a model with static matter terms).
    template <class Model, int KD>
    OPG_HD inline void fused_dH(const FusedState<Model::N, typename Model::Real>& s,
                                const Mat<Model::N, typename Model::Real>* dA, int k, bool st,
                                const Mat<Model::N, Dual<typename Model::Real, KD>>& HD,
                                Mat<Model::N, typename Model::Real>& dAt,
                                Mat<Model::N, typename Model::Real>& dHt)
    {
      using R         = typename Model::Real;
      constexpr int N = Model::N;
      dAt             = sandwich(s.V, dA[k]);
      if (st)
        for (int i = 0; i < N; i++)
          for (int j = 0; j < N; j++) dHt(i, j) = dAt(i, j) * s.u0;
      else dHt = sandwich(s.V, dual_dir<N, R, KD>(HD, k));
    }

    /// Whether chunk c only has model parameters of a model with static
    /// matter terms (dH = u dA).
    template <class Model, int KD>
    OPG_HD inline bool fused_static(const GradPrepared<Model, KD>& G, int count)
    {
      bool st = has_static_matter<Model>::value;
      for (int k = 0; k < count; k++) st = st && G.zoa_type[k] < 0 && G.rho_type[k] < 0;
      return st;
    }

    /// A cell of a one-segment path: pieces (sub-bins sub[first .. first+n))
    /// that share the expansion at u = uc; piece i is averaged with AvgFn
    /// shift c = u0_i - uc. A sub-bin on its own is the cell uc = u0, c = 0.
    template <class R> struct CellView {
        R                     uc;
        const AnalyticSubBin* sub;
        int                   first, n;
        R                     fast_begin, fast_end;
        OPG_HD AvgFn<R>       fn(int j) const
        {
          const AnalyticSubBin& s = sub[first + j];
          return AvgFn<R>{R(s.h), R(s.beta), fast_begin, fast_end, R(s.u0) - uc};
        }
    };

    /// Values and all derivatives of the pieces of a cell of a one-segment
    /// path, without per-segment scratch (same results as subbin +
    /// subbin_grad to round-off for one-piece cells): for sub-bin i of the
    /// cell, sp[i*N*N + a*N + b] and sg[i*npar*N*N + (p*N + a)*N + b]
    /// (sg = nullptr or nchunk = 0: values only), for the initial flavours a
    /// in `rows` (others are set to zero).
    ///
    /// Everything is kept in the eigenbasis of H (FusedState), and the
    /// derivatives are contracted through parameter-independent tables per
    /// channel: with y_m = B_bm conj(W_am), Y_n = sum_m F_nm conj(y_m) and
    /// Q_nl = sum_m F[mu_n - mu_m, mu_l - mu_m] conj(y_m),
    ///   Pbar_ab  = Re sum_n y_n Y_n,
    ///   dPbar_ab = 2 Re [sum_n dB_bn conj(W_an) Y_n +
    ///                    sum_nl dKk_nl B_bn conj(W_al) Q_nl],
    /// dB = V dSt X, dKk = X^dag dKt X (dKt hermitian). Parameters go in
    /// blocks of at most FUSED_MAXP: dKk and dB of the block first (per
    /// cell), then one contraction per piece with its channel tables
    /// (parameters x channels in registers). The last final flavour follows
    /// from unitarity: sum_b Pbar_ab = F(0) = 1 exactly, so sum_b dPbar_ab = 0.
    template <class Model, int KD>
    OPG_HD void cell_fused(const typename Model::Prepared& P,
                           const Mat<Model::N, typename Model::Real>& A,
                           const GradPrepared<Model, KD>* G, const int* off, const int* cnt,
                           int nchunk, const Mat<Model::N, typename Model::Real>* dA, int npar,
                           const Segment<typename Model::Real>& seg, bool nubar,
                           const CellView<typename Model::Real>& cell, unsigned rows,
                           typename Model::Real* sp, typename Model::Real* sg)
    {
      using R          = typename Model::Real;
      constexpr int N  = Model::N;
      constexpr int NK = N * N;
      using D          = Dual<R, KD>;
      static_assert(KD <= FUSED_MAXP, "cell_fused: gradient chunk too large");

      FusedState<N, R> s;
      fused_state<Model>(P, A, seg, nubar, cell.uc, s);
      for (int j = 0; j < cell.n; j++) {
        fused_F(s, cell.fn(j));
        fused_values(s, rows, sp + size_t(cell.first + j) * NK);
      }
      if (sg == nullptr || nchunk <= 0) return;
      FusedTables<N, R> t;
      fused_tables_H(s, t);

      for (int c0 = 0; c0 < nchunk;) {
        int c1 = c0, nq = 0;
        while (c1 < nchunk && nq + cnt[c1] <= FUSED_MAXP) nq += cnt[c1++];
        int        pid[FUSED_MAXP];
        R          dk[FUSED_MAXP][NK];
        Complex<R> db[FUSED_MAXP][N - 1][N];
        int        q = 0;
        for (int c = c0; c < c1; c++) {
          const bool st = fused_static(G[c], cnt[c]);
          Mat<N, D>  HD;
          if (!st)
            Model::hamiltonian(G[c].P, R(1) / cell.uc, nubar, seed_segment(G[c], seg), HD);
          for (int k = 0; k < cnt[c]; k++, q++) {
            pid[q] = off[c] + k;
            Mat<N, R> dAt, dHt;
            fused_dH<Model, KD>(s, dA + off[c], k, st, HD, dAt, dHt);
            fused_dparam(s, t, dAt, dHt, dk[q], db[q]);
          }
        }

        for (int j = 0; j < cell.n; j++) {
          const AvgFn<R> fn = cell.fn(j);
          fused_F(s, fn);
          fused_tables_F1(s, fn, t);
          R* dout = sg + size_t(cell.first + j) * size_t(npar) * NK;
          for (int a = 0; a < N; a++) {
            if (!row_on(rows, a)) {
              for (int r = 0; r < nq; r++)
                for (int b = 0; b < N; b++) dout[(size_t(pid[r]) * N + a) * N + b] = R(0);
              continue;
            }
            R rs[N - 1][NK], z[N - 1][2 * N];
            for (int b = 0; b < N - 1; b++) fused_channel(s, t, a, b, R(1), false, rs[b], z[b]);
            R acc[FUSED_MAXP][N - 1];
            OPG_UNROLL
            for (int r = 0; r < FUSED_MAXP; r++)
              OPG_UNROLL
            for (int b = 0; b < N - 1; b++) acc[r][b] = 0;
            OPG_UNROLL
            for (int k = 0; k < NK; k++) {
              R tk[N - 1];
              OPG_UNROLL
              for (int b = 0; b < N - 1; b++) tk[b] = rs[b][k];
              OPG_UNROLL
              for (int r = 0; r < FUSED_MAXP; r++)
                if (r < nq) {
                  const R d = dk[r][k];
                  OPG_UNROLL
                  for (int b = 0; b < N - 1; b++) acc[r][b] += d * tk[b];
                }
            }
            OPG_UNROLL
            for (int b = 0; b < N - 1; b++)
              OPG_UNROLL
            for (int n = 0; n < N; n++) {
              const R zr = z[b][2 * n], zi = z[b][2 * n + 1];
              OPG_UNROLL
              for (int r = 0; r < FUSED_MAXP; r++)
                if (r < nq) acc[r][b] += db[r][b][n].re * zr + db[r][b][n].im * zi;
            }
            OPG_UNROLL
            for (int r = 0; r < FUSED_MAXP; r++)
              if (r < nq) {
                R* o  = dout + (size_t(pid[r]) * N + a) * N;
                R  sm = 0;
                OPG_UNROLL
                for (int b = 0; b < N - 1; b++) {
                  o[b] = acc[r][b];
                  sm += acc[r][b];
                }
                o[N - 1] = -sm;
              }
          }
        }
        c0 = c1;
      }
    }

    /// Weighted derivatives of the pieces of a cell: g[p] = sum over its
    /// sub-bins i (bin bin_i) of subw[i] sum_ab
    /// w[(a*N + b)*nbins + bin_i] dPbar_ab(i)/dp, over the initial flavours a
    /// in `rows` (as cell_fused contracted with the weights, to round-off).
    /// The channel tables of all pieces are summed with the weights (w_ab -
    /// w_a,N-1 by unitarity) into one, so each parameter costs one dKk, dB
    /// and a dot product per cell.
    template <class Model, int KD>
    OPG_HD void cell_weighted(const typename Model::Prepared& P,
                              const Mat<Model::N, typename Model::Real>& A,
                              const GradPrepared<Model, KD>* G, const int* off, const int* cnt,
                              int nchunk, const Mat<Model::N, typename Model::Real>* dA,
                              const Segment<typename Model::Real>& seg, bool nubar,
                              const CellView<typename Model::Real>& cell, unsigned rows,
                              const typename Model::Real* w, size_t nbins,
                              const typename Model::Real* subw, typename Model::Real* g)
    {
      using R          = typename Model::Real;
      constexpr int N  = Model::N;
      constexpr int NK = N * N;
      using D          = Dual<R, KD>;

      FusedState<N, R> s;
      fused_state<Model>(P, A, seg, nubar, cell.uc, s);
      FusedTables<N, R> t;
      fused_tables_H(s, t);

      R rs[NK], z[N - 1][2 * N];
      OPG_UNROLL
      for (int i = 0; i < NK; i++) rs[i] = R(0);
      OPG_UNROLL
      for (int b = 0; b < N - 1; b++)
        OPG_UNROLL
      for (int i = 0; i < 2 * N; i++) z[b][i] = R(0);
      for (int j = 0; j < cell.n; j++) {
        const int      i  = cell.first + j;
        const AvgFn<R> fn = cell.fn(j);
        fused_F(s, fn);
        fused_tables_F1(s, fn, t);
        const R* wb = w + cell.sub[i].bin;
        for (int a = 0; a < N; a++) {
          if (!row_on(rows, a)) continue;
          const R wl = wb[size_t(a * N + N - 1) * nbins];
          for (int b = 0; b < N - 1; b++)
            fused_channel(s, t, a, b, (wb[size_t(a * N + b) * nbins] - wl) * subw[i], true, rs,
                          z[b]);
        }
      }
      for (int c = 0; c < nchunk; c++) {
        const bool st = fused_static(G[c], cnt[c]);
        Mat<N, D>  HD;
        if (!st) Model::hamiltonian(G[c].P, R(1) / cell.uc, nubar, seed_segment(G[c], seg), HD);
        for (int k = 0; k < cnt[c]; k++) {
          Mat<N, R> dAt, dHt;
          fused_dH<Model, KD>(s, dA + off[c], k, st, HD, dAt, dHt);
          R          dk[NK];
          Complex<R> db[N - 1][N];
          fused_dparam(s, t, dAt, dHt, dk, db);
          R acc = 0;
          OPG_UNROLL
          for (int i = 0; i < NK; i++) acc += dk[i] * rs[i];
          OPG_UNROLL
          for (int b = 0; b < N - 1; b++)
            OPG_UNROLL
          for (int n = 0; n < N; n++)
            acc += db[b][n].re * z[b][2 * n] + db[b][n].im * z[b][2 * n + 1];
          g[off[c] + k] = acc;
        }
      }
    }

    /// Cells of a one-segment path: the sub-bins sorted by u0 (perm: list
    /// index of each, in cell order) and grouped greedily into cells of
    /// relative width (u_hi / u_lo - 1) at most `width` and at most
    /// `max_pieces` sub-bins, uc = the middle of the cell (u0 for one
    /// sub-bin). Cell k holds sorted positions cell_start[k] ..
    /// cell_start[k+1]. width <= 0: one cell per sub-bin in list order.
    inline void make_cells(const std::vector<AnalyticSubBin>& sub, double width, int max_pieces,
                           std::vector<int>& perm, std::vector<double>& uc,
                           std::vector<int>& cell_start)
    {
      const size_t ns = sub.size();
      perm.resize(ns);
      for (size_t i = 0; i < ns; i++) perm[i] = int(i);
      uc.clear();
      cell_start.assign(1, 0);
      if (!(width > 0)) {
        for (size_t i = 0; i < ns; i++) {
          uc.push_back(sub[i].u0);
          cell_start.push_back(int(i) + 1);
        }
        return;
      }
      std::stable_sort(perm.begin(), perm.end(),
                       [&](int x, int y) { return sub[size_t(x)].u0 < sub[size_t(y)].u0; });
      for (size_t k0 = 0; k0 < ns;) {
        const auto&  s0 = sub[size_t(perm[k0])];
        const double lo = s0.u0 - s0.h;
        double       hi = s0.u0 + s0.h;
        size_t       k1 = k0 + 1;
        while (k1 < ns && int(k1 - k0) < max_pieces) {
          const auto&  s = sub[size_t(perm[k1])];
          const double h = std::max(hi, s.u0 + s.h);
          if (h / lo - 1 > width) break;
          hi = h;
          k1++;
        }
        uc.push_back(k1 == k0 + 1 ? s0.u0 : 0.5 * (lo + hi));
        cell_start.push_back(int(k1));
        k0 = k1;
      }
    }

    template <class R>
    double cell_width(const std::vector<Segment<R>>& path, const AnalyticAvgOptions& opt);

    /// Sub-bins of 1/E bins [ulo[i], uhi[i]] for a measure (uniform in E,
    /// log E or 1/E) and the options; r is the relative width of the
    /// geometric sub-bins (ignored if opt.nsub > 0).
    inline std::vector<AnalyticSubBin> make_subbins(const std::vector<double>& ulo,
                                                    const std::vector<double>& uhi,
                                                    EMeasure measure, double r,
                                                    const AnalyticAvgOptions& opt)
    {
      std::vector<AnalyticSubBin> sb;
      for (size_t b = 0; b < ulo.size(); b++) {
        const double lo = ulo[b], hi = uhi[b];
        int          n;
        if (opt.nsub > 0) n = opt.nsub;
        else n = std::max(1, int(std::ceil(std::log(hi / lo) / std::log1p(r) - 1e-9)));
        double e0 = lo;
        for (int k = 0; k < n; k++) {
          const double e1 = k + 1 == n ? hi
                            : opt.nsub > 0 ? lo + (hi - lo) * (k + 1) / n
                                           : lo * std::pow(hi / lo, double(k + 1) / n);
          const double u0 = (e0 + e1) / 2, h = (e1 - e0) / 2;
          double       w, beta;
          switch (measure) {
            case EMeasure::Linear: w = 1 / (u0 * u0); beta = -2 * h / u0; break;
            case EMeasure::Log: w = 1 / u0; beta = -h / u0; break;
            default: w = 1; beta = 0; break;
          }
          sb.push_back({int(b), u0, h, 2 * h * w, beta});
          e0 = e1;
        }
      }
      return sb;
    }

    /// The vacuum term A = dH/du (the vacuum Hamiltonian at E = 1 GeV) and,
    /// with chunks, its derivatives dA[p] for the npar parameters.
    template <class Model>
    void vacuum_term(const typename Model::Prepared& P, const std::vector<GradChunk<Model>>* chunks,
                     int npar, bool nubar, Mat<Model::N, typename Model::Real>& A,
                     std::vector<Mat<Model::N, typename Model::Real>>& dA)
    {
      using R         = typename Model::Real;
      constexpr int N = Model::N;
      const Segment<R> vac{R(0), R(0), R(0.5), -1};
      Model::hamiltonian(P, R(1), nubar, vac, A);
      A = full(A);
      dA.assign(size_t(std::max(npar, 0)), Mat<N, R>::zero());
      if constexpr (grad_traits<Model>::enabled) {
        constexpr int KD = grad_traits<Model>::K;
        using D          = Dual<R, KD>;
        if (chunks)
          for (const auto& c : *chunks) {
            Mat<N, D>            AD;
            const SegmentZ<R, D> vz{R(0), D(R(0)), D(R(0.5)), -1};
            Model::hamiltonian(c.P.P, R(1), nubar, vz, AD);
            for (int k = 0; k < c.count; k++) dA[c.offset + k] = dual_dir<N, R, KD>(AD, k);
          }
      }
      else {
        (void)chunks;
      }
    }

    /// Per-sub-bin results on the host (OpenMP): sp[i][a][b] and, with
    /// chunks, sg[i][p][a][b].
    template <class Model>
    void subbins_host(const typename Model::Prepared& P, const std::vector<GradChunk<Model>>* chunks,
                      int npar, const Mat<Model::N, typename Model::Real>& A,
                      const Mat<Model::N, typename Model::Real>* dA,
                      const std::vector<AnalyticSubBin>& sub,
                      const std::vector<Segment<typename Model::Real>>& path, bool nubar,
                      const AnalyticAvgOptions& opt, const std::vector<double>& uc,
                      const std::vector<int>& cstart, typename Model::Real* sp,
                      typename Model::Real* sg)
    {
      using R         = typename Model::Real;
      constexpr int N = Model::N;
      const size_t  ns = sub.size(), nch = size_t(N) * N;
      const size_t  ng = chunks ? size_t(npar) * nch : 0;
      const int     nseg = int(path.size());
      constexpr int KD   = grad_traits<Model>::K;
      // One segment: value and all gradient passes in one fused pass.
      std::vector<GradPrepared<Model, KD>> G;
      std::vector<int>                     off, cnt;
      if constexpr (grad_traits<Model>::enabled)
        if (chunks)
          for (const auto& c : *chunks) {
            G.push_back(c.P);
            off.push_back(c.offset);
            cnt.push_back(c.count);
          }
      if (nseg == 1) {  // cells uc, cstart over the list order of sub
#ifdef _OPENMP
        const int nthr1 = opt.threads > 0 ? opt.threads : omp_get_max_threads();
#pragma omp parallel for schedule(dynamic, 4) num_threads(nthr1)
#endif
        for (long k = 0; k < long(uc.size()); k++) {
          const CellView<R> cell{R(uc[size_t(k)]),
                                 sub.data(),
                                 cstart[size_t(k)],
                                 cstart[size_t(k) + 1] - cstart[size_t(k)],
                                 R(opt.fast_begin),
                                 R(opt.fast_end)};
          cell_fused<Model, KD>(P, A, G.data(), off.data(), cnt.data(), int(G.size()), dA, npar,
                                path[0], nubar, cell, opt.rows, sp, chunks ? sg : nullptr);
        }
        return;
      }
#ifdef _OPENMP
      const int nthr = opt.threads > 0 ? opt.threads : omp_get_max_threads();
#pragma omp parallel num_threads(nthr)
#endif
      {
        std::vector<SegData<N, R>> seg(static_cast<size_t>(nseg));
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 4)
#endif
        for (long i = 0; i < long(ns); i++) {
          const AnalyticSubBin& s = sub[size_t(i)];
          const AvgFn<R>        fn{R(s.h), R(s.beta), R(opt.fast_begin), R(opt.fast_end)};
          KState<N, R>          ks;
          subbin<Model>(P, A, path.data(), nseg, nubar, R(s.u0), fn, sp + size_t(i) * nch,
                        chunks != nullptr, seg.data(), ks);
          if constexpr (grad_traits<Model>::enabled)
            if (chunks)
              for (const auto& c : *chunks)
                subbin_grad<Model, grad_traits<Model>::K>(c.P, c.offset, c.count, dA,
                                                          path.data(), nseg, nubar, R(s.u0),
                                                          seg.data(), ks, sg + size_t(i) * ng);
        }
      }
    }

    /// Averages over explicit sub-bins: out[a][b][bin] (and
    /// dout[p][a][b][bin] with chunks), normalised by the sum of the
    /// sub-bin weights of each bin. With an engine that has a device
    /// implementation (and !opt.host), the sub-bins are computed there;
    /// the reduction is always on the host, in a fixed order.
    template <class Model>
    void average(const typename Model::Prepared& P, const std::vector<GradChunk<Model>>* chunks,
                 int npar, const std::vector<AnalyticSubBin>& sub_in, size_t nbins,
                 const std::vector<Segment<typename Model::Real>>& path, bool nubar,
                 const AnalyticAvgOptions& opt, std::vector<typename Model::Real>& out,
                 std::vector<typename Model::Real>* dout, EngineBase<Model>* engine = nullptr)
    {
      using R         = typename Model::Real;
      constexpr int N = Model::N;
      static_assert(has_analytic_avg<Model>::value,
                    "analytic averages need a hermitian model affine in 1/E "
                    "(Model::analytic_avg)");
      if (path.empty()) throw std::invalid_argument("avg_path_analytic: empty path");
      for (const auto& s : sub_in)
        if (s.bin < 0 || size_t(s.bin) >= nbins || !(s.u0 > 0) || !(s.h >= 0))
          throw std::invalid_argument("avg_path_analytic: bad sub-bin");

      Mat<N, R>              A;
      std::vector<Mat<N, R>> dA;
      vacuum_term<Model>(P, chunks, npar, nubar, A, dA);

      // One-segment paths: cells (sub-bins sharing an expansion, contiguous
      // in the reordered list subc), else one cell per sub-bin.
      std::vector<int>            perm, cstart;
      std::vector<double>         uc;
      std::vector<AnalyticSubBin> subc;
      const bool                  one = path.size() == 1;
      if (one) {
        make_cells(sub_in, opt.cells ? cell_width(path, opt) : 0.0, opt.cell_pieces, perm, uc,
                   cstart);
        subc.reserve(sub_in.size());
        for (int i : perm) subc.push_back(sub_in[size_t(i)]);
      }
      const std::vector<AnalyticSubBin>& sub = one ? subc : sub_in;

      const size_t   ns = sub.size(), nch = size_t(N) * N;
      const size_t   ng = chunks ? size_t(npar) * nch : 0;
      std::vector<R> sp(ns * nch), sg(ns * ng);
      const bool     dev = engine && !opt.host &&
                       engine->analytic_subbins(P, chunks, npar, A, dA.data(), sub.data(), ns,
                                                path.data(), int(path.size()), nubar,
                                                opt.fast_begin, opt.fast_end, opt.rows,
                                                uc.data(), cstart.data(), uc.size(), sp.data(),
                                                sg.data());
      if (!dev)
        subbins_host<Model>(P, chunks, npar, A, dA.data(), sub, path, nubar, opt, uc, cstart,
                            sp.data(), sg.data());

      // Fixed-order reduction (independent of the thread count and devices).
      std::vector<double> wsum(nbins, 0.0);
      out.assign(nch * nbins, R(0));
      if (dout) dout->assign(ng * nbins, R(0));
      for (size_t i = 0; i < ns; i++) {
        const size_t b = size_t(sub[i].bin);
        const R      w = R(sub[i].weight);
        wsum[b] += sub[i].weight;
        for (size_t ch = 0; ch < nch; ch++) out[ch * nbins + b] += w * sp[i * nch + ch];
        if (dout)
          for (size_t q = 0; q < ng; q++) (*dout)[q * nbins + b] += w * sg[i * ng + q];
      }
      for (size_t b = 0; b < nbins; b++) {
        if (!(wsum[b] > 0)) continue;
        const R inv = R(1 / wsum[b]);
        for (size_t ch = 0; ch < nch; ch++) out[ch * nbins + b] *= inv;
        if (dout)
          for (size_t q = 0; q < ng; q++) (*dout)[q * nbins + b] *= inv;
      }
      // Initial flavours outside opt.rows are zero (whichever path ran).
      if (opt.rows)
        for (size_t q = 0; q < nch + ng; q++) {
          if (row_on(opt.rows, int((q % nch) / N))) continue;
          R* o = q < nch ? &out[q * nbins] : &(*dout)[(q - nch) * nbins];
          std::fill(o, o + nbins, R(0));
        }
    }

    /// Relative sub-bin width for a path: sqrt(tol / (0.005 sum V L)) with
    /// V = sqrt(2) G_F N_A rho (Z/A = 1), capped at opt.max_width.
    template <class R>
    inline double subbin_width(const std::vector<Segment<R>>& path, const AnalyticAvgOptions& opt)
    {
      double phi = 0;
      for (const auto& s : path)
        phi += constants::matter_prefactor() * std::fabs(double(s.density)) *
               length_in_eV(double(s.length));
      if (!(opt.max_width > 0)) throw std::invalid_argument("avg_path_analytic: max_width <= 0");
      if (phi <= 0 || !(opt.tol > 0)) return opt.max_width;
      return std::min(opt.max_width, std::sqrt(opt.tol / (0.005 * phi)));
    }

    /// Relative width of the cells of a one-segment path (opt.cells): half
    /// the uncapped sub-bin width rule (pieces sit off the expansion point:
    /// ~3x the mean d^2 of a centred sub-bin), at most opt.cell_max_width.
    template <class R>
    inline double cell_width(const std::vector<Segment<R>>& path, const AnalyticAvgOptions& opt)
    {
      AnalyticAvgOptions o = opt;
      o.max_width          = 1e300;
      return std::min(opt.cell_max_width, 0.5 * subbin_width(path, o));
    }

  } // namespace analytic

} // namespace opg

#endif
