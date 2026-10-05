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

  /// Variable of the bin edges given to the analytic averages.
  enum class BinVar : int {
    E   = 0,  ///< edges in E (GeV)
    LoE = 1   ///< edges in L/E (km/GeV), L = total path length
  };

  struct AnalyticAvgOptions {
      /// Target for the first-order error (sets the sub-bin width r from
      /// the matter phase of the path: r = sqrt(tol / (0.005 sum V L)), V
      /// at Z/A = 1); observed errors are about 0.5-0.8 tol. The cost grows
      /// as 1/sqrt(tol).
      double tol = 1e-6;
      /// Largest relative sub-bin width r = u_hi / u_lo - 1.
      double max_width = 0.1;
      /// > 0: this many sub-bins per bin, uniform in 1/E (tol, max_width
      /// unused).
      int nsub = 0;
      /// Fade-out of fully fast pairs, in |mu_n - mu_m| h (see the file
      /// comment).
      double fast_begin = 1e6;
      double fast_end   = 1e7;
      /// OpenMP threads (0: default).
      int threads = 0;
  };

  /// One sub-bin: uniform in u = 1/E (GeV^-1) on [u0 - h, u0 + h] with
  /// weight weight * (1 + beta (u - u0) / h), added to bin `bin`.
  struct AnalyticSubBin {
      int    bin;
      double u0, h, weight, beta;
  };

  namespace analytic {

    /// J_n(z) = int_0^1 t^n exp(izt) dt for n = 0..nmax (nmax <= 15).
    template <class R> inline void jn_all(R z, int nmax, Complex<R>* J)
    {
      if (std::fabs(z) <= R(4)) {
        for (int n = 0; n <= nmax; n++) J[n] = Complex<R>(0, 0);
        Complex<R> term(1, 0);  // (iz)^k / k!
        for (int k = 0; k < 64; k++) {
          for (int n = 0; n <= nmax; n++) J[n] += term / R(n + k + 1);
          term = term * Complex<R>(0, z) / R(k + 1);
          if (norm(term) < R(1e-40)) break;
        }
      }
      else {
        const Complex<R> e = expi(z), iz(0, z);
        J[0]               = (e - R(1)) / iz;
        for (int n = 1; n <= nmax; n++) J[n] = (e - R(n) * J[n - 1]) / iz;
      }
    }

    template <class R> inline R sinc(R w)
    {
      return std::fabs(w) < R(1e-4) ? R(1) - w * w / R(6) : std::sin(w) / w;
    }

    /// g(x) = int_0^L exp(ixs) ds and its divided differences.
    template <class R> struct SegFn {
        R L;
        Complex<R> g(R x) const
        {
          const R w = x * L / 2;
          R       s, c;
          sin_cos(w, s, c);
          return Complex<R>(c, s) * (L * sinc(w));
        }
        /// (g(x) - g(y)) / (x - y), g'(x) at x = y.
        Complex<R> dd(R x, R y) const
        {
          const R d = x - y;
          if (std::fabs(d) * L > R(0.1)) return (g(x) - g(y)) / d;
          const R    m = (x + y) / 2, lt = L * d / 2;
          Complex<R> J[8];
          const Complex<R> I(0, 1);
          if (d == R(0)) {
            jn_all(m * L, 1, J);
            return I * J[1] * (L * L);
          }
          // sum over odd n of L^{n+1} i^n J_n(mL) t^{n-1} / n!, t = d / 2
          jn_all(m * L, 7, J);
          const R    t2 = lt * lt;
          Complex<R> acc =
              J[1] - J[3] * (t2 / 6) + J[5] * (t2 * t2 / 120) - J[7] * (t2 * t2 * t2 / 5040);
          return I * acc * (L * L);
        }
    };

    /// F(x) = <(1 + beta d/h) exp(-ixd)> over d in [-h, h], faded out for
    /// |x h| in [z1, z2], and its divided differences.
    template <class R> struct AvgFn {
        R h, beta, z1, z2;

        static void j01(R z, R& j0, R& j1)
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
        void fade(R s, R& chi, R& dchi) const
        {
          dchi = 0;
          if (s <= z1) { chi = 1; return; }
          if (s >= z2) { chi = 0; return; }
          const R lr = std::log(z2 / z1), t = std::log(s / z1) / lr;
          chi        = R(1) - t * t * t * (R(10) - R(15) * t + R(6) * t * t);
          dchi       = -R(30) * t * t * (R(1) - t) * (R(1) - t) / (s * lr);
        }
        Complex<R> F(R x) const
        {
          const R z = x * h;
          R       chi, dchi, j0, j1;
          fade(std::fabs(z), chi, dchi);
          if (chi == R(0)) return Complex<R>(0, 0);
          j01(z, j0, j1);
          return Complex<R>(j0, -beta * j1) * chi;
        }
        /// dF/dx (used inside the fade region only).
        Complex<R> dF(R x) const
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
        /// (F(x) - F(y)) / (x - y), F'(x) at x = y.
        Complex<R> dd(R x, R y) const
        {
          const R d = x - y;
          if (std::fabs(d) * h > R(0.1)) return (F(x) - F(y)) / d;
          const R m = (x + y) / 2;
          if (std::fabs(m * h) + R(0.1) >= z1) return dF(m);  // |F| < 1/z1 here
          // F^(n)(m) = h^n (-i)^n [E_n + beta E_{n+1}](mh),
          // E_n(z) = (conj J_n(z) + (-1)^n J_n(z)) / 2; odd n: -i Im J_n,
          // even n: Re J_n.
          Complex<R> J[9];
          const int  nmax = d == R(0) ? 2 : 8;
          jn_all(m * h, nmax, J);
          auto G = [&](int n) {  // (-i)^n [E_n + beta E_{n+1}], n odd
            const R sg = (n % 4 == 1) ? R(-1) : R(1);  // (-i)^{n+1}
            return Complex<R>(sg * J[n].im, sg * beta * J[n + 1].re);
          };
          if (d == R(0)) return G(1) * h;
          const R t2 = (h * d / 2) * (h * d / 2);
          return (G(1) + G(3) * (t2 / 6) + G(5) * (t2 * t2 / 120) +
                  G(7) * (t2 * t2 * t2 / 5040)) *
                 h;
        }
    };

    /// Full hermitian matrix from the upper triangle (values).
    template <int N, class R> inline Mat<N, R> full(Mat<N, R> H)
    {
      hermitize_from_upper(H);
      return H;
    }

    /// Derivative direction k of a dual matrix (upper triangle), completed
    /// to a hermitian matrix.
    template <int N, class R, int K>
    inline Mat<N, R> dual_dir(const Mat<N, Dual<R, K>>& HD, int k)
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

    template <int N, class R> inline Mat<N, R> mul_ah(const Mat<N, R>& A, const Mat<N, R>& B)
    {
      return matmul(adjoint(A), B);  // A^dag B
    }
    template <int N, class R>
    inline Mat<N, R> sandwich(const Mat<N, R>& V, const Mat<N, R>& X)  // V^dag X V
    {
      return matmul(adjoint(V), matmul(X, V));
    }
    template <int N, class R>
    inline Mat<N, R> unsandwich(const Mat<N, R>& V, const Mat<N, R>& X)  // V X V^dag
    {
      return matmul(V, matmul(X, adjoint(V)));
    }
    template <int N, class R> inline void add_to(Mat<N, R>& A, const Mat<N, R>& B)
    {
      for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) A(i, j) += B(i, j);
    }
    template <int N, class R> inline Mat<N, R> herm_part(const Mat<N, R>& A)
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
        Complex<R> gm[N][N], Gam[N][N], G1[N][N][N], G2[N][N][N];
    };

    /// Value-pass results of a sub-bin reused by the derivatives: K =
    /// W diag(mu) W^dag, B = S0 W, rho_a = W^dag e_a e_a^dag W, M_a = rho_a o F.
    template <int N, class R> struct KState {
        Mat<N, R> W, B, M[N], rho[N];
        R         mu[N];
    };

    /// Average over one sub-bin: out[a*N + b]. seg: scratch of size nseg
    /// (with want_grad, also filled with what subbin_grad needs).
    template <class Model>
    void subbin(const typename Model::Prepared& P, const Mat<Model::N, typename Model::Real>& A,
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
          // G1[i][l][j] = g[lam_i - lam_j, lam_l - lam_j] (symmetric in i, l);
          // G2[i][l][j] = g[lam_i - lam_l, lam_i - lam_j] = -conj G1[l][j][i]
          // since g(-x) = conj g(x).
          for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
              for (int l = 0; l <= i; l++)
                d.G1[i][l][j] = d.G1[l][i][j] =
                    g.dd(d.lam[i] - d.lam[j], d.lam[l] - d.lam[j]);
          for (int i = 0; i < N; i++)
            for (int l = 0; l < N; l++)
              for (int j = 0; j < N; j++) d.G2[i][l][j] = -conj(d.G1[l][j][i]);
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
    }

    /// Derivatives of a sub-bin average after subbin(..., want_grad = true):
    /// dout[(p*N + a)*N + b] for the parameters of the chunks, dA[p] being
    /// dA/dp.
    template <class Model>
    void subbin_grad(const std::vector<GradChunk<Model>>& chunks,
                     const Mat<Model::N, typename Model::Real>* dA,
                     const Segment<typename Model::Real>* path, int nseg, bool nubar,
                     typename Model::Real u0, const AvgFn<typename Model::Real>& fn,
                     const SegData<Model::N, typename Model::Real>* seg,
                     const KState<Model::N, typename Model::Real>& ks,
                     typename Model::Real* dout)
    {
      using R          = typename Model::Real;
      constexpr int N  = Model::N;
      constexpr int KD = grad_traits<Model>::K;
      using D          = Dual<R, KD>;
      const R E0       = R(1) / u0;

      // F1[n][l][m] = F[mu_n - mu_m, mu_l - mu_m], F2 = -conj F1[l][m][n] (as
      // G1, G2 in subbin(); F(-x) = conj F(x))
      Complex<R> F1[N][N][N], F2[N][N][N];
      for (int m = 0; m < N; m++)
        for (int n = 0; n < N; n++)
          for (int l = 0; l <= n; l++)
            F1[n][l][m] = F1[l][n][m] = fn.dd(ks.mu[n] - ks.mu[m], ks.mu[l] - ks.mu[m]);
      for (int n = 0; n < N; n++)
        for (int l = 0; l < N; l++)
          for (int m = 0; m < N; m++) F2[n][l][m] = -conj(F1[l][m][n]);

      for (const auto& c : chunks) {
        Mat<N, R> dT[KD], dK[KD];  // dS before the segment, dK so far
        for (int k = 0; k < KD; k++) dT[k] = dK[k] = Mat<N, R>::zero();
        for (int s = 0; s < nseg; s++) {
          const SegData<N, R>& d = seg[s];
          Mat<N, D>            HD;
          Model::hamiltonian(c.P.P, E0, nubar, seed_segment(c.P, path[s]), HD);
          for (int k = 0; k < c.count; k++) {
            // In the eigenbasis of H: dS_seg = Gamma o dH, and
            // dK_seg = dA o g + sum_l (dH_il A_lj G1_ilj - A_il dH_lj G2_ilj).
            const Mat<N, R> dHt = sandwich(d.V, dual_dir<N, R, KD>(HD, k));
            const Mat<N, R> dAt = sandwich(d.V, dA[c.offset + k]);
            Mat<N, R>       dSt, dKt;
            for (int i = 0; i < N; i++)
              for (int j = 0; j < N; j++) {
                dSt(i, j)      = d.Gam[i][j] * dHt(i, j);
                Complex<R> acc = dAt(i, j) * d.gm[i][j];
                for (int l = 0; l < N; l++)
                  acc += dHt(i, l) * d.At(l, j) * d.G1[i][l][j] -
                         d.At(i, l) * dHt(l, j) * d.G2[i][l][j];
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
        for (int k = 0; k < c.count; k++) {
          // In the eigenbasis of K: dM_a = sum_l (dK_nl rho_lm F1_nlm -
          // rho_nl dK_lm F2_nlm); dPbar_ab = 2 Re (dB M_a B^dag)_bb +
          // (B dM_a B^dag)_bb with dB = dS0 W.
          const Mat<N, R> dKt = sandwich(ks.W, herm_part(dK[k]));
          const Mat<N, R> dB  = matmul(dT[k], ks.W);
          R*              o   = dout + size_t(c.offset + k) * N * N;
          for (int a = 0; a < N; a++) {
            Mat<N, R> dM;
            for (int n = 0; n < N; n++)
              for (int m = 0; m < N; m++) {
                Complex<R> acc(0, 0);
                for (int l = 0; l < N; l++)
                  acc += dKt(n, l) * ks.rho[a](l, m) * F1[n][l][m] -
                         ks.rho[a](n, l) * dKt(l, m) * F2[n][l][m];
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
    }

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

    /// Averages over explicit sub-bins: out[a][b][bin] (and
    /// dout[p][a][b][bin] with chunks), normalised by the sum of the
    /// sub-bin weights of each bin. pd: the model's dual prepared state of
    /// each chunk gives dA.
    template <class Model>
    void average(const typename Model::Prepared& P, const std::vector<GradChunk<Model>>* chunks,
                 int npar, const std::vector<AnalyticSubBin>& sub, size_t nbins,
                 const std::vector<Segment<typename Model::Real>>& path, bool nubar,
                 const AnalyticAvgOptions& opt, std::vector<typename Model::Real>& out,
                 std::vector<typename Model::Real>* dout)
    {
      using R         = typename Model::Real;
      constexpr int N = Model::N;
      static_assert(has_analytic_avg<Model>::value,
                    "analytic averages need a hermitian model affine in 1/E "
                    "(Model::analytic_avg)");
      if (path.empty()) throw std::invalid_argument("avg_path_analytic: empty path");
      for (const auto& s : sub)
        if (s.bin < 0 || size_t(s.bin) >= nbins || !(s.u0 > 0) || !(s.h >= 0))
          throw std::invalid_argument("avg_path_analytic: bad sub-bin");

      // A = dH/du: the vacuum Hamiltonian at E = 1 GeV (zero density).
      const Segment<R> vac{R(0), R(0), R(0.5), -1};
      Mat<N, R>        A;
      Model::hamiltonian(P, R(1), nubar, vac, A);
      A = full(A);
      std::vector<Mat<N, R>> dA(size_t(std::max(npar, 0)));
      if constexpr (grad_traits<Model>::enabled) {
        constexpr int KD = grad_traits<Model>::K;
        using D          = Dual<R, KD>;
        if (chunks)
          for (const auto& c : *chunks) {
            Mat<N, D>          AD;
            const SegmentZ<R, D> vz{R(0), D(R(0)), D(R(0.5)), -1};
            Model::hamiltonian(c.P.P, R(1), nubar, vz, AD);
            for (int k = 0; k < c.count; k++) dA[c.offset + k] = dual_dir<N, R, KD>(AD, k);
          }
      }

      const size_t   ns = sub.size(), nch = size_t(N) * N;
      const size_t   ng = chunks ? size_t(npar) * nch : 0;
      std::vector<R> sp(ns * nch), sg(ns * ng);
      const int      nseg = int(path.size());

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
          subbin<Model>(P, A, path.data(), nseg, nubar, R(s.u0), fn, &sp[size_t(i) * nch],
                        chunks != nullptr, seg.data(), ks);
          if constexpr (grad_traits<Model>::enabled)
            if (chunks)
              subbin_grad<Model>(*chunks, dA.data(), path.data(), nseg, nubar, R(s.u0), fn,
                                 seg.data(), ks, &sg[size_t(i) * ng]);
        }
      }

      // Fixed-order reduction (independent of the thread count).
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

  } // namespace analytic

} // namespace opg

#endif
