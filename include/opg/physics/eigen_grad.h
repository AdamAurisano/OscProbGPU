///////////////////////////////////////////////////////////////////////////////
/// \file eigen_grad.h
///
/// \brief Derivative of a segment evolution operator U = exp(-i H L) for a
///        hermitian H, from its eigen-decomposition (Daleckii-Krein).
///
/// With H = V diag(lam) V^dag and f(lam) = exp(-i lam L),
///
///   dU = V [ Gamma o (V^dag dH V) ] V^dag,
///   Gamma_ij = (f(lam_i) - f(lam_j)) / (lam_i - lam_j),  Gamma_ii = f'(lam_i)
///
/// (o is the element-wise product). The divided differences are evaluated
/// in a form that is accurate for any eigenvalue separation, including
/// exact degeneracies, so no derivative is lost or amplified at zero mixing
/// or near resonances. The result does not depend on the phase convention
/// of the eigenvectors.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_PHYSICS_EIGEN_GRAD_H
#define OPG_PHYSICS_EIGEN_GRAD_H

#include <cmath>

#include "opg/core/dual.h"
#include "opg/core/matrix.h"
#include "opg/physics/propagate.h"

namespace opg {

  /// sin and cos together (one call on the device).
  template <class R> OPG_HD OPG_INLINE void sin_cos(R x, R& s, R& c)
  {
#if OPG_ON_DEVICE
    sincos(x, &s, &c);
#else
    using std::cos;
    using std::sin;
    s = sin(x);
    c = cos(x);
#endif
  }

  /// Gamma_ij of the Daleckii-Krein formula for f(lam) = exp(-i lam L), and
  /// the phases f_i. Gamma is symmetric, so only i <= j is evaluated.
  template <int N, class R>
  OPG_HD OPG_INLINE void dk_gamma(const R lam[N], R L, Complex<R> G[N][N],
                                  Complex<R> f[N])
  {
    using std::fabs;
    OPG_UNROLL
    for (int i = 0; i < N; i++) {
      R sn, cs;
      sin_cos(lam[i] * L, sn, cs);
      f[i] = Complex<R>(cs, -sn);
    }
    OPG_UNROLL
    for (int i = 0; i < N; i++) {
      G[i][i] = f[i] * Complex<R>(R(0), -L);  // f'(lam_i) = -iL f_i
      OPG_UNROLL
      for (int j = i + 1; j < N; j++) {
        const R dl = lam[i] - lam[j];
        const R x  = dl * L;
        Complex<R> g;
        if (fabs(x) < R(1e-3)) {
          // f_j * (-iL) * phi(z), phi(z) = (e^z - 1)/z, z = -i x
          // phi = 1 + z/2 + z^2/6 + z^3/24 + z^4/120 (error ~ x^5/720)
          const R          x2 = x * x;
          const Complex<R> phi(R(1) - x2 / R(6) + x2 * x2 / R(120),
                               -x / R(2) + x * x2 / R(24));
          g = f[j] * Complex<R>(R(0), -L) * phi;
        }
        else {
          // f_i - f_j = f_j (e^{-ix} - 1), e^{-ix} - 1 = -2 sin(x/2) (sin(x/2), cos(x/2))
          R sh, ch;
          sin_cos(x / R(2), sh, ch);
          const Complex<R> em1(R(-2) * sh * sh, R(-2) * sh * ch);
          g = (f[j] * em1) / dl;
        }
        G[i][j] = g;
        G[j][i] = g;
      }
    }
  }

  /// One segment step with derivatives:
  ///   dS_k <- U dS_k + dU_k S,   S <- U S,
  /// where U = V diag(phi) V^dag, phi = exp(-i lam L), and dU_k follows from
  /// the derivative parts of HD (only the diagonal and upper triangle of HD
  /// are read). Evaluated as
  ///   dS_k <- V [ phi o (V^dag dS_k) + (Gamma o M_k) (V^dag S) ],
  ///   M_k   = V^dag dH_k V   (hermitian: only half is computed).
  /// The value update of S is exactly apply_eigen_step.
  template <int N, class R, int K>
  OPG_HD inline void eigen_step_grad(const Mat<N, R>& V, const R lam[N],
                                     R LengthIneV, const Mat<N, Dual<R, K>>& HD,
                                     Mat<N, R>& S, Mat<N, R> (&dS)[K])
  {
    Complex<R> G[N][N], phi[N];
    dk_gamma<N, R>(lam, LengthIneV, G, phi);

    // W = V^dag S (before the step)
    Mat<N, R> W;
    OPG_UNROLL
    for (int i = 0; i < N; i++)
      OPG_UNROLL
    for (int a = 0; a < N; a++) {
      Complex<R> acc(0, 0);
      OPG_UNROLL
      for (int l = 0; l < N; l++) acc += conj(V(l, i)) * S(l, a);
      W(i, a) = acc;
    }

    OPG_UNROLL
    for (int k = 0; k < K; k++) {
      // Gm = Gamma o (V^dag dH_k V), dH_k hermitian from its upper triangle
      Mat<N, R> Gm;
      OPG_UNROLL
      for (int j = 0; j < N; j++) {
        Complex<R> t[N];  // column j of dH_k V
        OPG_UNROLL
        for (int i = 0; i < N; i++) {
          Complex<R> acc(0, 0);
          OPG_UNROLL
          for (int l = 0; l < N; l++) {
            Complex<R> h;
            if (l == i)
              h = Complex<R>(HD(i, i).re.d[k], R(0));
            else if (l > i)
              h = Complex<R>(HD(i, l).re.d[k], HD(i, l).im.d[k]);
            else
              h = Complex<R>(HD(l, i).re.d[k], -HD(l, i).im.d[k]);
            acc += h * V(l, j);
          }
          t[i] = acc;
        }
        OPG_UNROLL
        for (int i = 0; i <= j; i++) {
          Complex<R> m(0, 0);
          OPG_UNROLL
          for (int l = 0; l < N; l++) m += conj(V(l, i)) * t[l];
          if (i == j) m.im = 0;
          Gm(i, j) = G[i][j] * m;
          if (i < j) Gm(j, i) = G[j][i] * conj(m);
        }
      }
      // dS_k = V [ phi o (V^dag dS_k) + Gm W ], one column at a time
      OPG_UNROLL
      for (int a = 0; a < N; a++) {
        Complex<R> y[N];
        OPG_UNROLL
        for (int i = 0; i < N; i++) {
          Complex<R> x(0, 0);
          OPG_UNROLL
          for (int l = 0; l < N; l++) x += conj(V(l, i)) * dS[k](l, a);
          Complex<R> yy = phi[i] * x;
          OPG_UNROLL
          for (int l = 0; l < N; l++) yy += Gm(i, l) * W(l, a);
          y[i] = yy;
        }
        OPG_UNROLL
        for (int i = 0; i < N; i++) {
          Complex<R> acc(0, 0);
          OPG_UNROLL
          for (int l = 0; l < N; l++) acc += V(i, l) * y[l];
          dS[k](i, a) = acc;
        }
      }
    }

    apply_eigen_step<N, R>(V, lam, LengthIneV, S);
  }

  /// Write dP(a -> b)/dp_k = 2 Re(conj(S_ba) dS_k,ba) for k < count to
  /// out[((k * N + a) * N + b) * stride].
  template <int N, class R, int K>
  OPG_HD OPG_INLINE void store_grads(const Mat<N, R>& S, const Mat<N, R> (&dS)[K],
                                     int count, R* out, size_t stride)
  {
    OPG_UNROLL
    for (int k = 0; k < K; k++) {
      if (k >= count) break;
      OPG_UNROLL
      for (int a = 0; a < N; a++)
        OPG_UNROLL
      for (int b = 0; b < N; b++) {
        const Complex<R>& s = S(b, a);
        const Complex<R>& d = dS[k](b, a);
        out[((size_t(k) * N + a) * N + b) * stride] = 2 * (s.re * d.re + s.im * d.im);
      }
    }
  }

  /// Sum_ab w_ab dP_ab/dp_k for k < K, with w read from w[(a*N+b)*stride].
  template <int N, class R, int K>
  OPG_HD OPG_INLINE void contract_grads(const Mat<N, R>& S, const Mat<N, R> (&dS)[K],
                                        const R* w, size_t stride, R (&acc)[K])
  {
    OPG_UNROLL
    for (int k = 0; k < K; k++) acc[k] = 0;
    OPG_UNROLL
    for (int a = 0; a < N; a++)
      OPG_UNROLL
    for (int b = 0; b < N; b++) {
      const R wab = w[(a * N + b) * stride];
      if (wab == R(0)) continue;
      const Complex<R>& s = S(b, a);
      OPG_UNROLL
      for (int k = 0; k < K; k++) {
        const Complex<R>& d = dS[k](b, a);
        acc[k] += wab * (2 * (s.re * d.re + s.im * d.im));
      }
    }
  }

} // namespace opg

#endif
