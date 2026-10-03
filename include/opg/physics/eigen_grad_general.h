///////////////////////////////////////////////////////////////////////////////
/// \file eigen_grad_general.h
///
/// \brief Derivative of a segment evolution operator U = exp(-i H L) for a
///        non-hermitian, diagonalisable H (Daleckii-Krein with complex
///        eigenvalues).
///
/// With H = X diag(lam) X^-1 and f(lam) = exp(-i lam L),
///
///   dU = X [ Gamma o (X^-1 dH X) ] X^-1,
///   Gamma_ij = (f(lam_i) - f(lam_j)) / (lam_i - lam_j),  Gamma_ii = f'(lam_i),
///
/// evaluated stably for any eigenvalue separation. The eigensystem's
/// quality is checked by reconstructing U and comparing it with the value
/// operator computed independently (Pade); callers fall back to another
/// method when the check fails.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_PHYSICS_EIGEN_GRAD_GENERAL_H
#define OPG_PHYSICS_EIGEN_GRAD_GENERAL_H

#include <cmath>

#include "opg/core/dual.h"
#include "opg/core/matrix.h"
#include "opg/linalg/eig3_general.h"
#include "opg/physics/eigen_grad.h"

namespace opg {

  /// exp(z) - 1, accurate for small |z|.
  template <class R> OPG_HD OPG_INLINE Complex<R> cexpm1(const Complex<R>& z)
  {
    using std::exp;
    using std::expm1;
    R sy, cy, sh, ch;
    sin_cos(z.im, sy, cy);
    sin_cos(z.im / R(2), sh, ch);
    return Complex<R>(expm1(z.re) * cy - R(2) * sh * sh, exp(z.re) * sy);
  }

  /// Gamma_ij and f_i for f(lam) = exp(-i lam L), complex eigenvalues
  /// (Im lam <= 0 for decaying states, so |f_i| <= 1).
  template <int N, class R>
  OPG_HD OPG_INLINE void dk_gamma_complex(const Complex<R> lam[N], R L,
                                          Complex<R> G[N][N], Complex<R> f[N])
  {
    using C = Complex<R>;
    const C mil(R(0), -L);
    OPG_UNROLL
    for (int i = 0; i < N; i++) {
      // f = exp(-i lam L) = exp(Im(lam) L) * exp(-i Re(lam) L)
      using std::exp;
      R sn, cs;
      sin_cos(lam[i].re * L, sn, cs);
      const R m = exp(lam[i].im * L);
      f[i]      = C(m * cs, -m * sn);
    }
    OPG_UNROLL
    for (int i = 0; i < N; i++) {
      G[i][i] = f[i] * mil;
      OPG_UNROLL
      for (int j = i + 1; j < N; j++) {
        // (f_a - f_b) / (lam_a - lam_b) = f_b (e^z - 1) / dl, z = -i dl L,
        // with the roles chosen so that Re z <= 0 (no overflow).
        C dl = lam[i] - lam[j];
        C fb = f[j];
        if (dl.im > R(0)) {
          dl = -dl;
          fb = f[i];
        }
        const C z = dl * mil;
        C       g;
        if (abs(z) < R(1e-3)) {
          // phi(z) = (e^z - 1)/z = 1 + z/2 + z^2/6 + z^3/24 + z^4/120
          const C phi = C(R(1), R(0)) +
                        z * (C(R(0.5), R(0)) +
                             z * (C(R(1) / R(6), R(0)) +
                                  z * (C(R(1) / R(24), R(0)) + z * (R(1) / R(120)))));
          g = fb * mil * phi;
        }
        else
          g = fb * cexpm1(z) / dl;
        G[i][j] = g;
        G[j][i] = g;
      }
    }
  }

  /// One segment step with derivatives for a non-hermitian 3x3 H:
  ///   dS_k <- U dS_k + dU_k S,   (S itself is not modified)
  /// where U is the value operator (computed by the caller) and dU_k follow
  /// from the derivative parts of HD (full matrix). Returns false, leaving
  /// dS untouched, if the eigensystem of H is not accurate enough to
  /// reproduce U to tol.
  template <class R, int K>
  OPG_HD inline bool general_eigen_step_grad(const Mat<3, R>& H, R L,
                                             const Mat<3, Dual<R, K>>& HD,
                                             const Mat<3, R>& U, const Mat<3, R>& S,
                                             Mat<3, R> (&dS)[K], R tol = R(1e-11))
  {
    using C = Complex<R>;
    Mat<3, R> X, Xi;
    C         lam[3];
    if (!eig3_general<R>(H, X, Xi, lam)) return false;

    C G[3][3], f[3];
    dk_gamma_complex<3, R>(lam, L, G, f);

    // check: X diag(f) X^-1 == U
    R err = 0;
    OPG_UNROLL
    for (int i = 0; i < 3; i++)
      OPG_UNROLL
    for (int j = 0; j < 3; j++) {
      C acc(0, 0);
      OPG_UNROLL
      for (int l = 0; l < 3; l++) acc += X(i, l) * f[l] * Xi(l, j);
      using std::fmax;
      err = fmax(err, abs(acc - U(i, j)));
    }
    if (!(err <= tol)) return false;

    // W = X^-1 S
    const Mat<3, R> W = matmul(Xi, S);
    OPG_UNROLL
    for (int k = 0; k < K; k++) {
      Mat<3, R> dH;
      OPG_UNROLL
      for (int i = 0; i < 3; i++)
        OPG_UNROLL
      for (int j = 0; j < 3; j++) dH(i, j) = C(HD(i, j).re.d[k], HD(i, j).im.d[k]);
      // Gm = Gamma o (X^-1 dH X)
      Mat<3, R> Gm = matmul(Xi, matmul(dH, X));
      OPG_UNROLL
      for (int i = 0; i < 3; i++)
        OPG_UNROLL
      for (int j = 0; j < 3; j++) Gm(i, j) = G[i][j] * Gm(i, j);
      // dS_k = U dS_k + X (Gm W)
      const Mat<3, R> T = matmul(X, matmul(Gm, W));
      Mat<3, R>       N = matmul(U, dS[k]);
      OPG_UNROLL
      for (int i = 0; i < 3; i++)
        OPG_UNROLL
      for (int a = 0; a < 3; a++) dS[k](i, a) = N(i, a) + T(i, a);
    }
    return true;
  }

} // namespace opg

#endif
