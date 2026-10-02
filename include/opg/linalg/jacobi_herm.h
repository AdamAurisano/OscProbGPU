///////////////////////////////////////////////////////////////////////////////
/// \file jacobi_herm.h
///
/// \brief Cyclic Jacobi eigensolver for small complex hermitian matrices,
///        usable on host and device.
///
/// Each rotation first removes the phase of A(p,q) with a diagonal unitary
/// and then applies the real symmetric Jacobi rotation of Numerical
/// Recipes, so that the combined J = D R satisfies (J^dag A J)(p,q) = 0.
/// Jacobi is slower than tridiagonal QR for large N, but for N = 4 it is
/// short, branch-light, register-resident and accurate to a few ulp
/// relative to the matrix norm.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_LINALG_JACOBI_HERM_H
#define OPG_LINALG_JACOBI_HERM_H

#include <cmath>

#include "opg/core/matrix.h"
#include "opg/linalg/kopp/zheevh3.h"  // kopp::Eps

namespace opg {

  /// Diagonalise the hermitian matrix A (only the diagonal and the upper
  /// triangle are read). On output V holds the eigenvectors as columns and
  /// w the eigenvalues (unsorted). Returns the number of sweeps used, or -1
  /// if not converged within max_sweeps.
  template <int N, class R>
  OPG_HD inline int jacobi_hermitian(Mat<N, R> A, Mat<N, R>& V, R w[N],
                                     int max_sweeps = 20)
  {
    hermitize_from_upper(A);
    V = Mat<N, R>::identity();

    // Convergence threshold relative to the Frobenius norm.
    R fro2 = 0;
    OPG_UNROLL
    for (int i = 0; i < N; i++)
      OPG_UNROLL
    for (int j = 0; j < N; j++) fro2 += norm(A(i, j));
    const R eps  = kopp::Eps<R>::value;
    const R tol2 = eps * eps * fro2 * R(1e-2);

    int sweep = 0;
    for (; sweep < max_sweeps; sweep++) {
      R off2 = 0;
      OPG_UNROLL
      for (int p = 0; p < N; p++)
        OPG_UNROLL
      for (int q = p + 1; q < N; q++) off2 += norm(A(p, q));
      if (off2 <= tol2) break;

      // All loops below are fully unrolled so that every matrix index is a
      // compile-time constant and A, V stay in registers on the GPU.
      OPG_UNROLL
      for (int p = 0; p < N - 1; p++) {
        OPG_UNROLL
        for (int q = p + 1; q < N; q++) {
          const Complex<R> apq = A(p, q);
          const R          r   = abs(apq);
          if (r == R(0)) continue;

          // e = exp(i phi), phi = arg A(p,q)
          const Complex<R> e = apq / r;

          // Real Jacobi rotation (Numerical Recipes, jacobi)
          const R theta = (A(q, q).re - A(p, p).re) / (2 * r);
          R       t;
          if (std::fabs(theta) > R(1e150))
            t = R(0.5) / theta;
          else
            t = (theta >= 0 ? R(1) : R(-1)) /
                (std::fabs(theta) + std::sqrt(theta * theta + 1));
          const R c = R(1) / std::sqrt(t * t + 1);
          const R s = t * c;

          // J restricted to (p,q) is [[c, s], [-s conj(e), c conj(e)]].
          // Updates use the numerically stable form of Numerical Recipes
          // (tau = s / (1 + c)), applied after removing the phase of
          // column q: b_kq = a_kq conj(e).
          const R tau = s / (1 + c);

          A(p, p).re -= t * r;
          A(q, q).re += t * r;
          A(p, p).im = 0;
          A(q, q).im = 0;
          A(p, q)    = Complex<R>(0, 0);
          A(q, p)    = Complex<R>(0, 0);

          OPG_UNROLL
          for (int k = 0; k < N; k++) {
            if (k == p || k == q) continue;
            const Complex<R> g = A(k, p);
            const Complex<R> h = A(k, q) * conj(e);
            A(k, p)            = g - (h + g * tau) * s;
            A(k, q)            = h + (g - h * tau) * s;
            A(p, k)            = conj(A(k, p));
            A(q, k)            = conj(A(k, q));
          }

          // V <- V J
          OPG_UNROLL
          for (int k = 0; k < N; k++) {
            const Complex<R> g = V(k, p);
            const Complex<R> h = V(k, q) * conj(e);
            V(k, p)            = g - (h + g * tau) * s;
            V(k, q)            = h + (g - h * tau) * s;
          }
        }
      }
    }

    OPG_UNROLL
    for (int i = 0; i < N; i++) w[i] = A(i, i).re;
    return sweep < max_sweeps ? sweep : -1;
  }

} // namespace opg

#endif
