///////////////////////////////////////////////////////////////////////////////
/// \file eig3_general.h
///
/// \brief Eigen-decomposition A = X diag(lam) X^-1 of a general (not
///        necessarily hermitian) complex 3x3 matrix, for derivative
///        formulas. Not used for probabilities.
///
/// The matrix is scaled to O(1) and shifted to zero trace; the eigenvalues
/// are the roots of the characteristic cubic (Cardano, refined by Newton
/// steps) and the right eigenvectors the largest bilinear cross product of
/// two rows of (A - lam I). The decomposition reports failure when the
/// eigenvectors are (close to) linearly dependent, e.g. for degenerate or
/// defective matrices; callers then use another method.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_LINALG_EIG3_GENERAL_H
#define OPG_LINALG_EIG3_GENERAL_H

#include <cmath>

#include "opg/core/matrix.h"
#include "opg/physics/eigen_grad.h"  // sin_cos

namespace opg {

  namespace detail {

    /// Principal square root.
    template <class R> OPG_HD OPG_INLINE Complex<R> csqrt(const Complex<R>& z)
    {
      using std::fabs;
      using std::sqrt;
      const R r = abs(z);
      if (r == R(0)) return Complex<R>(0, 0);
      const R t = sqrt((r + fabs(z.re)) / 2);
      if (z.re >= R(0)) return Complex<R>(t, z.im / (2 * t));
      return Complex<R>(fabs(z.im) / (2 * t), z.im >= R(0) ? t : -t);
    }

    /// Principal cube root.
    template <class R> OPG_HD OPG_INLINE Complex<R> ccbrt(const Complex<R>& z)
    {
      using std::atan2;
      using std::cbrt;
      const R r = abs(z);
      if (r == R(0)) return Complex<R>(0, 0);
      R sn, cs;
      sin_cos(atan2(z.im, z.re) / R(3), sn, cs);
      const R m = cbrt(r);
      return Complex<R>(m * cs, m * sn);
    }

    /// Bilinear cross product a x b (no conjugation).
    template <class R>
    OPG_HD OPG_INLINE void ccross(const Complex<R> a[3], const Complex<R> b[3],
                                  Complex<R> c[3])
    {
      c[0] = a[1] * b[2] - a[2] * b[1];
      c[1] = a[2] * b[0] - a[0] * b[2];
      c[2] = a[0] * b[1] - a[1] * b[0];
    }

  } // namespace detail

  /// A = X diag(lam) X^-1. Returns false (outputs unusable) if the
  /// eigenvector matrix is too ill-conditioned (|det X| < min_det with unit
  /// columns).
  template <class R>
  OPG_HD inline bool eig3_general(const Mat<3, R>& A, Mat<3, R>& X, Mat<3, R>& Xinv,
                                  Complex<R> lam[3], R min_det = R(1e-4))
  {
    using C = Complex<R>;
    using std::fmax;
    using std::sqrt;

    // scale to O(1) and remove the trace
    R sc = 0;
    OPG_UNROLL
    for (int i = 0; i < 3; i++)
      OPG_UNROLL
    for (int j = 0; j < 3; j++) sc = fmax(sc, abs(A(i, j)));
    if (sc == R(0)) return false;
    const R inv = R(1) / sc;
    Mat<3, R> B;
    OPG_UNROLL
    for (int i = 0; i < 3; i++)
      OPG_UNROLL
    for (int j = 0; j < 3; j++) B(i, j) = A(i, j) * inv;
    const C t = (B(0, 0) + B(1, 1) + B(2, 2)) * (R(1) / R(3));
    OPG_UNROLL
    for (int i = 0; i < 3; i++) B(i, i) -= t;

    // characteristic polynomial mu^3 + p mu + q
    const C m00 = B(1, 1) * B(2, 2) - B(1, 2) * B(2, 1);
    const C m11 = B(0, 0) * B(2, 2) - B(0, 2) * B(2, 0);
    const C m22 = B(0, 0) * B(1, 1) - B(0, 1) * B(1, 0);
    const C p   = m00 + m11 + m22;
    const C det = B(0, 0) * m00 - B(0, 1) * (B(1, 0) * B(2, 2) - B(1, 2) * B(2, 0)) +
                  B(0, 2) * (B(1, 0) * B(2, 1) - B(1, 1) * B(2, 0));
    const C q = -det;

    // Cardano
    const C hq = q * R(0.5);
    const C p3 = p * (R(1) / R(3));
    const C s  = detail::csqrt(hq * hq + p3 * p3 * p3);
    const C w1 = s - hq, w2 = -hq - s;
    const C w  = abs(w1) >= abs(w2) ? w1 : w2;
    if (abs(w) == R(0)) return false;  // triple root
    const C u = detail::ccbrt(w);
    const C v = -p3 / u;
    const C om(R(-0.5), R(0.86602540378443864676));
    const C om2 = conj(om);
    C       mu[3] = {u + v, om * u + om2 * v, om2 * u + om * v};

    // Newton refinement
    OPG_UNROLL
    for (int k = 0; k < 3; k++)
      OPG_UNROLL
    for (int it = 0; it < 2; it++) {
      const C f  = (mu[k] * mu[k] + p) * mu[k] + q;
      const C fp = mu[k] * mu[k] * R(3) + p;
      if (abs(fp) > R(1e-8)) mu[k] -= f / fp;
    }

    // eigenvectors: largest cross product of two rows of (B - mu I)
    OPG_UNROLL
    for (int k = 0; k < 3; k++) {
      C r[3][3];
      OPG_UNROLL
      for (int i = 0; i < 3; i++)
        OPG_UNROLL
      for (int j = 0; j < 3; j++) r[i][j] = i == j ? B(i, j) - mu[k] : B(i, j);
      C c01[3], c02[3], c12[3];
      detail::ccross(r[0], r[1], c01);
      detail::ccross(r[0], r[2], c02);
      detail::ccross(r[1], r[2], c12);
      const R n01 = norm(c01[0]) + norm(c01[1]) + norm(c01[2]);
      const R n02 = norm(c02[0]) + norm(c02[1]) + norm(c02[2]);
      const R n12 = norm(c12[0]) + norm(c12[1]) + norm(c12[2]);
      R       nb  = n01;
      C       x[3] = {c01[0], c01[1], c01[2]};
      if (n02 > nb) {
        nb = n02;
        OPG_UNROLL
        for (int i = 0; i < 3; i++) x[i] = c02[i];
      }
      if (n12 > nb) {
        nb = n12;
        OPG_UNROLL
        for (int i = 0; i < 3; i++) x[i] = c12[i];
      }
      if (!(nb > R(0))) return false;
      const R in = R(1) / sqrt(nb);
      OPG_UNROLL
      for (int i = 0; i < 3; i++) X(i, k) = x[i] * in;
      lam[k] = (mu[k] + t) * sc;
    }

    // inverse by cofactors
    const C c00 = X(1, 1) * X(2, 2) - X(1, 2) * X(2, 1);
    const C c01 = X(1, 2) * X(2, 0) - X(1, 0) * X(2, 2);
    const C c02 = X(1, 0) * X(2, 1) - X(1, 1) * X(2, 0);
    const C dX  = X(0, 0) * c00 + X(0, 1) * c01 + X(0, 2) * c02;
    if (!(abs(dX) >= min_det)) return false;
    const C id = C(1, 0) / dX;
    Xinv(0, 0) = c00 * id;
    Xinv(1, 0) = c01 * id;
    Xinv(2, 0) = c02 * id;
    Xinv(0, 1) = (X(0, 2) * X(2, 1) - X(0, 1) * X(2, 2)) * id;
    Xinv(1, 1) = (X(0, 0) * X(2, 2) - X(0, 2) * X(2, 0)) * id;
    Xinv(2, 1) = (X(0, 1) * X(2, 0) - X(0, 0) * X(2, 1)) * id;
    Xinv(0, 2) = (X(0, 1) * X(1, 2) - X(0, 2) * X(1, 1)) * id;
    Xinv(1, 2) = (X(0, 2) * X(1, 0) - X(0, 0) * X(1, 2)) * id;
    Xinv(2, 2) = (X(0, 0) * X(1, 1) - X(0, 1) * X(1, 0)) * id;
    return true;
  }

} // namespace opg

#endif
