// ----------------------------------------------------------------------------
// Numerical diagonalization of 3x3 hermitian matrices
// Copyright (C) 2006  Joachim Kopp
// ----------------------------------------------------------------------------
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
// ----------------------------------------------------------------------------
//
// Port of zheevc3 / zhetrd3 / zheevq3 / zheevh3 (as shipped with OscProb in
// MatrixDecomp/) to header-only templates usable on host and CUDA devices.
// The sequence of floating point operations is kept identical to the
// original so that, on the host, results agree bit-for-bit (with the
// possible exception of the complex division in the QL fallback, where the
// compiler runtime's complex division algorithm may differ in the last ulp).
//
// Changes from the original:
//  * templated on the real type, operating on opg::Mat<3,Real>;
//  * A is passed by const reference; Q and w are outputs;
//  * DBL_EPSILON is replaced by the epsilon of the real type.
// ----------------------------------------------------------------------------

#ifndef OPG_LINALG_KOPP_ZHEEVH3_H
#define OPG_LINALG_KOPP_ZHEEVH3_H

#include <cmath>

#include "opg/core/matrix.h"

namespace opg {
  namespace kopp {

    template <class R> struct Eps;
    template <> struct Eps<double> {
        static constexpr double value = 2.2204460492503131e-16;
    };
    template <> struct Eps<float> {
        static constexpr float value = 1.19209290e-07f;
    };
    template <> struct Eps<long double> {
        static constexpr long double value = 1.0842021724855044340e-19L;
    };

    template <class R> OPG_HD OPG_INLINE R sqr(R x) { return x * x; }
    template <class R> OPG_HD OPG_INLINE R sqr_abs(const Complex<R>& x)
    {
      return sqr(x.re) + sqr(x.im);
    }

    // ------------------------------------------------------------------------
    /// Eigenvalues of a hermitian 3x3 matrix by Cardano's method.
    /// Only the diagonal and upper triangle of A are accessed.
    // ------------------------------------------------------------------------
    template <class R> OPG_HD inline int zheevc3(const Mat<3, R>& A, R w[3])
    {
      using C = Complex<R>;
      const R M_SQRT3_ = R(1.73205080756887729352744634151);

      R m, c1, c0;

      C de = A(0, 1) * A(1, 2);  // d * e
      R dd = sqr_abs(A(0, 1));   // d * conj(d)
      R ee = sqr_abs(A(1, 2));   // e * conj(e)
      R ff = sqr_abs(A(0, 2));   // f * conj(f)
      m    = A(0, 0).re + A(1, 1).re + A(2, 2).re;
      c1   = (A(0, 0).re * A(1, 1).re + A(0, 0).re * A(2, 2).re +
            A(1, 1).re * A(2, 2).re) -
           (dd + ee + ff);
      c0 = A(2, 2).re * dd + A(0, 0).re * ee + A(1, 1).re * ff -
           A(0, 0).re * A(1, 1).re * A(2, 2).re -
           R(2.0) * (A(0, 2).re * de.re + A(0, 2).im * de.im);

      R p, sqrt_p, q, c, s, phi;
      p      = sqr(m) - R(3.0) * c1;
      q      = m * (p - (R(3.0) / R(2.0)) * c1) - (R(27.0) / R(2.0)) * c0;
      sqrt_p = std::sqrt(std::fabs(p));

      phi = R(27.0) * (R(0.25) * sqr(c1) * (p - c1) +
                       c0 * (q + R(27.0) / R(4.0) * c0));
      phi = (R(1.0) / R(3.0)) * std::atan2(std::sqrt(std::fabs(phi)), q);

      c = sqrt_p * std::cos(phi);
      s = (R(1.0) / M_SQRT3_) * sqrt_p * std::sin(phi);

      w[1] = (R(1.0) / R(3.0)) * (m - c);
      w[2] = w[1] + s;
      w[0] = w[1] + c;
      w[1] -= s;

      return 0;
    }

    // ------------------------------------------------------------------------
    /// Householder reduction of a hermitian 3x3 matrix to real tridiagonal
    /// form: A = Q . T . Q^H with T = tridiag(e, d, e).
    // ------------------------------------------------------------------------
    template <class R>
    OPG_HD inline void zhetrd3(const Mat<3, R>& A, Mat<3, R>& Q, R d[3], R e[2])
    {
      using C     = Complex<R>;
      const int n = 3;
      C         u[n], q[n];
      C         omega, f;
      R         K, h, g;

      for (int i = 0; i < n; i++) {
        Q(i, i) = C(1.0);
        for (int j = 0; j < i; j++) Q(i, j) = Q(j, i) = C(0.0);
      }

      // Bring first row and column to the desired form
      h = sqr_abs(A(0, 1)) + sqr_abs(A(0, 2));
      if (A(0, 1).re > 0)
        g = -std::sqrt(h);
      else
        g = std::sqrt(h);
      e[0] = g;
      f    = g * A(0, 1);
      u[1] = conj(A(0, 1)) - C(g);
      u[2] = conj(A(0, 2));

      omega = C(h) - f;
      if (omega.re > R(0.0)) {
        omega = (R(0.5) * (C(1.0) + conj(omega) / omega)) / omega.re;
        K     = 0.0;
        for (int i = 1; i < n; i++) {
          f    = conj(A(1, i)) * u[1] + A(i, 2) * u[2];
          q[i] = omega * f;          // p
          K += (conj(u[i]) * f).re;  // u* A u
        }
        K *= R(0.5) * sqr_abs(omega);

        for (int i = 1; i < n; i++) q[i] = q[i] - K * u[i];

        d[0] = A(0, 0).re;
        d[1] = A(1, 1).re - R(2.0) * (q[1] * conj(u[1])).re;
        d[2] = A(2, 2).re - R(2.0) * (q[2] * conj(u[2])).re;

        // Store inverse Householder transformation in Q
        for (int j = 1; j < n; j++) {
          f = omega * conj(u[j]);
          for (int i = 1; i < n; i++) Q(i, j) = Q(i, j) - f * u[i];
        }

        // Calculate updated A[1][2] and store it in f
        f = A(1, 2) - q[1] * conj(u[2]) - u[1] * conj(q[2]);
      }
      else {
        for (int i = 0; i < n; i++) d[i] = A(i, i).re;
        f = A(1, 2);
      }

      // Make (23) element real
      e[1] = abs(f);
      if (e[1] != R(0.0)) {
        f = conj(f) / e[1];
        for (int i = 1; i < n; i++) Q(i, n - 1) = Q(i, n - 1) * f;
      }
    }

    // ------------------------------------------------------------------------
    /// Eigensystem of a hermitian 3x3 matrix by the QL algorithm with
    /// implicit shifts, preceded by Householder tridiagonalisation.
    /// Returns -1 on non-convergence.
    // ------------------------------------------------------------------------
    template <class R>
    OPG_HD inline int zheevq3(const Mat<3, R>& A, Mat<3, R>& Q, R w[3])
    {
      using C     = Complex<R>;
      const int n = 3;
      R         e[3];  // The third element is used only as temporary workspace
      R         g, r, p, f, b, s, c;
      C         t;
      int       nIter;
      int       m;

      zhetrd3(A, Q, w, e);

      for (int l = 0; l < n - 1; l++) {
        nIter = 0;
        while (1) {
          for (m = l; m <= n - 2; m++) {
            g = std::fabs(w[m]) + std::fabs(w[m + 1]);
            if (std::fabs(e[m]) + g == g) break;
          }
          if (m == l) break;

          if (nIter++ >= 30) return -1;

          g = (w[l + 1] - w[l]) / (e[l] + e[l]);
          r = std::sqrt(sqr(g) + R(1.0));
          if (g > 0)
            g = w[m] - w[l] + e[l] / (g + r);
          else
            g = w[m] - w[l] + e[l] / (g - r);

          s = c = R(1.0);
          p     = R(0.0);
          for (int i = m - 1; i >= l; i--) {
            f = s * e[i];
            b = c * e[i];
            if (std::fabs(f) > std::fabs(g)) {
              c        = g / f;
              r        = std::sqrt(sqr(c) + R(1.0));
              e[i + 1] = f * r;
              c *= (s = R(1.0) / r);
            }
            else {
              s        = f / g;
              r        = std::sqrt(sqr(s) + R(1.0));
              e[i + 1] = g * r;
              s *= (c = R(1.0) / r);
            }

            g        = w[i + 1] - p;
            r        = (w[i] - g) * s + R(2.0) * c * b;
            p        = s * r;
            w[i + 1] = g + p;
            g        = c * r - b;

            for (int k = 0; k < n; k++) {
              t           = Q(k, i + 1);
              Q(k, i + 1) = s * Q(k, i) + c * t;
              Q(k, i)     = c * Q(k, i) - s * t;
            }
          }
          w[l] -= p;
          e[l] = g;
          e[m] = R(0.0);
        }
      }

      return 0;
    }

    // ------------------------------------------------------------------------
    /// Eigenvalues and normalized eigenvectors (columns of Q) of a hermitian
    /// 3x3 matrix, using Cardano's method and vector cross products, with a
    /// fallback to QL when large round-off errors are expected.
    /// Only the diagonal and upper triangle of A are accessed.
    // ------------------------------------------------------------------------
    template <class R>
    OPG_HD inline int zheevh3(const Mat<3, R>& A, Mat<3, R>& Q, R w[3])
    {
      using C = Complex<R>;
      R norm;   // Squared norm or inverse norm of current eigenvector
      R error;  // Estimated maximum roundoff error
      R t, u;   // Intermediate storage
      int j;

      zheevc3(A, w);

      t = std::fabs(w[0]);
      if ((u = std::fabs(w[1])) > t) t = u;
      if ((u = std::fabs(w[2])) > t) t = u;
      if (t < R(1.0))
        u = t;
      else
        u = sqr(t);
      error = R(256.0) * Eps<R>::value * sqr(u);

      Q(0, 1) = A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1).re;
      Q(1, 1) = A(0, 2) * conj(A(0, 1)) - A(1, 2) * A(0, 0).re;
      Q(2, 1) = C(sqr_abs(A(0, 1)));

      // v[0] = conj( (A - w[0]).e1 x (A - w[0]).e2 )
      Q(0, 0) = Q(0, 1) + A(0, 2) * w[0];
      Q(1, 0) = Q(1, 1) + A(1, 2) * w[0];
      Q(2, 0) = C((A(0, 0).re - w[0]) * (A(1, 1).re - w[0])) - Q(2, 1);
      norm    = sqr_abs(Q(0, 0)) + sqr_abs(Q(1, 0)) + sqr(Q(2, 0).re);

      if (norm <= error)
        return zheevq3(A, Q, w);
      else {
        norm = std::sqrt(R(1.0) / norm);
        for (j = 0; j < 3; j++) Q(j, 0) = Q(j, 0) * norm;
      }

      // v[1] = conj( (A - w[1]).e1 x (A - w[1]).e2 )
      Q(0, 1) = Q(0, 1) + A(0, 2) * w[1];
      Q(1, 1) = Q(1, 1) + A(1, 2) * w[1];
      Q(2, 1) = C((A(0, 0).re - w[1]) * (A(1, 1).re - w[1]) - Q(2, 1).re);
      norm    = sqr_abs(Q(0, 1)) + sqr_abs(Q(1, 1)) + sqr(Q(2, 1).re);
      if (norm <= error)
        return zheevq3(A, Q, w);
      else {
        norm = std::sqrt(R(1.0) / norm);
        for (j = 0; j < 3; j++) Q(j, 1) = Q(j, 1) * norm;
      }

      // v[2] = conj(v[0] x v[1])
      Q(0, 2) = conj(Q(1, 0) * Q(2, 1) - Q(2, 0) * Q(1, 1));
      Q(1, 2) = conj(Q(2, 0) * Q(0, 1) - Q(0, 0) * Q(2, 1));
      Q(2, 2) = conj(Q(0, 0) * Q(1, 1) - Q(1, 0) * Q(0, 1));

      return 0;
    }

  } // namespace kopp
} // namespace opg

#endif
