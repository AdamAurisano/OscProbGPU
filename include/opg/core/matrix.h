///////////////////////////////////////////////////////////////////////////////
/// \file matrix.h
///
/// \brief Fixed-size complex matrices and vectors for host and device.
///
/// These are plain aggregates (trivially copyable) so they can be passed by
/// value as CUDA kernel arguments and live in registers.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_CORE_MATRIX_H
#define OPG_CORE_MATRIX_H

#include "opg/core/complex.h"

namespace opg {

  template <int N, class Real> struct Vec {
      Complex<Real> v[N];

      OPG_HD OPG_INLINE Complex<Real>&       operator[](int i) { return v[i]; }
      OPG_HD OPG_INLINE const Complex<Real>& operator[](int i) const
      {
        return v[i];
      }
  };

  template <int N, class Real> struct Mat {
      Complex<Real> m[N][N];

      OPG_HD OPG_INLINE Complex<Real>& operator()(int i, int j) { return m[i][j]; }
      OPG_HD OPG_INLINE const Complex<Real>& operator()(int i, int j) const
      {
        return m[i][j];
      }

      OPG_HD OPG_INLINE static Mat zero()
      {
        Mat a;
        OPG_UNROLL
        for (int i = 0; i < N; i++)
          OPG_UNROLL
        for (int j = 0; j < N; j++) a.m[i][j] = Complex<Real>(0, 0);
        return a;
      }

      OPG_HD OPG_INLINE static Mat identity()
      {
        Mat a = zero();
        OPG_UNROLL
        for (int i = 0; i < N; i++) a.m[i][i] = Complex<Real>(1, 0);
        return a;
      }
  };

  /// C = A * B
  template <int N, class R>
  OPG_HD OPG_INLINE Mat<N, R> matmul(const Mat<N, R>& A, const Mat<N, R>& B)
  {
    Mat<N, R> C;
    OPG_UNROLL
    for (int i = 0; i < N; i++) {
      OPG_UNROLL
      for (int j = 0; j < N; j++) {
        Complex<R> s(0, 0);
        OPG_UNROLL
        for (int k = 0; k < N; k++) s += A.m[i][k] * B.m[k][j];
        C.m[i][j] = s;
      }
    }
    return C;
  }

  /// Conjugate transpose
  template <int N, class R>
  OPG_HD OPG_INLINE Mat<N, R> adjoint(const Mat<N, R>& A)
  {
    Mat<N, R> C;
    OPG_UNROLL
    for (int i = 0; i < N; i++)
      OPG_UNROLL
    for (int j = 0; j < N; j++) C.m[i][j] = conj(A.m[j][i]);
    return C;
  }

  /// Fill the lower triangle from the upper triangle (Hermitian completion).
  template <int N, class R> OPG_HD OPG_INLINE void hermitize_from_upper(Mat<N, R>& A)
  {
    OPG_UNROLL
    for (int i = 0; i < N; i++) {
      A.m[i][i].im = 0;
      OPG_UNROLL
      for (int j = i + 1; j < N; j++) A.m[j][i] = conj(A.m[i][j]);
    }
  }

} // namespace opg

#endif
