///////////////////////////////////////////////////////////////////////////////
/// \file expm.h
///
/// \brief Matrix exponential of small complex matrices on host and device.
///
/// Scaling and squaring with Pade approximants (N. J. Higham, "The scaling
/// and squaring method for the matrix exponential revisited", SIAM J.
/// Matrix Anal. Appl. 26 (2005) 1179), following the same choices as
/// Eigen's MatrixExponential (used by OscProb's PMNS_Decay): degree 3, 5,
/// 7, 9 or 13 selected from the 1-norm, the same thresholds, and the
/// rational approximant evaluated with an LU solve with partial pivoting.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_LINALG_EXPM_H
#define OPG_LINALG_EXPM_H

#include <cmath>

#include "opg/core/matrix.h"

namespace opg {

  namespace detail {

    /// sum_k c[k] * M[k] + c0 * I, with real coefficients.
    template <int N, class R, int K>
    OPG_HD OPG_INLINE Mat<N, R> lincomb(const R (&c)[K], const Mat<N, R>* (&M)[K],
                                        R c0)
    {
      Mat<N, R> out;
      OPG_UNROLL
      for (int i = 0; i < N; i++)
        OPG_UNROLL
      for (int j = 0; j < N; j++) {
        Complex<R> s(0, 0);
        OPG_UNROLL
        for (int k = 0; k < K; k++) s += (*M[k])(i, j) * c[k];
        if (i == j) s.re += c0;
        out(i, j) = s;
      }
      return out;
    }

    /// Solve D X = B for X (N x N right-hand side) by LU with partial
    /// pivoting. D and B are overwritten.
    template <int N, class R>
    OPG_HD inline void lu_solve(Mat<N, R>& D, Mat<N, R>& B)
    {
      for (int k = 0; k < N; k++) {
        // pivot
        int p    = k;
        R   best = abs(D(k, k));
        for (int i = k + 1; i < N; i++) {
          R v = abs(D(i, k));
          if (v > best) {
            best = v;
            p    = i;
          }
        }
        if (p != k) {
          for (int j = 0; j < N; j++) {
            Complex<R> t = D(k, j);
            D(k, j)      = D(p, j);
            D(p, j)      = t;
            t            = B(k, j);
            B(k, j)      = B(p, j);
            B(p, j)      = t;
          }
        }
        // eliminate
        for (int i = k + 1; i < N; i++) {
          Complex<R> f = D(i, k) / D(k, k);
          D(i, k)      = f;
          for (int j = k + 1; j < N; j++) D(i, j) -= f * D(k, j);
          for (int j = 0; j < N; j++) B(i, j) -= f * B(k, j);
        }
      }
      // back substitution
      for (int j = 0; j < N; j++) {
        for (int i = N - 1; i >= 0; i--) {
          Complex<R> s = B(i, j);
          for (int k = i + 1; k < N; k++) s -= D(i, k) * B(k, j);
          B(i, j) = s / D(i, i);
        }
      }
    }

    /// Eigen's degree-selection thresholds (Higham 2005, Table 2.3).
    template <class R> struct ExpmTraits;
    template <> struct ExpmTraits<double> {
        static constexpr int nthr = 4;
        OPG_HD static constexpr double thr(int i)
        {
          return i == 0   ? 1.495585217958292e-002
                 : i == 1 ? 2.539398330063230e-001
                 : i == 2 ? 9.504178996162932e-001
                          : 2.097847961257068e+000;
        }
        OPG_HD static constexpr double maxnorm() { return 5.371920351148152; }
    };
    template <> struct ExpmTraits<float> {
        static constexpr int nthr = 2;
        OPG_HD static constexpr float thr(int i)
        {
          return i == 0 ? 4.258730016922831e-001f : 1.880152677804762e+000f;
        }
        OPG_HD static constexpr float maxnorm() { return 3.925724783138660f; }
    };

  } // namespace detail

  /// exp(A) for a small complex matrix.
  template <int N, class R> OPG_HD inline Mat<N, R> expm(const Mat<N, R>& A0)
  {
    using detail::lincomb;
    using T = detail::ExpmTraits<R>;

    // 1-norm: maximum absolute column sum
    R l1norm = 0;
    for (int j = 0; j < N; j++) {
      R s = 0;
      for (int i = 0; i < N; i++) s += abs(A0(i, j));
      if (s > l1norm) l1norm = s;
    }

    Mat<N, R> A = A0;
    Mat<N, R> U, V;
    int       squarings = 0;

    if (T::nthr >= 1 && l1norm < T::thr(0)) {  // Pade 3
      const double b[] = {120, 60, 12, 1};
      Mat<N, R> A2  = matmul(A, A);
      const Mat<N, R>* m1[] = {&A2};
      const R          c1[] = {R(b[3])};
      U                     = matmul(A, lincomb<N, R, 1>(c1, m1, R(b[1])));
      const R c2[]          = {R(b[2])};
      V                     = lincomb<N, R, 1>(c2, m1, R(b[0]));
    }
    else if (T::nthr >= 2 && l1norm < T::thr(1)) {  // Pade 5
      const double b[] = {30240, 15120, 3360, 420, 30, 1};
      Mat<N, R> A2  = matmul(A, A);
      Mat<N, R> A4  = matmul(A2, A2);
      const Mat<N, R>* m[] = {&A4, &A2};
      const R          cu[] = {R(b[5]), R(b[3])};
      const R          cv[] = {R(b[4]), R(b[2])};
      U = matmul(A, lincomb<N, R, 2>(cu, m, R(b[1])));
      V = lincomb<N, R, 2>(cv, m, R(b[0]));
    }
    else if (T::nthr >= 3 && l1norm < T::thr(2)) {  // Pade 7 (double)
      const double b[] = {17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1};
      Mat<N, R> A2  = matmul(A, A);
      Mat<N, R> A4  = matmul(A2, A2);
      Mat<N, R> A6  = matmul(A4, A2);
      const Mat<N, R>* m[] = {&A6, &A4, &A2};
      const R          cu[] = {R(b[7]), R(b[5]), R(b[3])};
      const R          cv[] = {R(b[6]), R(b[4]), R(b[2])};
      U = matmul(A, lincomb<N, R, 3>(cu, m, R(b[1])));
      V = lincomb<N, R, 3>(cv, m, R(b[0]));
    }
    else if (T::nthr >= 4 && l1norm < T::thr(3)) {  // Pade 9 (double)
      const double b[] = {17643225600., 8821612800., 2075673600., 302702400.,
                       30270240.,    2162160.,    110880.,     3960.,
                       90.,          1.};
      Mat<N, R> A2  = matmul(A, A);
      Mat<N, R> A4  = matmul(A2, A2);
      Mat<N, R> A6  = matmul(A4, A2);
      Mat<N, R> A8  = matmul(A6, A2);
      const Mat<N, R>* m[] = {&A8, &A6, &A4, &A2};
      const R          cu[] = {R(b[9]), R(b[7]), R(b[5]), R(b[3])};
      const R          cv[] = {R(b[8]), R(b[6]), R(b[4]), R(b[2])};
      U = matmul(A, lincomb<N, R, 4>(cu, m, R(b[1])));
      V = lincomb<N, R, 4>(cv, m, R(b[0]));
    }
    else {
      // Scale so that the norm is below maxnorm, then Pade 13 (double) or
      // Pade 7 (float).
      std::frexp(l1norm / T::maxnorm(), &squarings);
      if (squarings < 0) squarings = 0;
      for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
          A(i, j) = Complex<R>(std::ldexp(A(i, j).re, -squarings),
                               std::ldexp(A(i, j).im, -squarings));

      if constexpr (T::nthr == 4) {
        const double b[] = {64764752532480000., 32382376266240000.,
                            7771770303897600.,  1187353796428800.,
                            129060195264000.,   10559470521600.,
                            670442572800.,      33522128640.,
                            1323241920.,        40840800.,
                            960960.,            16380.,
                            182.,               1.};
        Mat<N, R> A2 = matmul(A, A);
        Mat<N, R> A4 = matmul(A2, A2);
        Mat<N, R> A6 = matmul(A4, A2);
        const Mat<N, R>* m3[] = {&A6, &A4, &A2};
        {
          const R   c[]  = {R(b[13]), R(b[11]), R(b[9])};
          Mat<N, R> t    = lincomb<N, R, 3>(c, m3, R(0));
          Mat<N, R> tmp  = matmul(A6, t);
          const R   c2[] = {R(b[7]), R(b[5]), R(b[3])};
          Mat<N, R> t2   = lincomb<N, R, 3>(c2, m3, R(b[1]));
          for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) tmp(i, j) += t2(i, j);
          U = matmul(A, tmp);
        }
        {
          const R   c[]  = {R(b[12]), R(b[10]), R(b[8])};
          Mat<N, R> t    = lincomb<N, R, 3>(c, m3, R(0));
          V              = matmul(A6, t);
          const R   c2[] = {R(b[6]), R(b[4]), R(b[2])};
          Mat<N, R> t2   = lincomb<N, R, 3>(c2, m3, R(b[0]));
          for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) V(i, j) += t2(i, j);
        }
      }
      else {
        const double b[] = {17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1};
        Mat<N, R> A2  = matmul(A, A);
        Mat<N, R> A4  = matmul(A2, A2);
        Mat<N, R> A6  = matmul(A4, A2);
        const Mat<N, R>* m[] = {&A6, &A4, &A2};
        const R          cu[] = {R(b[7]), R(b[5]), R(b[3])};
        const R          cv[] = {R(b[6]), R(b[4]), R(b[2])};
        U = matmul(A, lincomb<N, R, 3>(cu, m, R(b[1])));
        V = lincomb<N, R, 3>(cv, m, R(b[0]));
      }
    }

    // exp(A) ~ (V - U)^-1 (V + U)
    Mat<N, R> numer, denom;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        numer(i, j) = U(i, j) + V(i, j);
        denom(i, j) = V(i, j) - U(i, j);
      }
    detail::lu_solve<N, R>(denom, numer);

    for (int k = 0; k < squarings; k++) numer = matmul(numer, numer);
    return numer;
  }

} // namespace opg

#endif
