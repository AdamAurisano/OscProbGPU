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
///
/// The scalar type S may be Dual<R, K>: all decisions (degree, number of
/// squarings, pivots) are then taken from the value parts, and the
/// derivative parts are the exact derivatives of the approximant, i.e. the
/// Frechet derivative of exp to the same accuracy as the value (Al-Mohy and
/// Higham, SIAM J. Matrix Anal. Appl. 30 (2009) 1639).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_LINALG_EXPM_H
#define OPG_LINALG_EXPM_H

#include <cmath>

#include "opg/core/dual.h"
#include "opg/core/matrix.h"

namespace opg {

  namespace detail {

    /// Accumulate c * M into s (element (i, j)).
    template <int N, class S, class V>
    OPG_HD OPG_INLINE void lc_add(Complex<S>& s, int i, int j, V c, const Mat<N, S>& M)
    {
      const Complex<S>& m = M(i, j);
      s += Complex<S>(m.re * c, m.im * c);
    }

    /// sum_k c[k] * M_k + c0 * I, with real (value-type) coefficients. The
    /// matrices are passed by reference (no pointer arrays, which would
    /// force them out of registers on the GPU).
    template <int N, class S, class V, class... Ms>
    OPG_HD OPG_INLINE Mat<N, S> lincomb(const V (&c)[sizeof...(Ms)], V c0,
                                        const Ms&... M)
    {
      Mat<N, S> out;
      OPG_UNROLL
      for (int i = 0; i < N; i++)
        OPG_UNROLL
      for (int j = 0; j < N; j++) {
        Complex<S> s(S(0), S(0));
        int        k = 0;
        (lc_add<N, S, V>(s, i, j, c[k++], M), ...);
        if (i == j) s.re += c0;
        out(i, j) = s;
      }
      return out;
    }

    /// |z| of the value part.
    template <class S>
    OPG_HD OPG_INLINE value_type_t<S> abs_value(const Complex<S>& z)
    {
      using std::hypot;
      return hypot(value_of(z.re), value_of(z.im));
    }

    /// Solve D X = B for X (N x N right-hand side) by LU with partial
    /// pivoting (pivots chosen from the values). D and B are overwritten.
    template <int N, class S>
    OPG_HD OPG_INLINE void lu_solve(Mat<N, S>& D, Mat<N, S>& B)
    {
      using V = value_type_t<S>;
      OPG_UNROLL
      for (int k = 0; k < N; k++) {
        // pivot
        int p    = k;
        V   best = abs_value(D(k, k));
        OPG_UNROLL
        for (int i = k + 1; i < N; i++) {
          V v = abs_value(D(i, k));
          if (v > best) {
            best = v;
            p    = i;
          }
        }
        // row swap with compile-time indices (keeps D, B in registers)
        OPG_UNROLL
        for (int i = k + 1; i < N; i++) {
          if (p == i) {
            OPG_UNROLL
            for (int j = 0; j < N; j++) {
              Complex<S> t = D(k, j);
              D(k, j)      = D(i, j);
              D(i, j)      = t;
              t            = B(k, j);
              B(k, j)      = B(i, j);
              B(i, j)      = t;
            }
          }
        }
        // eliminate
        OPG_UNROLL
        for (int i = k + 1; i < N; i++) {
          Complex<S> f = D(i, k) / D(k, k);
          D(i, k)      = f;
          OPG_UNROLL
          for (int j = k + 1; j < N; j++) D(i, j) -= f * D(k, j);
          OPG_UNROLL
          for (int j = 0; j < N; j++) B(i, j) -= f * B(k, j);
        }
      }
      // back substitution
      OPG_UNROLL
      for (int j = 0; j < N; j++) {
        OPG_UNROLL
        for (int i = N - 1; i >= 0; i--) {
          Complex<S> s = B(i, j);
          OPG_UNROLL
          for (int k = i + 1; k < N; k++) s -= D(i, k) * B(k, j);
          B(i, j) = s / D(i, i);
        }
      }
    }

    /// Eigen's degree-selection thresholds (Higham 2005, Table 2.3).
    /// pade13: use degree 13 (else 7) after scaling to maxnorm.
    template <class R> struct ExpmTraits;
    template <> struct ExpmTraits<double> {
        static constexpr int  nthr   = 4;
        static constexpr bool pade13 = true;
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
        static constexpr int  nthr   = 2;
        static constexpr bool pade13 = false;
        OPG_HD static constexpr float thr(int i)
        {
          return i == 0 ? 4.258730016922831e-001f : 1.880152677804762e+000f;
        }
        OPG_HD static constexpr float maxnorm() { return 3.925724783138660f; }
    };
    /// Long double (host-side references): always Pade 13 after scaling to
    /// norm <= 1, where its truncation error (~ (1/5.37)^27 * 1e-16) is far
    /// below long-double round-off.
    template <> struct ExpmTraits<long double> {
        static constexpr int  nthr   = 0;
        static constexpr bool pade13 = true;
        static constexpr long double thr(int) { return 0; }
        static constexpr long double maxnorm() { return 1; }
    };

  } // namespace detail

  /// exp(A) for a small complex matrix with scalar type S (real or dual).
  template <int N, class S> OPG_HD inline Mat<N, S> expm(const Mat<N, S>& A0)
  {
    using detail::lincomb;
    using V = value_type_t<S>;
    using T = detail::ExpmTraits<V>;

    // 1-norm of the values: maximum absolute column sum
    V l1norm = 0;
    OPG_UNROLL
    for (int j = 0; j < N; j++) {
      V s = 0;
      OPG_UNROLL
      for (int i = 0; i < N; i++) s += detail::abs_value(A0(i, j));
      if (s > l1norm) l1norm = s;
    }

    Mat<N, S> A = A0;
    Mat<N, S> U, W;  // Pade numerator/denominator parts (W is Eigen's V)
    int       squarings = 0;

    if (T::nthr >= 1 && l1norm < T::thr(0)) {  // Pade 3
      const double b[] = {120, 60, 12, 1};
      Mat<N, S> A2  = matmul(A, A);
      const V   c1[] = {V(b[3])};
      U              = matmul(A, lincomb<N, S>(c1, V(b[1]), A2));
      const V c2[]   = {V(b[2])};
      W              = lincomb<N, S>(c2, V(b[0]), A2);
    }
    else if (T::nthr >= 2 && l1norm < T::thr(1)) {  // Pade 5
      const double b[] = {30240, 15120, 3360, 420, 30, 1};
      Mat<N, S> A2  = matmul(A, A);
      Mat<N, S> A4  = matmul(A2, A2);
      const V   cu[] = {V(b[5]), V(b[3])};
      const V   cv[] = {V(b[4]), V(b[2])};
      U = matmul(A, lincomb<N, S>(cu, V(b[1]), A4, A2));
      W = lincomb<N, S>(cv, V(b[0]), A4, A2);
    }
    else if (T::nthr >= 3 && l1norm < T::thr(2)) {  // Pade 7 (double)
      const double b[] = {17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1};
      Mat<N, S> A2  = matmul(A, A);
      Mat<N, S> A4  = matmul(A2, A2);
      Mat<N, S> A6  = matmul(A4, A2);
      const V   cu[] = {V(b[7]), V(b[5]), V(b[3])};
      const V   cv[] = {V(b[6]), V(b[4]), V(b[2])};
      U = matmul(A, lincomb<N, S>(cu, V(b[1]), A6, A4, A2));
      W = lincomb<N, S>(cv, V(b[0]), A6, A4, A2);
    }
    else if (T::nthr >= 4 && l1norm < T::thr(3)) {  // Pade 9 (double)
      const double b[] = {17643225600., 8821612800., 2075673600., 302702400.,
                       30270240.,    2162160.,    110880.,     3960.,
                       90.,          1.};
      Mat<N, S> A2  = matmul(A, A);
      Mat<N, S> A4  = matmul(A2, A2);
      Mat<N, S> A6  = matmul(A4, A2);
      Mat<N, S> A8  = matmul(A6, A2);
      const V   cu[] = {V(b[9]), V(b[7]), V(b[5]), V(b[3])};
      const V   cv[] = {V(b[8]), V(b[6]), V(b[4]), V(b[2])};
      U = matmul(A, lincomb<N, S>(cu, V(b[1]), A8, A6, A4, A2));
      W = lincomb<N, S>(cv, V(b[0]), A8, A6, A4, A2);
    }
    else {
      // Scale so that the norm is below maxnorm, then Pade 13 (double) or
      // Pade 7 (float).
      std::frexp(l1norm / T::maxnorm(), &squarings);
      if (squarings < 0) squarings = 0;
      OPG_UNROLL
      for (int i = 0; i < N; i++)
        OPG_UNROLL
        for (int j = 0; j < N; j++)
          A(i, j) = Complex<S>(ldexp_s(A(i, j).re, -squarings),
                               ldexp_s(A(i, j).im, -squarings));

      if constexpr (T::pade13) {
        const double b[] = {64764752532480000., 32382376266240000.,
                            7771770303897600.,  1187353796428800.,
                            129060195264000.,   10559470521600.,
                            670442572800.,      33522128640.,
                            1323241920.,        40840800.,
                            960960.,            16380.,
                            182.,               1.};
        Mat<N, S> A2 = matmul(A, A);
        Mat<N, S> A4 = matmul(A2, A2);
        Mat<N, S> A6 = matmul(A4, A2);
        {
          const V   c[]  = {V(b[13]), V(b[11]), V(b[9])};
          Mat<N, S> t    = lincomb<N, S>(c, V(0), A6, A4, A2);
          Mat<N, S> tmp  = matmul(A6, t);
          const V   c2[] = {V(b[7]), V(b[5]), V(b[3])};
          Mat<N, S> t2   = lincomb<N, S>(c2, V(b[1]), A6, A4, A2);
          OPG_UNROLL
          for (int i = 0; i < N; i++)
            OPG_UNROLL
            for (int j = 0; j < N; j++) tmp(i, j) += t2(i, j);
          U = matmul(A, tmp);
        }
        {
          const V   c[]  = {V(b[12]), V(b[10]), V(b[8])};
          Mat<N, S> t    = lincomb<N, S>(c, V(0), A6, A4, A2);
          W              = matmul(A6, t);
          const V   c2[] = {V(b[6]), V(b[4]), V(b[2])};
          Mat<N, S> t2   = lincomb<N, S>(c2, V(b[0]), A6, A4, A2);
          OPG_UNROLL
          for (int i = 0; i < N; i++)
            OPG_UNROLL
            for (int j = 0; j < N; j++) W(i, j) += t2(i, j);
        }
      }
      else {
        const double b[] = {17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1};
        Mat<N, S> A2  = matmul(A, A);
        Mat<N, S> A4  = matmul(A2, A2);
        Mat<N, S> A6  = matmul(A4, A2);
        const V   cu[] = {V(b[7]), V(b[5]), V(b[3])};
        const V   cv[] = {V(b[6]), V(b[4]), V(b[2])};
        U = matmul(A, lincomb<N, S>(cu, V(b[1]), A6, A4, A2));
        W = lincomb<N, S>(cv, V(b[0]), A6, A4, A2);
      }
    }

    // exp(A) ~ (W - U)^-1 (W + U)
    Mat<N, S> numer, denom;
    OPG_UNROLL
    for (int i = 0; i < N; i++)
      OPG_UNROLL
      for (int j = 0; j < N; j++) {
        numer(i, j) = U(i, j) + W(i, j);
        denom(i, j) = W(i, j) - U(i, j);
      }
    detail::lu_solve<N, S>(denom, numer);

    for (int k = 0; k < squarings; k++) numer = matmul(numer, numer);
    return numer;
  }

} // namespace opg

#endif
