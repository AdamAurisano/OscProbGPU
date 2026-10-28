///////////////////////////////////////////////////////////////////////////////
/// \file eigh3.h
///
/// \brief Analytic eigen-decomposition of a complex hermitian 3x3 matrix,
///        usable on host and device, with a Jacobi fallback.
///
/// The matrix is scaled to O(1) and shifted by its trace, B = A/s - m I.
/// The eigenvalues of B follow from the trigonometric solution of its
/// characteristic polynomial, lambda_k = 2 p cos(phi + 2 pi k / 3) with
/// p^2 = tr(B^2) / 6, cos(3 phi) = det(B) / (2 p^3).
///
/// For an eigenvalue lambda, every row r_i of M = B - lambda I satisfies
/// sum_j M_ij v_j = 0, so v is parallel to the (bilinear) cross product of
/// any two independent rows; the largest of the three is used. This is done
/// for the largest and the smallest eigenvalue, which are the best separated
/// from the rest. The second vector is orthogonalised against the first and
/// the middle eigenvector is conj(v_max x v_min), which is orthonormal to
/// both by construction. (Refining the eigenvalues as Rayleigh quotients
/// v^dag B v was measured: ~15% slower for no useful gain in accuracy.)
///
/// Near-degenerate matrices, where the cross products lose precision, fall
/// back to the cyclic Jacobi solver (jacobi_herm.h). Written for OscProbGPU
/// from the mathematics above (BSD 3-Clause, as the rest of the code).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_LINALG_EIGH3_H
#define OPG_LINALG_EIGH3_H

#include <cmath>
#include <type_traits>

#include "opg/core/matrix.h"
#include "opg/linalg/jacobi_herm.h"

namespace opg {

  /// Below this squared norm of the best row cross product (relative to
  /// tr(B^2)^2) the eigenvector would lose more than ~3e-13 (double) and
  /// eigh3 falls back to Jacobi.
  template <class R> struct Eigh3Threshold;
  template <> struct Eigh3Threshold<double> {
      static constexpr double value = 1e-7;
  };
  template <> struct Eigh3Threshold<float> {
      static constexpr float value = 1e-4f;
  };
  template <> struct Eigh3Threshold<long double> {
      static constexpr long double value = 1e-10L;
  };

  namespace detail {

    // Products that the compiler may not fuse into an FMA. eigh3 is inlined
    // into many CUDA kernels, and nvcc contracted its a*b - c*d differences
    // differently from kernel to kernel: the same matrix gave eigen-systems
    // differing in the last bits (probabilities with and without gradients
    // were no longer identical), amplified near degeneracy. Rounding every
    // product makes the result independent of the calling kernel.
    template <class R> OPG_HD OPG_INLINE R e3mul(R a, R b)
    {
#ifdef __CUDA_ARCH__
      if constexpr (std::is_same_v<R, double>) return __dmul_rn(a, b);
      if constexpr (std::is_same_v<R, float>) return __fmul_rn(a, b);
#endif
      return a * b;
    }
    template <class R> OPG_HD OPG_INLINE Complex<R> e3mul(const Complex<R>& a, const Complex<R>& b)
    {
      return Complex<R>(e3mul(a.re, b.re) - e3mul(a.im, b.im), e3mul(a.re, b.im) + e3mul(a.im, b.re));
    }
    template <class R> OPG_HD OPG_INLINE Complex<R> e3mul(const Complex<R>& a, R s)
    {
      return Complex<R>(e3mul(a.re, s), e3mul(a.im, s));
    }
    template <class R> OPG_HD OPG_INLINE R e3norm(const Complex<R>& a)
    {
      return e3mul(a.re, a.re) + e3mul(a.im, a.im);
    }

    /// The largest of the three row cross products of the hermitian
    /// B - lambda I (only the upper triangle of B is read), and its squared
    /// norm.
    template <class R>
    OPG_HD OPG_INLINE R eigh3_kernel_vector(const Mat<3, R>& B, R lambda,
                                            Complex<R> v[3])
    {
      Complex<R> r[3][3];
      OPG_UNROLL
      for (int i = 0; i < 3; i++) {
        OPG_UNROLL
        for (int j = 0; j < 3; j++) r[i][j] = i <= j ? B(i, j) : conj(B(j, i));
        r[i][i] = Complex<R>(B(i, i).re - lambda, 0);
      }
      Complex<R> c[3][3];
      R          n[3];
      OPG_UNROLL
      for (int k = 0; k < 3; k++) {
        const int a = k == 2 ? 1 : 0, b = k == 0 ? 1 : 2;  // row pairs (0,1), (0,2), (1,2)
        c[k][0]     = e3mul(r[a][1], r[b][2]) - e3mul(r[a][2], r[b][1]);
        c[k][1]     = e3mul(r[a][2], r[b][0]) - e3mul(r[a][0], r[b][2]);
        c[k][2]     = e3mul(r[a][0], r[b][1]) - e3mul(r[a][1], r[b][0]);
        n[k]        = e3norm(c[k][0]) + e3norm(c[k][1]) + e3norm(c[k][2]);
      }
      const int best = n[0] >= n[1] ? (n[0] >= n[2] ? 0 : 2) : (n[1] >= n[2] ? 1 : 2);
      OPG_UNROLL
      for (int j = 0; j < 3; j++) v[j] = c[best][j];
      return n[best];
    }

  } // namespace detail

  /// Diagonalise the hermitian matrix A (only the diagonal and the upper
  /// triangle are read). On output V holds orthonormal eigenvectors as
  /// columns and w the eigenvalues (largest, middle, smallest when the
  /// analytic path is taken; unsorted after the fallback). Returns 0 for the
  /// analytic path, 1 when it fell back to Jacobi.
  template <class R>
  OPG_HD inline int eigh3(const Mat<3, R>& A, Mat<3, R>& V, R w[3])
  {
    using detail::e3mul;
    using detail::e3norm;
    using std::atan2;
    using std::cos;
    using std::fabs;
    using std::fmax;
    using std::sin;
    using std::sqrt;

    // Scale to O(1).
    R sc = 0;
    OPG_UNROLL
    for (int i = 0; i < 3; i++)
      OPG_UNROLL
    for (int j = i; j < 3; j++) sc = fmax(sc, fmax(fabs(A(i, j).re), fabs(A(i, j).im)));
    if (sc == R(0)) {
      V    = Mat<3, R>::identity();
      w[0] = w[1] = w[2] = 0;
      return 0;
    }
    const R inv = R(1) / sc;

    // B = A / sc - m I (upper triangle).
    Mat<3, R> B;
    OPG_UNROLL
    for (int i = 0; i < 3; i++)
      OPG_UNROLL
    for (int j = i; j < 3; j++) B(i, j) = e3mul(A(i, j), inv);
    const R m = (B(0, 0).re + B(1, 1).re + B(2, 2).re) / 3;
    OPG_UNROLL
    for (int i = 0; i < 3; i++) B(i, i) = Complex<R>(B(i, i).re - m, 0);

    const R b0 = B(0, 0).re, b1 = B(1, 1).re, b2 = B(2, 2).re;
    const R n01 = e3norm(B(0, 1)), n02 = e3norm(B(0, 2)), n12 = e3norm(B(1, 2));
    const R tr2 = e3mul(b0, b0) + e3mul(b1, b1) + e3mul(b2, b2) + 2 * (n01 + n02 + n12);
    if (tr2 == R(0)) {  // a multiple of the identity
      V    = Mat<3, R>::identity();
      w[0] = w[1] = w[2] = e3mul(m, sc);
      return 0;
    }
    const R det = e3mul(e3mul(b0, b1), b2) + 2 * e3mul(e3mul(B(0, 1), B(1, 2)), conj(B(0, 2))).re
                  - e3mul(b0, n12) - e3mul(b1, n02) - e3mul(b2, n01);

    // lambda_k = 2 p cos(phi + 2 pi k / 3), phi = atan2(sqrt(p^6 - q^2), q) / 3.
    const R p2  = tr2 / 6;
    const R p   = sqrt(p2);
    const R q   = det / 2;
    const R p3  = e3mul(p2, p);
    const R phi = atan2(sqrt(fmax(e3mul(p3, p3) - e3mul(q, q), R(0))), q) / 3;
    const R c = cos(phi), s = sin(phi);
    const R lmax = e3mul(2 * p, c);
    const R lmin = -e3mul(p, c + e3mul(R(1.7320508075688772935), s));

    Complex<R> vmax[3], vmin[3];
    const R    nmax = detail::eigh3_kernel_vector(B, lmax, vmax);
    const R    nmin = detail::eigh3_kernel_vector(B, lmin, vmin);
    const R    tol  = e3mul(e3mul(Eigh3Threshold<R>::value, tr2), tr2);
    if (!(nmax > tol && nmin > tol)) {
      jacobi_hermitian<3, R>(B, V, w);
      OPG_UNROLL
      for (int i = 0; i < 3; i++) w[i] = e3mul(w[i] + m, sc);
      return 1;
    }

    // Normalise v_max; orthogonalise v_min against it and normalise.
    const R imax = R(1) / sqrt(nmax);
    OPG_UNROLL
    for (int j = 0; j < 3; j++) vmax[j] = e3mul(vmax[j], imax);
    const Complex<R> d = e3mul(conj(vmax[0]), vmin[0]) + e3mul(conj(vmax[1]), vmin[1])
                         + e3mul(conj(vmax[2]), vmin[2]);
    OPG_UNROLL
    for (int j = 0; j < 3; j++) vmin[j] = vmin[j] - e3mul(d, vmax[j]);
    const R imin = R(1) / sqrt(e3norm(vmin[0]) + e3norm(vmin[1]) + e3norm(vmin[2]));
    OPG_UNROLL
    for (int j = 0; j < 3; j++) vmin[j] = e3mul(vmin[j], imin);
    // v_mid = conj(v_max x v_min): orthonormal to both.
    Complex<R> vmid[3];
    vmid[0] = conj(e3mul(vmax[1], vmin[2]) - e3mul(vmax[2], vmin[1]));
    vmid[1] = conj(e3mul(vmax[2], vmin[0]) - e3mul(vmax[0], vmin[2]));
    vmid[2] = conj(e3mul(vmax[0], vmin[1]) - e3mul(vmax[1], vmin[0]));

    OPG_UNROLL
    for (int i = 0; i < 3; i++) {
      V(i, 0) = vmax[i];
      V(i, 1) = vmid[i];
      V(i, 2) = vmin[i];
    }
    w[0] = e3mul(lmax + m, sc);
    w[1] = e3mul(-lmax - lmin + m, sc);
    w[2] = e3mul(lmin + m, sc);
    return 0;
  }

} // namespace opg

#endif
