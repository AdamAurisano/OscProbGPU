///////////////////////////////////////////////////////////////////////////////
/// \file complex.h
///
/// \brief Minimal trivially-copyable complex type usable on host and device.
///
/// std::complex is not usable in device code and cuda::std::complex would
/// make the CPU-only build depend on the CUDA toolkit, so we use our own.
/// Arithmetic follows the textbook formulas used by libstdc++ for
/// std::complex<double> with -ffast-math disabled (no Annex G inf/nan
/// recovery, which never matters for oscillation physics).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_CORE_COMPLEX_H
#define OPG_CORE_COMPLEX_H

#include <cmath>

#include "opg/core/dual.h"
#include "opg/core/macros.h"

namespace opg {

  template <class Real> struct Complex {
      Real re;
      Real im;

      Complex() = default;
      OPG_HD constexpr Complex(Real r, Real i = Real(0)) : re(r), im(i) {}

      OPG_HD OPG_INLINE Complex& operator+=(const Complex& o)
      {
        re += o.re;
        im += o.im;
        return *this;
      }
      OPG_HD OPG_INLINE Complex& operator-=(const Complex& o)
      {
        re -= o.re;
        im -= o.im;
        return *this;
      }
      OPG_HD OPG_INLINE Complex& operator*=(const Complex& o)
      {
        Real r = re * o.re - im * o.im;
        im     = re * o.im + im * o.re;
        re     = r;
        return *this;
      }
      OPG_HD OPG_INLINE Complex& operator*=(Real s)
      {
        re *= s;
        im *= s;
        return *this;
      }
      OPG_HD OPG_INLINE Complex& operator/=(Real s)
      {
        re /= s;
        im /= s;
        return *this;
      }
      /// Adds a real number (to the real part), as std::complex.
      OPG_HD OPG_INLINE Complex& operator+=(Real s)
      {
        re += s;
        return *this;
      }
      OPG_HD OPG_INLINE Complex& operator-=(Real s)
      {
        re -= s;
        return *this;
      }
  };

  template <class R> OPG_HD OPG_INLINE Complex<R> operator+(Complex<R> a, R s)
  {
    return a += s;
  }
  template <class R> OPG_HD OPG_INLINE Complex<R> operator-(Complex<R> a, R s)
  {
    return a -= s;
  }
  template <class R> OPG_HD OPG_INLINE Complex<R> operator+(R s, const Complex<R>& a)
  {
    return Complex<R>(s + a.re, a.im);
  }
  /// real - complex, as std::complex: (s - re, -im)
  template <class R> OPG_HD OPG_INLINE Complex<R> operator-(R s, const Complex<R>& a)
  {
    return Complex<R>(s - a.re, -a.im);
  }

  template <class R>
  OPG_HD OPG_INLINE Complex<R> operator+(Complex<R> a, const Complex<R>& b)
  {
    return a += b;
  }
  template <class R>
  OPG_HD OPG_INLINE Complex<R> operator-(Complex<R> a, const Complex<R>& b)
  {
    return a -= b;
  }
  template <class R>
  OPG_HD OPG_INLINE Complex<R> operator*(Complex<R> a, const Complex<R>& b)
  {
    return a *= b;
  }
  template <class R> OPG_HD OPG_INLINE Complex<R> operator*(Complex<R> a, R s)
  {
    return a *= s;
  }
  template <class R> OPG_HD OPG_INLINE Complex<R> operator*(R s, Complex<R> a)
  {
    return a *= s;
  }
  template <class R> OPG_HD OPG_INLINE Complex<R> operator/(Complex<R> a, R s)
  {
    return a /= s;
  }
  template <class R>
  OPG_HD OPG_INLINE Complex<R> operator/(const Complex<R>& a, const Complex<R>& b)
  {
    // Smith's algorithm (robust against overflow), as used by most libms.
    using std::fabs;
    if (fabs(b.re) >= fabs(b.im)) {
      R r = b.im / b.re;
      R d = b.re + b.im * r;
      return Complex<R>((a.re + a.im * r) / d, (a.im - a.re * r) / d);
    }
    R r = b.re / b.im;
    R d = b.re * r + b.im;
    return Complex<R>((a.re * r + a.im) / d, (a.im * r - a.re) / d);
  }
  template <class R> OPG_HD OPG_INLINE Complex<R> operator-(const Complex<R>& a)
  {
    return Complex<R>(-a.re, -a.im);
  }
  template <class R>
  OPG_HD OPG_INLINE bool operator==(const Complex<R>& a, const Complex<R>& b)
  {
    return a.re == b.re && a.im == b.im;
  }

  template <class R> OPG_HD OPG_INLINE Complex<R> conj(const Complex<R>& a)
  {
    return Complex<R>(a.re, -a.im);
  }
  template <class R> OPG_HD OPG_INLINE R real(const Complex<R>& a) { return a.re; }
  template <class R> OPG_HD OPG_INLINE R imag(const Complex<R>& a) { return a.im; }
  /// Squared modulus (std::norm convention).
  template <class R> OPG_HD OPG_INLINE R norm(const Complex<R>& a)
  {
    return a.re * a.re + a.im * a.im;
  }
  template <class R> OPG_HD OPG_INLINE R abs(const Complex<R>& a)
  {
    using std::hypot;
    return hypot(a.re, a.im);
  }
  /// exp(i*phi)
  template <class R> OPG_HD OPG_INLINE Complex<R> expi(R phi)
  {
    using std::cos;
    using std::sin;
    return Complex<R>(cos(phi), sin(phi));
  }

} // namespace opg

#endif
