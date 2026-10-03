///////////////////////////////////////////////////////////////////////////////
/// \file dual.h
///
/// \brief Forward-mode dual numbers with K derivative directions, usable on
///        host and device.
///
/// Dual<T, K> holds a value v and K partial derivatives d[k]. Arithmetic on
/// the value part performs exactly the same floating point operations as
/// plain T arithmetic, so running code with Dual<T, K> instead of T leaves
/// the values unchanged.
///
/// Comparisons (<, ==, ...) compare values only, which makes branches in
/// generic code follow the value computation. Code that skips work for
/// exact zeros must use is_exact_zero(), which also checks the derivatives
/// (otherwise derivatives at e.g. theta = 0 would be lost).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_CORE_DUAL_H
#define OPG_CORE_DUAL_H

#include <cmath>

#include "opg/core/macros.h"

namespace opg {

  template <class T, int K> struct Dual {
      T v;
      T d[K];

      Dual() = default;
      OPG_HD constexpr Dual(T x) : v(x), d{} {}

      OPG_HD OPG_INLINE Dual& operator+=(const Dual& o)
      {
        v += o.v;
        OPG_UNROLL
        for (int k = 0; k < K; k++) d[k] += o.d[k];
        return *this;
      }
      OPG_HD OPG_INLINE Dual& operator-=(const Dual& o)
      {
        v -= o.v;
        OPG_UNROLL
        for (int k = 0; k < K; k++) d[k] -= o.d[k];
        return *this;
      }
      OPG_HD OPG_INLINE Dual& operator*=(const Dual& o)
      {
        OPG_UNROLL
        for (int k = 0; k < K; k++) d[k] = d[k] * o.v + v * o.d[k];
        v *= o.v;
        return *this;
      }
      OPG_HD OPG_INLINE Dual& operator/=(const Dual& o)
      {
        const T q = v / o.v;
        OPG_UNROLL
        for (int k = 0; k < K; k++) d[k] = (d[k] - q * o.d[k]) / o.v;
        v = q;
        return *this;
      }
      OPG_HD OPG_INLINE Dual& operator+=(T s)
      {
        v += s;
        return *this;
      }
      OPG_HD OPG_INLINE Dual& operator-=(T s)
      {
        v -= s;
        return *this;
      }
      OPG_HD OPG_INLINE Dual& operator*=(T s)
      {
        v *= s;
        OPG_UNROLL
        for (int k = 0; k < K; k++) d[k] *= s;
        return *this;
      }
      OPG_HD OPG_INLINE Dual& operator/=(T s)
      {
        v /= s;
        OPG_UNROLL
        for (int k = 0; k < K; k++) d[k] /= s;
        return *this;
      }
  };

  // --- arithmetic -----------------------------------------------------------
#define OPG_DUAL_BINOP(OP)                                                    \
  template <class T, int K>                                                   \
  OPG_HD OPG_INLINE Dual<T, K> operator OP(Dual<T, K> a, const Dual<T, K>& b) \
  {                                                                           \
    return a OP## = b;                                                        \
  }                                                                           \
  template <class T, int K>                                                   \
  OPG_HD OPG_INLINE Dual<T, K> operator OP(Dual<T, K> a, T s)                 \
  {                                                                           \
    return a OP## = s;                                                        \
  }
  OPG_DUAL_BINOP(+)
  OPG_DUAL_BINOP(-)
  OPG_DUAL_BINOP(*)
  OPG_DUAL_BINOP(/)
#undef OPG_DUAL_BINOP

  template <class T, int K>
  OPG_HD OPG_INLINE Dual<T, K> operator+(T s, Dual<T, K> a)
  {
    Dual<T, K> r = a;
    r.v          = s + a.v;
    return r;
  }
  template <class T, int K>
  OPG_HD OPG_INLINE Dual<T, K> operator-(T s, const Dual<T, K>& a)
  {
    Dual<T, K> r;
    r.v = s - a.v;
    OPG_UNROLL
    for (int k = 0; k < K; k++) r.d[k] = -a.d[k];
    return r;
  }
  template <class T, int K>
  OPG_HD OPG_INLINE Dual<T, K> operator*(T s, Dual<T, K> a)
  {
    Dual<T, K> r;
    r.v = s * a.v;
    OPG_UNROLL
    for (int k = 0; k < K; k++) r.d[k] = s * a.d[k];
    return r;
  }
  template <class T, int K>
  OPG_HD OPG_INLINE Dual<T, K> operator/(T s, const Dual<T, K>& a)
  {
    Dual<T, K> r;
    r.v       = s / a.v;
    const T f = -r.v / a.v;
    OPG_UNROLL
    for (int k = 0; k < K; k++) r.d[k] = f * a.d[k];
    return r;
  }
  template <class T, int K>
  OPG_HD OPG_INLINE Dual<T, K> operator-(const Dual<T, K>& a)
  {
    Dual<T, K> r;
    r.v = -a.v;
    OPG_UNROLL
    for (int k = 0; k < K; k++) r.d[k] = -a.d[k];
    return r;
  }
  template <class T, int K>
  OPG_HD OPG_INLINE Dual<T, K> operator+(const Dual<T, K>& a)
  {
    return a;
  }

  // --- comparisons (value only) ---------------------------------------------
#define OPG_DUAL_CMP(OP)                                                      \
  template <class T, int K>                                                   \
  OPG_HD OPG_INLINE bool operator OP(const Dual<T, K>& a, const Dual<T, K>& b) \
  {                                                                           \
    return a.v OP b.v;                                                        \
  }                                                                           \
  template <class T, int K>                                                   \
  OPG_HD OPG_INLINE bool operator OP(const Dual<T, K>& a, T b)                \
  {                                                                           \
    return a.v OP b;                                                          \
  }                                                                           \
  template <class T, int K>                                                   \
  OPG_HD OPG_INLINE bool operator OP(T a, const Dual<T, K>& b)                \
  {                                                                           \
    return a OP b.v;                                                          \
  }
  OPG_DUAL_CMP(<)
  OPG_DUAL_CMP(>)
  OPG_DUAL_CMP(<=)
  OPG_DUAL_CMP(>=)
  OPG_DUAL_CMP(==)
  OPG_DUAL_CMP(!=)
#undef OPG_DUAL_CMP

  // --- functions (found by ADL; generic code calls them unqualified) ----------
  template <class T, int K>
  OPG_HD OPG_INLINE Dual<T, K> chain(const Dual<T, K>& a, T fv, T dfv)
  {
    Dual<T, K> r;
    r.v = fv;
    OPG_UNROLL
    for (int k = 0; k < K; k++) r.d[k] = dfv * a.d[k];
    return r;
  }

  template <class T, int K> OPG_HD OPG_INLINE Dual<T, K> sin(const Dual<T, K>& a)
  {
    using std::cos;
    using std::sin;
    return chain(a, T(sin(a.v)), T(cos(a.v)));
  }
  template <class T, int K> OPG_HD OPG_INLINE Dual<T, K> cos(const Dual<T, K>& a)
  {
    using std::cos;
    using std::sin;
    return chain(a, T(cos(a.v)), T(-sin(a.v)));
  }
  template <class T, int K> OPG_HD OPG_INLINE Dual<T, K> sqrt(const Dual<T, K>& a)
  {
    using std::sqrt;
    const T s = sqrt(a.v);
    return chain(a, s, T(0.5) / s);
  }
  template <class T, int K> OPG_HD OPG_INLINE Dual<T, K> exp(const Dual<T, K>& a)
  {
    using std::exp;
    const T e = exp(a.v);
    return chain(a, e, e);
  }
  template <class T, int K> OPG_HD OPG_INLINE Dual<T, K> fabs(const Dual<T, K>& a)
  {
    return a.v < T(0) ? -a : a;
  }
  template <class T, int K>
  OPG_HD OPG_INLINE Dual<T, K> hypot(const Dual<T, K>& a, const Dual<T, K>& b)
  {
    using std::hypot;
    Dual<T, K> r;
    r.v = hypot(a.v, b.v);
    if (r.v == T(0)) {
      OPG_UNROLL
      for (int k = 0; k < K; k++) r.d[k] = T(0);
      return r;
    }
    OPG_UNROLL
    for (int k = 0; k < K; k++) r.d[k] = (a.v * a.d[k] + b.v * b.d[k]) / r.v;
    return r;
  }
  template <class T, int K>
  OPG_HD OPG_INLINE Dual<T, K> fmax(const Dual<T, K>& a, const Dual<T, K>& b)
  {
    return a.v >= b.v ? a : b;
  }

  // --- helpers for generic code ---------------------------------------------
  /// Value part (identity for plain numbers).
  template <class T> OPG_HD OPG_INLINE T value_of(T x) { return x; }
  template <class T, int K> OPG_HD OPG_INLINE T value_of(const Dual<T, K>& x)
  {
    return x.v;
  }

  /// True if x is exactly zero including all derivatives.
  template <class T> OPG_HD OPG_INLINE bool is_exact_zero(T x) { return x == T(0); }
  template <class T, int K>
  OPG_HD OPG_INLINE bool is_exact_zero(const Dual<T, K>& x)
  {
    if (x.v != T(0)) return false;
    for (int k = 0; k < K; k++)
      if (x.d[k] != T(0)) return false;
    return true;
  }

  /// Value type of a scalar type (T for Dual<T, K>, identity otherwise).
  template <class T> struct value_type { using type = T; };
  template <class T, int K> struct value_type<Dual<T, K>> { using type = T; };
  template <class T> using value_type_t = typename value_type<T>::type;

  /// x * 2^e (exact; applied to the value and all derivatives).
  template <class T> OPG_HD OPG_INLINE T ldexp_s(T x, int e)
  {
    using std::ldexp;
    return ldexp(x, e);
  }
  template <class T, int K>
  OPG_HD OPG_INLINE Dual<T, K> ldexp_s(Dual<T, K> x, int e)
  {
    using std::ldexp;
    x.v = ldexp(x.v, e);
    OPG_UNROLL
    for (int k = 0; k < K; k++) x.d[k] = ldexp(x.d[k], e);
    return x;
  }

  /// Number of derivative directions of a scalar type (0 for plain numbers).
  template <class T> struct dual_size { static constexpr int value = 0; };
  template <class T, int K> struct dual_size<Dual<T, K>> {
      static constexpr int value = K;
  };

} // namespace opg

#endif
