///////////////////////////////////////////////////////////////////////////////
/// \file prod_height.h
///
/// \brief Exact averaging of oscillation probabilities over the production
///        height of atmospheric neutrinos.
///
/// The path through the atmosphere is a single segment of constant matter
/// (the Earth model's top layer, e.g. PREM's 0.001 g/cm^3 shell) whose
/// length L depends on the production height h and on cos(zenith); the rest
/// of the path does not depend on h. With the atmosphere Hamiltonian
/// H = V diag(lam) V^dag, the segment's evolution is
/// U(L) = V diag(exp(-i lam_j L)) V^dag, and the amplitude of a fixed height
/// is S(L) = Rest V diag(exp(-i lam L)) V^dag I0 (I0: the model's initial
/// state, Rest: the remaining segments and the model's final rotation).
/// Averaging P(a -> b) = |S(L)_ba|^2 over L is then exact and needs one
/// propagation:
///
///   <P_ab> = sum_jk A_bj B_ja conj(A_bk B_ka) phi(lam_j - lam_k),
///   A = Rest V,  B = V^dag I0,  phi(w) = <exp(-i w L)>,
///
/// i.e. the averaged density matrix in the atmosphere's eigenbasis is
/// rho_jk phi(lam_j - lam_k) (decoherence factors), propagated through the
/// rest of the path.
///
/// Height distributions are histograms, uniform in h within each bin (a bin
/// of zero width is a fixed height). The geometry (production radius r_s + h,
/// atmosphere base r_s, detector radius r_d <= r_s, sin^2 = 1 - cosZ^2):
///
///   L(h) = sqrt((r_s + h)^2 - r_d^2 sin^2) - c,  c = sqrt(r_s^2 - r_d^2 sin^2),
///   dh/dL = g(L) = (L + c) / sqrt((L + c)^2 + r_d^2 sin^2),
///
/// so a bin contributes p_b / dh_b * int exp(-i w L) g(L) dL over its range
/// in L. g is analytic with its singularities at L = -c +- i r_d sin, at
/// least r_s (6371 km) from any point of the range, so on each bin it is
/// replaced by its degree-11 Legendre interpolant at 12 Gauss-Legendre nodes
/// (relative error below ~(hw / 2 r_s)^12, hw the half width of the bin in L:
/// < 1e-15 for bins up to ~1000 km of path, i.e. ~80 km of height at the
/// horizon). The oscillating factor is integrated exactly against that
/// polynomial (Filon-type quadrature):
///
///   int_{-1}^{1} P_k(t) exp(-i theta t) dt = 2 (-i)^k j_k(theta),
///
/// with spherical Bessel functions j_k, so the result has no error growing
/// with the phase w L and no sampling. Each bin is normalised by the
/// integral of its interpolant (phi(0) = 1 exactly, probabilities sum as
/// for a fixed height).
///
/// Gradients: dH (from the model's dual Hamiltonian) enters through the
/// two-sided Daleckii-Krein formula for F(H) rho = sum_jk phi(lam_j - lam_k)
/// P_j rho P_k:
///
///   (V^dag dF V)_jk = sum_l G(j,l,k) M_jl rho_lk + sum_l conj(G(l,k,j)) rho_jl M_lk,
///   G(j,l,k) = [phi(lam_j - lam_k) - phi(lam_l - lam_k)] / (lam_j - lam_l),
///   M = V^dag dH V,
///
/// with G(j,j,k) = phi'(lam_j - lam_k) = <-i L exp(-i (lam_j - lam_k) L)>
/// (from the Bessel functions of phi). Divided differences of close
/// arguments (|lam_j - lam_l| L_max < 2e-3) are evaluated as
/// G = <-i L sinc(d L) exp(-i mu L)> (mu, d the mid point and half
/// difference) with sinc expanded until the terms are below 1e-18, from
/// Legendre series of g L^p (p odd); other ones directly (relative error
/// < 5e-13 from cancellation). The rest of the path is
/// differentiated with the models' existing forward (step_grad) or reverse
/// (adj_step) machinery.
///
/// Models: those whose state is the amplitude matrix and which provide the
/// segment eigensystem hooks (seg_eigen, seg_hamiltonian_dual): Fast, NSI,
/// NUNM (non-unitary initial state and final rotation included), LIV, SNSI
/// and Sterile (any N). Others (Decay, Deco, OQS, SiderealLIV) are refused.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_AVG_PROD_HEIGHT_H
#define OPG_AVG_PROD_HEIGHT_H

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../core/complex.h"
#include "../core/constants.h"
#include "../core/matrix.h"
#include "../earth/earth.h"
#include "../earth/prem.h"
#include "../physics/adjoint.h"
#include "../physics/grad.h"
#include "../physics/propagate.h"

namespace opg {

  //...........................................................................
  // Model support
  //...........................................................................
  namespace detail {
    template <class Model, class = void> struct has_seg_eigen : std::false_type {};
    template <class Model>
    struct has_seg_eigen<Model, std::void_t<decltype(Model::has_seg_eigen)>>
        : std::bool_constant<Model::has_seg_eigen> {};
  } // namespace detail

  /// Whether production-height averaging is available for a model: its
  /// state is the amplitude matrix and it provides seg_eigen (value
  /// eigensystem of a constant-matter segment) and seg_hamiltonian_dual.
  template <class Model>
  constexpr bool supports_prod_height_v =
      detail::has_seg_eigen<Model>::value && !detail::state_of<Model>::custom;

  /// ... and gradients through it (forward mode).
  template <class Model>
  constexpr bool supports_prod_height_grad_v =
      supports_prod_height_v<Model> && grad_traits<Model>::enabled;

  /// ... and reverse mode (adjoint) weighted gradients.
  template <class Model>
  constexpr bool supports_prod_height_adjoint_v =
      supports_prod_height_grad_v<Model> && has_adjoint_v<Model>;

  //...........................................................................
  // Device view of a table of height distributions
  //...........................................................................

  /// Height distributions and atmosphere geometry (non-owning; host or
  /// device pointers). Distribution d has bins off[d] .. off[d+1]-1, bin b
  /// uniform in h on [lo[b], hi[b]] (km above r_surf) with probability
  /// prob[b] (normalised per distribution).
  template <class R> struct ProdHeightView {
      int           ndist  = 0;
      const int*    off    = nullptr;
      const double* lo     = nullptr;
      const double* hi     = nullptr;
      const double* prob   = nullptr;
      double        r_surf = 0;  ///< radius of the atmosphere's base (km)
      double        r_det  = 0;  ///< detector radius (km)
      R             atm_density = 0, atm_zoa = 0;
      int           atm_type    = -1;  ///< layer type (region) of the atmosphere

      OPG_HD OPG_INLINE Segment<R> atm_segment() const
      {
        return Segment<R>{R(0), atm_density, atm_zoa, atm_type};
      }
  };

  namespace prodh {

    /// Gauss-Legendre nodes per height bin (degree of the interpolant + 1).
    constexpr int kNodes = 12;
    /// Highest power of L in the expansion of close divided differences.
    constexpr int kMomDeg = 9;
    /// Highest Legendre order used.
    constexpr int kMaxOrder = kNodes - 1 + kMomDeg;
    /// Divided differences with |lam_j - lam_l| / 2 * L_max below this use
    /// the series (to x^4, truncation < 1e-18), others the plain difference
    /// (relative cancellation error ~ 2 eps / x < 5e-13).
    constexpr double kNear = 1e-3;

    /// Positive half of the 12-point Gauss-Legendre rule.
    OPG_HD OPG_INLINE double gl_x(int i)
    {
      const double x[6] = {0.1252334085114689, 0.3678314989981802, 0.5873179542866175,
                           0.7699026741943047, 0.9041172563704748, 0.9815606342467192};
      return i < 6 ? -x[5 - i] : x[i - 6];
    }
    OPG_HD OPG_INLINE double gl_w(int i)
    {
      const double w[6] = {0.2491470458134027, 0.23349253653835464, 0.20316742672306565,
                           0.1600783285433461, 0.10693932599531888, 0.04717533638651202};
      return i < 6 ? w[5 - i] : w[i - 6];
    }

    /// Spherical Bessel functions j_0 .. j_n at x (n <= kMaxOrder): series
    /// for |x| < 1, upward recurrence for |x| >= n, Miller's downward
    /// recurrence in between (normalised to j_0 or j_1, whichever is larger).
    OPG_HD inline void sph_bessel(double x, int n, double* j)
    {
      const double ax = std::fabs(x);
      if (ax < 1) {
        // power series (terms fall by >= 6x); for ax >= 1e-6 only for the top
        // two orders, then downward recurrence (stable)
        const double hx2  = 0.5 * ax * ax;
        auto         sers = [&](int k) {
          double pref = 1;  // ax^k / (2k+1)!!
          for (int q = 1; q <= k; q++) pref *= ax / (2 * q + 1);
          double s = 1, t = 1;
          for (int m = 1; m < 30; m++) {
            t *= -hx2 / (m * (2.0 * k + 2 * m + 1));
            s += t;
            if (std::fabs(t) < 1e-17 * std::fabs(s)) break;
          }
          return pref * s;
        };
        if (ax < 1e-6 || n < 2) {
          for (int k = 0; k <= n; k++) j[k] = sers(k);
        }
        else {
          const double inv = 1 / ax;
          j[n]             = sers(n);
          j[n - 1]         = sers(n - 1);
          for (int k = n - 1; k >= 1; k--) j[k - 1] = (2 * k + 1) * inv * j[k] - j[k + 1];
        }
      }
      else {
        double sn, cs;
#if OPG_ON_DEVICE
        sincos(ax, &sn, &cs);
#else
        sn = std::sin(ax);
        cs = std::cos(ax);
#endif
        const double j0 = sn / ax, j1 = (j0 - cs) / ax;
        if (ax >= n) {
          j[0] = j0;
          if (n >= 1) j[1] = j1;
          const double inv = 1 / ax;
          for (int k = 1; k < n; k++) j[k + 1] = (2 * k + 1) * inv * j[k] - j[k - 1];
        }
        else {
          const int ks = n + 16 + int(ax);
          const double inv = 1 / ax;
          double       fk1 = 0, fk = 1e-30;  // f_{k+1}, f_k
          for (int k = ks; k >= 1; k--) {
            const double fm = (2 * k + 1) * inv * fk - fk1;  // f_{k-1}
            fk1             = fk;
            fk              = fm;
            if (k - 1 <= n) j[k - 1] = fm;
          }
          // here j[0] = f_0, j[1] = f_1 (unnormalised)
          const double sc = std::fabs(j0) > std::fabs(j1) ? j0 / j[0] : j1 / j[1];
          for (int k = 0; k <= n; k++) j[k] *= sc;
        }
      }
      if (x < 0)
        for (int k = 1; k <= n; k += 2) j[k] = -j[k];
    }

    /// sum_k c_k int_{-1}^{1} P_k(t) exp(-i theta t) dt = sum_k 2 c_k (-i)^k j_k(theta)
    OPG_HD OPG_INLINE Complex<double> legendre_fourier(const double* c, int deg,
                                                       const double* jb)
    {
      double re = 0, im = 0;
      for (int k = 0; k <= deg; k += 4) re += c[k] * jb[k];
      for (int k = 2; k <= deg; k += 4) re -= c[k] * jb[k];
      for (int k = 1; k <= deg; k += 4) im -= c[k] * jb[k];
      for (int k = 3; k <= deg; k += 4) im += c[k] * jb[k];
      return Complex<double>(2 * re, 2 * im);
    }

    /// c <- (m + r t) c for a Legendre series c of degree deg (becomes deg+1).
    OPG_HD OPG_INLINE void mul_affine(double* c, int deg, double m, double r)
    {
      double prev = 0;  // c_{k-1} before update
      double out[kMaxOrder + 2];
      for (int k = 0; k <= deg + 1; k++) {
        const double ck  = k <= deg ? c[k] : 0;
        const double ck1 = k + 1 <= deg ? c[k + 1] : 0;
        out[k] = m * ck + r * (k > 0 ? prev * k / (2.0 * k - 1) : 0) +
                 r * ck1 * (k + 1) / (2.0 * k + 3);
        prev = ck;
      }
      for (int k = 0; k <= deg + 1; k++) c[k] = out[k];
    }

    /// Per-point geometry of the atmosphere segment for direction cosZ.
    struct AtmGeom {
        double rs, minRsq, c;
    };

    template <class R>
    OPG_HD OPG_INLINE AtmGeom atm_geom(const ProdHeightView<R>& hv, double cosZ)
    {
      AtmGeom g;
      g.rs            = hv.r_surf;
      const double mr = hv.r_det * std::sqrt(1 - cosZ * cosZ);
      g.minRsq        = mr * mr;
      const double c2 = g.rs * g.rs - g.minRsq;
      g.c             = c2 > 0 ? std::sqrt(c2) : 0;
      return g;
    }

    /// Path length (km) in the atmosphere for production height h (km).
    OPG_HD OPG_INLINE double atm_length(const AtmGeom& g, double h)
    {
      if (h == 0) return 0;
      const double rt = g.rs + h;
      return h * (2 * g.rs + h) / (std::sqrt(rt * rt - g.minRsq) + g.c);
    }

    /// One height bin as a Legendre series in t (L = m + r t, eV^-1) of
    /// p_b g(L) / int g, normalised so that its integral over t is p_b.
    struct BinSeries {
        double m, r;
        double c[kNodes];
        int    deg;  ///< last coefficient kept (|c_k| > 1e-18 c_0 beyond it are dropped)
    };

    template <class R>
    OPG_HD inline void bin_series(const ProdHeightView<R>& hv, const AtmGeom& g, int b,
                                  BinSeries& s)
    {
      const double kmeV = constants::kKm2eV;
      const double Llo = atm_length(g, hv.lo[b]), Lhi = atm_length(g, hv.hi[b]);
      const double m = 0.5 * (Llo + Lhi), r = 0.5 * (Lhi - Llo);
      // nodes come in pairs +-t (i and kNodes-1-i) and P_k(-t) = (-1)^k P_k(t):
      // even orders take w (g(t) + g(-t)), odd ones w (g(t) - g(-t))
      constexpr int H = kNodes / 2;
      double        ge[H], go[H];
      for (int i = 0; i < H; i++) {
        const double t  = gl_x(H + i), w = gl_w(H + i);
        const double up = m + r * t + g.c, um = m - r * t + g.c;
        double       gp = up / std::sqrt(up * up + g.minRsq);
        double       gm = um / std::sqrt(um * um + g.minRsq);
        if (!(up > 0)) gp = 1;  // r_d = r_s at the horizon, h = 0: g -> 1
        if (!(um > 0)) gm = 1;
        ge[i] = w * (gp + gm);
        go[i] = w * (gp - gm);
      }
      for (int k = 0; k < kNodes; k++) s.c[k] = 0;
      for (int i = 0; i < H; i++) {
        const double t  = gl_x(H + i);
        double       p0 = 1, p1 = t;
        s.c[0] += ge[i];
        s.c[1] += go[i] * t;
        for (int k = 2; k < kNodes; k++) {
          const double p2 = ((2 * k - 1) * t * p1 - (k - 1) * p0) / k;
          p0              = p1;
          p1              = p2;
          s.c[k] += (k % 2 ? go[i] : ge[i]) * p2;
        }
      }
      // c_k = (k + 1/2) sum_i w_i g_i P_k(t_i); normalise by int = 2 c_0
      const double norm = hv.prob[b] / s.c[0];
      for (int k = 0; k < kNodes; k++) s.c[k] *= (k + 0.5) * norm;
      // The coefficients fall like (hw / 2 r_s)^k: drop the negligible ones
      // (|F_k| <= 2, so the error is below 1e-17 of the bin's weight).
      s.deg = kNodes - 1;
      while (s.deg > 0 && std::fabs(s.c[s.deg]) < 1e-18 * std::fabs(s.c[0])) s.deg--;
      s.m = m * kmeV;
      s.r = r * kmeV;
    }

    /// The series of every bin of distribution d for direction cosZ
    /// (out[b - off[d]]), to share between points with the same direction
    /// and distribution; bins first .. first+count-1 of them only.
    template <class R>
    OPG_HD inline void series_table(const ProdHeightView<R>& hv, int d, double cosZ,
                                    BinSeries* out, int first = 0, int stride = 1)
    {
      const AtmGeom g = atm_geom(hv, cosZ);
      for (int b = hv.off[d] + first; b < hv.off[d + 1]; b += stride)
        bin_series(hv, g, b, out[b - hv.off[d]]);
    }

    /// exp(-i mu m) sum_k c_k int P_k exp(-i mu r t) dt.
    /// (jb: j_0 .. j_nbes(mu r) on return, nbes >= deg; the values of
    /// j_0 .. j_deg depend on nbes in the last bits, so callers that must
    /// agree bit for bit use the same nbes.)
    OPG_HD OPG_INLINE Complex<double> bin_ft(double mu, double m, double r, const double* c,
                                             int deg, double* jb, int nbes,
                                             Complex<double>* phase = nullptr)
    {
      sph_bessel(mu * r, nbes, jb);
      const Complex<double> f = legendre_fourier(c, deg, jb);
      double                sn, cs;
#if OPG_ON_DEVICE
      sincos(mu * m, &sn, &cs);
#else
      sn = std::sin(mu * m);
      cs = std::cos(mu * m);
#endif
      if (phase) *phase = Complex<double>(cs, -sn);
      return f * Complex<double>(cs, -sn);
    }

    /// Largest path length (eV^-1) of distribution d.
    template <class R>
    OPG_HD OPG_INLINE double max_length(const ProdHeightView<R>& hv, const AtmGeom& g, int d)
    {
      double h = 0;
      for (int b = hv.off[d]; b < hv.off[d + 1]; b++) h = std::fmax(h, hv.hi[b]);
      return atm_length(g, h) * constants::kKm2eV;
    }

    /// Phi_jk = phi(lam_j - lam_k) (Phi_jj = 1, Phi_kj = conj Phi_jk).
    template <int N, class R>
    OPG_HD OPG_NOINLINE void phi_table(const ProdHeightView<R>& hv, int d, double cosZ,
                                 const R lam[N], Complex<double> (&Phi)[N][N],
                                 const BinSeries* pre = nullptr)
    {
      const AtmGeom g = atm_geom(hv, cosZ);
      for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++) Phi[j][k] = Complex<double>(j == k ? 1 : 0, 0);
      double jb[kNodes + 1];
      for (int b = hv.off[d]; b < hv.off[d + 1]; b++) {
        if (hv.prob[b] == 0) continue;
        BinSeries s;
        if (pre)
          s = pre[b - hv.off[d]];
        else
          bin_series(hv, g, b, s);
        for (int j = 0; j < N; j++)
          for (int k = j + 1; k < N; k++) {
            const double mu = double(lam[j]) - double(lam[k]);
            Phi[j][k] += bin_ft(mu, s.m, s.r, s.c, s.deg, jb, s.deg + 1);
          }
      }
      for (int j = 0; j < N; j++)
        for (int k = j + 1; k < N; k++) Phi[k][j] = conj(Phi[j][k]);
    }

    /// Phi as phi_table, and G[j][l][k] = Gamma(lam_j - lam_k, lam_l - lam_k)
    /// (divided difference of phi; G[j][j][k] = phi'(lam_j - lam_k)).
    ///
    /// Not inlined on the device: inlined into its callers, nvcc 13.0 (sm_75,
    /// -O3) produced wrong tables (the first bins' contributions lost) in
    /// some contexts, while -G, -Xcicc -O0 and the host were right and
    /// compute-sanitizer, ASan and UBSan found nothing; as a function it is
    /// right everywhere.
    template <int N, class R>
    OPG_HD OPG_NOINLINE void phi_grad_tables(const ProdHeightView<R>& hv, int d, double cosZ,
                                       const R lam[N], Complex<double> (&Phi)[N][N],
                                       Complex<double> (&G)[N][N][N],
                                       const BinSeries* pre = nullptr)
    {
      const AtmGeom g  = atm_geom(hv, cosZ);
      const double  sL = std::fmax(max_length(hv, g, d), 1e-300);  // scale of L
      double        l[N];
      for (int j = 0; j < N; j++) l[j] = double(lam[j]);
      bool near[N][N];
      for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++) {
          Phi[j][k] = Complex<double>(j == k ? 1 : 0, 0);
          near[j][k] = std::fabs(0.5 * (l[j] - l[k])) * sL < kNear;
          for (int q = 0; q < N; q++) G[j][k][q] = Complex<double>(0, 0);
        }
      // terms of the sinc series needed per pair: dd^2M / (2M+1)! < 1e-18
      constexpr int NP = (kMomDeg + 1) / 2;
      int           nterm[N][N], mmax = 1;
      for (int j = 0; j < N; j++)
        for (int q = 0; q < N; q++) {
          const double dd = 0.5 * (l[j] - l[q]) * sL;
          int          M  = 1;
          double       t  = dd * dd / 6;  // dd^2 / 3!
          while (M < NP && t >= 1e-18) {
            t *= dd * dd / ((2 * M + 2) * (2 * M + 3));
            M++;
          }
          nterm[j][q] = M;
          if (near[j][q]) mmax = M > mmax ? M : mmax;
        }
      double jb[kMaxOrder + 1];
      for (int b = hv.off[d]; b < hv.off[d + 1]; b++) {
        if (hv.prob[b] == 0) continue;
        BinSeries s;
        if (pre)
          s = pre[b - hv.off[d]];
        else
          bin_series(hv, g, b, s);
        // series of g (L / sL)^p, p = 1, 3, .., 2 mmax - 1
        double        mom[NP][kMaxOrder + 1];
        double        cur[kMaxOrder + 2];
        for (int k = 0; k <= s.deg; k++) cur[k] = s.c[k];
        int deg = s.deg;
        for (int p = 1; p <= 2 * mmax - 1; p++) {
          mul_affine(cur, deg, s.m / sL, s.r / sL);
          deg++;
          if (p % 2 == 1)
            for (int k = 0; k <= deg; k++) mom[p / 2][k] = cur[k];
        }
        // phi(lam_j - lam_k) and phi'(lam_j - lam_k) = <-iL exp(..)> (= G[j][j][k])
        // from one set of Bessel functions; phi'(0) = -i <L>
        for (int j = 0; j < N; j++) {
          for (int k = j + 1; k < N; k++) {
            Complex<double>       e;
            const double          mu = l[j] - l[k];
            const Complex<double> f  = bin_ft(mu, s.m, s.r, s.c, s.deg, jb, s.deg + 1, &e);
            const Complex<double> f1 = legendre_fourier(mom[0], s.deg + 1, jb) * e;
            Phi[j][k] += f;
            G[j][j][k] += Complex<double>(sL * f1.im, -sL * f1.re);
          }
          G[j][j][j] += Complex<double>(0, -sL * 2 * mom[0][0]);
        }
        // close pairs j != q: G = <-i L sinc(d L) exp(-i mu L)>
        for (int j = 0; j < N; j++)
          for (int q = j + 1; q < N; q++) {
            if (!near[j][q]) continue;
            const double dd = 0.5 * (l[j] - l[q]) * sL;
            // sinc(x) = sum_m (-1)^m x^2m / (2m+1)!
            const int M = nterm[j][q];
            double    a[NP];
            a[0] = 1;
            for (int m = 1; m < M; m++) a[m] = -a[m - 1] * dd * dd / ((2 * m) * (2 * m + 1));
            // (mom[m] has degree s.deg + 2m + 1; higher entries are unset)
            const int dmax = s.deg + 2 * M - 1;
            double    cc[kMaxOrder + 1];
            for (int k = 0; k <= dmax; k++) {
              double v = 0;
              for (int m = 0; m < M; m++)
                if (k <= s.deg + 2 * m + 1) v += a[m] * mom[m][k];
              cc[k] = v;
            }
            for (int k = 0; k < N; k++) {
              const double          mu = 0.5 * (l[j] + l[q]) - l[k];
              const Complex<double> e  = bin_ft(mu, s.m, s.r, cc, dmax, jb, dmax);
              // -i sL e
              G[j][q][k] += Complex<double>(sL * e.im, -sL * e.re);
            }
          }
      }
      for (int j = 0; j < N; j++)
        for (int k = j + 1; k < N; k++) {
          Phi[k][j] = conj(Phi[j][k]);
          G[k][k][j] = -conj(G[j][j][k]);  // phi'(-x) = -conj(phi'(x))
        }
      for (int j = 0; j < N; j++)
        for (int q = j + 1; q < N; q++)
          for (int k = 0; k < N; k++) {
            if (!near[j][q]) G[j][q][k] = (Phi[j][k] - Phi[q][k]) / (l[j] - l[q]);
            G[q][j][k] = G[j][q][k];
          }
    }

    /// Rest of the path: segments after the leading ones in the atmosphere
    /// (layer type atm_type), which the averaged segment replaces.
    template <class R, class F>
    OPG_HD OPG_INLINE int for_each_rest_segment(const EarthView<R>& e, R cosZ, int atm_type,
                                                F&& f)
    {
      bool lead = true;
      int  n    = 0;
      for_each_segment(e, cosZ, [&](const Segment<R>& s) {
        if (lead && s.layer == atm_type) return;
        lead = false;
        n++;
        f(s);
      });
      return n;
    }

    /// Hermitian matrix from the derivative k of a dual matrix given by its
    /// upper triangle.
    template <int N, class R, int K>
    OPG_HD OPG_INLINE Mat<N, R> dual_part(const Mat<N, Dual<R, K>>& HD, int k)
    {
      Mat<N, R> D;
      for (int i = 0; i < N; i++) {
        D(i, i) = Complex<R>(HD(i, i).re.d[k], R(0));
        for (int j = i + 1; j < N; j++) {
          D(i, j) = Complex<R>(HD(i, j).re.d[k], HD(i, j).im.d[k]);
          D(j, i) = conj(D(i, j));
        }
      }
      return D;
    }

    template <int N, class R>
    OPG_HD OPG_INLINE Complex<double> cd(const Complex<R>& z)
    {
      return Complex<double>(double(z.re), double(z.im));
    }

    /// <P_ab> = sum_j |x_j|^2 + 2 Re sum_{j<k} x_j conj(x_k) Phi_jk with
    /// x_j = A_bj B_ja, written to out[(a*N + b) * stride].
    template <int N, class R>
    OPG_HD OPG_INLINE void avg_probs(const Mat<N, R>& A, const Mat<N, R>& B,
                                     const Complex<double> (&Phi)[N][N], R* out, size_t stride)
    {
      for (int a = 0; a < N; a++)
        for (int b = 0; b < N; b++) {
          Complex<double> x[N];
          double          p = 0;
          for (int j = 0; j < N; j++) {
            x[j] = cd<N, R>(A(b, j) * B(j, a));
            p += norm(x[j]);
          }
          double o = 0;
          for (int j = 0; j < N; j++)
            for (int k = j + 1; k < N; k++) {
              const Complex<double> t = x[j] * conj(x[k]);
              o += t.re * Phi[j][k].re - t.im * Phi[j][k].im;
            }
          out[(a * N + b) * stride] = R(p + 2 * o);
        }
    }

  } // namespace prodh

  //...........................................................................
  // Per-point evaluation
  //...........................................................................

  /// P(a -> b) averaged over the production heights of distribution `dist`
  /// (dist < 0: the Earth model's own atmosphere, as evolve_prem), written
  /// to out[(a*N + b) * stride]. pre: optionally the series_table of (dist,
  /// cosZ), shared by points of one direction (same results).
  template <class Model, class R>
  OPG_HD inline void prod_height_probs(const typename Model::Prepared& P,
                                       const EarthView<R>& earth, const ProdHeightView<R>& hv,
                                       int dist, R E, R cosZ, bool nubar, R* out, size_t stride,
                                       const prodh::BinSeries* pre = nullptr)
  {
    constexpr int N = Model::N;
    if (dist < 0 || !(std::fabs(cosZ) <= 1)) {
      auto S = evolve_prem<Model, R>(P, earth, E, cosZ, nubar);
      store_model_probs<Model, R>(S, out, stride);
      return;
    }
    Mat<N, R> V;
    R         lam[N];
    Model::seg_eigen(P, E, nubar, hv.atm_segment(), V, lam);
    Mat<N, R> S = Mat<N, R>::identity();
    prodh::for_each_rest_segment(earth, cosZ, hv.atm_type,
                                 [&](const Segment<R>& s) { Model::step(P, E, nubar, s, S); });
    Model::finalize(P, nubar, S);
    const Mat<N, R> I0 = initial_state<Model, R>(P, nubar, cosZ);
    const Mat<N, R> A = matmul(S, V), B = matmul(adjoint(V), I0);

    Complex<double> Phi[N][N];
    prodh::phi_table<N, R>(hv, dist, double(cosZ), lam, Phi, pre);
    prodh::avg_probs<N, R>(A, B, Phi, out, stride);
  }

  /// Averaged probabilities (out, if not null) and their derivatives for
  /// one gradient pass: dP_ab/dp_k written to outG[((k*N + a)*N + b) *
  /// stride] for k < count (if outG is not null), and, if w is not null,
  /// acc[k] = sum_ab w[(a*N + b) * stride] dP_ab/dp_k (else acc is zeroed).
  template <class Model, class R, int K>
  OPG_HD inline void prod_height_grad(const typename Model::Prepared& P,
                                      const GradPrepared<Model, K>& G,
                                      const EarthView<R>& earth, const ProdHeightView<R>& hv,
                                      int dist, R E, R cosZ, bool nubar, R* out, R* outG,
                                      int count, const R* w, size_t stride, R (&acc)[K],
                                      const prodh::BinSeries* pre = nullptr)
  {
    constexpr int N = Model::N;
    for (int k = 0; k < K; k++) acc[k] = 0;
    if (dist < 0 || !(std::fabs(cosZ) <= 1)) {
      Mat<N, R> S, dS[K];
      evolve_prem_grad<Model, R, K>(P, G, earth, E, cosZ, nubar, S, dS);
      if (out) store_model_probs<Model, R>(S, out, stride);
      if (outG) store_model_grads<Model, R, K>(S, dS, count, outG, stride);
      if (w) contract_model_grads<Model, R, K>(S, dS, w, stride, acc);
      return;
    }
    const Segment<R> atm = hv.atm_segment();
    Mat<N, R>        V;
    R                lam[N];
    Model::seg_eigen(P, E, nubar, atm, V, lam);
    Mat<N, Dual<R, K>> HD;
    Model::template seg_hamiltonian_dual<K>(G.P, E, nubar, seed_segment(G, atm), HD);

    // rest of the path from the identity, with derivatives
    Mat<N, R> S = Mat<N, R>::identity(), dS[K];
    for (int k = 0; k < K; k++) dS[k] = Mat<N, R>::zero();
    prodh::for_each_rest_segment(earth, cosZ, hv.atm_type, [&](const Segment<R>& s) {
      Model::template step_grad<K>(P, G.P, E, nubar, seed_segment(G, s), S, dS);
    });
    Model::template finalize_grad<K>(P, G.P, nubar, S, dS);
    Model::finalize(P, nubar, S);

    const Mat<N, R> I0 = initial_state<Model, R>(P, nubar, cosZ);
    Mat<N, R>       dI0[K];
    Model::template initial_grad<K>(G.P, nubar, dI0);

    Complex<double> Phi[N][N], Gm[N][N][N];
    prodh::phi_grad_tables<N, R>(hv, dist, double(cosZ), lam, Phi, Gm, pre);

    using C = Complex<double>;
    const Mat<N, R> Vh = adjoint(V);
    const Mat<N, R> A = matmul(S, V), B = matmul(Vh, I0);
    Mat<N, R>       dA[K], dB[K], M[K];
    for (int k = 0; k < K; k++) {
      dA[k] = matmul(dS[k], V);
      dB[k] = matmul(Vh, dI0[k]);
      M[k]  = matmul(Vh, matmul(prodh::dual_part<N, R, K>(HD, k), V));
    }

    // (probabilities as prod_height_probs, bit-identical)
    if (out) prodh::avg_probs<N, R>(A, B, Phi, out, stride);
    for (int a = 0; a < N; a++) {
      C bv[N];
      for (int j = 0; j < N; j++) bv[j] = prodh::cd<N, R>(B(j, a));
      C rb[N][N];  // averaged state in the eigenbasis
      for (int j = 0; j < N; j++)
        for (int q = 0; q < N; q++) rb[j][q] = Phi[j][q] * (bv[j] * conj(bv[q]));
      for (int k = 0; k < K; k++) {
        if (k >= count && !w) break;
        // X = V^dag d(rho_bar) V
        C db[N];
        for (int j = 0; j < N; j++) db[j] = prodh::cd<N, R>(dB[k](j, a));
        C X[N][N];
        for (int j = 0; j < N; j++)
          for (int q = 0; q < N; q++) {
            C x1(0, 0), x2(0, 0);
            for (int l = 0; l < N; l++) {
              x1 += Gm[j][l][q] * prodh::cd<N, R>(M[k](j, l)) * bv[l];
              x2 += conj(Gm[l][q][j]) * conj(bv[l]) * prodh::cd<N, R>(M[k](l, q));
            }
            X[j][q] = x1 * conj(bv[q]) + bv[j] * x2 +
                      Phi[j][q] * (db[j] * conj(bv[q]) + bv[j] * conj(db[q]));
          }
        for (int b = 0; b < N; b++) {
          double d = 0;
          for (int j = 0; j < N; j++) {
            C t1(0, 0), t2(0, 0);
            for (int q = 0; q < N; q++) {
              const C aq = conj(prodh::cd<N, R>(A(b, q)));
              t1 += rb[j][q] * aq;
              t2 += X[j][q] * aq;
            }
            d += 2 * (prodh::cd<N, R>(dA[k](b, j)) * t1).re +
                 (prodh::cd<N, R>(A(b, j)) * t2).re;
          }
          if (outG && k < count) outG[((size_t(k) * N + a) * N + b) * stride] = R(d);
          if (w) acc[k] += w[(a * N + b) * stride] * R(d);
        }
      }
    }
  }

  /// Reverse-mode weighted gradient of the averaged probabilities:
  /// acc[p] += sum_ab w[(a*N + b)*stride] d<P_ab>/dp for p < npar (chunk(c)
  /// as for adjoint_prem). Returns false (acc untouched) if the path has
  /// more than kAdjMaxSeg segments.
  template <class Model, class R, int K, class Chunk>
  OPG_HD inline bool prod_height_adjoint(const typename Model::Prepared& P, const Chunk& chunk,
                                         int npar, const EarthView<R>& earth,
                                         const ProdHeightView<R>& hv, int dist, R E, R cosZ,
                                         bool nubar, const R* w, size_t stride, R* acc,
                                         const prodh::BinSeries* pre = nullptr)
  {
    if (dist < 0 || !(std::fabs(cosZ) <= 1))
      return adjoint_prem<Model, R, K>(P, chunk, npar, earth, E, cosZ, nubar, w, stride, acc);
    constexpr int N      = Model::N;
    using State          = Mat<N, R>;
    using C              = Complex<double>;
    const int  nchunk    = (npar + K - 1) / K;
    const int  atm_type  = hv.atm_type;
    const auto atm       = hv.atm_segment();

    // forward pass over the rest of the path, with checkpoints
    State ckpt[kAdjMaxCkpt];
    State S    = State::identity();
    int   nseg = 0;
    bool  ok   = true;
    ckpt[0]    = S;
    prodh::for_each_rest_segment(earth, cosZ, atm_type, [&](const Segment<R>& s) {
      if (nseg % kAdjBlock == 0) {
        if (nseg / kAdjBlock >= kAdjMaxCkpt) ok = false;
        else ckpt[nseg / kAdjBlock] = S;
      }
      if (ok) Model::step(P, E, nubar, s, S);
      nseg++;
    });
    if (!ok) return false;

    State Rf = S;
    Model::finalize(P, nubar, Rf);
    State V;
    R     lam[N];
    Model::seg_eigen(P, E, nubar, atm, V, lam);
    C Phi[N][N], Gm[N][N][N];
    prodh::phi_grad_tables<N, R>(hv, dist, double(cosZ), lam, Phi, Gm, pre);
    const State I0 = initial_state<Model, R>(P, nubar, cosZ);
    const State A = matmul(Rf, V), B = matmul(adjoint(V), I0);

    // cotangents: Rb of the final amplitude, Cm of M = V^dag dH V, Ib of I0
    State Rb = State::zero();
    C     Cm[N][N], u[N][N];  // u[a][j]
    for (int j = 0; j < N; j++)
      for (int l = 0; l < N; l++) Cm[j][l] = C(0, 0);
    for (int a = 0; a < N; a++) {
      C bv[N];
      for (int j = 0; j < N; j++) bv[j] = prodh::cd<N, R>(B(j, a));
      C rb[N][N];
      for (int j = 0; j < N; j++)
        for (int q = 0; q < N; q++) rb[j][q] = Phi[j][q] * (bv[j] * conj(bv[q]));
      // Q = A^dag W_a A
      C Q[N][N];
      for (int q = 0; q < N; q++)
        for (int j = 0; j < N; j++) {
          C t(0, 0);
          for (int b = 0; b < N; b++) {
            const double wab = double(w[(a * N + b) * stride]);
            if (wab != 0)
              t += wab * (conj(prodh::cd<N, R>(A(b, q))) * prodh::cd<N, R>(A(b, j)));
          }
          Q[q][j] = t;
        }
      // Rb += 2 W_a A rho_bar V^dag
      for (int b = 0; b < N; b++) {
        const double wab = double(w[(a * N + b) * stride]);
        if (wab == 0) continue;
        C y[N];
        for (int q = 0; q < N; q++) {
          C t(0, 0);
          for (int j = 0; j < N; j++) t += prodh::cd<N, R>(A(b, j)) * rb[j][q];
          y[q] = t;
        }
        for (int i = 0; i < N; i++) {
          C t(0, 0);
          for (int q = 0; q < N; q++) t += y[q] * conj(prodh::cd<N, R>(V(i, q)));
          Rb(b, i) += Complex<R>(R(2 * wab * t.re), R(2 * wab * t.im));
        }
      }
      // M cotangent (two Daleckii-Krein terms) and the initial state's
      for (int j = 0; j < N; j++)
        for (int l = 0; l < N; l++) {
          C t1(0, 0), t2(0, 0);
          for (int q = 0; q < N; q++) {
            t1 += Q[q][j] * Gm[j][l][q] * (bv[l] * conj(bv[q]));      // term 1: M_jl
            t2 += Q[l][q] * conj(Gm[j][l][q]) * (bv[q] * conj(bv[j])); // term 2: M_jl
          }
          Cm[j][l] += t1 + t2;
        }
      for (int j = 0; j < N; j++) {
        C t(0, 0);
        for (int q = 0; q < N; q++) t += Q[q][j] * Phi[j][q] * conj(bv[q]);
        u[a][j] = t;
      }
    }

    // final rotation
    State lamS;
    Model::adj_final_cot(P, nubar, S, Rb, lamS);
    for (int c = 0; c < nchunk; c++) {
      R a[K] = {};
      Model::template adj_contract_final_cot<K>(chunk(c).P, nubar, S, Rb, a);
      for (int k = 0; k < K && c * K + k < npar; k++) acc[c * K + k] += a[k];
    }

    // backward pass through the rest of the path (as adjoint_prem)
    constexpr bool kAffine = detail::adj_affine<Model>::value;
    State          A0 = State::zero(), A1 = A0, A2 = A0, H1 = A0, H2 = A0;
    if constexpr (kAffine) {
      State H00, H10, H11;
      Model::template adj_hamiltonian<R>(P, ckpt[0], E, nubar, R(0), R(0), H00);
      Model::template adj_hamiltonian<R>(P, ckpt[0], E, nubar, R(1), R(0), H10);
      Model::template adj_hamiltonian<R>(P, ckpt[0], E, nubar, R(1), R(1), H11);
      for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++) {
          H1(i, j) = H10(i, j) - H00(i, j);
          H2(i, j) = H11(i, j) - H10(i, j);
        }
    }
    const int nblk = (nseg + kAdjBlock - 1) / kAdjBlock;
    for (int bk = nblk - 1; bk >= 0; bk--) {
      const int  first = bk * kAdjBlock;
      const int  len   = nseg - first < kAdjBlock ? nseg - first : kAdjBlock;
      State      buf[kAdjBlock];
      Segment<R> seg[kAdjBlock];
      int        i = 0;
      State      T = ckpt[bk];
      prodh::for_each_rest_segment(earth, cosZ, atm_type, [&](const Segment<R>& s) {
        const int j = i++ - first;
        if (j < 0 || j >= len) return;
        buf[j] = T;
        seg[j] = s;
        if (j + 1 < len) Model::step(P, E, nubar, s, T);
      });
      for (int j = len - 1; j >= 0; j--) {
        typename Model::AdjSeg Gs;
        Model::adj_step(P, E, nubar, seg[j], buf[j], lamS, Gs);
        if constexpr (kAffine) {
          R rho;
          if (Model::adj_affine_segment(seg[j], rho)) {
            const State& Hb = Model::adj_hbar(Gs);
            const R      z  = seg[j].zoa;
            for (int p = 0; p < N; p++)
              for (int l = 0; l < N; l++) {
                A0(p, l) += Hb(p, l);
                A1(p, l) += rho * Hb(p, l);
                A2(p, l) += (rho * z) * Hb(p, l);
              }
            const R h1 = hbar_dot<N, R>(Hb, H1), h2 = hbar_dot<N, R>(Hb, H2);
            const R sz = rho * h2, sr = rho * (h1 + z * h2);
            for (int c = 0; c < nchunk; c++)
              for (int k = 0; k < K && c * K + k < npar; k++) {
                if (chunk(c).zoa_type[k] == seg[j].layer) acc[c * K + k] += sz;
                if (chunk(c).rho_type[k] == seg[j].layer) acc[c * K + k] += sr;
              }
            continue;
          }
        }
        for (int c = 0; c < nchunk; c++) {
          R a[K] = {};
          Model::template adj_contract_step<K>(chunk(c).P, E, nubar,
                                               seed_segment(chunk(c), seg[j]), Gs, a);
          for (int k = 0; k < K && c * K + k < npar; k++) acc[c * K + k] += a[k];
        }
      }
    }
    if constexpr (kAffine) {
      using D = Dual<R, K>;
      for (int c = 0; c < nchunk; c++) {
        Mat<N, D> D00, D10, D11;
        Model::template adj_hamiltonian<D>(chunk(c).P, ckpt[0], E, nubar, D(0), D(0), D00);
        Model::template adj_hamiltonian<D>(chunk(c).P, ckpt[0], E, nubar, D(1), D(0), D10);
        Model::template adj_hamiltonian<D>(chunk(c).P, ckpt[0], E, nubar, D(1), D(1), D11);
        for (int p = 0; p < N; p++)
          for (int l = p; l < N; l++) {
            const Complex<D> d1 = D10(p, l) - D00(p, l), d2 = D11(p, l) - D10(p, l);
            D10(p, l)           = d1;
            D11(p, l)           = d2;
          }
        R a[K] = {};
        contract_hbar<N, R, K>(A0, D00, a);
        contract_hbar<N, R, K>(A1, D10, a);
        contract_hbar<N, R, K>(A2, D11, a);
        for (int k = 0; k < K && c * K + k < npar; k++) acc[c * K + k] += a[k];
      }
    }

    // atmosphere: Hb = V conj(Cm) V^dag; initial state: Ib = 2 V conj(u)
    State Hb, Ib;
    for (int p = 0; p < N; p++)
      for (int q = 0; q < N; q++) {
        C h(0, 0);
        for (int j = 0; j < N; j++)
          for (int l = 0; l < N; l++)
            h += prodh::cd<N, R>(V(p, j)) * conj(Cm[j][l]) * conj(prodh::cd<N, R>(V(q, l)));
        Hb(p, q) = Complex<R>(R(h.re), R(h.im));
      }
    for (int p = 0; p < N; p++)
      for (int a = 0; a < N; a++) {
        C t(0, 0);
        for (int j = 0; j < N; j++) t += prodh::cd<N, R>(V(p, j)) * conj(u[a][j]);
        Ib(p, a) = Complex<R>(R(2 * t.re), R(2 * t.im));
      }
    for (int c = 0; c < nchunk; c++) {
      Mat<N, Dual<R, K>> HD;
      Model::template seg_hamiltonian_dual<K>(chunk(c).P, E, nubar,
                                              seed_segment(chunk(c), atm), HD);
      R a[K] = {};
      contract_hbar<N, R, K>(Hb, HD, a);
      Model::template adj_contract_initial<K>(chunk(c).P, nubar, Ib, a);
      for (int k = 0; k < K && c * K + k < npar; k++) acc[c * K + k] += a[k];
    }
    return true;
  }

  //...........................................................................
  // Host-side table
  //...........................................................................

  /// Owning host table of production-height distributions plus the
  /// atmosphere geometry of an Earth model (see ProdHeightView).
  struct ProdHeightTable {
      std::vector<int>    off{0};
      std::vector<double> lo, hi, prob;
      double              r_surf = 0, r_det = 0, atm_density = 0, atm_zoa = 0;
      int                 atm_type = -1;

      int ndist() const { return int(off.size()) - 1; }

      /// Add a histogram: bin edges (km above the atmosphere's base,
      /// non-decreasing, >= 0; equal consecutive edges make a fixed height)
      /// and bin probabilities (>= 0, normalised here). Returns its index.
      int add(const std::vector<double>& edges, const std::vector<double>& p)
      {
        if (edges.size() < 2 || p.size() != edges.size() - 1)
          throw std::invalid_argument("production heights: a distribution needs nbins + 1 "
                                      "edges and nbins probabilities");
        double sum = 0;
        for (size_t b = 0; b < p.size(); b++) {
          if (!(edges[b] >= 0) || !(edges[b + 1] >= edges[b]) || !std::isfinite(edges[b + 1]))
            throw std::invalid_argument("production heights: edges must be finite, >= 0 and "
                                        "non-decreasing");
          if (!(p[b] >= 0) || !std::isfinite(p[b]))
            throw std::invalid_argument("production heights: probabilities must be >= 0");
          sum += p[b];
        }
        if (!(sum > 0))
          throw std::invalid_argument("production heights: probabilities sum to zero");
        for (size_t b = 0; b < p.size(); b++) {
          lo.push_back(edges[b]);
          hi.push_back(edges[b + 1]);
          prob.push_back(p[b] / sum);
        }
        off.push_back(int(lo.size()));
        return ndist() - 1;
      }

      /// A fixed production height h (km above the atmosphere's base).
      int add_fixed(double h) { return add({h, h}, {1}); }

      /// Take the atmosphere (top layer of a PREM model, top depth shell of
      /// a binned Earth) and the detector radius from an Earth model. Throws
      /// std::invalid_argument if the model has no such atmosphere: the top
      /// layer must be a single medium (one density, Z/A and type) whose
      /// type occurs nowhere else, with the detector below it.
      void set_earth(const EarthModel& e)
      {
        const EarthTable<double> t(e);
        const int                n = t.rows();
        if (n < 2) throw std::invalid_argument("production heights: the Earth model needs an "
                                               "atmosphere layer above other layers");
        if (!e.is_binned()) {
          r_surf      = t.radius[size_t(n - 2)];
          r_det       = t.det_radius;
          atm_density = t.density[size_t(n - 1)];
          atm_zoa     = t.zoa[size_t(n - 1)];
          atm_type    = t.type[size_t(n - 1)];
          if (t.det_layer >= n - 1)
            throw std::invalid_argument("production heights: the detector is in the top "
                                        "(atmosphere) layer");
          for (int l = 0; l < n - 1; l++)
            if (t.type[size_t(l)] == atm_type)
              throw std::invalid_argument("production heights: the atmosphere's layer type "
                                          "also occurs below it");
        }
        else {
          double rmax = 0;
          for (double r : t.rout) rmax = std::fmax(rmax, r);
          bool first = true;
          for (int b = 0; b < n; b++) {
            const bool top = t.rout[size_t(b)] == rmax;
            if (top && first) {
              r_surf      = t.rin[size_t(b)];
              atm_density = t.density[size_t(b)];
              atm_zoa     = t.zoa[size_t(b)];
              atm_type    = t.type[size_t(b)];
              first       = false;
            }
            else if (top) {
              if (t.rin[size_t(b)] != r_surf || t.density[size_t(b)] != atm_density ||
                  t.zoa[size_t(b)] != atm_zoa || t.type[size_t(b)] != atm_type)
                throw std::invalid_argument("production heights: the top depth shell of the "
                                            "binned Earth must be one medium (density, Z/A, "
                                            "region)");
            }
          }
          for (int b = 0; b < n; b++)
            if (t.rout[size_t(b)] != rmax && t.type[size_t(b)] == atm_type)
              throw std::invalid_argument("production heights: the atmosphere's region also "
                                          "occurs below it");
          r_det = t.geom.detRadius;
        }
        if (r_det > r_surf)
          throw std::invalid_argument("production heights: the detector is above the "
                                      "atmosphere's base");
      }

      /// View with host pointers.
      template <class R> ProdHeightView<R> view() const
      {
        ProdHeightView<R> v;
        v.ndist       = ndist();
        v.off         = off.data();
        v.lo          = lo.data();
        v.hi          = hi.data();
        v.prob        = prob.data();
        v.r_surf      = r_surf;
        v.r_det       = r_det;
        v.atm_density = R(atm_density);
        v.atm_zoa     = R(atm_zoa);
        v.atm_type    = atm_type;
        return v;
      }

      /// Path length (km) through the atmosphere for production height h
      /// and direction cosZ, as used by the averages.
      double path_length(double cosZ, double h) const
      {
        ProdHeightView<double> v = view<double>();
        return prodh::atm_length(prodh::atm_geom(v, cosZ), h);
      }
  };

} // namespace opg

#endif
