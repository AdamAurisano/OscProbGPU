// Phase 2: core numerics against OscProb.

#include <complex>
#include <random>

#include "doctest.h"

#include "npy.h"
#include "variants.h"

#include "opg/avg/gauss_legendre.h"
#include "opg/earth/prem.h"
#include "opg/linalg/expm.h"
#include "opg/linalg/jacobi_herm.h"
#include "opg/models/decay.h"
#include "opg/linalg/kopp/zheevh3.h"
#include "opg/physics/mixing.h"

#ifdef OPG_HAVE_EIGEN
#include <unsupported/Eigen/MatrixFunctions>
#endif

#ifdef OPG_HAVE_KOPP_ORIG
#include "zheevc3.h"
#include "zheevh3.h"
#include "zheevq3.h"
#endif

using M3 = opg::Mat<3, double>;
using C  = opg::Complex<double>;

namespace {

  M3 random_hermitian(std::mt19937_64& rng, double scale, bool degenerate)
  {
    std::uniform_real_distribution<double> u(-1, 1);
    M3                                     A = M3::zero();
    for (int i = 0; i < 3; i++) {
      A(i, i) = C(scale * u(rng));
      for (int j = i + 1; j < 3; j++) A(i, j) = C(scale * u(rng), scale * u(rng));
    }
    if (degenerate) {
      // Rank-1 perturbation of a multiple of the identity: two equal
      // eigenvalues, which forces the QL fallback in zheevh3.
      C v[3] = {C(u(rng), u(rng)), C(u(rng), u(rng)), C(u(rng), u(rng))};
      double d = scale * u(rng);
      for (int i = 0; i < 3; i++)
        for (int j = i; j < 3; j++) {
          A(i, j) = v[i] * opg::conj(v[j]) * scale;
          if (i == j) A(i, j) = C(A(i, j).re + d);
        }
    }
    hermitize_from_upper(A);
    return A;
  }

  // max |A Q - Q diag(w)|
  double eig_residual(const M3& A, const M3& Q, const double w[3])
  {
    double r = 0;
    for (int i = 0; i < 3; i++)
      for (int k = 0; k < 3; k++) {
        C s(0, 0);
        for (int j = 0; j < 3; j++) s += A(i, j) * Q(j, k);
        s -= Q(i, k) * w[k];
        r = std::max(r, opg::abs(s));
      }
    return r;
  }

} // namespace

//.............................................................................
TEST_CASE("zheevh3 port: eigen-decomposition is correct")
{
  std::mt19937_64 rng(42);
  for (int t = 0; t < 2000; t++) {
    bool   deg   = (t % 4 == 3);
    double scale = std::pow(10.0, -14 + 2 * (t % 8));  // eV-ish to O(1)
    M3     A     = random_hermitian(rng, scale, deg);
    M3     Q;
    double w[3];
    REQUIRE(opg::kopp::zheevh3(A, Q, w) == 0);
    double amax = 0;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) amax = std::max(amax, opg::abs(A(i, j)));
    CHECK(eig_residual(A, Q, w) <= 1e-12 * amax);
  }
}

#ifdef OPG_HAVE_KOPP_ORIG
TEST_CASE("zheevh3 port is bit-identical to the original MatrixDecomp")
{
  std::mt19937_64 rng(7);
  int             nfallback = 0, nfallback_mismatch = 0;
  for (int t = 0; t < 5000; t++) {
    bool   deg   = (t % 5 == 4);
    double scale = std::pow(10.0, -14 + 2 * (t % 8));
    M3     A     = random_hermitian(rng, scale, deg);

    std::complex<double> Ao[3][3], Qo[3][3];
    double               wo[3];
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) Ao[i][j] = {A(i, j).re, A(i, j).im};
    ::zheevh3(Ao, Qo, wo);

    M3     Q;
    double w[3];
    opg::kopp::zheevh3(A, Q, w);

    // Detect whether the QL fallback was taken (eigenvalues then differ
    // from the Cardano ones).
    double wc[3];
    ::zheevc3(Ao, wc);
    bool fallback = !(wc[0] == wo[0] && wc[1] == wo[1] && wc[2] == wo[2]);
    nfallback += fallback;

    bool same = true;
    for (int i = 0; i < 3; i++) {
      same = same && (w[i] == wo[i]);
      for (int j = 0; j < 3; j++)
        same = same && Q(i, j).re == Qo[i][j].real() &&
               Q(i, j).im == Qo[i][j].imag();
    }
    if (!fallback) { CHECK(same); }
    else if (!same) {
      // Complex division in zhetrd3 may differ in the last ulp between our
      // Smith division and libgcc's __divdc3; require near equality.
      nfallback_mismatch++;
      for (int i = 0; i < 3; i++) {
        CHECK(std::fabs(w[i] - wo[i]) <= 1e-14 * std::fabs(wo[i]) + 1e-300);
        for (int j = 0; j < 3; j++)
          CHECK(std::abs(std::complex<double>(Q(i, j).re, Q(i, j).im) -
                         Qo[i][j]) <= 1e-13);
      }
    }
  }
  MESSAGE("QL fallback taken in " << nfallback << " cases, "
                                  << nfallback_mismatch
                                  << " not bit-identical");
  CHECK(nfallback > 100);
  CHECK(nfallback < 4900);  // the Cardano branch is exercised too
}
#endif

//.............................................................................
template <int N>
void check_hms(const std::string& tag, const opg::MixingParams<N>& p)
{
  auto ref = npy::load<double>(tag + "_hms.npy");
  REQUIRE(ref.shape.size() == 3);
  REQUIRE(ref.shape[0] == size_t(N));
  auto H = opg::build_hms<N, double>(p);
  for (int i = 0; i < N; i++)
    for (int j = i; j < N; j++) {
      INFO(tag << " Hms(" << i << "," << j << ")");
      CHECK(H(i, j).re == ref[(i * N + j) * 2 + 0]);
      CHECK(H(i, j).im == ref[(i * N + j) * 2 + 1]);
    }
}

TEST_CASE("Decay effective mass matrix matches OscProb PMNS_Decay fHms")
{
  for (auto tag : {"decay", "decay_both"}) {
    auto ref = npy::load<double>(std::string(tag) + "_hms.npy");
    auto P   = opg::Decay<>::prepare(std::string(tag) == "decay"
                                         ? variants::decay()
                                         : variants::decay_both());
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        INFO(tag << " Heff(" << i << "," << j << ")");
        CHECK(P.Heff[0](i, j).re == ref[(i * 3 + j) * 2 + 0]);
        CHECK(P.Heff[0](i, j).im == ref[(i * 3 + j) * 2 + 1]);
      }
  }
}

TEST_CASE("build_hms is bit-identical to OscProb fHms")
{
  check_hms<3>("fast", variants::nominal_mix<3>());
  check_hms<3>("fast_io", variants::fast_io_mix());
  check_hms<3>("nsi", variants::nominal_mix<3>());
  check_hms<4>("sterile", variants::sterile_mix());
  check_hms<4>("sterile_phases", variants::sterile_phases_mix());
}

//.............................................................................
TEST_CASE("PREM paths are bit-identical to OscProb PremModel::FillPath")
{
  auto C     = npy::load<double>("grid_cosZ.npy");
  auto segs  = npy::load<double>("prem_paths.npy");
  auto nsegs = npy::load<int32_t>("prem_npaths.npy");
  const size_t maxseg = segs.shape[1];

  opg::PremModel prem;
  CHECK(prem.GetDetRadius() == 6368);

  for (size_t ic = 0; ic < C.size(); ic++) {
    auto path = prem.FillPath(C[ic]);
    INFO("cosZ = " << C[ic]);
    REQUIRE(int(path.size()) == nsegs[ic]);
    for (size_t k = 0; k < path.size(); k++) {
      const double* r = &segs.data[(ic * maxseg + k) * 4];
      CHECK(path[k].length == r[0]);
      CHECK(path[k].density == r[1]);
      CHECK(path[k].zoa == r[2]);
      CHECK(path[k].layer == int(r[3]));
    }
  }
}

TEST_CASE("PremModel helpers")
{
  opg::PremModel prem;
  // Total baseline for an upgoing vertical neutrino: Rdet + Rmax
  CHECK(prem.GetTotalL(-1) == doctest::Approx(6368 + 6386));
  CHECK(prem.GetCosT(prem.GetTotalL(-0.3)) == doctest::Approx(-0.3));
  // Sum of segment lengths equals the total baseline
  for (double c : {-0.99, -0.5, -0.1, 0.2, 0.9}) {
    double L = 0;
    for (auto& s : prem.FillPath(c)) L += s.length;
    CHECK(L == doctest::Approx(prem.GetTotalL(c)).epsilon(1e-10));
  }
}

//.............................................................................
TEST_CASE("Gauss-Legendre is exact up to degree 2n-1")
{
  for (int n = 1; n <= 24; n++) {
    opg::GaussLegendre gl(n);
    double             wsum = 0;
    for (double w : gl.w) wsum += w;
    CHECK(wsum == doctest::Approx(2).epsilon(1e-14));
    for (int d = 0; d <= 2 * n - 1; d++) {
      double s = 0;
      for (int i = 0; i < n; i++) s += gl.w[i] * std::pow(gl.x[i], d);
      double exact = (d % 2) ? 0 : 2.0 / (d + 1);
      INFO("n=" << n << " d=" << d);
      CHECK(std::fabs(s - exact) < 1e-13);
    }
  }
  // Mapping to [a, b]
  opg::GaussLegendre  gl(5);
  std::vector<double> x, w;
  gl.map(2, 5, x, w);
  double s = 0;
  for (int i = 0; i < 5; i++) s += w[i] * x[i] * x[i];
  CHECK(s == doctest::Approx((125.0 - 8.0) / 3));
}

//.............................................................................
TEST_CASE("Jacobi 4x4 hermitian eigensolver")
{
  using M4 = opg::Mat<4, double>;
  std::mt19937_64                        rng(99);
  std::uniform_real_distribution<double> u(-1, 1);
  int                                    maxsweeps = 0;
  for (int t = 0; t < 3000; t++) {
    double scale = std::pow(10.0, -14 + 2 * (t % 8));
    M4     A     = M4::zero();
    for (int i = 0; i < 4; i++) {
      A(i, i) = C(scale * u(rng));
      for (int j = i + 1; j < 4; j++) A(i, j) = C(scale * u(rng), scale * u(rng));
    }
    if (t % 3 == 0) {  // near-degenerate pair
      A(2, 2) = A(1, 1);
      A(1, 2) = C(scale * 1e-9, 0);
    }
    if (t % 5 == 0) A(0, 3) = C(0, 0);  // some exact zeros
    M4     V;
    double w[4];
    int    ns = opg::jacobi_hermitian<4, double>(A, V, w);
    REQUIRE(ns >= 0);
    maxsweeps = std::max(maxsweeps, ns);
    M4 Af = A;
    opg::hermitize_from_upper(Af);
    double amax = 0, res = 0, orth = 0;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) amax = std::max(amax, opg::abs(Af(i, j)));
    for (int i = 0; i < 4; i++)
      for (int k = 0; k < 4; k++) {
        C s(0, 0), o(0, 0);
        for (int j = 0; j < 4; j++) {
          s += Af(i, j) * V(j, k);
          o += opg::conj(V(j, i)) * V(j, k);
        }
        s -= V(i, k) * w[k];
        res  = std::max(res, opg::abs(s));
        orth = std::max(orth, opg::abs(o - C(i == k ? 1 : 0)));
      }
    CHECK(res <= 2e-15 * amax * 4);
    CHECK(orth <= 2e-15 * 4);
  }
  MESSAGE("Jacobi: max sweeps used = " << maxsweeps);
}

//.............................................................................
namespace {
  // exp(A) by Taylor series in long double with scaling and squaring
  opg::Mat<3, long double> expm_reference(const opg::Mat<3, double>& A)
  {
    using ML = opg::Mat<3, long double>;
    using CL = opg::Complex<long double>;
    ML   B;
    long double nrm = 0;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        B(i, j) = CL(A(i, j).re, A(i, j).im);
        nrm     = std::max(nrm, (long double)opg::abs(A(i, j)));
      }
    int s = 0;
    while (nrm > 0.05L) { nrm /= 2; s++; }
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) B(i, j) = B(i, j) * std::ldexp(1.0L, -s);
    ML sum = ML::identity(), term = ML::identity();
    for (int k = 1; k < 40; k++) {
      term = opg::matmul(term, B);
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) term(i, j) = term(i, j) / (long double)k;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) sum(i, j) += term(i, j);
    }
    for (int k = 0; k < s; k++) sum = opg::matmul(sum, sum);
    return sum;
  }
} // namespace

TEST_CASE("expm (Pade scaling and squaring) is accurate")
{
  std::mt19937_64                        rng(5);
  std::uniform_real_distribution<double> u(-1, 1);
  for (int t = 0; t < 3000; t++) {
    // norms spanning all Pade branches, up to ~1e3 (long baselines)
    double scale = std::pow(10.0, -3 + 6.0 * (t % 13) / 12.0);
    M3     A;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) A(i, j) = C(scale * u(rng), scale * u(rng));
    // decay-like: anti-hermitian (oscillation) plus small damping
    M3 B;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        C h  = (A(i, j) + opg::conj(A(j, i))) * 0.5;  // hermitian part
        C d  = C(i == j ? -0.01 * scale * std::fabs(u(rng)) : 0, 0);
        B(i, j) = C(h.im, -h.re) + d;                 // -i h + d
      }
    M3   X   = opg::expm<3, double>(B);
    auto Xr  = expm_reference(B);
    double err = 0, xmax = 0;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        err  = std::max(err, (double)opg::abs(opg::Complex<long double>(X(i, j).re, X(i, j).im) - Xr(i, j)));
        xmax = std::max(xmax, (double)opg::abs(Xr(i, j)));
      }
    INFO("scale = " << scale);
    // relative accuracy degrades ~linearly with the norm (squarings)
    CHECK(err <= 1e-14 * std::max(1.0, scale) * 10 * xmax);
  }
}

TEST_CASE("long-double expm matches the Taylor reference")
{
  std::mt19937_64                        rng(7);
  std::uniform_real_distribution<double> u(-1, 1);
  for (int t = 0; t < 200; t++) {
    double scale = std::pow(10.0, -3 + 6.0 * (t % 13) / 12.0);
    M3     A;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) A(i, j) = C(scale * u(rng), -scale * std::fabs(u(rng)));
    opg::Mat<3, long double> AL;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) AL(i, j) = {A(i, j).re, A(i, j).im};
    auto X = opg::expm<3, long double>(AL), Xr = expm_reference(A);
    long double err = 0, xmax = 0;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        err  = std::max(err, opg::abs(X(i, j) - Xr(i, j)));
        xmax = std::max(xmax, opg::abs(Xr(i, j)));
      }
    INFO("scale = " << scale);
    CHECK(double(err / xmax) < 1e-16 * std::max(1.0, scale));
  }
}

TEST_CASE("expm in dual arithmetic: unchanged values, exact Frechet derivative")
{
  using D  = opg::Dual<double, 2>;
  using MD = opg::Mat<3, D>;
  using ML = opg::Mat<3, long double>;
  std::mt19937_64                        rng(8);
  std::uniform_real_distribution<double> u(-1, 1);
  double                                 worst = 0;
  for (int t = 0; t < 600; t++) {
    // norms spanning all Pade branches and squarings
    double scale = std::pow(10.0, -3 + 6.0 * (t % 13) / 12.0);
    // decay-like exponents -i (h - i g) with h, g hermitian, g >= 0 small
    M3 A, B, E0, E1;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) A(i, j) = C(scale * u(rng), scale * u(rng));
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        C h      = (A(i, j) + opg::conj(A(j, i))) * 0.5;
        C d      = C(i == j ? -0.01 * scale * std::fabs(u(rng)) : 0, 0);
        B(i, j)  = C(h.im, -h.re) + d;
        E0(i, j) = C(u(rng), u(rng));            // generic direction
        E1(i, j) = C(i == j ? -std::fabs(u(rng)) : 0.0, 0.0);  // damping
      }
    MD BD;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        D re(B(i, j).re), im(B(i, j).im);
        re.d[0] = E0(i, j).re;
        im.d[0] = E0(i, j).im;
        re.d[1] = E1(i, j).re;
        im.d[1] = E1(i, j).im;
        BD(i, j) = opg::Complex<D>(re, im);
      }
    MD XD = opg::expm<3, D>(BD);
    M3 X  = opg::expm<3, double>(B);
    bool same = true;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        same = same && XD(i, j).re.v == X(i, j).re && XD(i, j).im.v == X(i, j).im;
    CHECK(same);

    // long-double 6-point central differences along E0 and E1
    for (int k = 0; k < 2; k++) {
      const M3&   Ek = k ? E1 : E0;
      long double h  = 1e-4L / std::max(1.0, scale);
      auto at = [&](long double x) {
        ML A;
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
            A(i, j) = opg::Complex<long double>(B(i, j).re + x * Ek(i, j).re,
                                                B(i, j).im + x * Ek(i, j).im);
        return opg::expm<3, long double>(A);
      };
      ML p1 = at(h), m1 = at(-h), p2 = at(2 * h), m2 = at(-2 * h), p3 = at(3 * h),
         m3 = at(-3 * h);
      long double err = 0, xmax = 0, fmax = 0;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
          auto fd = ((p1(i, j) - m1(i, j)) * 45.0L - (p2(i, j) - m2(i, j)) * 9.0L +
                     (p3(i, j) - m3(i, j))) /
                    (60 * h);
          opg::Complex<long double> g(XD(i, j).re.d[k], XD(i, j).im.d[k]);
          err  = std::max(err, opg::abs(g - fd));
          fmax = std::max(fmax, opg::abs(fd));
          xmax = std::max(xmax, opg::abs(p1(i, j)));
        }
      // relative to |exp(B)| (the derivative scales like it, times the norm)
      double e = double(err / (xmax * std::max(1.0, scale)));
      worst    = std::max(worst, e);
      INFO("scale = " << scale << ", direction " << k << ", |dX| = " << double(fmax));
      CHECK(e < 1e-13 * std::max(1.0, scale));
    }
  }
  MESSAGE("dual expm derivative vs long-double FD: worst " << worst);
}

#ifdef OPG_HAVE_EIGEN
TEST_CASE("expm agrees with Eigen MatrixExponential")
{
  std::mt19937_64                        rng(6);
  std::uniform_real_distribution<double> u(-1, 1);
  double                                 worst = 0;
  for (int t = 0; t < 3000; t++) {
    double            scale = std::pow(10.0, -3 + 6.0 * (t % 13) / 12.0);
    M3                A;
    Eigen::Matrix3cd  Ae;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        A(i, j)  = C(scale * u(rng), scale * u(rng));
        Ae(i, j) = {A(i, j).re, A(i, j).im};
      }
    M3               X  = opg::expm<3, double>(A);
    Eigen::Matrix3cd Xe = Ae.exp();
    double           nrm = Xe.cwiseAbs().maxCoeff(), err = 0;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        err = std::max(err, std::abs(std::complex<double>(X(i, j).re, X(i, j).im) - Xe(i, j)));
    worst = std::max(worst, err / nrm / std::max(1.0, scale));
  }
  MESSAGE("expm vs Eigen: worst relative difference / max(1, norm) = " << worst);
  CHECK(worst < 1e-13);
}
#endif
