// Phase 2: core numerics against OscProb.

#include <complex>
#include <random>

#include "doctest.h"

#include "npy.h"
#include "variants.h"

#include "opg/avg/gauss_legendre.h"
#include "opg/earth/prem.h"
#include "opg/linalg/kopp/zheevh3.h"
#include "opg/physics/mixing.h"

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
