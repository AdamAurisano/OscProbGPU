#include <complex>
#include <random>

#include "doctest.h"

#include "opg/core/complex.h"
#include "opg/core/constants.h"
#include "opg/core/matrix.h"

using C  = opg::Complex<double>;
using SC = std::complex<double>;

namespace {
  SC to_std(const C& a) { return SC(a.re, a.im); }
} // namespace

TEST_CASE("Complex arithmetic matches std::complex<double>")
{
  std::mt19937_64                        rng(1234);
  std::uniform_real_distribution<double> u(-3, 3);

  for (int t = 0; t < 1000; t++) {
    C  a(u(rng), u(rng)), b(u(rng), u(rng));
    SC sa = to_std(a), sb = to_std(b);
    double s = u(rng);

    CHECK(to_std(a + b) == sa + sb);
    CHECK(to_std(a - b) == sa - sb);
    CHECK(to_std(a * b) == sa * sb);
    CHECK(to_std(a * s) == sa * s);
    CHECK(to_std(conj(a)) == std::conj(sa));
    CHECK(opg::norm(a) == std::norm(sa));
    SC q = to_std(a / b) - sa / sb;
    CHECK(std::abs(q) <= 4e-16 * std::abs(sa / sb));
  }
}

TEST_CASE("Mat helpers")
{
  using M = opg::Mat<3, double>;
  M A     = M::zero();
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) A(i, j) = C(i + 1, j - 1);
  M I = M::identity();
  M B = opg::matmul(A, I);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) CHECK(B(i, j) == A(i, j));
  M Ad = opg::adjoint(A);
  CHECK(Ad(0, 2) == opg::conj(A(2, 0)));
}

TEST_CASE("Constants are those of OscProb")
{
  using namespace opg::constants;
  CHECK(kGeV2eV == 1.0e+09);
  CHECK(kKm2eV == 1.0 / 1.973269788e-10);
  // Value printed from OscProb: kK2 * sqrt(2) * Gf ~ 7.63e-14 eV/(g/cm^3)
  CHECK(matter_prefactor() == doctest::Approx(7.63e-14).epsilon(0.01));
}
