// Model-by-model comparison of the CPU backend against OscProb.

#include "doctest.h"

#include "ref_compare.h"
#include "variants.h"

#include "opg/models/fast.h"

namespace {

  constexpr double kTolCPU = 1e-11;

  template <class Model>
  void check_all(opg::Propagator<Model>& prop, const std::string& tag)
  {
    auto rt = refcmp::compare_path(prop, tag + "_testpath.npy", refcmp::test_path());
    auto rv = refcmp::compare_path(prop, tag + "_vacuum.npy", refcmp::vacuum_path());
    auto rp = refcmp::compare_prem(prop, tag + "_prem.npy");
    auto re = refcmp::compare_points(prop, tag + "_prem.npy");
    MESSAGE(tag << ": testpath max|dP| = " << rt.max_abs << " (" << rt.n_exact
                << "/" << rt.n << " exact), vacuum " << rv.max_abs << " ("
                << rv.n_exact << "/" << rv.n << "), prem " << rp.max_abs
                << " (" << rp.n_exact << "/" << rp.n << "), points "
                << re.max_abs);
    CHECK(rt.max_abs < kTolCPU);
    CHECK(rv.max_abs < kTolCPU);
    CHECK(rp.max_abs < kTolCPU);
    CHECK(re.max_abs < kTolCPU);
  }

} // namespace

TEST_CASE("Fast (CPU) matches OscProb PMNS_Fast")
{
  using M = opg::Fast<double>;
  opg::Propagator<M> prop;

  M::Params p;
  p.mix = variants::nominal_mix<3>();
  prop.set_params(p);
  check_all(prop, "fast");

  p.mix = variants::fast_io_mix();
  prop.set_params(p);
  check_all(prop, "fast_io");
}
