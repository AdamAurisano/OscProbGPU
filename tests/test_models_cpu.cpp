// Model-by-model comparison of the CPU backend against OscProb.

#include "doctest.h"

#include "ref_compare.h"
#include "variants.h"

#include "opg/models/all.h"

namespace {

  constexpr double kTolCPU = 1e-11;

  template <class Model>
  void check_all(opg::Propagator<Model>& prop, const std::string& tag,
                 double tol = kTolCPU)
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
    CHECK(rt.max_abs < tol);
    CHECK(rv.max_abs < tol);
    CHECK(rp.max_abs < tol);
    CHECK(re.max_abs < tol);
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

TEST_CASE("NSI (CPU) matches OscProb PMNS_NSI")
{
  opg::Propagator<opg::NSI<>> prop;
  prop.set_params(variants::nsi());
  check_all(prop, "nsi");
  prop.set_params(variants::nsi_phases());
  check_all(prop, "nsi_phases");
}

TEST_CASE("NUNM (CPU) matches OscProb PMNS_NUNM")
{
  opg::Propagator<opg::NUNM<>> prop;
  prop.set_params(variants::nunm(0));
  check_all(prop, "nunm");
  prop.set_params(variants::nunm_phases());
  check_all(prop, "nunm_phases");
  prop.set_params(variants::nunm(1));
  check_all(prop, "nunm_high");
}

TEST_CASE("Sterile 3+1 (CPU) matches OscProb PMNS_Sterile(4)")
{
  opg::Propagator<opg::Sterile<>> prop;
  opg::Sterile<>::Params           p;
  // A different eigensolver (Jacobi vs Eigen's QR) gives round-off level
  // differences; with Dm41 ~ 1 eV^2 the phases reach ~1e4 rad, so OscProb's
  // own error is ~1e-11 (see the long-double test below).
  const double tol = 1e-10;
  p.mix = variants::sterile_mix();
  prop.set_params(p);
  check_all(prop, "sterile", tol);
  p.mix = variants::sterile_phases_mix();
  prop.set_params(p);
  check_all(prop, "sterile_phases", tol);
}

TEST_CASE("Sterile 3+1 (CPU) is accurate against a long-double calculation")
{
  using LD = long double;
  auto E   = npy::load<double>("grid_E_test.npy");
  std::vector<LD> EL(E.data.begin(), E.data.end());
  for (int v = 0; v < 2; v++) {
    opg::Sterile<LD>::Params     pl;
    opg::Sterile<double>::Params pd;
    pl.mix = pd.mix = v ? variants::sterile_phases_mix() : variants::sterile_mix();
    opg::Propagator<opg::Sterile<LD>>     pL;
    opg::Propagator<opg::Sterile<double>> pD;
    pL.set_params(pl);
    pD.set_params(pd);
    for (int w = 0; w < 2; w++) {
      std::vector<opg::Segment<LD>> sl;
      for (auto& sg : w ? refcmp::vacuum_path() : refcmp::test_path())
        sl.push_back({sg.length, sg.density, sg.zoa, sg.layer});
      const auto& sd = w ? refcmp::vacuum_path() : refcmp::test_path();
      double      m  = 0;
      for (int nb = 0; nb < 2; nb++) {
        auto oL = pL.prob_path(EL, sl, nb);
        auto oD = pD.prob_path(E.data, sd, nb);
        for (size_t i = 0; i < oD.size(); i++)
          m = std::max(m, std::fabs(oD[i] - double(oL[i])));
      }
      MESSAGE("Sterile variant " << v << std::string(w ? " vacuum" : " testpath")
                                 << ": max |P - P_longdouble| = " << m);
      CHECK(m < 1e-11);
    }
  }
}

TEST_CASE("Decay (CPU) matches OscProb PMNS_Decay")
{
  // Our expm follows Eigen's algorithm, but not its exact operation order.
  const double tol = 1e-10;
  opg::Propagator<opg::Decay<>> prop;
  prop.set_params(variants::decay());
  check_all(prop, "decay", tol);
  prop.set_params(variants::decay_both());
  check_all(prop, "decay_both", tol);
}
