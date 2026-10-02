// Gradients (CPU backend).

#include "gradients.h"

#ifdef OPG_DISABLE_GRADIENTS
TEST_CASE("Gradients are disabled at build time")
{
  CHECK_FALSE(opg::Propagator<gradtest::Fast>::has_gradients());
  CHECK(opg::Propagator<gradtest::Fast>::parameter_names().empty());
  opg::Propagator<gradtest::Fast> prop;
  CHECK_THROWS_AS(prop.set_gradient_params({"th12"}), std::logic_error);
}
#else

TEST_CASE("Gradient parameter registry")
{
  auto n = opg::Propagator<gradtest::Fast>::parameter_names();
  CHECK(n == std::vector<std::string>{"th12", "th13", "th23", "d13", "dm21", "dm31"});
  CHECK(opg::Propagator<opg::Sterile<>>::parameter_names().empty());  // not yet

  opg::Propagator<gradtest::Fast> prop;
  gradtest::Fast::Params          p;
  p.mix = variants::nominal_mix<3>();
  prop.set_params(p);
  CHECK_THROWS_AS(prop.set_gradient_params({"th99"}), std::invalid_argument);
  prop.set_gradient_params({});
  prop.set_grid({1.0, 2.0}, {-1.0, 0.5});
  CHECK_THROWS_AS(prop.calculate(opg::Flavor::Both, true), std::logic_error);
  prop.calculate();  // probabilities still fine with gradients off

  opg::Propagator<opg::NSI<>> nsi;
  CHECK_THROWS_AS(nsi.set_gradient_params({"th12"}), std::logic_error);
}

TEST_CASE("Dual-number preparation has bit-identical value parts")
{
  using M = gradtest::Fast;
  using D = opg::Dual<double, 3>;
  for (auto& [label, par] : gradtest::param_points()) {
    auto P  = M::prepare(par);
    auto pd = M::cast<D>(par);
    for (int k = 0; k < 3; k++) M::param_ref(pd, k).d[k] = 1;
    auto PD = M::prepare_generic<D>(pd);
    bool same = PD.common.vfac.v == P.common.vfac;
    for (int i = 0; i < 3; i++) {
      same = same && PD.common.dm[i].v == P.common.dm[i];
      for (int j = 0; j < 3; j++) {
        same = same && PD.common.Hms(i, j).re.v == P.common.Hms(i, j).re &&
               PD.common.Hms(i, j).im.v == P.common.Hms(i, j).im;
        for (int nb = 0; nb < 2; nb++)
          same = same && PD.common.Uvac[nb](i, j).re.v == P.common.Uvac[nb](i, j).re &&
                 PD.common.Uvac[nb](i, j).im.v == P.common.Uvac[nb](i, j).im;
      }
    }
    INFO(label);
    CHECK(same);
  }
}

TEST_CASE("Probabilities are unchanged when gradients are on (CPU)")
{
  opg::Propagator<gradtest::Fast> prop;
  gradtest::check_values_unchanged(prop);
}

TEST_CASE("Fast gradients match long-double finite differences (CPU)")
{
  opg::Propagator<gradtest::Fast> prop;
  gradtest::check_against_ld(prop, 1e-9);
}

TEST_CASE("Grid, event-list and weighted gradients are consistent (CPU)")
{
  opg::Propagator<gradtest::Fast> prop;
  gradtest::check_grid_and_weighted(prop, 1e-13);

  // weighted results do not depend on the number of threads
  opg::Propagator<gradtest::Fast> p1(opg::PremModel(), {}, 1), p4(opg::PremModel(), {}, 4);
  gradtest::Fast::Params          par;
  par.mix = variants::nominal_mix<3>();
  std::vector<double> E = {0.7, 1.3, 2.9, 7.0, 20.0}, C = {-0.95, -0.4, 0.3};
  std::vector<double> w(2 * 9 * E.size() * C.size());
  for (size_t i = 0; i < w.size(); i++) w[i] = std::sin(1.0 + i);
  std::vector<std::vector<double>> g;
  for (auto* p : {&p1, &p4}) {
    p->set_params(par);
    p->set_gradient_params(gradtest::Fast::param_names());
    p->set_grid(E, C);
    g.push_back(p->weighted_gradient(w));
  }
  CHECK(g[0] == g[1]);
}

#endif  // OPG_DISABLE_GRADIENTS
