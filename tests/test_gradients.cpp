// Gradients (CPU backend).

#include <cstring>
#include <type_traits>

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
  CHECK(opg::Propagator<opg::Sterile<>>::parameter_names() ==
        std::vector<std::string>{"th12", "th13", "th23", "th14", "th24", "th34", "d13",
                                 "d14", "d24", "dm21", "dm31", "dm41"});
  CHECK(opg::Propagator<opg::NSI<>>::parameter_names() ==
        std::vector<std::string>{"th12", "th13", "th23", "d13", "dm21", "dm31",
                                 "eps_ee", "eps_emu", "eps_etau", "eps_mumu",
                                 "eps_mutau", "eps_tautau", "ph_emu", "ph_etau",
                                 "ph_mutau", "coup_e", "coup_u", "coup_d"});
  CHECK(opg::Propagator<opg::NUNM<>>::parameter_names() ==
        std::vector<std::string>{"th12", "th13", "th23", "d13", "dm21", "dm31",
                                 "alpha_ee", "alpha_mue", "alpha_taue", "alpha_mumu",
                                 "alpha_taumu", "alpha_tautau", "ph_mue", "ph_taue",
                                 "ph_taumu", "frac_vnc"});
  CHECK(opg::Propagator<opg::Decay<>>::parameter_names() ==
        std::vector<std::string>{"th12", "th13", "th23", "d13", "dm21", "dm31", "alpha2",
                                 "alpha3"});

  // defaults: everything except NSI's fermion couplings
  CHECK(opg::Propagator<gradtest::Fast>::default_gradient_params() ==
        gradtest::Fast::param_names());
  auto nsi_def = opg::Propagator<opg::NSI<>>::default_gradient_params();
  CHECK(nsi_def.size() == 15);
  CHECK(std::find(nsi_def.begin(), nsi_def.end(), "coup_e") == nsi_def.end());
  CHECK(nsi_def.back() == "ph_mutau");
  {
    opg::Propagator<opg::NSI<>> nsi;
    nsi.set_gradient_params();
    CHECK(nsi.gradient_params() == nsi_def);
    nsi.set_gradient_params({"coup_u"});  // still selectable by name
    CHECK(nsi.n_gradient_params() == 1);
  }

  {
    opg::Propagator<gradtest::Fast> cpu;
    cpu.set_params(gradtest::param_points<gradtest::Fast>()[0].second);
    cpu.set_gradient_params();
    CHECK_THROWS_AS(cpu.weighted_gradient_device({nullptr}), std::logic_error);
  }

  opg::Propagator<gradtest::Fast> prop;
  gradtest::Fast::Params          p;
  p.mix = variants::nominal_mix<3>();
  prop.set_params(p);
  CHECK_THROWS_AS(prop.set_gradient_params({"th99"}), std::invalid_argument);
  prop.set_gradient_params({});
  prop.set_grid({1.0, 2.0}, {-1.0, 0.5});
  CHECK_THROWS_AS(prop.calculate(opg::Flavor::Both, true), std::logic_error);
  prop.calculate();  // probabilities still fine with gradients off

}

TEST_CASE("Dual-number preparation has bit-identical value parts")
{
  using M = gradtest::Fast;
  using D = opg::Dual<double, 3>;
  for (auto& [label, par] : gradtest::param_points<M>()) {
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

TEST_CASE_TEMPLATE("G3/G4 gradients match long-double finite differences (CPU)", M,
                   gradtest::NSI, gradtest::NUNM, gradtest::Sterile,
                   gradtest::Decay)
{
  opg::Propagator<M> prop;
  gradtest::check_against_ld(prop, 1e-9);
}

TEST_CASE_TEMPLATE("G3/G4 probabilities are unchanged when gradients are on (CPU)", M,
                   gradtest::NSI, gradtest::NUNM, gradtest::Sterile,
                   gradtest::Decay)
{
  opg::Propagator<M> prop;
  gradtest::check_values_unchanged(prop);
}

TEST_CASE_TEMPLATE("G3/G4 grid, event-list and weighted gradients are consistent (CPU)", M,
                   gradtest::NSI, gradtest::NUNM, gradtest::Sterile,
                   gradtest::Decay)
{
  opg::Propagator<M> prop;
  gradtest::check_grid_and_weighted(prop, 1e-13);
}

TEST_CASE("NSI/NUNM prepared values do not depend on the derivative seeds")
{
  // value parts of the dual preparation equal prepare() bit for bit
  auto same_bits = [](const auto& PD, const auto& P) {
    using PDT = std::decay_t<decltype(PD)>;
    using PT  = std::decay_t<decltype(P)>;
    using D   = std::decay_t<decltype(PD.common.vfac)>;
    constexpr size_t nd = sizeof(D) / sizeof(double);
    static_assert(sizeof(PDT) == nd * sizeof(PT), "layout");
    const double* a = reinterpret_cast<const double*>(&PD);
    const double* b = reinterpret_cast<const double*>(&P);
    for (size_t i = 0; i < sizeof(PT) / sizeof(double); i++)
      if (std::memcmp(&a[i * nd], &b[i], sizeof(double)) != 0) return false;
    return true;
  };
  using D = opg::Dual<double, 2>;
  for (auto& [label, par] : gradtest::param_points<gradtest::NSI>()) {
    using M = gradtest::NSI;
    auto pd = M::cast<D>(par);
    M::param_ref(pd, 7).d[0]  = 1;
    M::param_ref(pd, 13).d[1] = 1;
    INFO(label);
    CHECK(same_bits(M::prepare_generic<D>(pd), M::prepare(par)));
  }
  for (auto& [label, par] : gradtest::param_points<gradtest::NUNM>()) {
    using M = gradtest::NUNM;
    auto pd = M::cast<D>(par);
    M::param_ref(pd, 7).d[0]  = 1;
    M::param_ref(pd, 15).d[1] = 1;
    INFO(label);
    CHECK(same_bits(M::prepare_generic<D>(pd), M::prepare(par)));
  }
}


TEST_CASE_TEMPLATE("Binned and avg_path gradients (CPU)", M, gradtest::Fast,
                   gradtest::NSI, gradtest::NUNM, gradtest::Sterile, gradtest::Decay)
{
  opg::Propagator<M> prop;
  gradtest::check_binned(prop, 1e-6, 1e-13);
}


TEST_CASE_TEMPLATE("Earth Z/A gradients (CPU)", M, gradtest::Fast, gradtest::NSI, gradtest::NUNM,
                   gradtest::Sterile, gradtest::Decay)
{
  opg::Propagator<M> prop;
  gradtest::check_zoa(prop, 1e-9);
}

#endif  // OPG_DISABLE_GRADIENTS
