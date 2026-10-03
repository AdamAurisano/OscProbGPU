// Parameter sets used to generate the reference files. These must match
// reference/dump_reference.cxx (which follows OscProb/test/Utils.h).

#ifndef OPG_TESTS_VARIANTS_H
#define OPG_TESTS_VARIANTS_H

#include <cmath>

#include "opg/models/all.h"
#include "opg/physics/mixing.h"

namespace variants {

  /// NuFIT 5.2 NO, as SetNominalPars in OscProb/test/Utils.h
  template <int N> opg::MixingParams<N> nominal_mix()
  {
    opg::MixingParams<N> p;
    p.SetDm(2, 7.41e-5);
    p.SetDm(3, 2.507e-3);
    p.SetAngle(1, 2, std::asin(std::sqrt(0.303)));
    p.SetAngle(1, 3, std::asin(std::sqrt(0.02225)));
    p.SetAngle(2, 3, std::asin(std::sqrt(0.451)));
    p.SetDelta(1, 3, 232 * M_PI / 180);
    return p;
  }

  inline opg::MixingParams<3> fast_io_mix()
  {
    auto p = nominal_mix<3>();
    p.SetDm(3, -2.465e-3 + 7.41e-5);
    p.SetDelta(1, 3, 1.1);
    return p;
  }

  inline opg::MixingParams<4> sterile_mix()
  {
    auto p = nominal_mix<4>();
    p.SetDm(4, 0.1);
    p.SetAngle(1, 4, 0.1);
    p.SetAngle(2, 4, 0.1);
    p.SetAngle(3, 4, 0.1);
    return p;
  }

  inline opg::MixingParams<4> sterile_phases_mix()
  {
    auto p = nominal_mix<4>();
    p.SetDm(4, 1.3);
    p.SetAngle(1, 4, 0.15);
    p.SetAngle(2, 4, 0.2);
    p.SetAngle(3, 4, 0.3);
    p.SetDelta(1, 4, 0.9);
    p.SetDelta(2, 4, -1.7);
    return p;
  }

  //...........................................................................
  inline opg::NSI<>::Params nsi()
  {
    opg::NSI<>::Params p;
    p.mix = nominal_mix<3>();
    p.SetEps(0, 0, 0.1, 0);
    p.SetEps(0, 1, 0.2, 0);
    p.SetEps(0, 2, 0.3, 0);
    p.SetEps(1, 1, 0.4, 0);
    p.SetEps(1, 2, 0.5, 0);
    p.SetEps(2, 2, 0.6, 0);
    return p;
  }

  inline opg::NSI<>::Params nsi_phases()
  {
    opg::NSI<>::Params p;
    p.mix = nominal_mix<3>();
    p.SetEps(0, 0, -0.2, 0);
    p.SetEps(0, 1, 0.05, 0.7);
    p.SetEps(0, 2, 0.15, -1.3);
    p.SetEps(1, 1, 0.02, 0);
    p.SetEps(1, 2, 0.03, 2.1);
    p.SetEps(2, 2, 0.1, 0);
    p.SetFermCoup(0.5, 1.0, 0.8);
    return p;
  }

  inline opg::NUNM<>::Params nunm(int scale = 0)
  {
    opg::NUNM<>::Params p;
    p.mix   = nominal_mix<3>();
    p.scale = scale;
    p.SetAlpha(0, 0, 0.05, 0);
    p.SetAlpha(1, 0, 0.06, 0);
    p.SetAlpha(2, 0, 0.07, 0);
    p.SetAlpha(1, 1, 0.08, 0);
    p.SetAlpha(2, 1, 0.09, 0);
    p.SetAlpha(2, 2, 0.1, 0);
    return p;
  }

  inline opg::NUNM<>::Params nunm_phases()
  {
    opg::NUNM<>::Params p;
    p.mix = nominal_mix<3>();
    p.SetAlpha(0, 0, -0.03, 0);
    p.SetAlpha(1, 0, 0.02, 0.4);
    p.SetAlpha(2, 0, 0.04, -2.0);
    p.SetAlpha(1, 1, -0.01, 0);
    p.SetAlpha(2, 1, 0.05, 1.2);
    p.SetAlpha(2, 2, -0.02, 0);
    p.SetFracVnc(0.7);
    return p;
  }

  inline opg::Decay<>::Params decay()
  {
    opg::Decay<>::Params p;
    p.mix = nominal_mix<3>();
    p.SetAlpha3(1e-4);
    return p;
  }

  inline opg::Decay<>::Params decay_both()
  {
    opg::Decay<>::Params p;
    p.mix = nominal_mix<3>();
    p.SetAlpha2(3e-5);
    p.SetAlpha3(2e-4);
    return p;
  }

  //...........................................................................
  inline opg::LIV<>::Params liv()
  {
    opg::LIV<>::Params p;
    p.mix = nominal_mix<3>();
    p.SetaT(0, 0, 3, 1e-21, 0);
    p.SetaT(0, 1, 3, 2e-21, 0);
    p.SetaT(1, 2, 3, -1e-21, 0);
    p.SetcT(1, 1, 4, 5e-23, 0);
    p.SetcT(0, 2, 4, 1e-22, 0);
    p.SetaT(1, 2, 5, 1e-24, 0);
    p.SetcT(2, 2, 6, 2e-26, 0);
    p.SetaT(0, 1, 7, 1e-28, 0);
    p.SetcT(1, 2, 8, 1e-30, 0);
    return p;
  }

  inline opg::LIV<>::Params liv_phases()
  {
    opg::LIV<>::Params p;
    p.mix = nominal_mix<3>();
    p.SetaT(0, 1, 3, 1.5e-21, 0.7);
    p.SetaT(0, 2, 3, 8e-22, -1.9);
    p.SetaT(2, 2, 3, -6e-22, 0);
    p.SetcT(1, 2, 4, 1e-22, 2.3);
    p.SetcT(0, 0, 4, -4e-23, 0);
    p.SetcT(0, 1, 6, 3e-26, 1.1);
    return p;
  }

  inline opg::SNSI<>::Params snsi()
  {
    opg::SNSI<>::Params p;
    p.mix = nominal_mix<3>();
    p.SetLowestMass(0.05);
    p.SetEps(0, 0, 0.5, 0);
    p.SetEps(0, 1, 0.3, 0);
    p.SetEps(1, 1, -0.2, 0);
    p.SetEps(1, 2, 0.4, 0);
    p.SetEps(2, 2, 0.1, 0);
    return p;
  }

  inline opg::SNSI<>::Params snsi_io()
  {
    opg::SNSI<>::Params p;
    p.mix = nominal_mix<3>();
    p.mix.SetDm(3, -2.465e-3 + 7.41e-5);
    p.SetLowestMass(0);
    p.SetEps(0, 1, 0.4, 0.9);
    p.SetEps(0, 2, 0.2, -2.1);
    p.SetEps(1, 1, 0.3, 0);
    p.SetFermCoup(0.5, 1.0, 0.8);
    return p;
  }

} // namespace variants

#endif
