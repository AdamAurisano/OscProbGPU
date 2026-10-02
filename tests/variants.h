// Parameter sets used to generate the reference files. These must match
// reference/dump_reference.cxx (which follows OscProb/test/Utils.h).

#ifndef OPG_TESTS_VARIANTS_H
#define OPG_TESTS_VARIANTS_H

#include <cmath>

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

} // namespace variants

#endif
