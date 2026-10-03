///////////////////////////////////////////////////////////////////////////////
/// \file absorption.h
///
/// \brief Absorption of neutrinos along their path (OscProb::Absorption).
///
/// The probability that a neutrino with a cross section xsec (cm^2 per
/// nucleon) crosses a path without interacting is
///   T = prod_segments exp(-L rho / u * xsec),
/// with L in cm and u the atomic mass unit. It depends on the path only
/// through the column depth X = sum rho L (g/cm^2): T = exp(-X xsec / u).
/// transmission() reproduces OscProb's per-segment product.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_EARTH_ABSORPTION_H
#define OPG_EARTH_ABSORPTION_H

#include <cmath>
#include <vector>

#include "opg/earth/prem.h"

namespace opg {

  /// Atomic mass unit in g (OscProb::Absorption::kU).
  constexpr double kAtomicMassUnit = 1.660539066e-24;

  /// Column depth sum rho L in g/cm^2 (L converted from km).
  template <class Real> double column_depth(const std::vector<Segment<Real>>& path)
  {
    double x = 0;
    for (const auto& s : path) x += double(s.density) * double(s.length) * 1e5;
    return x;
  }

  /// Probability of no absorption along the path (Absorption::Trans).
  template <class Real>
  double transmission(const std::vector<Segment<Real>>& path, double xsec)
  {
    double t = 1;
    for (const auto& s : path) {
      const double n = double(s.density) / kAtomicMassUnit;
      const double l = double(s.length) * 1e5;  // l in km -> cm
      t *= std::exp(-l * n * xsec);
    }
    return t;
  }

} // namespace opg

#endif
