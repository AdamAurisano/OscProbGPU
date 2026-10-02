///////////////////////////////////////////////////////////////////////////////
/// \file constants.h
///
/// \brief Physical constants and unit conversions, copied verbatim from
///        OscProb (PMNS_Base.cxx, "PDG 2015") so results agree bit-for-bit.
///
/// kK2 is defined in OscProb through std::pow(kKm2eV, 3). Since x*x*x can
/// differ from pow(x, 3) in the last bit, the combination that is actually
/// used on the device (the matter-potential prefactor kK2*sqrt(2)*Gf) is
/// computed on the host with std::pow and shipped inside each model's
/// Prepared struct. See matter_prefactor().
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_CORE_CONSTANTS_H
#define OPG_CORE_CONSTANTS_H

#include <cmath>

namespace opg {
  namespace constants {

    constexpr double kGeV2eV = 1.0e+09;               ///< GeV to eV
    constexpr double kKm2eV  = 1.0 / 1.973269788e-10; ///< km to eV^-1 (1/hbar.c)
    constexpr double kNA     = 6.022140857e23;        ///< Avogadro constant
    constexpr double kGf     = 1.1663787e-05; ///< G_F/(hbar*c)^3 [GeV^-2]

    /// N_A * (hbar*c [GeV.cm])^3 * kGeV2eV  (mol/GeV^2/cm^3 to eV)
    inline double kK2() { return 1e-3 * kNA / std::pow(kKm2eV, 3); }

    /// Matter potential prefactor kK2*sqrt(2)*G_F in eV per (g/cm^3),
    /// evaluated in the same order of operations as OscProb.
    inline double matter_prefactor() { return kK2() * M_SQRT2 * kGf; }

  } // namespace constants
} // namespace opg

#endif
