///////////////////////////////////////////////////////////////////////////////
/// \file hermitian3.h
///
/// \brief Shared machinery for 3-flavour models whose Hamiltonian is
///        hermitian and which inherit PMNS_Fast::SolveHam in OscProb
///        (PMNS_Fast, PMNS_NSI, PMNS_NUNM).
///
/// In matter, the Hamiltonian built by the model's hamiltonian() is
/// diagonalised with the ported Kopp zheevh3. For densities below 1e-6
/// g/cm^3, OscProb uses the analytic vacuum eigensystem
/// (PMNS_Fast::SetVacuumEigensystem), which we precompute in prepare().
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_HERMITIAN3_H
#define OPG_MODELS_HERMITIAN3_H

#include <cmath>
#include <complex>

#include "opg/core/constants.h"
#include "opg/linalg/kopp/zheevh3.h"
#include "opg/physics/mixing.h"
#include "opg/physics/propagate.h"

namespace opg {

  /// Parts of the prepared state common to all hermitian 3-flavour models.
  template <class R> struct Hermitian3Common {
      Mat<3, R> Hms;      ///< U diag(dm) U^dagger (upper triangle), eV^2
      Mat<3, R> Uvac[2];  ///< vacuum eigenvectors for nu / nubar
      R         dm[3];    ///< mass splittings dm_j1 (dm[0] = 0)
      R         vfac;     ///< kK2*sqrt(2)*G_F, matter potential prefactor
  };

  /// Port of PMNS_Fast::SetVacuumEigensystem (host, std::complex).
  template <class R>
  void prepare_vacuum3(const MixingParams<3>& p, Mat<3, R> Uvac[2])
  {
    using cplx = std::complex<double>;
    for (int nb = 0; nb < 2; nb++) {
      double s12, s23, s13, c12, c23, c13;
      cplx   idelta(0.0, p.dcp[0][2]);
      if (nb) idelta = conj(idelta);

      s12 = std::sin(p.th[0][1]);
      s23 = std::sin(p.th[1][2]);
      s13 = std::sin(p.th[0][2]);
      c12 = std::cos(p.th[0][1]);
      c23 = std::cos(p.th[1][2]);
      c13 = std::cos(p.th[0][2]);

      cplx E[3][3];
      E[0][0] = c12 * c13;
      E[0][1] = s12 * c13;
      E[0][2] = s13 * exp(-idelta);

      E[1][0] = -s12 * c23 - c12 * s23 * s13 * exp(idelta);
      E[1][1] = c12 * c23 - s12 * s23 * s13 * exp(idelta);
      E[1][2] = s23 * c13;

      E[2][0] = s12 * s23 - c12 * c23 * s13 * exp(idelta);
      E[2][1] = -c12 * s23 - s12 * c23 * s13 * exp(idelta);
      E[2][2] = c23 * c13;

      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          Uvac[nb](i, j) = Complex<R>(R(E[i][j].real()), R(E[i][j].imag()));
    }
  }

  template <class R>
  void prepare_hermitian3(const MixingParams<3>& p, Hermitian3Common<R>& c)
  {
    c.Hms = build_hms<3, R>(p);
    prepare_vacuum3<R>(p, c.Uvac);
    c.dm[0] = 0;
    c.dm[1] = R(p.dm[1]);
    c.dm[2] = R(p.dm[2]);
    c.vfac  = R(constants::matter_prefactor());
  }

  /// One segment step for a hermitian 3-flavour model. Model must provide
  ///   OPG_HD static void hamiltonian(const Prepared&, R E, bool nubar,
  ///                                  const Segment<R>&, Mat<3,R>& H)
  /// filling at least the diagonal and upper triangle of H, and Prepared
  /// must have a member `common` of type Hermitian3Common<R>.
  template <class Model, class R>
  OPG_HD OPG_INLINE void hermitian3_step(const typename Model::Prepared& P,
                                         R E, bool nubar, const Segment<R>& s,
                                         Mat<3, R>& S)
  {
    Mat<3, R> V;
    R         lam[3];

    // PMNS_Fast::SolveHam: do vacuum oscillation in low density
    if (s.density < R(1.0e-6)) {
      V      = P.common.Uvac[nubar ? 1 : 0];
      lam[0] = 0;
      lam[1] = P.common.dm[1] / (2 * R(constants::kGeV2eV) * E);
      lam[2] = P.common.dm[2] / (2 * R(constants::kGeV2eV) * E);
    }
    else {
      Mat<3, R> H;
      Model::hamiltonian(P, E, nubar, s, H);
      kopp::zheevh3(H, V, lam);
    }

    apply_eigen_step<3, R>(V, lam, length_in_eV(s.length), S);
  }

} // namespace opg

#endif
