///////////////////////////////////////////////////////////////////////////////
/// \file mixing.h
///
/// \brief Host-side construction of the vacuum mass matrix
///        Hms = U diag(m^2) U^dagger in the flavour basis.
///
/// This is a verbatim port of PMNS_Base::RotateH and PMNS_Base::BuildHms
/// from OscProb, using std::complex<double> so that the result is
/// bit-for-bit identical. It depends only on the oscillation parameters
/// (not on energy or path), so it runs once per parameter set on the host
/// and the result is shipped to the device in each model's Prepared struct.
///
/// Conventions (as OscProb, but 0-based):
///   dm[j]       = m_j^2 - m_1^2 (dm[0] = 0)
///   th[i][j]    = theta_{i+1,j+1} for i < j
///   dcp[i][j]   = delta_{i+1,j+1} for i < j
/// Only the diagonal and upper triangle of the output are meaningful; the
/// lower triangle is left at zero exactly like OscProb's fHms.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_PHYSICS_MIXING_H
#define OPG_PHYSICS_MIXING_H

#include <cmath>
#include <complex>

#include "opg/core/matrix.h"

namespace opg {

  /// Mixing parameters for N neutrinos.
  template <int N> struct MixingParams {
      double dm[N]     = {};  ///< m_j^2 - m_1^2 in eV^2 (dm[0] ignored)
      double th[N][N]  = {};  ///< mixing angles theta_ij (i<j) in rad
      double dcp[N][N] = {};  ///< CP phases delta_ij (i<j) in rad

      /// Set theta_ij with OscProb's 1-based indices.
      void SetAngle(int i, int j, double v) { th[i - 1][j - 1] = v; }
      /// Set delta_ij with OscProb's 1-based indices.
      void SetDelta(int i, int j, double v) { dcp[i - 1][j - 1] = v; }
      /// Set dm_j1 with OscProb's 1-based index.
      void SetDm(int j, double v) { dm[j - 1] = v; }
  };

  namespace detail {

    using cplx = std::complex<double>;

    /// Port of PMNS_Base::RotateH: rotate Ham by theta_ij and delta_ij.
    template <int N>
    void rotate_h(int i, int j, const double th[N][N], const double dcp[N][N],
                  cplx Ham[N][N])
    {
      // Do nothing if angle is zero
      if (th[i][j] == 0) return;

      double fSinBuffer = std::sin(th[i][j]);
      double fCosBuffer = std::cos(th[i][j]);

      double fHmsBufferD;
      cplx   fHmsBufferC;

      // With Delta
      if (i + 1 < j) {
        cplx fExpBuffer = cplx(std::cos(dcp[i][j]), -std::sin(dcp[i][j]));

        // General case
        if (i > 0) {
          // Top columns
          for (int k = 0; k < i; k++) {
            fHmsBufferC = Ham[k][i];

            Ham[k][i] *= fCosBuffer;
            Ham[k][i] += Ham[k][j] * fSinBuffer * conj(fExpBuffer);

            Ham[k][j] *= fCosBuffer;
            Ham[k][j] -= fHmsBufferC * fSinBuffer * fExpBuffer;
          }

          // Middle row and column
          for (int k = i + 1; k < j; k++) {
            fHmsBufferC = Ham[k][j];

            Ham[k][j] *= fCosBuffer;
            Ham[k][j] -= conj(Ham[i][k]) * fSinBuffer * fExpBuffer;

            Ham[i][k] *= fCosBuffer;
            Ham[i][k] += fSinBuffer * fExpBuffer * conj(fHmsBufferC);
          }

          // Nodes ij
          fHmsBufferC = Ham[i][i];
          fHmsBufferD = real(Ham[j][j]);

          Ham[i][i] *= fCosBuffer * fCosBuffer;
          Ham[i][i] +=
              2 * fSinBuffer * fCosBuffer * real(Ham[i][j] * conj(fExpBuffer));
          Ham[i][i] += fSinBuffer * Ham[j][j] * fSinBuffer;

          Ham[j][j] *= fCosBuffer * fCosBuffer;
          Ham[j][j] += fSinBuffer * fHmsBufferC * fSinBuffer;
          Ham[j][j] -=
              2 * fSinBuffer * fCosBuffer * real(Ham[i][j] * conj(fExpBuffer));

          Ham[i][j] -= 2 * fSinBuffer * real(Ham[i][j] * conj(fExpBuffer)) *
                       fSinBuffer * fExpBuffer;
          Ham[i][j] += fSinBuffer * fCosBuffer * (fHmsBufferD - fHmsBufferC) *
                       fExpBuffer;
        }
        // First rotation on j (No top columns)
        else {
          // Middle rows and columns
          for (int k = i + 1; k < j; k++) {
            Ham[k][j] = -conj(Ham[i][k]) * fSinBuffer * fExpBuffer;

            Ham[i][k] *= fCosBuffer;
          }

          // Nodes ij
          fHmsBufferD = real(Ham[i][i]);

          Ham[i][j] =
              fSinBuffer * fCosBuffer * (Ham[j][j] - fHmsBufferD) * fExpBuffer;

          Ham[i][i] *= fCosBuffer * fCosBuffer;
          Ham[i][i] += fSinBuffer * Ham[j][j] * fSinBuffer;

          Ham[j][j] *= fCosBuffer * fCosBuffer;
          Ham[j][j] += fSinBuffer * fHmsBufferD * fSinBuffer;
        }
      }
      // Without Delta (No middle rows or columns: j = i+1)
      else {
        // General case
        if (i > 0) {
          // Top columns
          for (int k = 0; k < i; k++) {
            fHmsBufferC = Ham[k][i];

            Ham[k][i] *= fCosBuffer;
            Ham[k][i] += Ham[k][j] * fSinBuffer;

            Ham[k][j] *= fCosBuffer;
            Ham[k][j] -= fHmsBufferC * fSinBuffer;
          }

          // Nodes ij
          fHmsBufferC = Ham[i][i];
          fHmsBufferD = real(Ham[j][j]);

          Ham[i][i] *= fCosBuffer * fCosBuffer;
          Ham[i][i] += 2 * fSinBuffer * fCosBuffer * real(Ham[i][j]);
          Ham[i][i] += fSinBuffer * Ham[j][j] * fSinBuffer;

          Ham[j][j] *= fCosBuffer * fCosBuffer;
          Ham[j][j] += fSinBuffer * fHmsBufferC * fSinBuffer;
          Ham[j][j] -= 2 * fSinBuffer * fCosBuffer * real(Ham[i][j]);

          Ham[i][j] -= 2 * fSinBuffer * real(Ham[i][j]) * fSinBuffer;
          Ham[i][j] += fSinBuffer * fCosBuffer * (fHmsBufferD - fHmsBufferC);
        }
        // First rotation (theta12)
        else {
          Ham[i][j] = fSinBuffer * fCosBuffer * Ham[j][j];

          Ham[i][i] = fSinBuffer * Ham[j][j] * fSinBuffer;

          Ham[j][j] *= fCosBuffer * fCosBuffer;
        }
      }
    }

  } // namespace detail

  /// Port of PMNS_Base::BuildHms with an arbitrary diagonal: returns
  /// U diag(diag) U^dagger (upper triangle + diagonal; lower triangle zero).
  /// With diag = dm this is OscProb's fHms; PMNS_Decay also applies it to
  /// the decay constants alpha_j to build fHd.
  template <int N>
  void rotate_diagonal(const double diag[N], const double th[N][N],
                       const double dcp[N][N], std::complex<double> H[N][N])
  {
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) H[i][j] = 0;

    for (int j = 0; j < N; j++) {
      // Set mass splitting
      H[j][j] = diag[j];
      // Reset off-diagonal elements
      for (int i = 0; i < j; i++) H[i][j] = 0;
      // Rotate j neutrinos
      for (int i = 0; i < j; i++) detail::rotate_h<N>(i, j, th, dcp, H);
    }
  }

  /// Hms = U diag(dm) U^dagger as an opg::Mat (upper triangle + diagonal).
  template <int N, class Real>
  Mat<N, Real> build_hms(const MixingParams<N>& p)
  {
    std::complex<double> H[N][N];
    double               dm[N];
    for (int j = 0; j < N; j++) dm[j] = p.dm[j];
    dm[0] = 0;  // OscProb keeps fDm[0] = 0
    rotate_diagonal<N>(dm, p.th, p.dcp, H);
    Mat<N, Real> out;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        out(i, j) = Complex<Real>(Real(H[i][j].real()), Real(H[i][j].imag()));
    return out;
  }

  /// OscProb default/nominal 3-flavour helper (PMNS_Fast::SetMix and
  /// SetDeltaMsqrs conventions).
  template <int N>
  void set_standard_3nu(MixingParams<N>& p, double th12, double th23,
                        double th13, double dcp, double dm21, double dm31)
  {
    p.SetAngle(1, 2, th12);
    p.SetAngle(1, 3, th13);
    p.SetAngle(2, 3, th23);
    p.SetDelta(1, 3, dcp);
    p.SetDm(2, dm21);
    p.SetDm(3, dm31);
  }

} // namespace opg

#endif
