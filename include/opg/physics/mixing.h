///////////////////////////////////////////////////////////////////////////////
/// \file mixing.h
///
/// \brief Host-side construction of the vacuum mass matrix
///        Hms = U diag(m^2) U^dagger in the flavour basis.
///
/// This is a port of PMNS_Base::RotateH and PMNS_Base::BuildHms from
/// OscProb, written for a generic scalar type S. With S = double it
/// performs exactly OscProb's floating point operations (bit-identical
/// results); with S = Dual<double, K> it also yields derivatives with
/// respect to the mixing parameters. It depends only on the oscillation
/// parameters (not on energy or path), so it runs once per parameter set on
/// the host and the result is shipped to the device in each model's
/// Prepared struct.
///
/// Conventions (as OscProb, but 0-based):
///   dm[j]       = m_j^2 - m_1^2 (dm[0] = 0)
///   th[i][j]    = theta_{i+1,j+1} for i < j
///   dcp[i][j]   = delta_{i+1,j+1} for i + 1 < j
/// Only the diagonal and upper triangle of the output are meaningful; the
/// lower triangle is left at zero exactly like OscProb's fHms.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_PHYSICS_MIXING_H
#define OPG_PHYSICS_MIXING_H

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include "opg/core/dual.h"
#include "opg/core/matrix.h"

namespace opg {

  /// Mixing parameters for N neutrinos, with scalar type S.
  template <int N, class S = double> struct MixingParamsT {
      S dm[N]     = {};  ///< m_j^2 - m_1^2 in eV^2 (dm[0] ignored)
      S th[N][N]  = {};  ///< mixing angles theta_ij (i<j) in rad
      S dcp[N][N] = {};  ///< CP phases delta_ij (i+1<j) in rad

      /// Set theta_ij with OscProb's 1-based indices.
      void SetAngle(int i, int j, S v) { th[i - 1][j - 1] = v; }
      /// Set delta_ij with OscProb's 1-based indices.
      void SetDelta(int i, int j, S v) { dcp[i - 1][j - 1] = v; }
      /// Set dm_j1 with OscProb's 1-based index.
      void SetDm(int j, S v) { dm[j - 1] = v; }
  };

  template <int N> using MixingParams = MixingParamsT<N, double>;

  /// Convert the scalar type of mixing parameters.
  template <class S, int N, class S0>
  MixingParamsT<N, S> cast_mixing(const MixingParamsT<N, S0>& p)
  {
    MixingParamsT<N, S> q;
    for (int i = 0; i < N; i++) {
      q.dm[i] = S(p.dm[i]);
      for (int j = 0; j < N; j++) {
        q.th[i][j]  = S(p.th[i][j]);
        q.dcp[i][j] = S(p.dcp[i][j]);
      }
    }
    return q;
  }

  /// Names and addresses of the continuous mixing parameters:
  /// th<ij> (i<j), d<ij> (i+1<j), dm<j>1 (j>=2), with 1-based indices.
  template <int N> struct MixingRegistry {
      static std::vector<std::string> names()
      {
        std::vector<std::string> n;
        for (int j = 2; j <= N; j++)
          for (int i = 1; i < j; i++)
            n.push_back("th" + std::to_string(i) + std::to_string(j));
        for (int j = 3; j <= N; j++)
          for (int i = 1; i + 1 < j; i++)
            n.push_back("d" + std::to_string(i) + std::to_string(j));
        for (int j = 2; j <= N; j++) n.push_back("dm" + std::to_string(j) + "1");
        return n;
      }
      static int count() { return int(names().size()); }

      template <class S> static S& ref(MixingParamsT<N, S>& p, int idx)
      {
        int k = 0;
        for (int j = 2; j <= N; j++)
          for (int i = 1; i < j; i++)
            if (k++ == idx) return p.th[i - 1][j - 1];
        for (int j = 3; j <= N; j++)
          for (int i = 1; i + 1 < j; i++)
            if (k++ == idx) return p.dcp[i - 1][j - 1];
        for (int j = 2; j <= N; j++)
          if (k++ == idx) return p.dm[j - 1];
        throw std::out_of_range("MixingRegistry: bad parameter index");
      }
  };

  namespace detail {

    /// Port of PMNS_Base::RotateH: rotate Ham by theta_ij and delta_ij.
    template <int N, class S>
    void rotate_h(int i, int j, const S th[N][N], const S dcp[N][N],
                  Complex<S> Ham[N][N])
    {
      using std::cos;
      using std::sin;
      using C = Complex<S>;

      // Do nothing if angle is zero (and, for derivatives, its seed too)
      if (is_exact_zero(th[i][j])) return;

      S fSinBuffer = sin(th[i][j]);
      S fCosBuffer = cos(th[i][j]);
      const S two(2);

      S fHmsBufferD;
      C fHmsBufferC;

      // With Delta
      if (i + 1 < j) {
        C fExpBuffer = C(cos(dcp[i][j]), -sin(dcp[i][j]));

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
              two * fSinBuffer * fCosBuffer * real(Ham[i][j] * conj(fExpBuffer));
          Ham[i][i] += fSinBuffer * Ham[j][j] * fSinBuffer;

          Ham[j][j] *= fCosBuffer * fCosBuffer;
          Ham[j][j] += fSinBuffer * fHmsBufferC * fSinBuffer;
          Ham[j][j] -=
              two * fSinBuffer * fCosBuffer * real(Ham[i][j] * conj(fExpBuffer));

          Ham[i][j] -= two * fSinBuffer * real(Ham[i][j] * conj(fExpBuffer)) *
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
          Ham[i][i] += two * fSinBuffer * fCosBuffer * real(Ham[i][j]);
          Ham[i][i] += fSinBuffer * Ham[j][j] * fSinBuffer;

          Ham[j][j] *= fCosBuffer * fCosBuffer;
          Ham[j][j] += fSinBuffer * fHmsBufferC * fSinBuffer;
          Ham[j][j] -= two * fSinBuffer * fCosBuffer * real(Ham[i][j]);

          Ham[i][j] -= two * fSinBuffer * real(Ham[i][j]) * fSinBuffer;
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
  template <int N, class S>
  void rotate_diagonal_generic(const S diag[N], const S th[N][N],
                               const S dcp[N][N], Complex<S> H[N][N])
  {
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) H[i][j] = Complex<S>(S(0), S(0));

    for (int j = 0; j < N; j++) {
      // Set mass splitting
      H[j][j] = Complex<S>(diag[j], S(0));
      // Reset off-diagonal elements
      for (int i = 0; i < j; i++) H[i][j] = Complex<S>(S(0), S(0));
      // Rotate j neutrinos
      for (int i = 0; i < j; i++) detail::rotate_h<N, S>(i, j, th, dcp, H);
    }
  }

  /// Same as rotate_diagonal_generic, with std::complex<double> output.
  template <int N>
  void rotate_diagonal(const double diag[N], const double th[N][N],
                       const double dcp[N][N], std::complex<double> H[N][N])
  {
    Complex<double> h[N][N];
    rotate_diagonal_generic<N, double>(diag, th, dcp, h);
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) H[i][j] = {h[i][j].re, h[i][j].im};
  }

  /// Hms = U diag(dm) U^dagger (upper triangle + diagonal), scalar type S.
  template <int N, class S>
  Mat<N, S> build_hms_generic(const MixingParamsT<N, S>& p)
  {
    Complex<S> H[N][N];
    S          dm[N];
    for (int j = 0; j < N; j++) dm[j] = p.dm[j];
    dm[0] = S(0);  // OscProb keeps fDm[0] = 0
    rotate_diagonal_generic<N, S>(dm, p.th, p.dcp, H);
    Mat<N, S> out;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) out(i, j) = H[i][j];
    return out;
  }

  /// Hms computed in double precision and converted to Real.
  template <int N, class Real>
  Mat<N, Real> build_hms(const MixingParams<N>& p)
  {
    Mat<N, double> h = build_hms_generic<N, double>(p);
    Mat<N, Real>   out;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        out(i, j) = Complex<Real>(Real(h(i, j).re), Real(h(i, j).im));
    return out;
  }

  /// OscProb's default parameters (PMNS_Base::SetStdPars, PDG): applies to
  /// 3 and 3+N neutrinos; extra angles, phases and splittings are zero.
  template <int N> void set_std_pars(MixingParams<N>& p)
  {
    p = MixingParams<N>();
    p.SetAngle(1, 2, std::asin(std::sqrt(0.304)));
    p.SetAngle(1, 3, std::asin(std::sqrt(0.0219)));
    p.SetAngle(2, 3, std::asin(std::sqrt(0.514)));
    p.SetDm(2, 7.53e-5);
    p.SetDm(3, 2.52e-3);
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
