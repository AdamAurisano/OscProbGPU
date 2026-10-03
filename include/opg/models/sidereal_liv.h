///////////////////////////////////////////////////////////////////////////////
/// \file sidereal_liv.h
///
/// \brief 3-flavour oscillations with direction- and time-dependent Lorentz
///        invariance violation in the minimal SME (OscProb::PMNS_SiderealLIV).
///
/// The real coefficients aT (mass dimension 3, a 3-vector per flavour pair)
/// and cT (dimension 4, a tensor of which XX, YY, XY, XZ, YZ enter) are
/// combined with the neutrino direction N (from the detector colatitude and
/// the zenith and azimuth of arrival) and the local sidereal time T into a
/// real term added to the Hamiltonian (see hamiltonian()). As in OscProb:
/// * the cT terms enter as E[GeV] * cT without the GeV -> eV factor that the
///   aT terms carry;
/// * in vacuum (density < 1e-6) PMNS_Fast's analytic eigensystem is used, so
///   the sidereal terms do not act there.
///
/// Direction: for paths through the Earth the zenith is that of the path,
/// zenith = acos(cosZ) (as SetNeutrinoDirection(acos(cosZ) [deg], azimuth)),
/// unless a fixed direction is set with SetNeutrinoDirection(); fixed paths
/// always use the set direction.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_SIDEREAL_LIV_H
#define OPG_MODELS_SIDEREAL_LIV_H

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "opg/models/hermitian3.h"

namespace opg {

  /// Sidereal LIV parameters with scalar type S (coefficients); geometry and
  /// time are plain numbers (degrees, hours).
  template <class S> struct SiderealLIVParams {
      MixingParamsT<3, S> mix;
      S                   a[3][3][3]    = {};  ///< aT_ij^(coord), GeV
      S                   c[3][3][3][3] = {};  ///< cT_ij^(coord1 coord2)
      double              chi       = 0;     ///< detector colatitude, deg
      double              zenith    = 0;     ///< fixed zenith, deg
      double              azimuth   = 0;     ///< azimuth, deg
      double              time      = 0;     ///< local sidereal time, hours
      bool                fixed_dir = false; ///< zenith fixed (else from path)

      /// As PMNS_SiderealLIV::SetA(flvi, flvj, coord, val).
      void SetA(int i, int j, int coord, double val)
      {
        order(i, j);
        if (coord < 0 || coord > 2) throw std::invalid_argument("SiderealLIV::SetA: coord");
        a[i][j][coord] = S(val);
      }
      /// As PMNS_SiderealLIV::SetC(flvi, flvj, coord1, coord2, val).
      void SetC(int i, int j, int c1, int c2, double val)
      {
        order(i, j);
        if (c1 < 0 || c1 > 2 || c2 < 0 || c2 > 2)
          throw std::invalid_argument("SiderealLIV::SetC: coord");
        c[i][j][c1][c2] = S(val);
      }
      /// As PMNS_SiderealLIV::SetColatitude(chi) (degrees).
      void SetColatitude(double x) { chi = x; }
      /// As PMNS_SiderealLIV::SetColatitude(deg, min, sec) (signed latitude).
      void SetColatitude(double deg, double min, double sec)
      {
        chi = 90.0 - (deg + min / 60.0 + sec / 3600.0);
      }
      /// As PMNS_SiderealLIV::SetNeutrinoDirection(zenith, azimuth) (degrees):
      /// a fixed direction for all paths.
      void SetNeutrinoDirection(double zen, double azi)
      {
        zenith    = zen;
        azimuth   = azi;
        fixed_dir = true;
      }
      /// Azimuth (degrees) with the zenith of each Earth path.
      void SetAzimuth(double azi)
      {
        azimuth   = azi;
        fixed_dir = false;
      }
      /// As PMNS_SiderealLIV::SetTimeHours(hours).
      void SetTimeHours(double h) { time = h; }

    private:
      static void order(int& i, int& j)
      {
        if (i > j) std::swap(i, j);
        if (i < 0 || j > 2) throw std::invalid_argument("SiderealLIV: bad flavours");
      }
  };

  /// Prepared state with scalar type S.
  template <class S> struct SiderealLIVPrepared {
      Hermitian3Common<S> common;
      S                   a[3][3][3];     ///< upper triangle
      S                   c[3][3][5];     ///< XX, YY, XY, XZ, YZ (upper triangle)
      double              chi, azi;       ///< radians
      double              zen;            ///< fixed zenith, radians
      bool                fixed_dir;
      double              sw, cw;         ///< sin, cos(omega_sidereal T)
  };

  /// Amplitudes plus the direction of the path.
  template <class R> struct SiderealLIVState {
      Mat<3, R> S;
      R         N[3];
  };

  template <class R = double> struct SiderealLIV {
      using Real                        = R;
      static constexpr int         N    = 3;
      static constexpr const char* name = "SiderealLIV";

      static constexpr double kSiderealDayHours = 23.9344696;
      static constexpr double kOmegaSidereal    = 2.0 * M_PI / kSiderealDayHours;

#ifdef OPG_SIDEREAL_LIV_GRAD_CHUNK
      static constexpr int grad_chunk = OPG_SIDEREAL_LIV_GRAD_CHUNK;
#else
      static constexpr int grad_chunk = 2;
#endif

      template <class S> using ParamsT   = SiderealLIVParams<S>;
      using Params                       = SiderealLIVParams<double>;
      template <class S> using PreparedT = SiderealLIVPrepared<S>;
      using Prepared                     = SiderealLIVPrepared<R>;
      using State                        = SiderealLIVState<R>;

      /// mixing, then per flavour pair <ab>: aX_<ab>, aY_<ab>, aZ_<ab>,
      /// cXX_<ab>, cYY_<ab>, cXY_<ab>, cXZ_<ab>, cYZ_<ab> (the coefficients
      /// that enter the Hamiltonian).
      static std::vector<std::string> param_names()
      {
        static const char* fl[3] = {"e", "mu", "tau"};
        static const char* k[8]  = {"aX", "aY", "aZ", "cXX", "cYY", "cXY", "cXZ", "cYZ"};
        auto n = MixingRegistry<3>::names();
        for (int i = 0; i < 3; i++)
          for (int j = i; j < 3; j++)
            for (int q = 0; q < 8; q++)
              n.push_back(std::string(k[q]) + "_" + fl[i] + fl[j]);
        return n;
      }
      template <class S> static S& param_ref(ParamsT<S>& p, int idx)
      {
        static const int cc[5][2] = {{0, 0}, {1, 1}, {0, 1}, {0, 2}, {1, 2}};
        const int nmix = MixingRegistry<3>::count();
        if (idx < nmix) return MixingRegistry<3>::ref(p.mix, idx);
        int k = nmix;
        for (int i = 0; i < 3; i++)
          for (int j = i; j < 3; j++)
            for (int q = 0; q < 8; q++)
              if (k++ == idx)
                return q < 3 ? p.a[i][j][q] : p.c[i][j][cc[q - 3][0]][cc[q - 3][1]];
        throw std::out_of_range("SiderealLIV: bad parameter index");
      }
      template <class S> static ParamsT<S> cast(const Params& p)
      {
        ParamsT<S> q;
        q.mix = cast_mixing<S>(p.mix);
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
            for (int x = 0; x < 3; x++) {
              q.a[i][j][x] = S(p.a[i][j][x]);
              for (int y = 0; y < 3; y++) q.c[i][j][x][y] = S(p.c[i][j][x][y]);
            }
        q.chi = p.chi, q.zenith = p.zenith, q.azimuth = p.azimuth, q.time = p.time;
        q.fixed_dir = p.fixed_dir;
        return q;
      }

      template <class S> static PreparedT<S> prepare_generic(const ParamsT<S>& p)
      {
        static const int cc[5][2] = {{0, 0}, {1, 1}, {0, 1}, {0, 2}, {1, 2}};
        PreparedT<S>     out;
        prepare_hermitian3_generic<S>(p.mix, out.common);
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) {
            for (int x = 0; x < 3; x++) out.a[i][j][x] = p.a[i][j][x];
            for (int q = 0; q < 5; q++) out.c[i][j][q] = p.c[i][j][cc[q][0]][cc[q][1]];
          }
        out.chi       = p.chi * M_PI / 180.0;
        out.zen       = p.zenith * M_PI / 180.0;
        out.azi       = p.azimuth * M_PI / 180.0;
        out.fixed_dir = p.fixed_dir;
        out.sw        = std::sin(kOmegaSidereal * p.time);
        out.cw        = std::cos(kOmegaSidereal * p.time);
        return out;
      }

      static Prepared prepare(const Params& p) { return prepare_generic<R>(cast<R>(p)); }

      /// Direction factors (PMNS_SiderealLIV::SetNeutrinoDirection).
      template <class S>
      OPG_HD OPG_INLINE static void direction(const PreparedT<S>& P, double zen, R n[3])
      {
        using std::cos;
        using std::sin;
        const double chi = P.chi, azi = P.azi;
        n[0] = R(cos(chi) * sin(zen) * cos(azi) + sin(chi) * cos(zen));
        n[1] = R(sin(zen) * sin(azi));
        n[2] = R(-sin(chi) * sin(zen) * cos(azi) + cos(chi) * cos(zen));
      }

      /// Port of PMNS_SiderealLIV::UpdateHam (upper triangle + diagonal), for
      /// any scalar type S of the prepared state and segment type Seg; with
      /// liv = false only the standard part (vacuum shortcut, see header).
      template <class S, class Seg>
      OPG_HD OPG_INLINE static void hamiltonian(const PreparedT<S>& P, const R n[3], R E,
                                                bool nubar, const Seg& s, Mat<3, S>& H,
                                                bool liv = true)
      {
        const S lv = S(2 * R(constants::kGeV2eV) * E);  // 2*E in eV
        const Mat<3, S>& Hms = P.common.Hms;
        OPG_UNROLL
        for (int i = 0; i < 3; i++)
          OPG_UNROLL
        for (int j = i; j < 3; j++) H(i, j) = Hms(i, j) / lv;

        // matter potential (kK2 * sqrt2 * Gf * rho * zoa, in this order)
        S kr2GNe = P.common.vfac;
        kr2GNe *= s.density;
        kr2GNe *= s.zoa;
        if (!nubar)
          H(0, 0).re += kr2GNe;
        else
          H(0, 0).re -= kr2GNe;

        if (liv) {
          const R sw = R(P.sw), cw = R(P.cw);
          const R sign = nubar ? -R(constants::kGeV2eV) : R(constants::kGeV2eV);
          const R N0 = n[0], N1 = n[1], N2 = n[2];
          OPG_UNROLL
          for (int i = 0; i < 3; i++)
            OPG_UNROLL
          for (int j = i; j < 3; j++) {
            const S& a0  = P.a[i][j][0];
            const S& a1  = P.a[i][j][1];
            const S& a2  = P.a[i][j][2];
            const S& c00 = P.c[i][j][0];
            const S& c11 = P.c[i][j][1];
            const S& c01 = P.c[i][j][2];
            const S& c02 = P.c[i][j][3];
            const S& c12 = P.c[i][j][4];

            const S C0  = -sign * a2 * N2;
            const S As0 = sign * (a0 * N1 - a1 * N0);
            const S Ac0 = -sign * (a0 * N0 + a1 * N1);
            const S As1 = R(2) * N1 * N2 * c02 - R(2) * N0 * N2 * c12;
            const S Ac1 = R(-2) * N0 * N2 * c02 - R(2) * N1 * N2 * c12;
            const S Bs1 = N0 * N1 * (c00 - c11) - (N0 * N0 - N1 * N1) * c01;
            const S Bs  = E * Bs1;
            const S Bc1 = R(-0.5) * (N0 * N0 - N1 * N1) * (c00 - c11) -
                          R(2.0) * N0 * N1 * c01;
            const S Bc  = E * Bc1;
            const S As  = As0 + E * As1;
            const S Ac  = Ac0 + E * Ac1;
            const S liv_term =
                C0 + As * sw + Ac * cw + Bs * R(2) * sw * cw + Bc * (cw * cw - sw * sw);
            H(i, j).re += liv_term;
          }
        }

        // Conjugate Hamiltonian for antineutrinos
        if (nubar) {
          OPG_UNROLL
          for (int i = 0; i < 3; i++)
            OPG_UNROLL
          for (int j = i + 1; j < 3; j++) H(i, j) = conj(H(i, j));
        }
      }

      // --- states --------------------------------------------------------------
      /// Fixed direction (fixed paths).
      OPG_HD OPG_INLINE static State initial(const Prepared& P, bool)
      {
        State st;
        st.S = Mat<3, R>::identity();
        direction(P, P.zen, st.N);
        return st;
      }
      /// Earth path with direction cosZ: zenith = acos(cosZ) unless fixed.
      OPG_HD OPG_INLINE static State initial(const Prepared& P, bool, R cosZ)
      {
        State st;
        st.S = Mat<3, R>::identity();
        using std::acos;
        // as SetNeutrinoDirection(acos(cosZ) [deg], ...)
        const double zdeg = acos(double(cosZ)) * 180.0 / M_PI;
        direction(P, P.fixed_dir ? P.zen : zdeg * M_PI / 180.0, st.N);
        return st;
      }

      /// Eigensystem of a segment (PMNS_Fast::SolveHam with the sidereal H).
      OPG_HD OPG_INLINE static void eigen(const Prepared& P, const R n[3], R E, bool nubar,
                                          const Segment<R>& s, Mat<3, R>& V, R lam[3])
      {
        if (s.density < R(1.0e-6)) {
          V      = P.common.Uvac[nubar ? 1 : 0];
          lam[0] = 0;
          lam[1] = P.common.dm[1] / (2 * R(constants::kGeV2eV) * E);
          lam[2] = P.common.dm[2] / (2 * R(constants::kGeV2eV) * E);
        }
        else {
          Mat<3, R> H;
          hamiltonian(P, n, E, nubar, s, H);
          diagonalize3(H, V, lam);
        }
      }

      OPG_HD OPG_INLINE static void step(const Prepared& P, R E, bool nubar,
                                         const Segment<R>& s, State& st)
      {
        Mat<3, R> V;
        R         lam[3];
        eigen(P, st.N, E, nubar, s, V, lam);
        apply_eigen_step<3, R>(V, lam, length_in_eV(s.length), st.S);
      }

      OPG_HD OPG_INLINE static void finalize(const Prepared&, bool, State&) {}

      template <class Rr>
      OPG_HD OPG_INLINE static void store_probs(const State& st, Rr* out, size_t stride)
      {
        opg::store_probs<3, Rr>(st.S, out, stride);
      }

      // --- gradients ---------------------------------------------------------
      template <int K>
      OPG_HD OPG_INLINE static void initial_grad(const PreparedT<Dual<R, K>>&, bool,
                                                 State (&dS)[K])
      {
        OPG_UNROLL
        for (int k = 0; k < K; k++) {
          dS[k].S = Mat<3, R>::zero();
          dS[k].N[0] = dS[k].N[1] = dS[k].N[2] = R(0);
        }
      }

      template <int K>
      OPG_HD OPG_INLINE static void step_grad(const Prepared&               P,
                                              const PreparedT<Dual<R, K>>& PD,
                                              R E, bool nubar,
                                              const SegmentZ<R, Dual<R, K>>& sz,
                                              State& st, State (&dS)[K])
      {
        const Segment<R> s{sz.length, sz.density, sz.zoa.v, sz.layer};
        Mat<3, R>        V;
        R                lam[3];
        eigen(P, st.N, E, nubar, s, V, lam);

        // dual Hamiltonian; in vacuum H = Hms/2E (no sidereal terms)
        const bool              vac = s.density < R(1.0e-6);
        SegmentZ<R, Dual<R, K>> sd  = sz;
        if (vac) sd.density = 0;
        Mat<3, Dual<R, K>> HD;
        hamiltonian(PD, st.N, E, nubar, sd, HD, !vac);

        Mat<3, R> dSm[K];
        OPG_UNROLL
        for (int k = 0; k < K; k++) dSm[k] = dS[k].S;
        eigen_step_grad<3, R, K>(V, lam, length_in_eV(s.length), HD, st.S, dSm);
        OPG_UNROLL
        for (int k = 0; k < K; k++) dS[k].S = dSm[k];
      }

      template <int K>
      OPG_HD OPG_INLINE static void finalize_grad(const Prepared&,
                                                  const PreparedT<Dual<R, K>>&, bool,
                                                  State&, State (&)[K])
      {
      }

      template <int K>
      OPG_HD OPG_INLINE static void store_grads(const State& S, const State (&dS)[K],
                                                int count, R* out, size_t stride)
      {
        Mat<3, R> d[K];
        OPG_UNROLL
        for (int k = 0; k < K; k++) d[k] = dS[k].S;
        opg::store_grads<3, R, K>(S.S, d, count, out, stride);
      }

      template <int K>
      OPG_HD OPG_INLINE static void contract_grads(const State& S, const State (&dS)[K],
                                                   const R* w, size_t stride, R (&acc)[K])
      {
        Mat<3, R> d[K];
        OPG_UNROLL
        for (int k = 0; k < K; k++) d[k] = dS[k].S;
        opg::contract_grads<3, R, K>(S.S, d, w, stride, acc);
      }
  };

} // namespace opg

#endif
