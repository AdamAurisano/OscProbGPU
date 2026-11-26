///////////////////////////////////////////////////////////////////////////////
/// \file binned_walk.h
///
/// \brief Path walker of the 3D (depth x latitude x longitude) binned Earth,
///        ported from OscProb::EarthModelBinned::FillPath.
///
/// The walker runs on host and device over a BinnedGeom (a POD of scalars
/// and pointers into the bin table). It is a generator: binned_walk_init()
/// sets up the trajectory for (cosT, azimuth) and each binned_walk_next()
/// returns the next segment (length and bin index), in the order and with
/// the operations of FillPath, DetDistForNextLatBin, DetDistForNextLonBin,
/// RecordLatLonBinCrossings and LonBinIndex (OscProb 7c06b3e), so that host
/// segments are bit-identical to OscProb's. The geometry is always computed
/// in double precision.
///
/// The generator state lives in a struct (BinnedWalk) and the stepping
/// functions are not inlined on the device: the state then stays in local
/// memory instead of occupying registers across the model's step, which
/// keeps the register use of the kernels (and so PREM performance)
/// unchanged.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_EARTH_BINNED_WALK_H
#define OPG_EARTH_BINNED_WALK_H

#include <cmath>

#include "../core/macros.h"

namespace opg {

  /// Geometry of a binned Earth model (non-owning view of the bin table
  /// plus derived constants), as needed by the path walker. Bin b has outer
  /// and inner radius radius_out[b], radius_in[b] (km) and centre
  /// latitude[b], longitude[b] (radians); bins are ordered by decreasing
  /// depth, then increasing longitude, then increasing latitude (OscProb's
  /// table order). Densities, Z/A and region indices are in the EarthView.
  struct BinnedGeom {
      int    nbins        = 0;  ///< total number of bins
      int    nDepth       = 0;  ///< fnDepthBins
      int    nLon         = 0;  ///< fnLonBins
      int    nLat         = 0;  ///< fnLatBins
      int    max_seg      = 0;  ///< safety bound on the segments of a path
      double invLonW      = 0;  ///< fInvLonBinWidth
      double invLatW      = 0;  ///< fInvLatBinWidth
      double halfLonW     = 0;  ///< fHalfLonBinWidth
      double halfLatW     = 0;  ///< fHalfLatBinWidth
      double radiusMax    = 0;  ///< fRadiusMax
      double detRadius    = 0;  ///< fDetRadius
      double detLat       = 0;  ///< fDetLat (rad)
      double detLon       = 0;  ///< fDetLon (rad)
      // detector part of OscProb's TrajConstants (UpdateDetPos)
      double sinDetLat     = 0;
      double cosDetLat     = 0;
      double cosDetLon     = 0;
      double sinDetLon     = 0;
      double rDetSinDetLat = 0;
      double rDetCosDetLat = 0;
      const double* radius_out = nullptr;
      const double* radius_in  = nullptr;
      const double* latitude   = nullptr;
      const double* longitude  = nullptr;
  };

  /// State of a path walk through a binned Earth (OscProb's TrajConstants,
  /// LatBinInfo, LonBinInfo and the loop variables of FillPath).
  struct BinnedWalk {
      // TrajConstants (UpdateNuAngles, Recalculate)
      double cosT, sinT, cosA, sinA, sinTsinA, sinTcosA;
      double sinTsinAsinDetLon, sinTsinAcosDetLon, rDetCosT, rDetSinT;
      double alpha, beta, gamma, gammaSq, rDetGammaSinDetLat, maxSinSqLat, xlatextreme;
      double rDetSqSinSqT, minRadius;
      // LatBinInfo
      int    lat_bin, lat_next, lat_sign;
      double lat_dist, lat_dLat;
      bool   lat_maxreached;
      // LonBinInfo
      int    lon_bin, lon_next, lon_error;
      double lon_dist, lon_dLon, lon_min, lon_max;
      // FillPath loop state
      int    index;   ///< current Earth bin
      int    phase;   ///< 0: down, 1: up, 2: last segment, 3: done
      int    dBin;    ///< depth bin counter of the down loop
      int    nseg;    ///< segments returned so far
      bool   have_target;
      double target;  ///< detector distance of the next depth crossing
      double prev;    ///< current detector distance (prev_DetDistance)
  };

  namespace detail {

    /// EarthModelBinned::LonBinIndex (including its shift of londiff by
    /// floor(londiff / 2 pi), not times 2 pi).
    OPG_HD inline int binned_lon_bin_index(const BinnedGeom& g, double longitude)
    {
      double londiff = longitude - (g.longitude[0] - g.halfLonW);
      if (londiff < 0 || londiff >= 2 * M_PI) londiff -= std::floor(londiff / (2 * M_PI));
      return int(std::floor(londiff * g.invLonW));
    }

    /// EarthModelBinned::DetDistForNextLatBin.
    OPG_HD inline double binned_lat_dist(const BinnedGeom& g, BinnedWalk& w, int cur_index)
    {
      double NewLat = g.latitude[cur_index] + w.lat_sign * w.lat_dLat;
      if (std::fabs(NewLat) - 0.5 * M_PI > -1.0e-14) {
        if (w.lat_maxreached) return -1;
        w.lat_maxreached = true;
        w.lat_sign       = -w.lat_sign;
        NewLat           = g.latitude[cur_index] + w.lat_sign * w.lat_dLat;
        if (std::fabs(NewLat) - 0.5 * M_PI > -1.0e-14) return -1;
      }

      double sinNewLat    = std::sin(NewLat);
      double NsinSqNewLat = -sinNewLat * sinNewLat;
      double radicand     = w.maxSinSqLat + NsinSqNewLat;
      if (radicand < 1.0e-14) {
        if (w.lat_maxreached) return -1;
        w.lat_maxreached = true;
        w.lat_sign       = -w.lat_sign;
        NewLat           = g.latitude[cur_index] + w.lat_sign * w.lat_dLat;
        sinNewLat        = std::sin(NewLat);
        NsinSqNewLat     = -sinNewLat * sinNewLat;
        radicand         = w.maxSinSqLat + NsinSqNewLat;
        if (radicand < 1.0e-14) return -1;
      }
      double denominator = w.gammaSq + NsinSqNewLat;

      double answer;
      if (denominator == 0) { answer = 0.5 * (w.xlatextreme - g.rDetSinDetLat / w.gamma); }
      else {
        answer = -(w.rDetCosT * NsinSqNewLat + w.rDetGammaSinDetLat +
                   w.lat_sign * w.rDetSinT * sinNewLat * std::sqrt(radicand)) /
                 denominator;
      }

      if (w.lat_maxreached && answer >= w.xlatextreme) return -1;
      return answer;
    }

    /// EarthModelBinned::DetDistForNextLonBin.
    OPG_HD inline double binned_lon_dist(const BinnedGeom& g, BinnedWalk& w, double prev_lon)
    {
      double twoPi = 2 * M_PI;
      double lon   = prev_lon + w.lon_dLon;
      if (lon < 0 || lon >= twoPi) { lon -= std::floor(lon / twoPi) * twoPi; }
      if (w.lon_dLon > 0) {
        w.lon_next = w.lon_bin + 1;
        if (w.lon_next >= g.nLon) w.lon_next = 0;
      }
      else {
        w.lon_next = w.lon_bin - 1;
        if (w.lon_next < 0) w.lon_next = g.nLon - 1;
      }

      if (w.lon_max < w.lon_min) {
        if (0 > lon - w.lon_min && w.lon_max - lon < 0) return -1;
      }
      else {
        if (0 > lon - w.lon_min || w.lon_max - lon < 0) return -1;
      }

      double cosNewLon   = std::cos(lon);
      double sinNewLon   = std::sin(lon);
      double expression  = g.sinDetLon * cosNewLon - sinNewLon * g.cosDetLon;
      double denominator = w.alpha * expression + w.sinTsinAcosDetLon * cosNewLon +
                           w.sinTsinAsinDetLon * sinNewLon;

      if (denominator == 0) {
        w.lon_error = -1;
        return -1;
      }
      return g.rDetCosDetLat * expression / denominator;
    }

    /// One step of EarthModelBinned::RecordLatLonBinCrossings: if a latitude
    /// or longitude crossing comes before the detector distance `target`,
    /// returns true with the segment up to it (and moves to the next bin).
    OPG_HD inline bool binned_cross(const BinnedGeom& g, BinnedWalk& w, double target,
                                    double& len, int& bin)
    {
      if (!(target < w.lat_dist || target < w.lon_dist)) return false;
      if (w.lat_dist > w.lon_dist) {
        len = w.prev - w.lat_dist;
        bin = w.index;
        w.prev = w.lat_dist;
        w.index += w.lat_next - w.lat_bin;
        w.lat_bin = w.lat_next;
        if (w.index < 0 || w.index >= g.nbins) return true;  // stopped by binned_walk_next
        if (w.lat_dLat == 0) { w.lat_dist = -1; }
        else {
          w.lat_dist = binned_lat_dist(g, w, w.index);
          w.lat_next = w.lat_bin + w.lat_sign;
        }
      }
      else {
        len = w.prev - w.lon_dist;
        bin = w.index;
        w.prev = w.lon_dist;
        w.index += (w.lon_next - w.lon_bin) * g.nLat;
        w.lon_bin = w.lon_next;
        if (w.index < 0 || w.index >= g.nbins) return true;  // stopped by binned_walk_next
        if (w.lon_dLon == 0) { w.lon_dist = -1; }
        else {
          w.lon_dist = binned_lon_dist(g, w, g.longitude[w.index]);
        }
      }
      return true;
    }

  } // namespace detail

  //...........................................................................
  /// Start a walk for direction cosT (cosine of the zenith angle) and
  /// azimuth phi (degrees from North) of the vector from the detector to
  /// where the neutrino enters the Earth: the set-up part of
  /// EarthModelBinned::FillPath. Returns false (no segments) if |cosT| > 1.
  OPG_HD OPG_NOINLINE inline bool binned_walk_init(const BinnedGeom& g, BinnedWalk& w,
                                                   double cosT, double phi)
  {
    w.phase = 3;
    w.nseg  = 0;
    if (std::fabs(cosT) > 1) return false;

    // TrajConstants::UpdateNuAngles(cosT, Az) and Recalculate()
    const double Az = phi / 180.0 * M_PI;
    w.cosT          = cosT;
    const double sinSqT = 1 - cosT * cosT;
    w.sinT          = std::sqrt(sinSqT);
    w.cosA          = std::cos(Az);
    w.sinA          = std::sin(Az);
    w.sinTsinA      = w.sinT * w.sinA;
    w.sinTcosA      = w.sinT * w.cosA;
    const double cosTcosDetLat = w.cosT * g.cosDetLat;
    w.sinTsinAsinDetLon = w.sinTsinA * g.sinDetLon;
    w.sinTsinAcosDetLon = w.sinTsinA * g.cosDetLon;
    w.rDetCosT          = g.detRadius * w.cosT;
    w.rDetSinT          = g.detRadius * w.sinT;
    w.alpha             = w.sinTcosA * g.sinDetLat - cosTcosDetLat;
    w.beta              = w.sinT * g.sinDetLat - w.cosA * cosTcosDetLat;
    w.gamma             = w.sinTcosA * g.cosDetLat + w.cosT * g.sinDetLat;
    w.gammaSq           = w.gamma * w.gamma;
    w.rDetGammaSinDetLat = g.rDetSinDetLat * w.gamma;
    const double sAcL   = w.sinA * g.cosDetLat;
    w.maxSinSqLat       = 1 - sAcL * sAcL;  // 1 - pow(sinA * cosDetLat, 2)
    if (w.beta == 0) { w.xlatextreme = 0; }
    else {
      w.xlatextreme = w.cosA * g.rDetCosDetLat / w.beta;
    }

    w.rDetSqSinSqT = w.rDetSinT * w.rDetSinT;  // pow(rDetSinT, 2)
    const double distEdgeToMin = std::sqrt(g.radiusMax * g.radiusMax - w.rDetSqSinSqT);
    const double baseline      = distEdgeToMin - w.rDetCosT;

    w.minRadius = g.detRadius;
    if (cosT < 0) { w.minRadius = w.rDetSinT; }

    const double init_lat =
        std::asin((w.rDetSinT * w.beta + w.gamma * distEdgeToMin) / g.radiusMax);
    double init_lon = std::atan2(
        g.sinDetLon * g.rDetCosDetLat -
            baseline * (w.alpha * g.sinDetLon + w.sinTsinAcosDetLon),
        g.cosDetLon * g.rDetCosDetLat +
            baseline * (w.sinTsinAsinDetLon - w.alpha * g.cosDetLon));
    if (init_lon < 0) { init_lon += 2 * M_PI; }

    const int binsPerDepth = g.nbins / g.nDepth;
    const int init_lonBin  = detail::binned_lon_bin_index(g, init_lon);
    const int init_latBin  = int(std::floor((init_lat + M_PI / 2.0) * g.invLatW));
    w.index = g.nbins - binsPerDepth + init_lonBin * g.nLat + init_latBin;

    w.lat_bin   = init_latBin;
    w.lon_bin   = init_lonBin;
    w.lat_next  = 0;
    w.lon_next  = 0;
    w.lon_error = 0;

    w.lat_dLat       = g.halfLatW;
    w.lat_sign       = 1;
    w.lat_maxreached = false;
    if (w.beta < 0) w.lat_sign = -1;
    if (w.beta == 0) {
      if (w.cosA > 0) w.lat_sign = -1;
    }
    else if (baseline < w.xlatextreme) {
      w.lat_sign       = -(w.lat_sign);
      w.lat_maxreached = true;
    }
    w.lon_dLon = g.halfLonW;
    w.lon_min  = init_lon;
    w.lon_max  = g.detLon;
    if (w.sinA < 0) {
      w.lon_dLon = -(w.lon_dLon);
      w.lon_max  = init_lon;
      w.lon_min  = g.detLon;
    }
    if (w.index < 0 || w.index >= g.nbins) return false;  // (not in OscProb)
    if (w.sinT == 0) {
      w.lon_dist = -w.rDetCosT;
      w.lon_next = detail::binned_lon_bin_index(g, g.detLon);
      w.lon_dLon = 0;
      w.lat_dist = w.lon_dist;
      w.lat_next = int(std::floor((g.detLat + 0.5 * M_PI) * g.invLatW));
      w.lat_dLat = 0;
    }
    else {
      if (w.sinA == 0) {
        if (w.alpha == 0) { w.lon_dist = -1; }
        else {
          w.lon_dist = g.rDetCosDetLat / w.alpha;
          w.lon_next = detail::binned_lon_bin_index(g, g.detLon);
        }
        w.lon_dLon = 0;
      }
      else if (g.cosDetLat == 0) {
        w.lon_dist = -1;
      }
      else {
        w.lon_dist = detail::binned_lon_dist(g, w, g.longitude[w.index]);
      }
      w.lat_dist = detail::binned_lat_dist(g, w, w.index);
      w.lat_next = init_latBin + w.lat_sign;
    }

    w.prev        = baseline;
    w.phase       = 0;
    w.dBin        = 0;
    w.have_target = false;
    return true;
  }

  //...........................................................................
  /// Next segment of a walk: its length (km) and bin index. Returns false
  /// at the end of the path. Follows the loops of FillPath: depth bins down
  /// to the deepest point, back up to the detector's depth bin, then the
  /// last segment to the detector, each split at latitude and longitude
  /// bin crossings (RecordLatLonBinCrossings).
  OPG_HD OPG_NOINLINE inline bool binned_walk_next(const BinnedGeom& g, BinnedWalk& w,
                                                   double& len, int& bin)
  {
    const int binsPerDepth = g.nbins / g.nDepth;
    // guard against bin indices out of range or runaway crossings (not in
    // OscProb, whose FillPath would read outside the table)
    if (w.phase < 3 && (w.index < 0 || w.index >= g.nbins || w.nseg >= g.max_seg)) {
      w.phase = 3;
      return false;
    }
    for (;;) {
      switch (w.phase) {
        case 0:  // down: depth bins above the deepest point
          if (!w.have_target) {
            if (w.dBin >= g.nDepth - 1) {
              w.phase = 1;
              continue;
            }
            const double r_in = g.radius_in[w.index];
            if (r_in < w.minRadius) {
              w.phase = 1;
              continue;
            }
            w.target      = -w.rDetCosT + std::sqrt(r_in * r_in - w.rDetSqSinSqT);
            w.have_target = true;
          }
          if (detail::binned_cross(g, w, w.target, len, bin)) break;
          len           = w.prev - w.target;
          bin           = w.index;
          w.prev        = w.target;
          w.index      -= binsPerDepth;
          w.dBin++;
          w.have_target = false;
          break;
        case 1:  // up: deepest depth bin and back up to the detector's
          if (!w.have_target) {
            if (!(g.radius_out[w.index] < g.detRadius)) {
              w.phase = 2;
              continue;
            }
            const double r_out = g.radius_out[w.index];
            w.target      = -w.rDetCosT - std::sqrt(r_out * r_out - w.rDetSqSinSqT);
            w.have_target = true;
          }
          if (detail::binned_cross(g, w, w.target, len, bin)) break;
          len           = w.prev - w.target;
          bin           = w.index;
          w.prev        = w.target;
          w.index      += binsPerDepth;
          w.have_target = false;
          break;
        case 2:  // last segment, in the detector's bin
          if (detail::binned_cross(g, w, 0, len, bin)) break;
          len     = w.prev;
          bin     = w.index;
          w.phase = 3;
          break;
        default: return false;
      }
      w.nseg++;
      return true;
    }
  }

} // namespace opg

#endif
