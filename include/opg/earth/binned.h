///////////////////////////////////////////////////////////////////////////////
/// \file binned.h
///
/// \brief 3D Earth model of depth x latitude x longitude bins, ported from
///        OscProb::EarthModelBinned (with EarthModelBase).
///
/// The host class EarthModelBinned loads and edits the bin table exactly
/// like OscProb (reading the table as float; same ordering checks and bin
/// widths), places the detector at (radius, latitude, longitude) as
/// EarthModelBase::SetDetectorCoordinates, and supports SetRegionZoA,
/// GetRegionZoA and ScaleRegionDensity per region index. Paths are walked
/// by the OPG_HD walker of binned_walk.h (on host and device), which
/// reproduces FillPath(cosT, phi) operation by operation.
///
/// Additions to OscProb's interface: an azimuth stored with the model
/// (SetAzimuth), used for paths given by cosT alone (grids, bin averages,
/// event lists without per-event azimuths); FillPath returns the segments;
/// GetGrazingCosines() for the layer-adapted cos-zenith rule (depth-shell
/// boundaries only); errors throw std::runtime_error instead of printing
/// and clearing the model.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_EARTH_BINNED_H
#define OPG_EARTH_BINNED_H

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "binned_walk.h"
#include "prem.h"

namespace opg {

  /// A bin of the binned Earth (OscProb::EarthBin): part of a spherical
  /// shell, with centre latitude and longitude in radians.
  struct EarthBin {
      double radius_out;  ///< outer radius (km)
      double radius_in;   ///< inner radius (km)
      double latitude;    ///< latitude of the bin centre (rad)
      double longitude;   ///< longitude of the bin centre (rad)
      double density;     ///< g/cm^3
      double zoa;         ///< effective Z/A
      int    index;       ///< region index
  };

  class EarthModelBinned {
    public:
      /// Load a bin table (empty: data/earth_binned_default.txt) with the
      /// detector at radius 6368 km, latitude 0, longitude 0, as OscProb.
      explicit EarthModelBinned(const std::string& filename = "")
      {
        SetDetPos(6368, 0, 0);
        LoadModel(filename);
      }

      /// The default table: $OPG_DATA_DIR/earth_binned_default.txt if that
      /// environment variable is set, else the source tree's data/ for builds
      /// that compiled its path in, else the working directory.
      static std::string DefaultFile()
      {
        const char* env = std::getenv("OPG_DATA_DIR");
        if (env && *env) return std::string(env) + "/earth_binned_default.txt";
#ifdef OPG_DATA_DIR
        return std::string(OPG_DATA_DIR) + "/earth_binned_default.txt";
#else
        return "earth_binned_default.txt";
#endif
      }

      /// Load a bin table. Rows: longitude (deg), latitude (deg), outer
      /// depth (km), density (g/cm^3), Z/A, region index; ordered by
      /// decreasing depth, then increasing longitude, then increasing
      /// latitude; constant latitude and longitude bin widths; the centre
      /// of the Earth at a depth of 6371 km. Port of LoadModel.
      void LoadModel(std::string filename)
      {
        ClearModel();
        const double EarthRadius = 6371.0;
        if (filename.empty()) filename = DefaultFile();
        std::ifstream fin(filename);
        if (!fin) throw std::runtime_error("EarthModelBinned: cannot open " + filename);

        // OscProb reads the table as float
        float  depth, latitude, longitude, density, zoa, index;
        double radius_prev = 0;
        double lat_prev    = -90.0;
        double lon_prev    = 0;
        double radius_min  = radius_prev;
        fMaxRegIndex       = -1;

        while (fin >> longitude >> latitude >> depth >> density >> zoa >> index) {
          double outer_radius = EarthRadius - depth;
          if (outer_radius > radius_prev) {
            radius_min = radius_prev;
            lat_prev   = -90.0;
            lon_prev   = 0.0;
            fnDepthBins++;
          }
          else if (outer_radius < radius_prev) {
            ClearModel();
            throw std::runtime_error("EarthModelBinned: depths are not sorted in decreasing "
                                     "order (or depth > 6371 km) in " + filename);
          }
          else if (longitude > lon_prev) {
            lat_prev = -90.0;
            fnLonBins++;
          }
          else if (longitude < lon_prev) {
            ClearModel();
            throw std::runtime_error("EarthModelBinned: longitudes are not sorted in "
                                     "increasing order (after depths) in " + filename);
          }
          else if (latitude < lat_prev) {
            ClearModel();
            throw std::runtime_error("EarthModelBinned: latitudes are not sorted in "
                                     "increasing order (after longitudes) in " + filename);
          }

          AddBin(outer_radius, radius_min, latitude, longitude, density, zoa, index);
          if (index > fMaxRegIndex) fMaxRegIndex = int(index);

          radius_prev = outer_radius;
          lat_prev    = latitude;
          lon_prev    = longitude;
        }
        if (fBins.empty() || fnDepthBins == 0) {
          ClearModel();
          throw std::runtime_error("EarthModelBinned: no bins in " + filename);
        }

        fRadiusMax = radius_prev;

        fnLonBins        = fnLonBins / fnDepthBins + 1;
        fInvLonBinWidth  = 0.5 * fnLonBins / M_PI;
        fHalfLonBinWidth = M_PI / fnLonBins;

        fnLatBins        = int(fBins.size()) / (fnLonBins * fnDepthBins);
        fInvLatBinWidth  = fnLatBins / M_PI;
        fHalfLatBinWidth = 0.5 * M_PI / fnLatBins;

        if (std::fabs(fHalfLatBinWidth - (fBins[0].latitude + 0.5 * M_PI)) >
            fHalfLatBinWidth * 0.01) {
          ClearModel();
          throw std::runtime_error("EarthModelBinned: latitude of the first bin centre is "
                                   "not in the middle of the first bin in " + filename);
        }
        // (not checked by OscProb) the bins must fill depth x lon x lat
        if (size_t(fnDepthBins) * fnLonBins * fnLatBins != fBins.size()) {
          ClearModel();
          throw std::runtime_error("EarthModelBinned: the bins do not form a depth x "
                                   "longitude x latitude grid in " + filename);
        }
      }

      /// Set the detector position: radius (km), latitude (deg N) and
      /// longitude (deg E). Port of EarthModelBase::SetDetectorCoordinates
      /// (the radius is made non-negative, the latitude folded into
      /// [-90, 90] and the longitude into [0, 360)).
      void SetDetPos(double rad, double lat = 0, double lon = 0)
      {
        if (rad < 0) rad = -rad;
        fDetRadius = rad;
        lat -= std::floor((lat + 90.0) / 360.0) * 360.0;
        if (lat > 90) { lat = 180.0 - lat; }
        fDetLat = lat / 180.0 * M_PI;
        lon -= std::floor(lon / 360.0) * 360.0;
        fDetLon = lon / 180.0 * M_PI;
      }

      /// Azimuth (degrees from North, of the direction from the detector to
      /// where the neutrino enters the Earth) used for paths given by their
      /// cos zenith alone: grids, bin averages, event lists without
      /// per-event azimuths, FillPath(cosT). Default 0. (Not in OscProb,
      /// whose FillPath takes it as an argument.)
      void   SetAzimuth(double phi) { fAzimuth = phi; }
      double GetAzimuth() const { return fAzimuth; }

      /// Path segments for direction (cosT, phi [deg]), from where the
      /// neutrino enters the Earth to the detector. Port of FillPath; the
      /// segment `layer` is the region index.
      std::vector<Segment<double>> FillPath(double cosT, double phi) const
      {
        std::vector<Segment<double>> out;
        const BinnedGeom             g = Geometry();
        BinnedWalk                   w;
        if (!binned_walk_init(g, w, cosT, phi)) {
          if (std::fabs(cosT) > 1) return out;
          throw std::runtime_error("EarthModelBinned::FillPath: starting bin outside the "
                                   "table");
        }
        double len;
        int    b;
        while (binned_walk_next(g, w, len, b))
          out.push_back(Segment<double>{len, fBins[size_t(b)].density, fBins[size_t(b)].zoa,
                                        fBins[size_t(b)].index});
        if (w.phase != 3)
          throw std::runtime_error("EarthModelBinned::FillPath: path left the table or "
                                   "exceeded the segment bound");
        return out;
      }
      /// FillPath for the model's azimuth (SetAzimuth).
      std::vector<Segment<double>> FillPath(double cosT) const
      {
        return FillPath(cosT, fAzimuth);
      }

      const std::vector<EarthBin>& GetEarthBins() const { return fBins; }

      /// Set Z/A of all bins of a region (no effect for an unknown index,
      /// as OscProb).
      void SetRegionZoA(int index, double zoa)
      {
        if (index > fMaxRegIndex) return;
        for (auto& b : fBins)
          if (b.index == index) b.zoa = zoa;
      }

      /// Z/A of the first bin of a region (0 if none, as OscProb).
      double GetRegionZoA(int index) const
      {
        if (index > fMaxRegIndex) return 0;
        for (auto& b : fBins)
          if (b.index == index) return b.zoa;
        return 0;
      }

      /// Multiply the density of all bins of a region by factor (the
      /// derivative parameter rho_<index> is d/d ln(factor)).
      void ScaleRegionDensity(int index, double factor)
      {
        if (index > fMaxRegIndex) return;
        for (auto& b : fBins)
          if (b.index == index) {
            double density_init = b.density;
            b.density           = factor * density_init;
          }
      }

      /// Region indices present in the table, sorted.
      std::vector<int> GetRegions() const
      {
        std::vector<int> r;
        for (auto& b : fBins)
          if (std::find(r.begin(), r.end(), b.index) == r.end()) r.push_back(b.index);
        std::sort(r.begin(), r.end());
        return r;
      }

      double GetDetRadius() const { return fDetRadius; }
      double GetDetLat() const { return fDetLat; }  ///< radians
      double GetDetLon() const { return fDetLon; }  ///< radians
      double GetRadiusMax() const { return fRadiusMax; }
      int    GetNDepthBins() const { return fnDepthBins; }
      int    GetNLatBins() const { return fnLatBins; }
      int    GetNLonBins() const { return fnLonBins; }

      /// Total baseline (km) for a given cosZ (EarthModelBase::GetTotalL).
      double GetTotalL(double cosT) const
      {
        if (std::fabs(cosT) > 1) return 0;
        double sinsqrT = 1 - cosT * cosT;
        return -fDetRadius * cosT +
               std::sqrt(fRadiusMax * fRadiusMax - fDetRadius * fDetRadius * sinsqrT);
      }

      /// cosZ for a given total baseline (EarthModelBase::GetCosT).
      double GetCosT(double L) const
      {
        if (L < fRadiusMax - fDetRadius) return 1;
        if (L > fRadiusMax + fDetRadius) return -1;
        return (fRadiusMax * fRadiusMax - fDetRadius * fDetRadius - L * L) /
               (2 * fDetRadius * L);
      }

      /// Cosines (<= 0) at which a trajectory grazes a depth-bin boundary
      /// below the detector, or the detector's own radius (cosZ = 0): the
      /// path length in a shell behaves like sqrt(cosT - c_k) there.
      /// Latitude and longitude bin boundaries are not included. Sorted.
      std::vector<double> GetGrazingCosines() const
      {
        std::vector<double> rad = {fDetRadius};
        const int           bpd = fnDepthBins ? int(fBins.size()) / fnDepthBins : 1;
        for (size_t i = 0; i < fBins.size(); i += size_t(bpd)) rad.push_back(fBins[i].radius_out);
        std::vector<double> c;
        for (double r : rad)
          if (r <= fDetRadius && r > 0 && fDetRadius > 0) {
            double s = r / fDetRadius;
            c.push_back(-std::sqrt(std::fmax(0.0, 1 - s * s)));
          }
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        return c;
      }

      /// Bound on the segments of a path: 2 per depth bin, and generously
      /// 4 per latitude and per longitude bin (a chord crosses each
      /// latitude cone at most twice and each meridian plane at most once).
      int MaxSegments() const { return 2 * fnDepthBins + 4 * fnLatBins + 4 * fnLonBins + 16; }

      /// Geometry for the walker, pointing into this model's bin arrays
      /// (valid until the model is changed or destroyed).
      BinnedGeom Geometry() const
      {
        BinnedGeom g = GeometryScalars();
        g.radius_out = fRout.data();
        g.radius_in  = fRin.data();
        g.latitude   = fLat.data();
        g.longitude  = fLon.data();
        return g;
      }

      /// The scalar part of Geometry() (pointers null).
      BinnedGeom GeometryScalars() const
      {
        BinnedGeom g;
        g.nbins     = int(fBins.size());
        g.nDepth    = fnDepthBins;
        g.nLon      = fnLonBins;
        g.nLat      = fnLatBins;
        g.max_seg   = MaxSegments();
        g.invLonW   = fInvLonBinWidth;
        g.invLatW   = fInvLatBinWidth;
        g.halfLonW  = fHalfLonBinWidth;
        g.halfLatW  = fHalfLatBinWidth;
        g.radiusMax = fRadiusMax;
        g.detRadius = fDetRadius;
        g.detLat    = fDetLat;
        g.detLon    = fDetLon;
        // TrajConstants::UpdateDetPos(fDetRadius, fDetLat, fDetLon)
        g.sinDetLat     = std::sin(fDetLat);
        g.cosDetLat     = std::cos(fDetLat);
        g.cosDetLon     = std::cos(fDetLon);
        g.sinDetLon     = std::sin(fDetLon);
        g.rDetSinDetLat = fDetRadius * g.sinDetLat;
        g.rDetCosDetLat = fDetRadius * g.cosDetLat;
        return g;
      }

      /// Per-bin geometry arrays (outer and inner radius, latitude,
      /// longitude), in table order.
      const std::vector<double>& RadiusOut() const { return fRout; }
      const std::vector<double>& RadiusIn() const { return fRin; }
      const std::vector<double>& Latitude() const { return fLat; }
      const std::vector<double>& Longitude() const { return fLon; }

    private:
      void ClearModel()
      {
        fnDepthBins = fnLonBins = fnLatBins = 0;
        fBins.clear();
        fRout.clear();
        fRin.clear();
        fLat.clear();
        fLon.clear();
      }

      void AddBin(double radius_out, double radius_in, double latitude, double longitude,
                  double density, double zoa, double index)
      {
        // EarthBin::SetBin: angles to radians as lat * M_PI / 180.0
        EarthBin b{radius_out,
                   radius_in,
                   latitude * M_PI / 180.0,
                   longitude * M_PI / 180.0,
                   density,
                   zoa,
                   int(index)};
        fBins.push_back(b);
        fRout.push_back(b.radius_out);
        fRin.push_back(b.radius_in);
        fLat.push_back(b.latitude);
        fLon.push_back(b.longitude);
      }

      std::vector<EarthBin> fBins;
      std::vector<double>   fRout, fRin, fLat, fLon;  ///< geometry arrays for the walker
      int                   fnDepthBins = 0, fnLonBins = 0, fnLatBins = 0;
      double                fInvLonBinWidth = 0, fInvLatBinWidth = 0;
      double                fHalfLonBinWidth = 0, fHalfLatBinWidth = 0;
      int                   fMaxRegIndex = -1;
      double                fRadiusMax   = 0;
      double                fDetRadius   = 6368;
      double                fDetLat      = 0;
      double                fDetLon      = 0;
      double                fAzimuth     = 0;
  };

} // namespace opg

#endif
