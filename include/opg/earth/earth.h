///////////////////////////////////////////////////////////////////////////////
/// \file earth.h
///
/// \brief EarthModel: either a PremModel (spherical layers) or an
///        EarthModelBinned (depth x latitude x longitude bins), as accepted
///        by opg::Propagator, and EarthTable, the flattened copy the
///        engines walk paths through.
///
/// Both kinds are walked by the same for_each_segment() (earth/prem.h),
/// which switches on the view at run time, so every kernel is compiled once.
/// The `type` of a segment is the PREM layer type or the binned region
/// index; the Earth gradient parameters zoa_<t> and rho_<t> refer to it.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_EARTH_EARTH_H
#define OPG_EARTH_EARTH_H

#include <algorithm>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "binned.h"
#include "prem.h"

namespace opg {

  class EarthModel {
    public:
      /// The default PREM model.
      EarthModel() : fM(PremModel()) {}
      EarthModel(const PremModel& m) : fM(m) {}               // NOLINT: implicit
      EarthModel(const EarthModelBinned& m) : fM(m) {}        // NOLINT: implicit

      bool is_binned() const { return std::holds_alternative<EarthModelBinned>(fM); }

      /// The PREM model (throws std::logic_error for a binned Earth).
      const PremModel& prem() const
      {
        if (is_binned())
          throw std::logic_error("opg: the Earth model is an EarthModelBinned, not a PremModel");
        return std::get<PremModel>(fM);
      }
      PremModel& prem()
      {
        if (is_binned())
          throw std::logic_error("opg: the Earth model is an EarthModelBinned, not a PremModel");
        return std::get<PremModel>(fM);
      }
      /// The binned model (throws std::logic_error for PREM).
      const EarthModelBinned& binned() const
      {
        if (!is_binned())
          throw std::logic_error("opg: the Earth model is a PremModel, not an EarthModelBinned");
        return std::get<EarthModelBinned>(fM);
      }
      EarthModelBinned& binned()
      {
        if (!is_binned())
          throw std::logic_error("opg: the Earth model is a PremModel, not an EarthModelBinned");
        return std::get<EarthModelBinned>(fM);
      }

      /// Path for cosT (binned: at the model's azimuth).
      std::vector<Segment<double>> FillPath(double cosT) const
      {
        return is_binned() ? binned().FillPath(cosT) : prem().FillPath(cosT);
      }

      std::vector<double> GetGrazingCosines() const
      {
        return is_binned() ? binned().GetGrazingCosines() : prem().GetGrazingCosines();
      }

      double GetTotalL(double cosT) const
      {
        return is_binned() ? binned().GetTotalL(cosT) : prem().GetTotalL(cosT);
      }

      /// Layer types (PREM) or region indices (binned) present, sorted.
      std::vector<int> types() const
      {
        if (is_binned()) return binned().GetRegions();
        std::vector<int> t;
        for (const auto& l : prem().GetLayers())
          if (std::find(t.begin(), t.end(), l.type) == t.end()) t.push_back(l.type);
        std::sort(t.begin(), t.end());
        return t;
      }

      /// Bound on the number of segments of any path.
      int max_segments() const
      {
        if (is_binned()) return binned().MaxSegments();
        const auto& p = prem();
        return int(p.GetLayers().size()) + p.GetDetLayer();
      }

    private:
      std::variant<PremModel, EarthModelBinned> fM;
  };

  //...........................................................................
  /// Flattened, owning copy of an Earth model in precision Real (per layer,
  /// or per bin with the binned geometry in double), from which an
  /// EarthView into host memory can be taken. Engines upload its arrays.
  template <class Real> struct EarthTable {
      std::vector<Real>   radius, density, zoa;  ///< radius: PREM only
      std::vector<int>    type;
      int                 det_layer  = 0;
      Real                det_radius = 0;
      bool                binned     = false;
      int                 max_seg    = 0;
      double              azimuth    = 0;
      BinnedGeom          geom;                  ///< scalars (binned)
      std::vector<double> rout, rin, lat, lon;   ///< bin geometry (binned)

      explicit EarthTable(const EarthModel& m)
      {
        max_seg = m.max_segments();
        if (!m.is_binned()) {
          const PremModel::HostTable<Real> t(m.prem());
          radius     = t.radius;
          density    = t.density;
          zoa        = t.zoa;
          type       = t.type;
          det_layer  = t.det_layer;
          det_radius = t.det_radius;
          return;
        }
        const EarthModelBinned& b = m.binned();
        if (b.GetDetRadius() > b.GetRadiusMax())
          throw std::invalid_argument("opg: EarthModelBinned detector radius above the top "
                                      "of the model");
        if (b.GetDetRadius() <= 0)
          throw std::invalid_argument("opg: EarthModelBinned detector at the centre");
        binned  = true;
        azimuth = b.GetAzimuth();
        geom    = b.GeometryScalars();
        for (const auto& e : b.GetEarthBins()) {
          density.push_back(Real(e.density));
          zoa.push_back(Real(e.zoa));
          type.push_back(e.index);
        }
        rout       = b.RadiusOut();
        rin        = b.RadiusIn();
        lat        = b.Latitude();
        lon        = b.Longitude();
        det_radius = Real(b.GetDetRadius());
      }

      /// Number of rows of density/zoa/type (layers or bins).
      int rows() const { return int(density.size()); }

      /// View into this table's host memory (geom_out: storage for the
      /// geometry the view points to, binned only).
      EarthView<Real> view(BinnedGeom& geom_out) const
      {
        EarthView<Real> v{rows(),         det_layer,   det_radius,
                          radius.empty() ? nullptr : radius.data(),
                          density.data(), zoa.data(),  type.data()};
        if (binned) {
          geom_out            = geom;
          geom_out.radius_out = rout.data();
          geom_out.radius_in  = rin.data();
          geom_out.latitude   = lat.data();
          geom_out.longitude  = lon.data();
          v.binned            = &geom_out;
          v.azimuth           = azimuth;
        }
        return v;
      }

      /// Whether two tables differ at most in densities and Z/A (the
      /// condition for per-point Earths of a batch).
      bool same_geometry(const EarthTable& o) const
      {
        if (binned != o.binned || radius != o.radius || type != o.type ||
            det_layer != o.det_layer || det_radius != o.det_radius)
          return false;
        if (!binned) return true;
        return rout == o.rout && rin == o.rin && lat == o.lat && lon == o.lon &&
               azimuth == o.azimuth && geom.detLat == o.geom.detLat &&
               geom.detLon == o.geom.detLon && geom.detRadius == o.geom.detRadius;
      }
  };

} // namespace opg

#endif
