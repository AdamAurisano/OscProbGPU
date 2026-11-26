///////////////////////////////////////////////////////////////////////////////
/// \file prem.h
///
/// \brief Spherical-shell Earth model, ported from OscProb::PremModel.
///
/// The host class PremModel loads and edits the layer table exactly like
/// OscProb (including reading the table as float, merging identical
/// consecutive layers, and inserting a detector layer). Path segments are
/// not stored: the OPG_HD function for_each_segment() walks the path for a
/// given cos(zenith) and calls a functor per segment, reproducing
/// PremModel::FillPath operation by operation. It runs on host and device
/// over an EarthView, a POD of pointers into the layer table.
///
/// An EarthView can also describe a binned 3D Earth (EarthModelBinned,
/// earth/binned.h): then `binned` points to its geometry, the density, Z/A
/// and type arrays hold one entry per bin (type = region index), and
/// for_each_segment() walks the path with the binned walker
/// (earth/binned_walk.h) for the view's azimuth. The switch is made at run
/// time, in one loop with a single call of the functor, so the kernels are
/// compiled once for both Earth models.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_EARTH_PREM_H
#define OPG_EARTH_PREM_H

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../core/macros.h"
#include "binned_walk.h"

namespace opg {

  /// A path segment of constant matter density.
  template <class Real> struct Segment {
      Real length;   ///< km
      Real density;  ///< g/cm^3
      Real zoa;      ///< effective Z/A
      int  layer;    ///< layer type index (informational)
  };

  /// A segment whose density and Z/A have scalar type Z (e.g. dual numbers
  /// carrying derivatives with respect to the density scale and the Z/A of
  /// its layer type).
  template <class Real, class Z> struct SegmentZ {
      Real length;
      Z    density;
      Z    zoa;
      int  layer;
  };

  /// Device/host view of a PREM layer table, or of a binned Earth
  /// (non-owning).
  template <class Real> struct EarthView {
      int         nlayers;     ///< number of layers (binned: number of bins)
      int         det_layer;   ///< index of the detector layer (PREM)
      Real        det_radius;  ///< km
      const Real* radius;      ///< outer radius of each layer (km; PREM)
      const Real* density;     ///< g/cm^3, per layer (binned: per bin)
      const Real* zoa;         ///< Z/A, per layer (binned: per bin)
      const int*  type;        ///< layer type index (binned: region index)
      /// Binned Earth geometry; nullptr: PREM layers.
      const BinnedGeom* binned = nullptr;
      /// Azimuth of the path (degrees, as EarthModelBinned::FillPath);
      /// binned Earth only.
      double azimuth = 0;
  };

  /// The view with the azimuth of event i (az: per-event azimuths in
  /// degrees, or nullptr for the view's own).
  template <class Real>
  OPG_HD OPG_INLINE EarthView<Real> event_earth(const EarthView<Real>& e, const Real* az,
                                                size_t i)
  {
    EarthView<Real> v = e;
    if (az) v.azimuth = double(az[i]);
    return v;
  }

  //...........................................................................
  /// Walk the neutrino path for direction cosT and call
  /// f(const Segment<Real>&) for each segment, from production to detector.
  /// Returns the number of segments. Port of PremModel::FillPath, or for a
  /// binned view of EarthModelBinned::FillPath(cosT, e.azimuth).
  template <class Real, class F>
  OPG_HD inline int for_each_segment(const EarthView<Real>& e, Real cosT, F&& f)
  {
    // Do nothing if cosine is unphysical
    if (std::fabs(cosT) > 1) return 0;

    // Binned Earth: generator state in memory (see binned_walk.h)
    const bool binned = e.binned != nullptr;
    BinnedWalk w;

    Real minRsq = 0;
    int  nsteps = 0, layer = 0, dl = -1;
    if (binned) {
      if (!binned_walk_init(*e.binned, w, double(cosT), e.azimuth)) return 0;
      nsteps = e.binned->max_seg;
    }
    else {
      // Define the minimum path radius
      Real minR = e.det_radius * std::sqrt(1 - cosT * cosT);
      minRsq    = minR * minR;

      // Set the top layer index
      int toplayer = e.nlayers - 1;

      // Find the inner-most crossed layer
      int minlayer = 0;
      while (e.radius[minlayer] < minR) minlayer++;

      // Compute the number of path segments needed
      nsteps = toplayer - e.det_layer;
      if (cosT < 0) nsteps += 2 * (e.det_layer - minlayer) + 1;

      // Start at the top layer and go down
      layer = toplayer;
    }

    int count = 0;
    for (int i = 0; i < nsteps; i++) {
      Segment<Real> s;
      bool          ismin = false;
      if (binned) {
        double len;
        int    b;
        if (!binned_walk_next(*e.binned, w, len, b)) return count;
        s = Segment<Real>{Real(len), e.density[b], e.zoa[b], e.type[b]};
      }
      else {
        // Square of the path length between this layer's outer radius and
        // the inner-most radius
        Real L1 = e.radius[layer] * e.radius[layer] - minRsq;

        // If L1 is negative, outer radius is not crossed. This only happens
        // if detector is at the top layer and the neutrino is coming from
        // above.
        if (L1 < 0) return count;

        // Square of the path length between this layer's inner radius and
        // the inner-most radius
        Real L2 = -minRsq;
        if (layer > 0) L2 += e.radius[layer - 1] * e.radius[layer - 1];

        // If L2 is negative, inner radius is not crossed, so set this as the
        // minimum layer.
        ismin = (L2 <= 0 && cosT < 0);

        Real dL;
        if (ismin)
          dL = 2 * std::sqrt(L1);
        else if (L2 >= 0)
          dL = std::sqrt(L1) - std::sqrt(L2);
        else
          dL = std::sqrt(L1);

        s = Segment<Real>{dL, e.density[layer], e.zoa[layer], e.type[layer]};
      }
      f(s);
      count++;

      if (!binned) {
        // If we reached the inner-most layer, start moving up again.
        if (ismin) dl = 1;

        layer += dl;
      }
    }

    return count;
  }

  //...........................................................................
  /// Host-side PREM model. Mirrors the public interface of OscProb's
  /// PremModel where it matters for path generation.
  class PremModel {
    public:
      struct Layer {
          double radius;
          double density;
          double zoa;
          int    type;
          bool   same_matter(const Layer& o) const
          {
            return o.density == density && o.zoa == zoa && o.type == type;
          }
      };

      static constexpr double DET_TOL = 0.2;  ///< km

      /// Load a model file (radius density zoa type per line). An empty
      /// filename loads the default model (data/prem_default.txt).
      explicit PremModel(const std::string& filename = "")
      {
        fDetRadius = 6368;
        LoadModel(filename);
      }

      /// The default model file: $OPG_DATA_DIR/prem_default.txt if that
      /// environment variable is set (the Python package sets it to its own
      /// data directory), else the source tree's data/ for builds that
      /// compiled its path in, else prem_default.txt in the working directory.
      static std::string DefaultFile()
      {
        const char* env = std::getenv("OPG_DATA_DIR");
        if (env && *env) return std::string(env) + "/prem_default.txt";
#ifdef OPG_DATA_DIR
        return std::string(OPG_DATA_DIR) + "/prem_default.txt";
#else
        return "prem_default.txt";
#endif
      }

      void LoadModel(std::string filename)
      {
        ClearModel();
        if (filename.empty()) filename = DefaultFile();
        std::ifstream fin(filename);
        if (!fin) throw std::runtime_error("PremModel: cannot open " + filename);

        // OscProb reads the table as float
        float  radius, density, zoa, layer;
        double rprev = 0;
        while (fin >> radius >> density >> zoa >> layer) {
          if (radius <= rprev) {
            ClearModel();
            throw std::runtime_error(
                "PremModel: radii not sorted in increasing order in " + filename);
          }
          AddLayer(radius, density, zoa, layer);
          // NB: OscProb never updates rprev, so only radius <= 0 is caught.
        }
        AddDetLayer();
        fRadiusMax = fLayers.back().radius;
      }

      /// Set the detector radius (km).
      void SetDetPos(double rad)
      {
        if (rad < 0) rad = -rad;
        fDetRadius = rad;
        if (fLayers.size()) AddDetLayer();
      }

      void SetLayerZoA(int type, double zoa)
      {
        for (auto& l : fLayers)
          if (l.type == type) l.zoa = zoa;
      }

      /// Multiply the density of all layers of a given type by factor (not in
      /// OscProb; the derivative parameter rho_<type> is d/d ln(factor)).
      void ScaleLayerDensity(int type, double factor)
      {
        for (auto& l : fLayers)
          if (l.type == type) l.density *= factor;
      }

      double GetLayerZoA(int type) const
      {
        for (auto& l : fLayers)
          if (l.type == type) return l.zoa;
        return 0;
      }

      void SetTopLayerSize(double thick)
      {
        if (thick <= 0 || fLayers.empty()) return;
        int    n            = fLayers.size();
        double bottomRadius = n > 1 ? fLayers[n - 2].radius : 0;
        fLayers[n - 1].radius = bottomRadius + thick;
        fRadiusMax            = fLayers.back().radius;
      }

      double GetDetRadius() const { return fDetRadius; }
      int    GetDetLayer() const { return fDetLayer; }
      double GetRadiusMax() const { return fRadiusMax; }
      const std::vector<Layer>& GetLayers() const { return fLayers; }

      /// Total baseline (km) for a given cosZ. Port of
      /// EarthModelBase::GetTotalL.
      double GetTotalL(double cosT) const
      {
        if (std::fabs(cosT) > 1) return 0;
        double sinsqrT = 1 - cosT * cosT;
        return -fDetRadius * cosT +
               std::sqrt(fRadiusMax * fRadiusMax -
                         fDetRadius * fDetRadius * sinsqrT);
      }

      /// cosZ for a given total baseline. Port of EarthModelBase::GetCosT.
      double GetCosT(double L) const
      {
        if (L < fRadiusMax - fDetRadius) return 1;
        if (L > fRadiusMax + fDetRadius) return -1;
        return (fRadiusMax * fRadiusMax - fDetRadius * fDetRadius - L * L) /
               (2 * fDetRadius * L);
      }

      /// Cosines (< 0) at which a trajectory grazes a layer boundary below
      /// the detector. The path length in that layer behaves like
      /// sqrt(cosT - c_k) there, so P(cosT) has a kink. Sorted increasing.
      std::vector<double> GetGrazingCosines() const
      {
        std::vector<double> c;
        for (auto& l : fLayers)
          if (l.radius <= fDetRadius && l.radius > 0) {
            double s = l.radius / fDetRadius;
            c.push_back(-std::sqrt(std::fmax(0.0, 1 - s * s)));
          }
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        return c;
      }

      /// Path segments for cosT, computed on the host (for tests/debugging).
      std::vector<Segment<double>> FillPath(double cosT) const
      {
        std::vector<Segment<double>> out;
        HostTable<double> t(*this);
        for_each_segment(t.view(), cosT,
                         [&](const Segment<double>& s) { out.push_back(s); });
        return out;
      }

      /// Flattened, owning copy of the layer table in precision Real, from
      /// which an EarthView can be taken (host memory).
      template <class Real> struct HostTable {
          std::vector<Real> radius, density, zoa;
          std::vector<int>  type;
          int               det_layer;
          Real              det_radius;

          explicit HostTable(const PremModel& m)
          {
            for (auto& l : m.fLayers) {
              radius.push_back(Real(l.radius));
              density.push_back(Real(l.density));
              zoa.push_back(Real(l.zoa));
              type.push_back(l.type);
            }
            det_layer  = m.fDetLayer;
            det_radius = Real(m.fDetRadius);
          }

          EarthView<Real> view() const
          {
            return EarthView<Real>{int(radius.size()), det_layer, det_radius,
                                   radius.data(),      density.data(),
                                   zoa.data(),         type.data()};
          }
      };

    private:
      void ClearModel()
      {
        fDetLayer = 0;
        fLayers.clear();
      }

      void AddLayer(double radius, double density, double zoa, double type)
      {
        fLayers.push_back(Layer{radius, density, zoa, int(type)});
      }

      void CleanIdentical()
      {
        int i = 0;
        while (i < int(fLayers.size()) - 1) {
          if (fLayers[i].same_matter(fLayers[i + 1])) {
            fLayers.erase(fLayers.begin() + i);
            continue;
          }
          i++;
        }
      }

      void AddDetLayer()
      {
        CleanIdentical();

        fDetLayer = fLayers.size() - 1;

        if (fLayers.size() && fDetRadius > fLayers.back().radius) {
          fDetRadius = fLayers.back().radius;
          return;
        }

        for (int i = 0; i < int(fLayers.size()); i++) {
          double radius = fLayers[i].radius;
          if (radius > fDetRadius - DET_TOL) {
            fDetLayer = i;
            if (radius > fDetRadius + DET_TOL) {
              Layer det_layer  = fLayers[i];
              det_layer.radius = fDetRadius;
              fLayers.insert(fLayers.begin() + fDetLayer, det_layer);
            }
            else if (radius != fDetRadius) {
              fDetRadius = radius;
            }
            break;
          }
        }
      }

      std::vector<Layer> fLayers;
      int                fDetLayer  = 0;
      double             fDetRadius = 6368;
      double             fRadiusMax = 0;
  };

} // namespace opg

#endif
