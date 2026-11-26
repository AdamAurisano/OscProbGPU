// Python bindings for OscProbGPU (nanobind).
//
// Each model class wraps an opg::Propagator<Model<double>> together with its
// parameter struct; parameters are pushed to the propagator lazily before
// the next computation, so setters are cheap.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "opg/propagator.h"

#ifdef OPG_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace {

  using Arr1  = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
  using Arr2  = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
  using Arr1u = nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

  std::vector<double> to_vec(const Arr1& a)
  {
    return std::vector<double>(a.data(), a.data() + a.shape(0));
  }

  /// An Earth model argument: PremModel or EarthModelBinned.
  using PyEarth = std::variant<opg::PremModel, opg::EarthModelBinned>;

  opg::EarthModel to_earth(const PyEarth& e)
  {
    if (auto* b = std::get_if<opg::EarthModelBinned>(&e)) return opg::EarthModel(*b);
    return opg::EarthModel(std::get<opg::PremModel>(e));
  }

  /// Per-event azimuths (one per event) or empty for None.
  std::vector<double> to_azimuth(const std::optional<Arr1>& a, size_t n)
  {
    if (!a) return {};
    if (a->shape(0) != n) throw std::invalid_argument("azimuth must have one value per event");
    return to_vec(*a);
  }

  /// Per-point Earths of a batch: all PremModel or all EarthModelBinned.
  struct BatchEarths {
      std::vector<opg::PremModel>        prem;
      std::vector<opg::EarthModelBinned> binned;
      bool                               is_binned = false;
  };
  std::optional<BatchEarths> to_batch_earths(const std::optional<std::vector<PyEarth>>& e)
  {
    if (!e) return std::nullopt;
    BatchEarths b;
    b.is_binned = !e->empty() && std::holds_alternative<opg::EarthModelBinned>((*e)[0]);
    for (const auto& x : *e) {
      if (std::holds_alternative<opg::EarthModelBinned>(x) != b.is_binned)
        throw std::invalid_argument("earths: all PremModel or all EarthModelBinned");
      if (b.is_binned) b.binned.push_back(std::get<opg::EarthModelBinned>(x));
      else b.prem.push_back(std::get<opg::PremModel>(x));
    }
    return b;
  }

  /// Per-event extra inputs extra[x, i] (x < nx, i < n) as a flat vector,
  /// or empty for None.
  std::vector<double> to_extra(const std::optional<Arr2>& a, int nx, size_t n)
  {
    if (!a) return {};
    if (nx == 0)
      throw std::invalid_argument("extra: this model takes no per-event inputs");
    if (a->shape(0) != size_t(nx) || a->shape(1) != n)
      throw std::invalid_argument("extra must have shape (n_event_extra, n_events)");
    return std::vector<double>(a->data(), a->data() + a->size());
  }

  /// numpy array that owns a copy of the data
  nb::ndarray<nb::numpy, double> make_array(std::vector<double>&& v,
                                            std::vector<size_t> shape)
  {
    auto* heap = new std::vector<double>(std::move(v));
    nb::capsule owner(heap, [](void* p) noexcept {
      delete static_cast<std::vector<double>*>(p);
    });
    return nb::ndarray<nb::numpy, double>(heap->data(), shape.size(),
                                          shape.data(), owner);
  }

  opg::Flavor parse_flavor(const std::string& s)
  {
    if (s == "both") return opg::Flavor::Both;
    if (s == "nu" || s == "neutrino") return opg::Flavor::Neutrino;
    if (s == "nubar" || s == "antineutrino") return opg::Flavor::Antineutrino;
    throw std::invalid_argument("flavor must be 'both', 'nu' or 'nubar'");
  }

  opg::EMeasure parse_measure(const std::string& s)
  {
    if (s == "linear") return opg::EMeasure::Linear;
    if (s == "log") return opg::EMeasure::Log;
    if (s == "inv" || s == "invE" || s == "loe") return opg::EMeasure::InvE;
    throw std::invalid_argument("measure must be 'linear', 'log' or 'invE'");
  }

  opg::CosZRule parse_rule(const std::string& s)
  {
    if (s == "layer" || s == "layer_adapted") return opg::CosZRule::LayerAdapted;
    if (s == "plain") return opg::CosZRule::Plain;
    throw std::invalid_argument("cosz_rule must be 'layer' or 'plain'");
  }

  std::vector<opg::Segment<double>> to_path(const Arr2& seg)
  {
    if (seg.shape(1) < 2 || seg.shape(1) > 4)
      throw std::invalid_argument("path must be (n, 2..4): length [km], "
                                  "density [g/cm3], zoa (0.5), layer");
    std::vector<opg::Segment<double>> p;
    for (size_t i = 0; i < seg.shape(0); i++) {
      const double* r = seg.data() + i * seg.shape(1);
      p.push_back({r[0], r[1], seg.shape(1) > 2 ? r[2] : 0.5,
                   seg.shape(1) > 3 ? int(r[3]) : 0});
    }
    return p;
  }

  //...........................................................................
  using AnyArr = nb::ndarray<const double, nb::c_contig>;

  /// DLPack capsule for device memory (owner kept alive by the capsule);
  /// read-only unless `writable` (buffers from cuda_empty).
  inline nb::object dlpack_capsule(uintptr_t ptr, const std::vector<size_t>& shape, int device,
                                   nb::handle owner, bool writable)
  {
    if (writable)
      return nb::cast(nb::ndarray<double, nb::device::cuda>(
          reinterpret_cast<double*>(ptr), shape.size(), shape.data(), owner, nullptr,
          nb::dtype<double>(), nb::device::cuda::value, device));
    return nb::cast(nb::ndarray<const double, nb::device::cuda>(
        reinterpret_cast<const double*>(ptr), shape.size(), shape.data(), owner, nullptr,
        nb::dtype<double>(), nb::device::cuda::value, device));
  }

  /// Single-GPU propagator check for the device-array API.
  template <class Prop> int single_device(const Prop& prop, const char* what)
  {
    if (prop.devices().size() != 1)
      throw std::invalid_argument(std::string(what) +
                                  ": CUDA arrays need a propagator on exactly one "
                                  "GPU (use the C++ API for several)");
    return prop.devices()[0];
  }

  using DevOut = nb::ndarray<double, nb::c_contig, nb::device::cuda>;
  using DevIn  = nb::ndarray<const double, nb::c_contig, nb::device::cuda>;

  void* to_stream(uintptr_t s) { return reinterpret_cast<void*>(s); }

  /// Check a CUDA array argument: on `dev` with `n` entries.
  template <class A> void check_dev(const A& a, int dev, size_t n, const char* what)
  {
    if (a.device_id() != dev)
      throw std::invalid_argument(std::string(what) + ": array on another GPU");
    if (a.size() != n)
      throw std::invalid_argument(std::string(what) + ": array must have " +
                                  std::to_string(n) + " entries");
  }

  /// AnalyticAvgOptions from None (defaults) or a bound options object.
  opg::AnalyticAvgOptions opts(const std::optional<opg::AnalyticAvgOptions>& o)
  {
    return o ? *o : opg::AnalyticAvgOptions{};
  }

  opg::BinVar parse_binvar(const std::string& s)
  {
    if (s == "E" || s == "e") return opg::BinVar::E;
    if (s == "LoE" || s == "loe" || s == "L/E") return opg::BinVar::LoE;
    throw std::invalid_argument("var must be 'E' or 'LoE'");
  }

  /// Sub-bins as rows (bin, u0, h, weight, beta).
  std::vector<opg::AnalyticSubBin> to_subbins(const Arr2& a)
  {
    if (a.shape(1) != 5)
      throw std::invalid_argument("sub-bins must be (n, 5): bin, u0, h, weight, beta");
    std::vector<opg::AnalyticSubBin> v(a.shape(0));
    for (size_t i = 0; i < v.size(); i++) {
      const double* r = a.data() + 5 * i;
      v[i]            = {int(r[0]), r[1], r[2], r[3], r[4]};
    }
    return v;
  }

  template <class Model> struct PyModel {
      using Params           = typename Model::Params;
      static constexpr int N = Model::N;
      static constexpr int NX = opg::Propagator<Model>::n_event_extra();

      opg::Propagator<Model> prop;
      Params                 params;
      bool                   dirty = true;
      /// Bumped whenever the device_probs()/device_binned() buffers may be
      /// freed or reallocated (set_grid, set_bins, set_earth): DeviceArray
      /// views taken before are then stale.
      uint64_t gen_probs = 0, gen_binned = 0;

      /// Parameter points: the model's current parameters with the named
      /// model parameters (parameter_names) set to values[p, k].
      std::vector<Params> points(const std::vector<std::string>& names, const Arr2& values) const
      {
        const auto all = opg::Propagator<Model>::parameter_names();
        if (values.shape(1) != names.size())
          throw std::invalid_argument("values must have shape (n_points, len(names))");
        std::vector<int> idx;
        for (const auto& n : names) {
          auto it = std::find(all.begin(), all.end(), n);
          if (it == all.end())
            throw std::invalid_argument("unknown model parameter '" + n +
                                        "' (see parameter_names; Earth parameters go "
                                        "through earths)");
          idx.push_back(int(it - all.begin()));
        }
        std::vector<Params> pts(values.shape(0), params);
        for (size_t p = 0; p < pts.size(); p++)
          for (size_t k = 0; k < idx.size(); k++)
            Model::template param_ref<double>(pts[p], idx[k]) =
                values.data()[p * names.size() + k];
        return pts;
      }

      PyModel(const opg::EarthModel& earth, const std::vector<int>& devices,
              int threads)
          : prop(earth, devices, threads)
      {
        opg::set_std_pars(params.mix);
      }

      void sync()
      {
        if (dirty) {
          prop.set_params(params);
          dirty = false;
        }
      }
      auto& mix()
      {
        dirty = true;
        return params.mix;
      }

      nb::ndarray<nb::numpy, double> probs()
      {
        const double* p;
        {
          nb::gil_scoped_release r;
          p = prop.probs();
        }
        size_t        n = 2 * N * N * prop.n_cosines() * prop.n_energies();
        return make_array(std::vector<double>(p, p + n),
                          {2, N, N, prop.n_cosines(), prop.n_energies()});
      }

      nb::ndarray<nb::numpy, double> binned()
      {
        const double* p;
        {
          nb::gil_scoped_release r;
          p = prop.binned();
        }
        size_t n = 2 * N * N * prop.n_cosine_bins() * prop.n_energy_bins();
        return make_array(std::vector<double>(p, p + n),
                          {2, N, N, prop.n_cosine_bins(), prop.n_energy_bins()});
      }
  };

  template <class Model>
  nb::class_<PyModel<Model>> bind_model(nb::module_& m, const char* name,
                                        const char* doc)
  {
    using W = PyModel<Model>;
    constexpr int N = Model::N;
    nb::class_<W> c(m, name, doc);
    c.def_prop_ro_static("n_flavours", [](nb::handle) { return N; })
        .def_prop_ro("on_gpu", [](W& w) { return w.prop.on_gpu(); })
        .def_prop_ro("devices", [](W& w) { return w.prop.devices(); })
        // mixing parameters (OscProb conventions, 1-based indices)
        .def("set_angle",
             [](W& w, int i, int j, double v) {
               if (i > j) std::swap(i, j);
               if (i < 1 || j > N || i == j)
                 throw std::invalid_argument("invalid angle indices");
               w.mix().SetAngle(i, j, v);
             },
             "i"_a, "j"_a, "theta"_a, "Set theta_ij in radians.")
        .def("set_delta",
             [](W& w, int i, int j, double v) {
               if (i > j) std::swap(i, j);
               if (i < 1 || j > N || j < i + 2)
                 throw std::invalid_argument("invalid delta indices (i+1 < j)");
               w.mix().SetDelta(i, j, v);
             },
             "i"_a, "j"_a, "delta"_a, "Set the CP phase delta_ij in radians.")
        .def("set_dm",
             [](W& w, int j, double v) {
               if (j < 2 || j > N) throw std::invalid_argument("invalid dm index");
               w.mix().SetDm(j, v);
             },
             "j"_a, "dm"_a, "Set dm^2_j1 in eV^2.")
        .def("get_angle",
             [](W& w, int i, int j) {
               if (i > j) std::swap(i, j);
               if (i < 1 || j > N || i == j) throw std::out_of_range("invalid angle indices");
               return w.params.mix.th[i - 1][j - 1];
             },
             "i"_a, "j"_a)
        .def("get_delta",
             [](W& w, int i, int j) {
               if (i > j) std::swap(i, j);
               if (i < 1 || j > N || j < i + 2)
                 throw std::out_of_range("invalid delta indices (i+1 < j)");
               return w.params.mix.dcp[i - 1][j - 1];
             },
             "i"_a, "j"_a)
        .def("get_dm",
             [](W& w, int j) {
               if (j < 1 || j > N) throw std::out_of_range("invalid dm index");
               return w.params.mix.dm[j - 1];
             },
             "j"_a)
        .def("set_std_pars", [](W& w) { opg::set_std_pars(w.mix()); },
             "Reset mixing parameters to OscProb's PDG defaults.")
        // earth
        .def("set_earth",
             [](W& w, const PyEarth& e) {
               const opg::EarthModel em = to_earth(e);
               w.gen_probs++;
               w.gen_binned++;
               nb::gil_scoped_release r;
               w.prop.set_earth(em);
             },
             "earth"_a, "Earth model: PremModel or EarthModelBinned.")
        .def("set_earth_azimuth",
             [](W& w, double phi) {
               w.gen_probs++;
               w.gen_binned++;
               nb::gil_scoped_release r;
               w.prop.set_earth_azimuth(phi);
             },
             "phi"_a,
             "Azimuth (degrees, EarthModelBinned convention) of the paths through a "
             "binned Earth for grids, bin averages and event lists without per-event "
             "azimuths (sets it on the propagator's copy of the Earth model). "
             "Raises for a PREM Earth.")
        .def_prop_ro("earth_azimuth", [](W& w) { return w.prop.earth_azimuth(); },
                     "Azimuth of the binned Earth's paths (degrees).")
        .def_prop_ro("earth_is_binned", [](W& w) { return w.prop.earth_is_binned(); })
        // grid mode
        .def("set_grid",
             [](W& w, Arr1 E, Arr1 C) {
               auto ev = to_vec(E), cv = to_vec(C);
               w.gen_probs++;
               nb::gil_scoped_release r;
               w.prop.set_grid(ev, cv);
             },
             "energies"_a, "cosines"_a,
             "Set the (E [GeV], cosZ) grid; uploads once.")
        .def("calculate",
             [](W& w, const std::string& fl) {
               const auto f = parse_flavor(fl);
               nb::gil_scoped_release r;
               w.sync();
               w.prop.calculate(f);
             },
             "flavor"_a = "both", "Launch the grid computation (asynchronous).")
        .def("wait", [](W& w) { w.prop.wait(); }, nb::call_guard<nb::gil_scoped_release>())
        .def("_device_generation",
             [](W& w, const std::string& what) -> uint64_t {
               if (what == "probs") return w.gen_probs;
               if (what == "binned") return w.gen_binned;
               throw std::invalid_argument("_device_generation: probs or binned");
             },
             "what"_a)
        .def("probs", &W::probs,
             "Grid probabilities P[nubar, a, b, iC, iE] = P(a -> b) (copy).")
        .def("prob",
             [](W& w, int a, int b, size_t ic, size_t ie, bool nubar) {
               return w.prop.prob(a, b, ic, ie, nubar);
             },
             "a"_a, "b"_a, "ic"_a, "ie"_a, "nubar"_a = false)
        // bin-averaged mode
        .def("set_bins",
             [](W& w, Arr1 Ee, Arr1 Ce, int nE, int nC, const std::string& meas,
                const std::string& rule) {
               auto       ev = to_vec(Ee), cv = to_vec(Ce);
               const auto m = parse_measure(meas);
               const auto ru = parse_rule(rule);
               w.gen_binned++;
               nb::gil_scoped_release r;
               w.prop.set_bins(ev, cv, nE, nC, m, ru);
             },
             "energy_edges"_a, "cosine_edges"_a, "n_gl_energy"_a = 8,
             "n_gl_cosine"_a = 8, "measure"_a = "linear", "cosz_rule"_a = "layer",
             "Set a 2D binning for Gauss-Legendre bin averages.")
        .def("calculate_binned",
             [](W& w, const std::string& fl) {
               const auto f = parse_flavor(fl);
               nb::gil_scoped_release r;
               w.sync();
               w.prop.calculate_binned(f);
             },
             "flavor"_a = "both")
        .def("binned", &W::binned,
             "Bin averages A[nubar, a, b, iCbin, iEbin] (copy).")
        // one-shot modes
        .def_prop_ro_static("n_event_extra",
                            [](nb::handle) { return opg::Propagator<Model>::n_event_extra(); },
                            "Number of per-event extra inputs in event lists (e.g. "
                            "SiderealLIV: azimuth [deg], local sidereal time [h]).")
        .def("prob_points",
             [](W& w, Arr1 E, Arr1 C, Arr1u nb_, std::optional<Arr2> ex,
                std::optional<Arr1> az) {
               if (E.shape(0) != C.shape(0) || E.shape(0) != nb_.shape(0))
                 throw std::invalid_argument("size mismatch");
               std::vector<uint8_t> nbv(nb_.data(), nb_.data() + nb_.shape(0));
               auto ev = to_vec(E), cv = to_vec(C), xv = to_extra(ex, W::NX, E.shape(0));
               auto av = to_azimuth(az, E.shape(0));
               std::vector<double> out;
               {
                 nb::gil_scoped_release r;
                 w.sync();
                 out = w.prop.prob_points(ev, cv, nbv, xv, av);
               }
               return make_array(std::move(out), {N, N, E.shape(0)});
             },
             "energies"_a, "cosines"_a, "nubar"_a, "extra"_a = nb::none(),
             "azimuth"_a = nb::none(),
             "Event list: returns P[a, b, i]. extra: optional per-event inputs "
             "extra[x, i], x < n_event_extra (None: the values set on the model). "
             "azimuth: optional per-event azimuths (degrees) of the paths through a "
             "binned Earth (None: earth_azimuth; never taken from extra); ignored for "
             "PREM.")
        .def("prob_path",
             [](W& w, Arr1 E, Arr2 seg, bool nubar) {
               auto ev = to_vec(E);
               auto pv = to_path(seg);
               std::vector<double> out;
               {
                 nb::gil_scoped_release r;
                 w.sync();
                 out = w.prop.prob_path(ev, pv, nubar);
               }
               return make_array(std::move(out), {N, N, E.shape(0)});
             },
             "energies"_a, "path"_a, "nubar"_a = false,
             "Fixed path given as rows (length km, density, [zoa, layer]); "
             "returns P[a, b, iE].")
        // gradients
        .def_prop_ro_static("parameter_names",
                            [](nb::handle) { return opg::Propagator<Model>::parameter_names(); },
                            "Names of the differentiable parameters.")
        .def_prop_ro_static("has_gradients",
                            [](nb::handle) { return opg::Propagator<Model>::has_gradients(); })
        .def_prop_ro_static("default_gradient_params",
                            [](nb::handle) {
                              return opg::Propagator<Model>::default_gradient_params();
                            },
                            "Parameters selected by set_gradient_params() without "
                            "arguments (e.g. NSI leaves out the fermion couplings).")
        .def("set_gradient_params",
             [](W& w, std::optional<std::vector<std::string>> names) {
               if (names)
                 w.prop.set_gradient_params(*names);
               else
                 w.prop.set_gradient_params();
             },
             "names"_a = nb::none(),
             "Select parameters to differentiate (empty list = gradients off; "
             "no argument = default_gradient_params).")
        .def_prop_ro("gradient_params", [](W& w) { return w.prop.gradient_params(); })
        .def_prop_ro_static("has_adjoint_gradients",
                            [](nb::handle) {
                              return opg::Propagator<Model>::has_adjoint_gradients();
                            },
                            "Whether the weighted gradient modes can use reverse mode.")
        .def_prop_rw(
            "adjoint_gradients", [](W& w) { return w.prop.adjoint_gradients(); },
            [](W& w, bool on) { w.prop.set_adjoint_gradients(on); },
            "Use reverse mode (cost independent of the number of parameters) in the "
            "weighted gradient modes where the model supports it (default True).")
        .def("calculate_gradient",
             [](W& w, const std::string& fl) {
               const auto f = parse_flavor(fl);
               nb::gil_scoped_release r;
               w.sync();
               w.prop.calculate(f, true);
             },
             "flavor"_a = "both",
             "Grid probabilities and gradients (see probs() and grad()).")
        .def("grad",
             [](W& w) {
               const double* g;
               {
                 nb::gil_scoped_release r;
                 g = w.prop.grad();
               }
               size_t        np = w.prop.n_gradient_params();
               size_t        n  = 2 * np * N * N * w.prop.n_cosines() * w.prop.n_energies();
               return make_array(std::vector<double>(g, g + n),
                                 {2, np, N, N, w.prop.n_cosines(), w.prop.n_energies()});
             },
             "Grid gradients G[nubar, p, a, b, iC, iE] = dP(a -> b)/dp (copy).")
        .def("weighted_gradient",
             [](W& w, AnyArr wt, const std::string& fl) {
               const auto          f = parse_flavor(fl);
               std::vector<double> g;
               if (wt.device_type() == nb::device::cuda::value) {
                 const int dev = single_device(w.prop, "weighted_gradient");
                 if (wt.device_id() != dev)
                   throw std::invalid_argument("weighted_gradient: weights on another GPU");
                 if (wt.size() != 2 * size_t(N) * N * w.prop.n_cosines() * w.prop.n_energies())
                   throw std::invalid_argument("weighted_gradient: weights must have "
                                               "the shape of probs()");
                 const double*          d = wt.data();
                 nb::gil_scoped_release r;
                 w.sync();
                 g = w.prop.weighted_gradient_device({d}, f);
               }
               else {
                 if (wt.device_type() != nb::device::cpu::value)
                   throw std::invalid_argument("weighted_gradient: weights must be a "
                                               "host or CUDA array");
                 std::vector<double>    v(wt.data(), wt.data() + wt.size());
                 nb::gil_scoped_release r;
                 w.sync();
                 g = w.prop.weighted_gradient(v, f);
               }
               return make_array(std::move(g), {g.size()});
             },
             "weights"_a, "flavor"_a = "both",
             "sum of weights * dP/dp over the grid; weights have the shape of "
             "probs() and may be a CUDA array (DLPack, e.g. CuPy or PyTorch) on "
             "the propagator's GPU, which avoids uploading them. Returns one "
             "value per gradient parameter.")
        .def("_device_view",
             [](W& w, const std::string& what) {
               const int dev = single_device(w.prop, ("device_" + what).c_str());
               const double* p;
               std::vector<size_t> shape = {2, size_t(N), size_t(N)};
               if (what == "probs") {
                 p = w.prop.device_probs(0);
                 shape.push_back(w.prop.n_cosines());
                 shape.push_back(w.prop.n_energies());
               }
               else if (what == "binned") {
                 p = w.prop.device_binned(0);
                 shape.push_back(w.prop.n_cosine_bins());
                 shape.push_back(w.prop.n_energy_bins());
               }
               else
                 throw std::invalid_argument("_device_view: probs or binned");
               if (!p) throw std::logic_error("device_" + what + ": nothing computed yet");
               const uint64_t gen = what == "probs" ? w.gen_probs : w.gen_binned;
               return nb::make_tuple(uintptr_t(p), nb::tuple(nb::cast(shape)), dev, gen);
             },
             "what"_a, "(pointer, shape, device, generation) of device-resident results "
             "(see device_probs()/device_binned()).")
        .def("prob_points_grad",
             [](W& w, Arr1 E, Arr1 C, Arr1u nb_, std::optional<Arr2> ex,
                std::optional<Arr1> az) {
               std::vector<uint8_t> nbv(nb_.data(), nb_.data() + nb_.shape(0));
               auto ev = to_vec(E), cv = to_vec(C), xv = to_extra(ex, W::NX, E.shape(0));
               auto av = to_azimuth(az, E.shape(0));
               std::vector<double>  P, dP;
               {
                 nb::gil_scoped_release r;
                 w.sync();
                 w.prop.prob_points_grad(ev, cv, nbv, P, dP, xv, av);
               }
               size_t np = w.prop.n_gradient_params();
               return nb::make_tuple(make_array(std::move(P), {N, N, E.shape(0)}),
                                     make_array(std::move(dP), {np, N, N, E.shape(0)}));
             },
             "energies"_a, "cosines"_a, "nubar"_a, "extra"_a = nb::none(),
             "azimuth"_a = nb::none(),
             "Event list: returns (P[a, b, i], dP[p, a, b, i]); extra and azimuth "
             "as in prob_points.")
        .def("weighted_gradient_points",
             [](W& w, Arr1 E, Arr1 C, Arr1u nb_,
                nb::ndarray<const double, nb::c_contig, nb::device::cpu> wt,
                std::optional<Arr2> ex, std::optional<Arr1> az) {
               std::vector<uint8_t> nbv(nb_.data(), nb_.data() + nb_.shape(0));
               std::vector<double>  v(wt.data(), wt.data() + wt.size());
               auto ev = to_vec(E), cv = to_vec(C), xv = to_extra(ex, W::NX, E.shape(0));
               auto av = to_azimuth(az, E.shape(0));
               std::vector<double> g;
               {
                 nb::gil_scoped_release r;
                 w.sync();
                 g = w.prop.weighted_gradient_points(ev, cv, nbv, v, xv, av);
               }
               return make_array(std::move(g), {g.size()});
             },
             "energies"_a, "cosines"_a, "nubar"_a, "weights"_a, "extra"_a = nb::none(),
             "azimuth"_a = nb::none(),
             "sum of weights[a, b, i] * dP_ab(i)/dp over an event list; extra "
             "and azimuth as in prob_points.")
        .def("weighted_gradient_points_binned",
             [](W& w, Arr1 E, Arr1 C, Arr1u nb_,
                nb::ndarray<const double, nb::c_contig, nb::device::cpu> wt,
                nb::ndarray<const int, nb::ndim<1>, nb::c_contig, nb::device::cpu> bins,
                int nbins, std::optional<Arr2> ex, std::optional<Arr1> az) {
               std::vector<uint8_t> nbv(nb_.data(), nb_.data() + nb_.shape(0));
               std::vector<double>  v(wt.data(), wt.data() + wt.size());
               std::vector<int>     bv(bins.data(), bins.data() + bins.shape(0));
               auto ev = to_vec(E), cv = to_vec(C), xv = to_extra(ex, W::NX, E.shape(0));
               auto av = to_azimuth(az, E.shape(0));
               std::vector<double> G;
               {
                 nb::gil_scoped_release r;
                 w.sync();
                 G = w.prop.weighted_gradient_points_binned(ev, cv, nbv, v, bv, nbins, xv,
                                                            av);
               }
               return make_array(std::move(G),
                                 {size_t(nbins), w.prop.n_gradient_params()});
             },
             "energies"_a, "cosines"_a, "nubar"_a, "weights"_a, "bins"_a, "nbins"_a,
             "extra"_a = nb::none(), "azimuth"_a = nb::none(),
             "Per-analysis-bin weighted gradient: G[b, p] = sum over events i "
             "with bins[i] == b of weights[a, b, i] * dP_ab(i)/dp (bins outside "
             "[0, nbins) are ignored); extra and azimuth as in prob_points.")
        .def("prob_path_grad",
             [](W& w, Arr1 E, Arr2 seg, bool nubar) {
               auto ev = to_vec(E);
               auto pv = to_path(seg);
               std::vector<double> P, dP;
               {
                 nb::gil_scoped_release r;
                 w.sync();
                 w.prop.prob_path_grad(ev, pv, nubar, P, dP);
               }
               size_t np = w.prop.n_gradient_params();
               return nb::make_tuple(make_array(std::move(P), {N, N, E.shape(0)}),
                                     make_array(std::move(dP), {np, N, N, E.shape(0)}));
             },
             "energies"_a, "path"_a, "nubar"_a = false,
             "Fixed path: returns (P[a, b, iE], dP[p, a, b, iE]).")
        .def_prop_ro("earth_parameter_names",
                     [](W& w) { return w.prop.earth_parameter_names(); },
                     "Earth gradient parameters of this propagator's Earth model: "
                     "zoa_<layer type> (Z/A of all layers of the type), then "
                     "rho_<layer type> (common relative scale of their densities, "
                     "dP/d ln rho); never selected by default.")
        .def_prop_ro("gradient_parameter_names",
                     [](W& w) { return w.prop.gradient_parameter_names(); },
                     "All selectable gradient parameters (model, then Earth).")
        .def("calculate_binned_gradient",
             [](W& w, const std::string& fl) {
               const auto f = parse_flavor(fl);
               nb::gil_scoped_release r;
               w.sync();
               w.prop.calculate_binned(f, true);
             },
             "flavor"_a = "both",
             "Bin averages and their gradients (see binned() and binned_grad()).")
        .def("binned_grad",
             [](W& w) {
               const double* g;
               {
                 nb::gil_scoped_release r;
                 g = w.prop.binned_grad();
               }
               size_t        np = w.prop.n_gradient_params();
               size_t        nc = w.prop.n_cosine_bins(), ne = w.prop.n_energy_bins();
               return make_array(std::vector<double>(g, g + 2 * np * N * N * nc * ne),
                                 {2, np, N, N, nc, ne});
             },
             "Binned gradients G[nubar, p, a, b, iCbin, iEbin] (copy).")
        .def("weighted_gradient_binned",
             [](W& w, AnyArr wt, const std::string& fl) {
               const auto          f = parse_flavor(fl);
               std::vector<double> g;
               if (wt.device_type() == nb::device::cuda::value) {
                 const int dev = single_device(w.prop, "weighted_gradient_binned");
                 if (wt.device_id() != dev)
                   throw std::invalid_argument("weighted_gradient_binned: weights on "
                                               "another GPU");
                 if (wt.size() !=
                     2 * size_t(N) * N * w.prop.n_cosine_bins() * w.prop.n_energy_bins())
                   throw std::invalid_argument("weighted_gradient_binned: weights must "
                                               "have the shape of binned()");
                 const double*          d = wt.data();
                 nb::gil_scoped_release r;
                 w.sync();
                 g = w.prop.weighted_gradient_binned_device({d}, f);
               }
               else {
                 if (wt.device_type() != nb::device::cpu::value)
                   throw std::invalid_argument("weighted_gradient_binned: weights must "
                                               "be a host or CUDA array");
                 std::vector<double>    v(wt.data(), wt.data() + wt.size());
                 nb::gil_scoped_release r;
                 w.sync();
                 g = w.prop.weighted_gradient_binned(v, f);
               }
               return make_array(std::move(g), {g.size()});
             },
             "weights"_a, "flavor"_a = "both",
             "sum of weights * d(bin average)/dp; weights have the shape of "
             "binned() (host or CUDA array, see weighted_gradient). Returns one "
             "value per gradient parameter.")
        .def("avg_path_grad",
             [](W& w, Arr1 Ee, int nE, Arr2 seg, bool nubar, const std::string& meas) {
               auto       ev = to_vec(Ee);
               auto       pv = to_path(seg);
               const auto m  = parse_measure(meas);
               std::vector<double> A, dA;
               {
                 nb::gil_scoped_release r;
                 w.sync();
                 w.prop.avg_path_grad(ev, nE, pv, nubar, A, dA, m);
               }
               size_t np = w.prop.n_gradient_params(), nb_ = Ee.shape(0) - 1;
               return nb::make_tuple(make_array(std::move(A), {N, N, nb_}),
                                     make_array(std::move(dA), {np, N, N, nb_}));
             },
             "energy_edges"_a, "n_gl"_a, "path"_a, "nubar"_a = false,
             "measure"_a = "linear",
             "1D bin averages for a fixed path with gradients: (A[a, b, iEbin], "
             "dA[p, a, b, iEbin]).")
        .def("avg_path",
             [](W& w, Arr1 Ee, int nE, Arr2 seg, bool nubar, const std::string& meas) {
               auto       ev = to_vec(Ee);
               auto       pv = to_path(seg);
               const auto m  = parse_measure(meas);
               std::vector<double> out;
               {
                 nb::gil_scoped_release r;
                 w.sync();
                 out = w.prop.avg_path(ev, nE, pv, nubar, m);
               }
               return make_array(std::move(out), {N, N, Ee.shape(0) - 1});
             },
             "energy_edges"_a, "n_gl"_a, "path"_a, "nubar"_a = false,
             "measure"_a = "linear", "1D bin averages for a fixed path.")
        // batched bin averages
        .def("binned_batch",
             [](W& w, const std::vector<std::string>& names, Arr2 values,
                std::optional<std::vector<PyEarth>> earths, bool gradient) {
               auto                pts = w.points(names, values);
               const auto          be  = to_batch_earths(earths);
               std::vector<double> out, dout;
               {
                 nb::gil_scoped_release r;
                 w.sync();
                 if (be && be->is_binned)
                   w.prop.binned_batch(pts, &be->binned, out, gradient ? &dout : nullptr);
                 else
                   w.prop.binned_batch(pts, be ? &be->prem : nullptr, out,
                                       gradient ? &dout : nullptr);
               }
               const size_t np = pts.size(), nc = w.prop.n_cosine_bins(),
                            ne = w.prop.n_energy_bins(), nq = w.prop.n_gradient_params();
               auto A = make_array(std::move(out), {np, 2, N, N, nc, ne});
               if (!gradient) return nb::cast(A);
               return nb::object(nb::make_tuple(
                   A, make_array(std::move(dout), {np, 2, nq, N, N, nc, ne})));
             },
             "names"_a, "values"_a, "earths"_a = nb::none(), "gradient"_a = false,
             "Bin averages (set_bins) at many parameter points, on any backend, "
             "one point at a time: point p is this model's parameters with "
             "names[k] (model parameters, see parameter_names) set to "
             "values[p, k], and earths[p] as Earth if given (same layers, or bins, "
             "detector and azimuth, as the propagator's Earth; only densities and Z/A "
             "may differ). Returns "
             "A[p, nubar, a, b, iCbin, iEbin], with gradient=True also "
             "dA[p, nubar, q, a, b, iCbin, iEbin] for the gradient_params. Each "
             "point equals setting it and calling calculate_binned(_gradient) "
             "bit for bit; the parameters and Earth are restored.")
        .def("calculate_binned_batch",
             [](W& w, const std::vector<std::string>& names, Arr2 values, DevOut out,
                std::optional<DevOut> dout, std::optional<std::vector<PyEarth>> earths,
                uintptr_t stream, const std::string& fl) {
               const int    dev = single_device(w.prop, "calculate_binned_batch");
               auto         pts = w.points(names, values);
               const size_t per = 2 * size_t(N) * N * w.prop.n_cosine_bins() *
                                  w.prop.n_energy_bins();
               check_dev(out, dev, pts.size() * per, "calculate_binned_batch: out");
               if (dout)
                 check_dev(*dout, dev, pts.size() * per * w.prop.n_gradient_params(),
                           "calculate_binned_batch: dout");
               const auto f  = parse_flavor(fl);
               double*    o  = out.data();
               double*    d  = dout ? dout->data() : nullptr;
               const auto be = to_batch_earths(earths);
               nb::gil_scoped_release r;
               w.sync();
               if (be && be->is_binned)
                 w.prop.calculate_binned_batch(pts, &be->binned, o, d, to_stream(stream), f);
               else
                 w.prop.calculate_binned_batch(pts, be ? &be->prem : nullptr, o, d,
                                               to_stream(stream), f);
             },
             "names"_a, "values"_a, "out"_a, "dout"_a = nb::none(), "earths"_a = nb::none(),
             "stream"_a = 0, "flavor"_a = "both",
             "As binned_batch, many points per launch on the propagator's GPU, "
             "into CUDA arrays (DLPack, e.g. cuda_empty, CuPy, PyTorch): out with "
             "n_points x binned() entries, dout (optional) with n_points x "
             "binned_grad() entries, layouts as binned_batch. Asynchronous on "
             "`stream` (a cudaStream_t as an integer; 0: the propagator's): "
             "call wait() (default stream) or synchronise it before reading. "
             "Bit-identical to calculate_binned per point. One GPU only.");
    return c;
  }

  /// Analytic bin averages and their batch handles (models with
  /// Model::analytic_avg: Fast, NSI, NUNM, Sterile).
  template <class Model>
  void bind_analytic(nb::module_& m, nb::class_<PyModel<Model>>& c, const char* name)
  {
    using W         = PyModel<Model>;
    using Batch     = typename opg::Propagator<Model>::AnalyticBatch;
    constexpr int N = Model::N;
    nb::class_<Batch>(m, (std::string(name) + "AnalyticBatch").c_str(),
                      "Batch handle of analytic averages: fixed path, binning, nu/nubar "
                      "and device (see analytic_batch).")
        .def_prop_ro("n_bins", &Batch::n_bins)
        .def_prop_ro("on_device", &Batch::on_device)
        .def_prop_ro("nubar", [](const Batch& b) { return b.nubar; })
        .def_prop_ro("device", [](const Batch& b) { return b.device; },
                     "Index into the propagator's devices (-1: host).");

    c.def("avg_path_analytic",
          [](W& w, Arr1 edges, Arr2 seg, bool nubar, const std::string& meas,
             const std::string& var, std::optional<opg::AnalyticAvgOptions> o) {
            auto       ev = to_vec(edges);
            auto       pv = to_path(seg);
            const auto mm = parse_measure(meas);
            const auto vv = parse_binvar(var);
            const auto op = opts(o);
            std::vector<double> out;
            {
              nb::gil_scoped_release r;
              w.sync();
              out = w.prop.avg_path_analytic(ev, pv, nubar, mm, vv, op);
            }
            return make_array(std::move(out), {N, N, ev.size() - 1});
          },
          "edges"_a, "path"_a, "nubar"_a = false, "measure"_a = "linear", "var"_a = "E",
          "options"_a = nb::none(),
          "Analytic bin averages for a fixed path (after OscProb's PMNS_Maltoni): "
          "A[a, b, bin]. edges ascending in E (var='E', GeV) or L/E (var='LoE', "
          "km/GeV); measure: averaging weight uniform in E ('linear'), log E ('log') "
          "or 1/E i.e. L/E ('invE'); options: AnalyticAvgOptions.")
        .def("avg_path_analytic_grad",
             [](W& w, Arr1 edges, Arr2 seg, bool nubar, const std::string& meas,
                const std::string& var, std::optional<opg::AnalyticAvgOptions> o) {
               auto       ev = to_vec(edges);
               auto       pv = to_path(seg);
               const auto mm = parse_measure(meas);
               const auto vv = parse_binvar(var);
               const auto op = opts(o);
               std::vector<double> A, dA;
               {
                 nb::gil_scoped_release r;
                 w.sync();
                 w.prop.avg_path_analytic_grad(ev, pv, nubar, A, dA, mm, vv, op);
               }
               const size_t nb_ = ev.size() - 1, nq = w.prop.n_gradient_params();
               return nb::make_tuple(make_array(std::move(A), {N, N, nb_}),
                                     make_array(std::move(dA), {nq, N, N, nb_}));
             },
             "edges"_a, "path"_a, "nubar"_a = false, "measure"_a = "linear", "var"_a = "E",
             "options"_a = nb::none(),
             "avg_path_analytic with gradients: (A[a, b, bin], dA[q, a, b, bin]) for "
             "the gradient_params (model and Earth zoa_/rho_).")
        .def_static("analytic_subbins",
                    [](Arr1 edges, Arr2 seg, const std::string& meas, const std::string& var,
                       std::optional<opg::AnalyticAvgOptions> o) {
                      auto sub = opg::Propagator<Model>::analytic_subbins(
                          to_vec(edges), to_path(seg), parse_measure(meas),
                          parse_binvar(var), opts(o));
                      std::vector<double> v;
                      for (const auto& s : sub)
                        v.insert(v.end(), {double(s.bin), s.u0, s.h, s.weight, s.beta});
                      return make_array(std::move(v), {sub.size(), 5});
                    },
                    "edges"_a, "path"_a, "measure"_a = "linear", "var"_a = "E",
                    "options"_a = nb::none(),
                    "The sub-bins avg_path_analytic uses, as rows (bin, u0, h, weight, "
                    "beta): uniform in u = 1/E on [u0 - h, u0 + h] with weight "
                    "weight * (1 + beta (u - u0) / h). Fix them at a nominal path "
                    "when fitting densities.")
        .def("avg_path_analytic_subbins",
             [](W& w, Arr2 sub, size_t nbins, Arr2 seg, bool nubar,
                std::optional<opg::AnalyticAvgOptions> o) {
               auto       sv = to_subbins(sub);
               auto       pv = to_path(seg);
               const auto op = opts(o);
               std::vector<double> out;
               {
                 nb::gil_scoped_release r;
                 w.sync();
                 out = w.prop.avg_path_analytic_subbins(sv, nbins, pv, nubar, op);
               }
               return make_array(std::move(out), {N, N, nbins});
             },
             "subbins"_a, "nbins"_a, "path"_a, "nubar"_a = false, "options"_a = nb::none(),
             "Analytic averages over explicit sub-bins (analytic_subbins rows): "
             "A[a, b, bin].")
        .def("avg_path_analytic_subbins_grad",
             [](W& w, Arr2 sub, size_t nbins, Arr2 seg, bool nubar,
                std::optional<opg::AnalyticAvgOptions> o) {
               auto       sv = to_subbins(sub);
               auto       pv = to_path(seg);
               const auto op = opts(o);
               std::vector<double> A, dA;
               {
                 nb::gil_scoped_release r;
                 w.sync();
                 w.prop.avg_path_analytic_subbins_grad(sv, nbins, pv, nubar, A, dA, op);
               }
               const size_t nq = w.prop.n_gradient_params();
               return nb::make_tuple(make_array(std::move(A), {N, N, nbins}),
                                     make_array(std::move(dA), {nq, N, N, nbins}));
             },
             "subbins"_a, "nbins"_a, "path"_a, "nubar"_a = false, "options"_a = nb::none(),
             "avg_path_analytic_subbins with gradients: (A[a, b, bin], dA[q, a, b, bin]).")
        .def("analytic_batch",
             [](W& w, Arr1 edges, Arr2 seg, bool nubar, const std::string& meas,
                const std::string& var, std::optional<opg::AnalyticAvgOptions> o, int device) {
               auto       ev = to_vec(edges);
               auto       pv = to_path(seg);
               const auto mm = parse_measure(meas);
               const auto vv = parse_binvar(var);
               const auto op = opts(o);
               nb::gil_scoped_release r;
               return w.prop.analytic_batch(ev, pv, nubar, mm, vv, op, device);
             },
             "edges"_a, "path"_a, "nubar"_a = false, "measure"_a = "linear", "var"_a = "E",
             "options"_a = nb::none(), "device"_a = 0, nb::keep_alive<0, 1>(),
             "Batch handle (path, binning, nu/nubar fixed) for "
             "avg_path_analytic_batch*. With the CUDA backend (and not "
             "options.host) its buffers live on devices[device]. Keeps the "
             "propagator alive; each call needs exclusive use of its handle.")
        .def("analytic_batch_subbins",
             [](W& w, Arr2 sub, size_t nbins, Arr2 seg, bool nubar,
                std::optional<opg::AnalyticAvgOptions> o, int device) {
               auto       sv = to_subbins(sub);
               auto       pv = to_path(seg);
               const auto op = opts(o);
               nb::gil_scoped_release r;
               return w.prop.analytic_batch(sv, nbins, pv, nubar, op, device);
             },
             "subbins"_a, "nbins"_a, "path"_a, "nubar"_a = false, "options"_a = nb::none(),
             "device"_a = 0, nb::keep_alive<0, 1>(),
             "Batch handle for explicit sub-bins (analytic_subbins rows).")
        .def("avg_path_analytic_batch",
             [](W& w, const Batch& b, const std::vector<std::string>& names, Arr2 values,
                bool gradient) {
               auto                pts = w.points(names, values);
               std::vector<double> out, dout;
               {
                 nb::gil_scoped_release r;
                 w.prop.avg_path_analytic_batch(b, pts, out, gradient ? &dout : nullptr);
               }
               const size_t np = pts.size(), nq = w.prop.n_gradient_params();
               auto A = make_array(std::move(out), {np, N, N, b.nbins});
               if (!gradient) return nb::cast(A);
               return nb::object(
                   nb::make_tuple(A, make_array(std::move(dout), {np, nq, N, N, b.nbins})));
             },
             "batch"_a, "names"_a, "values"_a, "gradient"_a = false,
             "Batched analytic averages into host arrays (any backend; blocking): "
             "point p is this model's parameters with names[k] set to values[p, k]. "
             "Returns A[p, a, b, bin], with gradient=True also dA[p, q, a, b, bin] "
             "for the gradient_params. Each point's result does not depend on the "
             "batch (size, order, chunking).")
        .def("avg_path_analytic_batch_device",
             [](W& w, const Batch& b, const std::vector<std::string>& names, Arr2 values,
                DevOut out, std::optional<DevOut> dout, uintptr_t stream) {
               if (!b.on_device())
                 throw std::logic_error("avg_path_analytic_batch_device: the handle is not "
                                        "on a GPU");
               const int    dev = w.prop.devices()[size_t(b.device)];
               auto         pts = w.points(names, values);
               const size_t per = size_t(N) * N * b.nbins;
               check_dev(out, dev, pts.size() * per, "avg_path_analytic_batch_device: out");
               if (dout)
                 check_dev(*dout, dev, pts.size() * per * w.prop.n_gradient_params(),
                           "avg_path_analytic_batch_device: dout");
               double* o = out.data();
               double* d = dout ? dout->data() : nullptr;
               nb::gil_scoped_release r;
               w.prop.avg_path_analytic_batch(b, pts, o, d, to_stream(stream));
             },
             "batch"_a, "names"_a, "values"_a, "out"_a, "dout"_a = nb::none(), "stream"_a = 0,
             "avg_path_analytic_batch into CUDA arrays on the handle's GPU (layouts "
             "as the host version), asynchronous on `stream` (0: the propagator's; "
             "call wait() or synchronise it before reading).")
        .def("avg_path_analytic_batch_weighted",
             [](W& w, const Batch& b, const std::vector<std::string>& names, Arr2 values,
                AnyArr wt, std::optional<DevOut> g, uintptr_t stream) {
               auto         pts = w.points(names, values);
               const size_t nq = w.prop.n_gradient_params();
               const size_t nw = pts.size() * size_t(N) * N * b.nbins;
               if (wt.device_type() == nb::device::cuda::value) {
                 if (!b.on_device())
                   throw std::logic_error("avg_path_analytic_batch_weighted: CUDA weights "
                                          "need a handle on a GPU");
                 if (!g)
                   throw std::invalid_argument("avg_path_analytic_batch_weighted: CUDA "
                                               "weights need a CUDA output g");
                 const int dev = w.prop.devices()[size_t(b.device)];
                 check_dev(wt, dev, nw, "avg_path_analytic_batch_weighted: weights");
                 check_dev(*g, dev, pts.size() * nq, "avg_path_analytic_batch_weighted: g");
                 const double* wd = wt.data();
                 double*       gd = g->data();
                 {
                   nb::gil_scoped_release r;
                   w.prop.avg_path_analytic_batch_weighted(b, pts, wd, gd, to_stream(stream));
                 }
                 return nb::object(nb::none());
               }
               if (wt.device_type() != nb::device::cpu::value)
                 throw std::invalid_argument("avg_path_analytic_batch_weighted: weights "
                                             "must be a host or CUDA array");
               if (g)
                 throw std::invalid_argument("avg_path_analytic_batch_weighted: g is only "
                                             "for CUDA weights");
               if (wt.size() != nw)
                 throw std::invalid_argument("avg_path_analytic_batch_weighted: weights "
                                             "must have the shape of the batch output");
               std::vector<double> v(wt.data(), wt.data() + wt.size()), gv;
               {
                 nb::gil_scoped_release r;
                 w.prop.avg_path_analytic_batch_weighted(b, pts, v, gv);
               }
               return nb::cast(make_array(std::move(gv), {pts.size(), nq}));
             },
             "batch"_a, "names"_a, "values"_a, "weights"_a, "g"_a = nb::none(),
             "stream"_a = 0,
             "Weighted gradients of batched averages: g[p, q] = sum over (a, b, bin) "
             "of weights[p, a, b, bin] * dA_ab(bin)/dq at point p, q over the "
             "gradient_params. Host weights: returns g (blocking). CUDA weights on "
             "the handle's GPU: writes the CUDA array g (n_points x n_q), "
             "asynchronous on `stream` as avg_path_analytic_batch_device.");
  }

  template <class Model>
  void bind_ctor(nb::class_<PyModel<Model>>& c)
  {
    c.def("__init__",
          [](PyModel<Model>* self, const PyEarth& e, const std::vector<int>& d, int t) {
            new (self) PyModel<Model>(to_earth(e), d, t);
          },
          "earth"_a = PyEarth(opg::PremModel()), "devices"_a = std::vector<int>{},
          "threads"_a = 0,
          "earth: PremModel or EarthModelBinned; devices: CUDA device ids (empty "
          "list = CPU backend); threads: OpenMP threads for the CPU backend (0 = "
          "default).");
  }

} // namespace

NB_MODULE(_oscprobgpu, m)
{
  m.attr("__version__") = OPG_VERSION;
  m.doc() = "OscProbGPU: CUDA port of OscProb oscillation calculators";
  m.def("cuda_device_count", &opg::cuda_device_count);
  m.def("_dlpack", &dlpack_capsule, "ptr"_a, "shape"_a, "device"_a, "owner"_a,
        "writable"_a = false);
  // Owned device buffers for cuda_empty() (no CuPy/PyTorch needed).
  m.def("_cuda_alloc",
        [](size_t n, int device) {
#ifdef OPG_HAVE_CUDA
          int prev = 0;
          cudaGetDevice(&prev);
          if (cudaSetDevice(device) != cudaSuccess)
            throw std::invalid_argument("cuda_empty: bad device");
          void*       p = nullptr;
          cudaError_t e = cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(double));
          if (e == cudaSuccess) e = cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(double));
          cudaSetDevice(prev);
          if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
          struct Buf {
              void* p;
              int   dev;
          };
          auto*       b = new Buf{p, device};
          nb::capsule owner(b, [](void* q) noexcept {
            auto* bb  = static_cast<Buf*>(q);
            int   cur = 0;
            cudaGetDevice(&cur);
            cudaSetDevice(bb->dev);
            cudaFree(bb->p);
            cudaSetDevice(cur);
            delete bb;
          });
          return nb::make_tuple(uintptr_t(p), owner);
#else
          (void)n;
          (void)device;
          throw std::runtime_error("cuda_empty: built without CUDA");
          return nb::make_tuple();
#endif
        },
        "n"_a, "device"_a);
  m.def("_cuda_to_host",
        [](uintptr_t ptr, std::vector<size_t> shape, int device) {
          size_t n = 1;
          for (size_t s : shape) n *= s;
          std::vector<double> v(n);
#ifdef OPG_HAVE_CUDA
          cudaError_t e;
          {
            nb::gil_scoped_release r;
            int                    prev = 0;
            cudaGetDevice(&prev);
            cudaSetDevice(device);
            e = cudaDeviceSynchronize();
            if (e == cudaSuccess)
              e = cudaMemcpy(v.data(), reinterpret_cast<const void*>(ptr),
                             n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaSetDevice(prev);
          }
          if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
#else
          (void)ptr;
          (void)device;
          throw std::runtime_error("DeviceArray.get: built without CUDA");
#endif
          return make_array(std::move(v), shape);
        },
        "ptr"_a, "shape"_a, "device"_a);
  m.def("path_transmission",
        [](Arr2 seg, double xsec) { return opg::transmission(to_path(seg), xsec); },
        "path"_a, "xsec"_a,
        "Probability of no absorption along a fixed path (rows: length km, "
        "density, ...); xsec in cm^2 per nucleon.");
  m.def("path_column_depth", [](Arr2 seg) { return opg::column_depth(to_path(seg)); },
        "path"_a, "Column depth sum rho L (g/cm^2) of a fixed path.");
#ifdef OPG_HAVE_CUDA
  m.attr("has_cuda") = true;
#else
  m.attr("has_cuda") = false;
#endif

  nb::class_<opg::PremModel>(m, "PremModel")
      .def(nb::init<const std::string&>(), "filename"_a = "",
           "Spherical-shell Earth model (default: PREM 44 layers).")
      .def("set_det_pos", &opg::PremModel::SetDetPos, "radius_km"_a)
      .def("column_depth",
           [](const opg::PremModel& e, Arr1 c) {
             std::vector<double> x(c.shape(0));
             for (size_t i = 0; i < x.size(); i++)
               x[i] = opg::column_depth(e.FillPath(c.data()[i]));
             return make_array(std::move(x), {x.size()});
           },
           "cosT"_a, "Column depth sum rho L (g/cm^2) for each cos(zenith).")
      .def("transmission",
           [](const opg::PremModel& e, Arr1 c, nb::handle xsec) {
             const size_t        n = c.shape(0);
             std::vector<double> xs;
             if (nb::isinstance<nb::float_>(xsec) || nb::isinstance<nb::int_>(xsec))
               xs.assign(n, nb::cast<double>(xsec));
             else {
               auto a = nb::cast<Arr1>(xsec);
               if (a.shape(0) != n) throw std::invalid_argument("transmission: size mismatch");
               xs.assign(a.data(), a.data() + n);
             }
             std::vector<double> t(n);
             for (size_t i = 0; i < n; i++)
               t[i] = opg::transmission(e.FillPath(c.data()[i]), xs[i]);
             return make_array(std::move(t), {n});
           },
           "cosT"_a, "xsec"_a,
           "Probability of no absorption (OscProb Absorption::Trans) for each "
           "cos(zenith); xsec in cm^2 per nucleon, scalar or one per cosine.")
      .def("set_layer_zoa", &opg::PremModel::SetLayerZoA, "layer"_a, "zoa"_a)
      .def("get_layer_zoa", &opg::PremModel::GetLayerZoA, "layer"_a)
      .def("set_top_layer_size", &opg::PremModel::SetTopLayerSize, "thickness_km"_a)
      .def("get_total_L", &opg::PremModel::GetTotalL, "cosT"_a)
      .def("get_cosT", &opg::PremModel::GetCosT, "L"_a)
      .def_prop_ro("det_radius", &opg::PremModel::GetDetRadius)
      .def("grazing_cosines", &opg::PremModel::GetGrazingCosines)
      .def("fill_path",
           [](const opg::PremModel& e, double c) {
             auto                p = e.FillPath(c);
             std::vector<double> v;
             for (auto& s : p) v.insert(v.end(), {s.length, s.density, s.zoa, double(s.layer)});
             return make_array(std::move(v), {p.size(), 4});
           },
           "cosT"_a, "Path segments (length, density, zoa, layer).");

  nb::class_<opg::EarthModelBinned>(m, "EarthModelBinned",
                                    "3D Earth of depth x latitude x longitude bins with "
                                    "region indices (OscProb EarthModelBinned).")
      .def(nb::init<const std::string&>(), "filename"_a = "",
           "Bin table (default: data/earth_binned_default.txt); detector at radius "
           "6368 km, latitude 0, longitude 0.")
      .def("set_det_pos", &opg::EarthModelBinned::SetDetPos, "radius_km"_a, "lat_deg"_a = 0.0,
           "lon_deg"_a = 0.0, "Detector radius (km), latitude (deg N), longitude (deg E).")
      .def("set_azimuth", &opg::EarthModelBinned::SetAzimuth, "phi"_a,
           "Azimuth (degrees from North) of paths given by cos(zenith) alone (grids, "
           "bin averages, event lists without per-event azimuths).")
      .def_prop_ro("azimuth", &opg::EarthModelBinned::GetAzimuth)
      .def("fill_path",
           [](const opg::EarthModelBinned& e, double c, std::optional<double> phi) {
             auto p = phi ? e.FillPath(c, *phi) : e.FillPath(c);
             std::vector<double> v;
             for (auto& s : p) v.insert(v.end(), {s.length, s.density, s.zoa, double(s.layer)});
             return make_array(std::move(v), {p.size(), 4});
           },
           "cosT"_a, "phi"_a = nb::none(),
           "Path segments (length, density, zoa, region) for cos(zenith) and azimuth "
           "phi in degrees (None: the model's azimuth).")
      .def("column_depth",
           [](const opg::EarthModelBinned& e, Arr1 c) {
             std::vector<double> x(c.shape(0));
             for (size_t i = 0; i < x.size(); i++)
               x[i] = opg::column_depth(e.FillPath(c.data()[i]));
             return make_array(std::move(x), {x.size()});
           },
           "cosT"_a, "Column depth (g/cm^2) for each cos(zenith), at the model's azimuth.")
      .def("set_region_zoa", &opg::EarthModelBinned::SetRegionZoA, "index"_a, "zoa"_a)
      .def("get_region_zoa", &opg::EarthModelBinned::GetRegionZoA, "index"_a)
      .def("scale_region_density", &opg::EarthModelBinned::ScaleRegionDensity, "index"_a,
           "factor"_a)
      .def_prop_ro("regions", &opg::EarthModelBinned::GetRegions)
      .def("get_total_L", &opg::EarthModelBinned::GetTotalL, "cosT"_a)
      .def("get_cosT", &opg::EarthModelBinned::GetCosT, "L"_a)
      .def_prop_ro("det_radius", &opg::EarthModelBinned::GetDetRadius)
      .def_prop_ro("det_lat", &opg::EarthModelBinned::GetDetLat, "Detector latitude (rad).")
      .def_prop_ro("det_lon", &opg::EarthModelBinned::GetDetLon, "Detector longitude (rad).")
      .def_prop_ro("n_depth_bins", &opg::EarthModelBinned::GetNDepthBins)
      .def_prop_ro("n_lat_bins", &opg::EarthModelBinned::GetNLatBins)
      .def_prop_ro("n_lon_bins", &opg::EarthModelBinned::GetNLonBins)
      .def("grazing_cosines", &opg::EarthModelBinned::GetGrazingCosines)
      .def("earth_bins",
           [](const opg::EarthModelBinned& e) {
             const auto&         b = e.GetEarthBins();
             std::vector<double> v;
             for (auto& x : b)
               v.insert(v.end(), {x.radius_out, x.radius_in, x.latitude, x.longitude,
                                  x.density, x.zoa, double(x.index)});
             return make_array(std::move(v), {b.size(), 7});
           },
           "Bins as rows (radius_out, radius_in, lat [rad], lon [rad], density, zoa, "
           "region).");

  nb::class_<opg::AnalyticAvgOptions>(m, "AnalyticAvgOptions",
                                      "Options of the analytic bin averages (see README).")
      .def(nb::init<>())
      .def_rw("tol", &opg::AnalyticAvgOptions::tol,
              "Target first-order error (sets the sub-bin width; default 1e-6).")
      .def_rw("max_width", &opg::AnalyticAvgOptions::max_width,
              "Largest relative sub-bin width u_hi/u_lo - 1 (0.1).")
      .def_rw("nsub", &opg::AnalyticAvgOptions::nsub,
              "> 0: this many sub-bins per bin, uniform in 1/E (tol unused).")
      .def_rw("fast_begin", &opg::AnalyticAvgOptions::fast_begin)
      .def_rw("fast_end", &opg::AnalyticAvgOptions::fast_end)
      .def_rw("threads", &opg::AnalyticAvgOptions::threads, "OpenMP threads (0: default).")
      .def_rw("host", &opg::AnalyticAvgOptions::host,
              "Compute on the host even with a GPU backend.")
      .def_rw("rows", &opg::AnalyticAvgOptions::rows,
              "Initial flavours as a bit mask (e.g. 3 = e and mu; 0: all).")
      .def_rw("batch_scratch_mb", &opg::AnalyticAvgOptions::batch_scratch_mb)
      .def_rw("cells", &opg::AnalyticAvgOptions::cells,
              "One-segment paths: share expansions over cells (faster for narrow bins).")
      .def_rw("cell_max_width", &opg::AnalyticAvgOptions::cell_max_width)
      .def_rw("cell_pieces", &opg::AnalyticAvgOptions::cell_pieces);

  using Fast    = opg::Fast<double>;
  using NSI     = opg::NSI<double>;
  using NUNM    = opg::NUNM<double>;
  using Sterile = opg::Sterile<double>;
  using Decay   = opg::Decay<double>;

  auto cf = bind_model<Fast>(m, "Fast", "Standard 3-flavour oscillations (PMNS_Fast).");
  bind_ctor<Fast>(cf);

  auto cn = bind_model<NSI>(m, "NSI", "3 flavours with vector NSI (PMNS_NSI).");
  bind_ctor<NSI>(cn);
  cn.def("set_eps",
         [](PyModel<NSI>& w, int i, int j, double v, double ph) {
           w.dirty = true;
           w.params.SetEps(i, j, v, ph);
         },
         "flvi"_a, "flvj"_a, "value"_a, "phase"_a = 0.0,
         "Set eps_ij (0-based flavours, i <= j).")
      .def("set_ferm_coup",
           [](PyModel<NSI>& w, double e, double u, double d) {
             w.dirty = true;
             w.params.SetFermCoup(e, u, d);
           },
           "e"_a, "u"_a, "d"_a);

  auto cu = bind_model<NUNM>(m, "NUNM", "3 flavours with non-unitary mixing (PMNS_NUNM).");
  cu.def("__init__",
         [](PyModel<NUNM>* self, const PyEarth& e, const std::vector<int>& d,
            int t, int scale) {
           new (self) PyModel<NUNM>(to_earth(e), d, t);
           self->params.scale = scale;
         },
         "earth"_a = PyEarth(opg::PremModel()), "devices"_a = std::vector<int>{},
         "threads"_a = 0, "scale"_a = 0,
         "scale: 0 = low-scale, 1 = high-scale scenario.")
      .def("set_alpha",
           [](PyModel<NUNM>& w, int i, int j, double v, double ph) {
             w.dirty = true;
             w.params.SetAlpha(i, j, v, ph);
           },
           "i"_a, "j"_a, "value"_a, "phase"_a = 0.0,
           "Set alpha_ij (0-based, i >= j); diagonal entries are 1 + value.")
      .def("set_frac_vnc",
           [](PyModel<NUNM>& w, double f) {
             w.dirty = true;
             w.params.SetFracVnc(f);
           },
           "f"_a);

  auto cs = bind_model<Sterile>(m, "Sterile", "3+1 oscillations (PMNS_Sterile with 4 flavours).");
  bind_ctor<Sterile>(cs);

  bind_analytic<Fast>(m, cf, "Fast");
  bind_analytic<NSI>(m, cn, "NSI");
  bind_analytic<NUNM>(m, cu, "NUNM");
  bind_analytic<Sterile>(m, cs, "Sterile");

  using LIV  = opg::LIV<double>;
  using SNSI = opg::SNSI<double>;

  auto cl = bind_model<LIV>(m, "LIV", "3 flavours with SME Lorentz invariance violation (PMNS_LIV).");
  bind_ctor<LIV>(cl);
  cl.def("set_aT",
         [](PyModel<LIV>& w, int i, int j, int dim, double v, double ph) {
           w.dirty = true;
           w.params.SetaT(i, j, dim, v, ph);
         },
         "flvi"_a, "flvj"_a, "dim"_a, "value"_a, "phase"_a = 0.0,
         "Set aT_ij of dimension 3, 5 or 7 (GeV^(4-dim); 0-based flavours).")
      .def("set_cT",
           [](PyModel<LIV>& w, int i, int j, int dim, double v, double ph) {
             w.dirty = true;
             w.params.SetcT(i, j, dim, v, ph);
           },
           "flvi"_a, "flvj"_a, "dim"_a, "value"_a, "phase"_a = 0.0,
           "Set cT_ij of dimension 4, 6 or 8 (GeV^(4-dim); 0-based flavours).");

  auto cs2 = bind_model<SNSI>(m, "SNSI", "3 flavours with scalar NSI (PMNS_SNSI).");
  bind_ctor<SNSI>(cs2);
  cs2.def("set_eps",
          [](PyModel<SNSI>& w, int i, int j, double v, double ph) {
            w.dirty = true;
            w.params.SetEps(i, j, v, ph);
          },
          "flvi"_a, "flvj"_a, "value"_a, "phase"_a = 0.0,
          "Set eps_ij in MeV^-2 (0-based flavours, i <= j).")
      .def("set_ferm_coup",
           [](PyModel<SNSI>& w, double e, double u, double d) {
             w.dirty = true;
             w.params.SetFermCoup(e, u, d);
           },
           "e"_a, "u"_a, "d"_a)
      .def("set_lowest_mass",
           [](PyModel<SNSI>& w, double mass) {
             w.dirty = true;
             w.params.SetLowestMass(mass);
           },
           "m"_a, "Lightest neutrino mass in eV.");

  using SLIV = opg::SiderealLIV<double>;
  auto csl  = bind_model<SLIV>(m, "SiderealLIV",
                              "3 flavours with sidereal SME LIV (PMNS_SiderealLIV).");
  bind_ctor<SLIV>(csl);
  csl.def("set_a",
          [](PyModel<SLIV>& w, int i, int j, int coord, double v) {
            w.dirty = true;
            w.params.SetA(i, j, coord, v);
          },
          "flvi"_a, "flvj"_a, "coord"_a, "value"_a, "aT_ij^coord in GeV (coord 0, 1, 2 = X, Y, Z).")
      .def("set_c",
           [](PyModel<SLIV>& w, int i, int j, int c1, int c2, double v) {
             w.dirty = true;
             w.params.SetC(i, j, c1, c2, v);
           },
           "flvi"_a, "flvj"_a, "coord1"_a, "coord2"_a, "value"_a)
      .def("set_colatitude",
           [](PyModel<SLIV>& w, double chi) {
             w.dirty = true;
             w.params.SetColatitude(chi);
           },
           "chi"_a, "Detector colatitude in degrees (90 - latitude).")
      .def("set_latitude",
           [](PyModel<SLIV>& w, double deg, double min, double sec) {
             w.dirty = true;
             w.params.SetColatitude(deg, min, sec);
           },
           "deg"_a, "min"_a = 0.0, "sec"_a = 0.0,
           "As SetColatitude(deg, min, sec): signed latitude (negative south).")
      .def("set_neutrino_direction",
           [](PyModel<SLIV>& w, double zen, double azi) {
             w.dirty = true;
             w.params.SetNeutrinoDirection(zen, azi);
           },
           "zenith"_a, "azimuth"_a,
           "Fixed direction (degrees) for all paths (set the colatitude first).")
      .def("set_azimuth",
           [](PyModel<SLIV>& w, double azi) {
             w.dirty = true;
             w.params.SetAzimuth(azi);
           },
           "azimuth"_a, "Azimuth (degrees); Earth paths use their own zenith.")
      .def("set_time_hours",
           [](PyModel<SLIV>& w, double h) {
             w.dirty = true;
             w.params.SetTimeHours(h);
           },
           "hours"_a, "Local sidereal time.");

  using Deco = opg::Deco<double>;
  auto ce = bind_model<Deco>(m, "Deco", "3 flavours with decoherence (PMNS_Deco).");
  bind_ctor<Deco>(ce);
  ce.def("set_gamma",
         [](PyModel<Deco>& w, int j, double v) {
           w.dirty = true;
           w.params.SetGamma(j, v);
         },
         "j"_a, "value"_a, "Set Gamma_j1 in GeV, j = 2 or 3.")
      .def("set_gamma32",
           [](PyModel<Deco>& w, double v) {
             w.dirty = true;
             w.params.SetGamma32(v);
           },
           "value"_a, "Set Gamma_32 (adjusts Gamma_31; see PMNS_Deco::SetGamma32).")
      .def("set_deco_angle",
           [](PyModel<Deco>& w, double th) {
             w.dirty = true;
             w.params.SetDecoAngle(th);
           },
           "theta"_a)
      .def("set_power",
           [](PyModel<Deco>& w, double n) {
             w.dirty = true;
             w.params.SetPower(n);
           },
           "n"_a, "Gamma_ij ~ (E/GeV)^n.")
      .def("get_gamma",
           [](PyModel<Deco>& w, int i, int j) {
             if (i < j) std::swap(i, j);
             if (j < 1 || i > 3 || i == j) throw std::out_of_range("invalid Gamma indices");
             return w.params.GetGamma(i, j);
           },
           "i"_a, "j"_a);

  using OQS = opg::OQS<double>;
  auto co   = bind_model<OQS>(m, "OQS",
                            "3 flavours as an open quantum system (PMNS_OQS).");
  bind_ctor<OQS>(co);
  co.def("set_deco_element",
         [](PyModel<OQS>& w, int i, double v) {
           w.dirty = true;
           w.params.SetDecoElement(i, v);
         },
         "i"_a, "value"_a, "Set |a_i|, i = 1..8 (Gell-Mann basis; |value| is used).")
      .def("set_deco_angle",
           [](PyModel<OQS>& w, int i, int j, double th) {
             w.dirty = true;
             w.params.SetDecoAngle(i, j, th);
           },
           "i"_a, "j"_a, "theta"_a, "Angle (radians) between a_i and a_j, i != j in 1..8.")
      .def("set_power",
           [](PyModel<OQS>& w, double n) {
             w.dirty = true;
             w.params.SetPower(n);
           },
           "n"_a, "Dissipator ~ (E/GeV)^n.")
      .def("get_deco_element",
           [](PyModel<OQS>& w, int i) {
             if (i < 1 || i > 8) throw std::invalid_argument("i in [1, 8]");
             return w.params.a[i];
           },
           "i"_a)
      .def("get_deco_angle",
           [](PyModel<OQS>& w, int i, int j) {
             if (i < 1 || i > 8 || j < 1 || j > 8 || i == j)
               throw std::invalid_argument("i != j in [1, 8]");
             return w.params.ang[std::min(i, j)][std::max(i, j)];
           },
           "i"_a, "j"_a)
      .def("get_power", [](PyModel<OQS>& w) { return w.params.power; });

  auto cd = bind_model<Decay>(m, "Decay", "3 flavours with invisible decay (PMNS_Decay).");
  bind_ctor<Decay>(cd);
  cd.def("set_alpha2",
         [](PyModel<Decay>& w, double a) {
           w.dirty = true;
           w.params.SetAlpha2(a);
         },
         "alpha2"_a, "alpha_2 = m_2/tau_2 in eV^2")
      .def("set_alpha3",
           [](PyModel<Decay>& w, double a) {
             w.dirty = true;
             w.params.SetAlpha3(a);
           },
           "alpha3"_a, "alpha_3 = m_3/tau_3 in eV^2");
}
