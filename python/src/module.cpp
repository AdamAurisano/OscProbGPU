// Python bindings for OscProbGPU (nanobind).
//
// Each model class wraps an opg::Propagator<Model<double>> together with its
// parameter struct; parameters are pushed to the propagator lazily before
// the next computation, so setters are cheap.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "opg/propagator.h"

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

  /// DLPack capsule for device memory (owner kept alive by the capsule).
  inline nb::ndarray<const double, nb::device::cuda>
  dlpack_capsule(uintptr_t ptr, const std::vector<size_t>& shape, int device,
                 nb::handle owner)
  {
    return nb::ndarray<const double, nb::device::cuda>(
        reinterpret_cast<const double*>(ptr), shape.size(), shape.data(), owner,
        nullptr, nb::dtype<double>(), nb::device::cuda::value, device);
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

  template <class Model> struct PyModel {
      using Params           = typename Model::Params;
      static constexpr int N = Model::N;
      static constexpr int NX = opg::Propagator<Model>::n_event_extra();

      opg::Propagator<Model> prop;
      Params                 params;
      bool                   dirty = true;

      PyModel(const opg::PremModel& earth, const std::vector<int>& devices,
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
        const double* p = prop.probs();
        size_t        n = 2 * N * N * prop.n_cosines() * prop.n_energies();
        return make_array(std::vector<double>(p, p + n),
                          {2, N, N, prop.n_cosines(), prop.n_energies()});
      }

      nb::ndarray<nb::numpy, double> binned()
      {
        const double* p = prop.binned();
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
        .def("get_angle", [](W& w, int i, int j) { return w.params.mix.th[i - 1][j - 1]; })
        .def("get_delta", [](W& w, int i, int j) { return w.params.mix.dcp[i - 1][j - 1]; })
        .def("get_dm", [](W& w, int j) { return w.params.mix.dm[j - 1]; })
        .def("set_std_pars", [](W& w) { opg::set_std_pars(w.mix()); },
             "Reset mixing parameters to OscProb's PDG defaults.")
        // earth
        .def("set_earth", [](W& w, const opg::PremModel& e) { w.prop.set_earth(e); })
        // grid mode
        .def("set_grid",
             [](W& w, Arr1 E, Arr1 C) { w.prop.set_grid(to_vec(E), to_vec(C)); },
             "energies"_a, "cosines"_a,
             "Set the (E [GeV], cosZ) grid; uploads once.")
        .def("calculate",
             [](W& w, const std::string& fl) {
               w.sync();
               w.prop.calculate(parse_flavor(fl));
             },
             "flavor"_a = "both", "Launch the grid computation (asynchronous).")
        .def("wait", [](W& w) { w.prop.wait(); })
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
               w.prop.set_bins(to_vec(Ee), to_vec(Ce), nE, nC, parse_measure(meas),
                               parse_rule(rule));
             },
             "energy_edges"_a, "cosine_edges"_a, "n_gl_energy"_a = 8,
             "n_gl_cosine"_a = 8, "measure"_a = "linear", "cosz_rule"_a = "layer",
             "Set a 2D binning for Gauss-Legendre bin averages.")
        .def("calculate_binned",
             [](W& w, const std::string& fl) {
               w.sync();
               w.prop.calculate_binned(parse_flavor(fl));
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
             [](W& w, Arr1 E, Arr1 C, Arr1u nb_, std::optional<Arr2> ex) {
               w.sync();
               if (E.shape(0) != C.shape(0) || E.shape(0) != nb_.shape(0))
                 throw std::invalid_argument("size mismatch");
               std::vector<uint8_t> nbv(nb_.data(), nb_.data() + nb_.shape(0));
               auto out = w.prop.prob_points(to_vec(E), to_vec(C), nbv,
                                             to_extra(ex, W::NX, E.shape(0)));
               return make_array(std::move(out), {N, N, E.shape(0)});
             },
             "energies"_a, "cosines"_a, "nubar"_a, "extra"_a = nb::none(),
             "Event list: returns P[a, b, i]. extra: optional per-event inputs "
             "extra[x, i], x < n_event_extra (None: the values set on the model).")
        .def("prob_path",
             [](W& w, Arr1 E, Arr2 seg, bool nubar) {
               w.sync();
               auto out = w.prop.prob_path(to_vec(E), to_path(seg), nubar);
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
        .def("calculate_gradient",
             [](W& w, const std::string& fl) {
               w.sync();
               w.prop.calculate(parse_flavor(fl), true);
             },
             "flavor"_a = "both",
             "Grid probabilities and gradients (see probs() and grad()).")
        .def("grad",
             [](W& w) {
               const double* g  = w.prop.grad();
               size_t        np = w.prop.n_gradient_params();
               size_t        n  = 2 * np * N * N * w.prop.n_cosines() * w.prop.n_energies();
               return make_array(std::vector<double>(g, g + n),
                                 {2, np, N, N, w.prop.n_cosines(), w.prop.n_energies()});
             },
             "Grid gradients G[nubar, p, a, b, iC, iE] = dP(a -> b)/dp (copy).")
        .def("weighted_gradient",
             [](W& w, AnyArr wt, const std::string& fl) {
               w.sync();
               std::vector<double> g;
               if (wt.device_type() == nb::device::cuda::value) {
                 const int dev = single_device(w.prop, "weighted_gradient");
                 if (wt.device_id() != dev)
                   throw std::invalid_argument("weighted_gradient: weights on another GPU");
                 if (wt.size() != 2 * size_t(N) * N * w.prop.n_cosines() * w.prop.n_energies())
                   throw std::invalid_argument("weighted_gradient: weights must have "
                                               "the shape of probs()");
                 g = w.prop.weighted_gradient_device({wt.data()}, parse_flavor(fl));
               }
               else {
                 if (wt.device_type() != nb::device::cpu::value)
                   throw std::invalid_argument("weighted_gradient: weights must be a "
                                               "host or CUDA array");
                 std::vector<double> v(wt.data(), wt.data() + wt.size());
                 g = w.prop.weighted_gradient(v, parse_flavor(fl));
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
               return nb::make_tuple(uintptr_t(p), nb::tuple(nb::cast(shape)), dev);
             },
             "what"_a, "(pointer, shape, device) of device-resident results "
             "(see device_probs()/device_binned()).")
        .def("prob_points_grad",
             [](W& w, Arr1 E, Arr1 C, Arr1u nb_, std::optional<Arr2> ex) {
               w.sync();
               std::vector<uint8_t> nbv(nb_.data(), nb_.data() + nb_.shape(0));
               std::vector<double>  P, dP;
               w.prop.prob_points_grad(to_vec(E), to_vec(C), nbv, P, dP,
                                       to_extra(ex, W::NX, E.shape(0)));
               size_t np = w.prop.n_gradient_params();
               return nb::make_tuple(make_array(std::move(P), {N, N, E.shape(0)}),
                                     make_array(std::move(dP), {np, N, N, E.shape(0)}));
             },
             "energies"_a, "cosines"_a, "nubar"_a, "extra"_a = nb::none(),
             "Event list: returns (P[a, b, i], dP[p, a, b, i]); extra as in "
             "prob_points.")
        .def("weighted_gradient_points",
             [](W& w, Arr1 E, Arr1 C, Arr1u nb_,
                nb::ndarray<const double, nb::c_contig, nb::device::cpu> wt,
                std::optional<Arr2> ex) {
               w.sync();
               std::vector<uint8_t> nbv(nb_.data(), nb_.data() + nb_.shape(0));
               std::vector<double>  v(wt.data(), wt.data() + wt.size());
               auto g = w.prop.weighted_gradient_points(to_vec(E), to_vec(C), nbv, v,
                                                        to_extra(ex, W::NX, E.shape(0)));
               return make_array(std::move(g), {g.size()});
             },
             "energies"_a, "cosines"_a, "nubar"_a, "weights"_a, "extra"_a = nb::none(),
             "sum of weights[a, b, i] * dP_ab(i)/dp over an event list; extra "
             "as in prob_points.")
        .def("weighted_gradient_points_binned",
             [](W& w, Arr1 E, Arr1 C, Arr1u nb_,
                nb::ndarray<const double, nb::c_contig, nb::device::cpu> wt,
                nb::ndarray<const int, nb::ndim<1>, nb::c_contig, nb::device::cpu> bins,
                int nbins, std::optional<Arr2> ex) {
               w.sync();
               std::vector<uint8_t> nbv(nb_.data(), nb_.data() + nb_.shape(0));
               std::vector<double>  v(wt.data(), wt.data() + wt.size());
               std::vector<int>     bv(bins.data(), bins.data() + bins.shape(0));
               auto G = w.prop.weighted_gradient_points_binned(
                   to_vec(E), to_vec(C), nbv, v, bv, nbins, to_extra(ex, W::NX, E.shape(0)));
               return make_array(std::move(G),
                                 {size_t(nbins), w.prop.n_gradient_params()});
             },
             "energies"_a, "cosines"_a, "nubar"_a, "weights"_a, "bins"_a, "nbins"_a,
             "extra"_a = nb::none(),
             "Per-analysis-bin weighted gradient: G[b, p] = sum over events i "
             "with bins[i] == b of weights[a, b, i] * dP_ab(i)/dp (bins outside "
             "[0, nbins) are ignored); extra as in prob_points.")
        .def("prob_path_grad",
             [](W& w, Arr1 E, Arr2 seg, bool nubar) {
               w.sync();
               std::vector<double> P, dP;
               w.prop.prob_path_grad(to_vec(E), to_path(seg), nubar, P, dP);
               size_t np = w.prop.n_gradient_params();
               return nb::make_tuple(make_array(std::move(P), {N, N, E.shape(0)}),
                                     make_array(std::move(dP), {np, N, N, E.shape(0)}));
             },
             "energies"_a, "path"_a, "nubar"_a = false,
             "Fixed path: returns (P[a, b, iE], dP[p, a, b, iE]).")
        .def_prop_ro("earth_parameter_names",
                     [](W& w) { return w.prop.earth_parameter_names(); },
                     "Earth Z/A gradient parameters zoa_<layer type> of this "
                     "propagator's Earth model (never selected by default).")
        .def_prop_ro("gradient_parameter_names",
                     [](W& w) { return w.prop.gradient_parameter_names(); },
                     "All selectable gradient parameters (model, then Earth).")
        .def("calculate_binned_gradient",
             [](W& w, const std::string& fl) {
               w.sync();
               w.prop.calculate_binned(parse_flavor(fl), true);
             },
             "flavor"_a = "both",
             "Bin averages and their gradients (see binned() and binned_grad()).")
        .def("binned_grad",
             [](W& w) {
               const double* g  = w.prop.binned_grad();
               size_t        np = w.prop.n_gradient_params();
               size_t        nc = w.prop.n_cosine_bins(), ne = w.prop.n_energy_bins();
               return make_array(std::vector<double>(g, g + 2 * np * N * N * nc * ne),
                                 {2, np, N, N, nc, ne});
             },
             "Binned gradients G[nubar, p, a, b, iCbin, iEbin] (copy).")
        .def("weighted_gradient_binned",
             [](W& w, AnyArr wt, const std::string& fl) {
               w.sync();
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
                 g = w.prop.weighted_gradient_binned_device({wt.data()}, parse_flavor(fl));
               }
               else {
                 if (wt.device_type() != nb::device::cpu::value)
                   throw std::invalid_argument("weighted_gradient_binned: weights must "
                                               "be a host or CUDA array");
                 std::vector<double> v(wt.data(), wt.data() + wt.size());
                 g = w.prop.weighted_gradient_binned(v, parse_flavor(fl));
               }
               return make_array(std::move(g), {g.size()});
             },
             "weights"_a, "flavor"_a = "both",
             "sum of weights * d(bin average)/dp; weights have the shape of "
             "binned() (host or CUDA array, see weighted_gradient). Returns one "
             "value per gradient parameter.")
        .def("avg_path_grad",
             [](W& w, Arr1 Ee, int nE, Arr2 seg, bool nubar, const std::string& meas) {
               w.sync();
               std::vector<double> A, dA;
               w.prop.avg_path_grad(to_vec(Ee), nE, to_path(seg), nubar, A, dA,
                                    parse_measure(meas));
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
               w.sync();
               auto out = w.prop.avg_path(to_vec(Ee), nE, to_path(seg), nubar,
                                          parse_measure(meas));
               return make_array(std::move(out), {N, N, Ee.shape(0) - 1});
             },
             "energy_edges"_a, "n_gl"_a, "path"_a, "nubar"_a = false,
             "measure"_a = "linear", "1D bin averages for a fixed path.");
    return c;
  }

  template <class Model>
  void bind_ctor(nb::class_<PyModel<Model>>& c)
  {
    c.def(nb::init<const opg::PremModel&, const std::vector<int>&, int>(),
          "earth"_a = opg::PremModel(), "devices"_a = std::vector<int>{},
          "threads"_a = 0,
          "devices: CUDA device ids (empty list = CPU backend); threads: "
          "OpenMP threads for the CPU backend (0 = default).");
  }

} // namespace

NB_MODULE(_oscprobgpu, m)
{
  m.doc() = "OscProbGPU: CUDA port of OscProb oscillation calculators";
  m.def("cuda_device_count", &opg::cuda_device_count);
  m.def("_dlpack", &dlpack_capsule, "ptr"_a, "shape"_a, "device"_a, "owner"_a);
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
         [](PyModel<NUNM>* self, const opg::PremModel& e, const std::vector<int>& d,
            int t, int scale) {
           new (self) PyModel<NUNM>(e, d, t);
           self->params.scale = scale;
         },
         "earth"_a = opg::PremModel(), "devices"_a = std::vector<int>{},
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
      .def("get_gamma", [](PyModel<Deco>& w, int i, int j) { return w.params.GetGamma(i, j); },
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
