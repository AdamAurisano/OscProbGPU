///////////////////////////////////////////////////////////////////////////////
/// \file propagator.h
///
/// \brief User-facing batched oscillation calculator.
///
/// Lifecycle (modelled on CUDAProb3's Propagator):
///
///   opg::Propagator<opg::Fast<>> prop(opg::PremModel(), {0});  // GPU 0
///   prop.set_grid(energies, cosines);     // once: upload + preallocate
///   prop.set_params(params);              // per fit iteration (cheap)
///   prop.calculate(opg::Flavor::Both);    // asynchronous launch
///   const double* P = prop.probs();       // waits, copies to host lazily
///   double pmue = prop.prob(1, 0, iC, iE, /*nubar=*/false);
///
/// An empty device list selects the multi-threaded CPU backend, which runs
/// the same physics code and is used to validate the GPU results.
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_PROPAGATOR_H
#define OPG_PROPAGATOR_H

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "opg/avg/gauss_legendre.h"
#include "opg/engine.h"
#include "opg/engine_cpu.h"
#include "opg/models/all.h"

namespace opg {

#ifdef OPG_HAVE_CUDA
  /// Defined in src/engine_cuda.cu, explicitly instantiated per model.
  template <class Model>
  std::unique_ptr<EngineBase<Model>> make_cuda_engine(const std::vector<int>& devices);
#endif

  namespace detail {
    template <class Model, class = void>
    struct has_default_param_names : std::false_type {};
    template <class Model>
    struct has_default_param_names<Model,
                                   std::void_t<decltype(Model::default_param_names())>>
        : std::true_type {};
  } // namespace detail

  /// Number of CUDA devices visible (0 if built without CUDA).
  int cuda_device_count();

  template <class Model> class Propagator {
    public:
      using R                = typename Model::Real;
      using Params           = typename Model::Params;
      using Prepared         = typename Model::Prepared;
      static constexpr int N = Model::N;

      /// devices: CUDA device ids to use; empty = CPU backend.
      /// cpu_threads: OpenMP threads for the CPU backend (0 = default).
      explicit Propagator(const PremModel& earth = PremModel(),
                          const std::vector<int>& devices = {},
                          int cpu_threads = 0)
          : fDevices(devices)
      {
        if (devices.empty()) { fEngine.reset(new CpuEngine<Model>(cpu_threads)); }
        else {
#ifdef OPG_HAVE_CUDA
          if constexpr (has_cuda_engine<Model>::value)
            fEngine = make_cuda_engine<Model>(devices);
          else
            throw std::runtime_error(
                "opg::Propagator: this model/precision is not compiled into "
                "the CUDA backend (see OPG_FOR_EACH_MODEL)");
#else
          throw std::runtime_error(
              "opg::Propagator: GPU devices requested but OscProbGPU was "
              "built without CUDA");
#endif
        }
        set_earth(earth);
      }

      bool on_gpu() const { return !fDevices.empty(); }
      const std::vector<int>& devices() const { return fDevices; }

      void set_earth(const PremModel& earth)
      {
        fEarth = earth;
        fEngine->set_earth(earth);
        fZoaTypes.clear();
        for (const auto& l : fEarth.GetLayers())
          if (std::find(fZoaTypes.begin(), fZoaTypes.end(), l.type) == fZoaTypes.end())
            fZoaTypes.push_back(l.type);
        std::sort(fZoaTypes.begin(), fZoaTypes.end());
        // the meaning of zoa_<type> parameters may have changed
        if (!fGradNames.empty()) set_gradient_params(std::vector<std::string>(fGradNames));
        fChunksValid = false;
      }

      const PremModel& earth() const { return fEarth; }

      /// Set the (E, cosZ) grid. E in GeV.
      void set_grid(const std::vector<R>& E, const std::vector<R>& cosZ)
      {
        fNE = E.size();
        fNC = cosZ.size();
        fEngine->set_grid(E, cosZ);
      }

      /// Set the model parameters (runs Model::prepare on the host).
      void set_params(const Params& p)
      {
        fParams     = p;
        fPrepared   = Model::prepare(p);
        fHaveParams = true;
        fChunksValid = false;
      }

      //.......................................................................
      // Gradients. Off unless parameters are selected with
      // set_gradient_params(); probability-only calls are unaffected.

      /// True if this model supports gradients (and they were compiled in).
      static constexpr bool has_gradients() { return grad_traits<Model>::enabled; }

      /// Names of the differentiable parameters of this model.
      static std::vector<std::string> parameter_names()
      {
        if constexpr (grad_traits<Model>::enabled) return Model::param_names();
        else return {};
      }

      /// Z/A parameters of the Earth model: zoa_<type> for each layer type
      /// (PremModel::SetLayerZoA). The derivative with respect to zoa_<t> is
      /// that of changing the Z/A of all layers of type t together; for
      /// fixed paths it applies to segments whose `layer` is t.
      std::vector<std::string> earth_parameter_names() const
      {
        if constexpr (!grad_traits<Model>::enabled) return {};
        std::vector<std::string> n;
        for (int t : fZoaTypes) n.push_back("zoa_" + std::to_string(t));
        return n;
      }

      /// All parameters set_gradient_params() accepts: parameter_names()
      /// followed by earth_parameter_names().
      std::vector<std::string> gradient_parameter_names() const
      {
        auto n = parameter_names();
        for (auto& e : earth_parameter_names()) n.push_back(e);
        return n;
      }

      /// The parameters set_gradient_params() selects when called without
      /// arguments: all of parameter_names() except those a model leaves
      /// out by default (Model::default_param_names(), e.g. NSI's fermion
      /// couplings, which are rarely fitted). Earth Z/A parameters are never
      /// selected by default.
      static std::vector<std::string> default_gradient_params()
      {
        if constexpr (!grad_traits<Model>::enabled) return {};
        else if constexpr (detail::has_default_param_names<Model>::value)
          return Model::default_param_names();
        else
          return Model::param_names();
      }

      /// Select the default parameters (see default_gradient_params()).
      void set_gradient_params() { set_gradient_params(default_gradient_params()); }

      /// Select the parameters to differentiate with respect to, by name
      /// (see gradient_parameter_names()). An empty list turns gradients off
      /// and frees the gradient buffers.
      void set_gradient_params(const std::vector<std::string>& names)
      {
        if (!names.empty() && !has_gradients())
          throw std::logic_error("opg::Propagator: gradients are not available "
                                 "for this model (or were disabled at build "
                                 "time)");
        auto all = gradient_parameter_names();
        std::vector<int> idx;
        for (auto& n : names) {
          auto it = std::find(all.begin(), all.end(), n);
          if (it == all.end())
            throw std::invalid_argument("opg::Propagator: unknown gradient "
                                        "parameter '" + n + "'");
          idx.push_back(int(it - all.begin()));
        }
        fGradNames   = names;
        fGradIdx     = idx;
        fChunksValid = false;
        if (names.empty()) fChunks.clear();
      }

      const std::vector<std::string>& gradient_params() const { return fGradNames; }
      size_t n_gradient_params() const { return fGradIdx.size(); }

      /// Launch the grid computation; with gradient = true also compute
      /// dP/dp for the selected parameters (see grad()).
      void calculate(Flavor which, bool gradient)
      {
        if (!gradient) return calculate(which);
        require_params();
        fEngine->calculate_grad(fPrepared, chunks(), int(fGradIdx.size()), which);
      }

      /// Grid gradients grad[nubar][p][a][b][iC][iE] = dP(a -> b)/dp, with p
      /// in the order given to set_gradient_params(). Waits for completion.
      const R* grad() { return fEngine->host_grad(); }
      const R* device_grad(int device_index = 0)
      {
        return fEngine->device_grad(device_index);
      }
      R grad(size_t p, int a, int b, size_t iC, size_t iE, bool nubar)
      {
        const R* g = grad();
        return g[((((size_t(nubar) * fGradIdx.size() + p) * N + a) * N + b) *
                      fNC + iC) * fNE + iE];
      }

      /// Weighted gradient on the grid: g[p] = sum w * dP/dp over all grid
      /// points and channels, with w[nubar][a][b][iC][iE] (the layout of
      /// probs()). Only the gradient vector is returned; no per-point
      /// derivatives are stored. Typically w = dchi2/dP.
      std::vector<R> weighted_gradient(const std::vector<R>& w,
                                       Flavor which = Flavor::Both)
      {
        require_params();
        if (w.size() != 2 * size_t(N) * N * fNC * fNE)
          throw std::invalid_argument("weighted_gradient: weights must have "
                                      "the layout of probs()");
        std::vector<R> g(fGradIdx.size(), R(0));
        fEngine->weighted_grad(fPrepared, chunks(), int(fGradIdx.size()), which,
                               w.data(), g.data());
        return g;
      }

      const Prepared& prepared() const { return fPrepared; }

      /// Launch the grid computation (asynchronous on GPU).
      void calculate(Flavor which = Flavor::Both)
      {
        require_params();
        fEngine->calculate(fPrepared, which);
      }

      void wait() { fEngine->wait(); }

      /// Host array probs[nubar][a][b][iC][iE]; waits for completion.
      const R* probs() { return fEngine->host_probs(); }

      /// Device array (same layout) on the given device index (GPU only).
      const R* device_probs(int device_index = 0)
      {
        return fEngine->device_probs(device_index);
      }

      /// P(a -> b) at grid point (iC, iE).
      R prob(int a, int b, size_t iC, size_t iE, bool nubar)
      {
        const R* p = probs();
        return p[(((size_t(nubar) * N + a) * N + b) * fNC + iC) * fNE + iE];
      }

      size_t n_energies() const { return fNE; }
      size_t n_cosines() const { return fNC; }

      //.......................................................................
      /// Bin-averaged probabilities on a 2D (E, cosZ) binning.
      ///
      /// E: nglE Gauss-Legendre nodes per bin, uniform in E, log E or 1/E
      /// according to `measure` (InvE is uniform in L/E at fixed baseline,
      /// the analogue of OscProb's AvgProbLoE).
      /// cosZ: uniform average with nglC Gauss-Legendre nodes; with
      /// CosZRule::LayerAdapted (default) the bin is first split at the
      /// layer-grazing cosines of the Earth model and nglC nodes are used on
      /// each piece (see CosZRule).
      void set_bins(const std::vector<double>& Eedges,
                    const std::vector<double>& Cedges, int nglE, int nglC,
                    EMeasure measure = EMeasure::Linear,
                    CosZRule rule    = CosZRule::LayerAdapted)
      {
        if (Eedges.size() < 2 || Cedges.size() < 2)
          throw std::invalid_argument("set_bins: need at least one bin");
        if (nglE < 1 || nglC < 1)
          throw std::invalid_argument("set_bins: GL orders must be >= 1");
        BinSpec<R> B;
        B.nEb  = Eedges.size() - 1;
        B.nCb  = Cedges.size() - 1;
        B.nglE = nglE;
        GaussLegendre       glE(nglE), glC(nglC);
        std::vector<double> x, w;
        for (size_t b = 0; b < B.nEb; b++) {
          double lo = Eedges[b], hi = Eedges[b + 1];
          if (!(hi > lo) || lo <= 0)
            throw std::invalid_argument("set_bins: E edges must be positive "
                                        "and increasing");
          double ulo, uhi;
          switch (measure) {
            case EMeasure::Linear: ulo = lo; uhi = hi; break;
            case EMeasure::Log: ulo = std::log(lo); uhi = std::log(hi); break;
            case EMeasure::InvE: ulo = 1 / hi; uhi = 1 / lo; break;
            default: throw std::invalid_argument("set_bins: bad measure");
          }
          glE.map(ulo, uhi, x, w);
          for (int i = 0; i < nglE; i++) {
            double e = measure == EMeasure::Linear ? x[i]
                       : measure == EMeasure::Log  ? std::exp(x[i])
                                                   : 1 / x[i];
            B.nodesE.push_back(R(e));
            B.wE.push_back(R(w[i] / (uhi - ulo)));
          }
        }

        const std::vector<double> kinks =
            rule == CosZRule::LayerAdapted ? fEarth.GetGrazingCosines()
                                           : std::vector<double>{};
        B.offC.push_back(0);
        for (size_t b = 0; b < B.nCb; b++) {
          double lo = Cedges[b], hi = Cedges[b + 1];
          if (!(hi > lo) || lo < -1 || hi > 1)
            throw std::invalid_argument("set_bins: cosZ edges must be "
                                        "increasing within [-1, 1]");
          std::vector<double> brk = {lo};
          for (double k : kinks)
            if (k > lo && k < hi) brk.push_back(k);
          brk.push_back(hi);
          for (size_t q = 0; q + 1 < brk.size(); q++) {
            double a = brk[q], c = brk[q + 1];
            if (rule == CosZRule::Plain) {
              glC.map(a, c, x, w);
              for (int j = 0; j < nglC; j++) {
                B.nodesC.push_back(R(x[j]));
                B.wC.push_back(R(w[j] / (hi - lo)));
              }
            }
            else {
              // cosZ = a + (c-a)(1 - cos(pi u))/2, u in [0, 1]
              glC.map(0, 1, x, w);
              for (int j = 0; j < nglC; j++) {
                double u   = x[j];
                double cz  = a + (c - a) * 0.5 * (1 - std::cos(M_PI * u));
                double jac = (c - a) * 0.5 * M_PI * std::sin(M_PI * u);
                B.nodesC.push_back(R(cz));
                B.wC.push_back(R(w[j] * jac / (hi - lo)));
              }
            }
          }
          B.offC.push_back(B.nodesC.size());
        }
        fNEb = B.nEb;
        fNCb = B.nCb;
        fEngine->set_bins(B);
      }

      /// Launch the bin-averaged computation (asynchronous on GPU).
      void calculate_binned(Flavor which = Flavor::Both)
      {
        require_params();
        fEngine->calculate_binned(fPrepared, which);
      }

      /// Launch the bin-averaged computation; with gradient = true also the
      /// bin-averaged derivatives for the selected parameters (see
      /// binned_grad()).
      void calculate_binned(Flavor which, bool gradient)
      {
        if (!gradient) return calculate_binned(which);
        require_params();
        fEngine->calculate_binned_grad(fPrepared, chunks(), int(fGradIdx.size()), which);
      }

      /// Bin-averaged gradients grad[nubar][p][a][b][iCb][iEb]; waits.
      const R* binned_grad() { return fEngine->host_binned_grad(); }
      const R* device_binned_grad(int device_index = 0)
      {
        return fEngine->device_binned_grad(device_index);
      }

      /// Weighted bin-averaged gradient: g[p] = sum w * d(avg P)/dp over all
      /// bins and channels, w in the layout of binned(). Nothing per bin or
      /// node is stored.
      std::vector<R> weighted_gradient_binned(const std::vector<R>& w,
                                              Flavor which = Flavor::Both)
      {
        require_params();
        if (w.size() != 2 * size_t(N) * N * fNCb * fNEb)
          throw std::invalid_argument("weighted_gradient_binned: weights must "
                                      "have the layout of binned()");
        std::vector<R> g(fGradIdx.size(), R(0));
        fEngine->weighted_grad_binned(fPrepared, chunks(), int(fGradIdx.size()), which,
                                      w.data(), g.data());
        return g;
      }

      /// Host array avg[nubar][a][b][iCbin][iEbin]; waits for completion.
      const R* binned() { return fEngine->host_binned(); }

      const R* device_binned(int device_index = 0)
      {
        return fEngine->device_binned(device_index);
      }

      /// Bin-averaged P(a -> b) in bin (iCb, iEb).
      R binned(int a, int b, size_t iCb, size_t iEb, bool nubar)
      {
        const R* p = binned();
        return p[(((size_t(nubar) * N + a) * N + b) * fNCb + iCb) * fNEb + iEb];
      }

      size_t n_energy_bins() const { return fNEb; }
      size_t n_cosine_bins() const { return fNCb; }

      /// 1D bin averages along E for a fixed path (e.g. long baseline),
      /// using nglE Gauss-Legendre nodes per bin. Returns out[a][b][iEbin].
      std::vector<R> avg_path(const std::vector<double>& Eedges, int nglE,
                              const std::vector<Segment<R>>& path, bool nubar,
                              EMeasure measure = EMeasure::Linear)
      {
        require_params();
        std::vector<R> nodes, wts;
        const size_t   nb = path_nodes(Eedges, nglE, measure, nodes, wts);
        std::vector<R> pn = prob_path(nodes, path, nubar);  // [a][b][node]
        return average_nodes(pn, size_t(N) * N, nb, nglE, wts);
      }

      /// avg_path with gradients: out[a][b][iEbin], dout[p][a][b][iEbin].
      void avg_path_grad(const std::vector<double>& Eedges, int nglE,
                         const std::vector<Segment<R>>& path, bool nubar,
                         std::vector<R>& out, std::vector<R>& dout,
                         EMeasure measure = EMeasure::Linear)
      {
        require_params();
        std::vector<R> nodes, wts, pn, dpn;
        const size_t   nb = path_nodes(Eedges, nglE, measure, nodes, wts);
        prob_path_grad(nodes, path, nubar, pn, dpn);
        out  = average_nodes(pn, size_t(N) * N, nb, nglE, wts);
        dout = average_nodes(dpn, fGradIdx.size() * N * N, nb, nglE, wts);
      }

      /// Event list through the Earth model. Returns out[a][b][i].
      std::vector<R> prob_points(const std::vector<R>& E,
                                 const std::vector<R>& cosZ,
                                 const std::vector<uint8_t>& nubar)
      {
        require_params();
        check_points(E, cosZ, nubar);
        std::vector<R> out(size_t(N) * N * E.size());
        fEngine->prob_points(fPrepared, E.data(), cosZ.data(), nubar.data(),
                             E.size(), out.data());
        return out;
      }

      /// Event list with gradients: P[a][b][i] and dP[p][a][b][i].
      void prob_points_grad(const std::vector<R>& E, const std::vector<R>& cosZ,
                            const std::vector<uint8_t>& nubar, std::vector<R>& P,
                            std::vector<R>& dP)
      {
        require_params();
        check_points(E, cosZ, nubar);
        P.assign(size_t(N) * N * E.size(), R(0));
        dP.assign(fGradIdx.size() * N * N * E.size(), R(0));
        fEngine->prob_points_grad(fPrepared, chunks(), int(fGradIdx.size()),
                                  E.data(), cosZ.data(), nubar.data(), E.size(),
                                  P.data(), dP.data());
      }

      /// Weighted gradient over an event list: g[p] = sum_i,ab w[a][b][i]
      /// dP_ab(i)/dp.
      std::vector<R> weighted_gradient_points(const std::vector<R>& E,
                                              const std::vector<R>& cosZ,
                                              const std::vector<uint8_t>& nubar,
                                              const std::vector<R>& w)
      {
        require_params();
        check_points(E, cosZ, nubar);
        if (w.size() != size_t(N) * N * E.size())
          throw std::invalid_argument("weighted_gradient_points: weights must "
                                      "be [a][b][i]");
        std::vector<R> g(fGradIdx.size(), R(0));
        fEngine->weighted_grad_points(fPrepared, chunks(), int(fGradIdx.size()),
                                      E.data(), cosZ.data(), nubar.data(),
                                      E.size(), w.data(), g.data());
        return g;
      }

      /// Per-analysis-bin weighted gradient over an event list:
      /// G[b][p] = sum over events i in bin b (bin[i] in [0, nbins); other
      /// events are ignored) of sum_ab w[a][b][i] dP_ab(i)/dp. Suited to
      /// likelihoods that need dN_b/dp for each analysis bin b.
      std::vector<R> weighted_gradient_points_binned(const std::vector<R>& E,
                                                     const std::vector<R>& cosZ,
                                                     const std::vector<uint8_t>& nubar,
                                                     const std::vector<R>& w,
                                                     const std::vector<int>& bin,
                                                     int nbins)
      {
        require_params();
        check_points(E, cosZ, nubar);
        if (w.size() != size_t(N) * N * E.size() || bin.size() != E.size())
          throw std::invalid_argument("weighted_gradient_points_binned: weights "
                                      "must be [N][N][n] and bins [n]");
        if (nbins < 0) throw std::invalid_argument("weighted_gradient_points_binned: nbins < 0");
        std::vector<R> G(size_t(nbins) * fGradIdx.size(), R(0));
        fEngine->weighted_grad_points_binned(fPrepared, chunks(), int(fGradIdx.size()),
                                             E.data(), cosZ.data(), nubar.data(), E.size(),
                                             w.data(), bin.data(), nbins, G.data());
        return G;
      }

      /// Fixed path with gradients: P[a][b][iE] and dP[p][a][b][iE].
      void prob_path_grad(const std::vector<R>& E, const std::vector<Segment<R>>& path,
                          bool nubar, std::vector<R>& P, std::vector<R>& dP)
      {
        require_params();
        P.assign(size_t(N) * N * E.size(), R(0));
        dP.assign(fGradIdx.size() * N * N * E.size(), R(0));
        fEngine->prob_path_grad(fPrepared, chunks(), int(fGradIdx.size()),
                                E.data(), E.size(), path.data(), int(path.size()),
                                nubar, P.data(), dP.data());
      }

      /// Fixed baseline through explicit segments. Returns out[a][b][iE].
      std::vector<R> prob_path(const std::vector<R>& E,
                               const std::vector<Segment<R>>& path, bool nubar)
      {
        require_params();
        std::vector<R> out(size_t(N) * N * E.size());
        fEngine->prob_path(fPrepared, E.data(), E.size(), path.data(),
                           int(path.size()), nubar, out.data());
        return out;
      }

    private:
      static void check_points(const std::vector<R>& E, const std::vector<R>& C,
                               const std::vector<uint8_t>& nb)
      {
        if (E.size() != C.size() || E.size() != nb.size())
          throw std::invalid_argument("prob_points: size mismatch");
      }

      /// Gradient passes for the selected parameters (cached until the
      /// parameters or the selection change).
      const std::vector<GradChunk<Model>>& chunks()
      {
        if (fGradIdx.empty())
          throw std::logic_error("opg::Propagator: no gradient parameters "
                                 "selected (set_gradient_params)");
        if constexpr (grad_traits<Model>::enabled) {
          if (!fChunksValid) {
            constexpr int K = grad_traits<Model>::K;
            using D         = Dual<R, K>;
            fChunks.clear();
            const int nmodel = int(parameter_names().size());
            for (size_t off = 0; off < fGradIdx.size(); off += K) {
              auto pd = Model::template cast<D>(fParams);
              int  cnt = int(std::min<size_t>(K, fGradIdx.size() - off));
              GradChunk<Model> c;
              for (int k = 0; k < K; k++) c.P.zoa_type[k] = -1;
              for (int k = 0; k < cnt; k++) {
                const int idx = fGradIdx[off + k];
                if (idx < nmodel)
                  Model::template param_ref<D>(pd, idx).d[k] += R(1);
                else
                  c.P.zoa_type[k] = fZoaTypes[idx - nmodel];
              }
              c.P.P    = Model::template prepare_generic<D>(pd);
              c.offset = int(off);
              c.count  = cnt;
              fChunks.push_back(c);
            }
            fChunksValid = true;
          }
        }
        return fChunks;
      }

      /// Gauss-Legendre nodes and normalised weights for 1D E bins.
      static size_t path_nodes(const std::vector<double>& Eedges, int nglE,
                               EMeasure measure, std::vector<R>& nodes,
                               std::vector<R>& wts)
      {
        if (Eedges.size() < 2 || nglE < 1)
          throw std::invalid_argument("avg_path: bad binning");
        const size_t        nb = Eedges.size() - 1;
        GaussLegendre       gl(nglE);
        std::vector<double> x, w;
        nodes.clear();
        wts.clear();
        for (size_t b = 0; b < nb; b++) {
          double lo = Eedges[b], hi = Eedges[b + 1], ulo, uhi;
          if (!(hi > lo) || lo <= 0)
            throw std::invalid_argument("avg_path: bad E edges");
          if (measure == EMeasure::Linear) { ulo = lo; uhi = hi; }
          else if (measure == EMeasure::Log) { ulo = std::log(lo); uhi = std::log(hi); }
          else { ulo = 1 / hi; uhi = 1 / lo; }
          gl.map(ulo, uhi, x, w);
          for (int i = 0; i < nglE; i++) {
            nodes.push_back(R(measure == EMeasure::Linear ? x[i]
                              : measure == EMeasure::Log  ? std::exp(x[i])
                                                          : 1 / x[i]));
            wts.push_back(R(w[i] / (uhi - ulo)));
          }
        }
        return nb;
      }

      /// out[ch][b] = sum_i wts[b nglE + i] in[ch][b nglE + i]
      static std::vector<R> average_nodes(const std::vector<R>& in, size_t nch,
                                          size_t nb, int nglE, const std::vector<R>& wts)
      {
        const size_t   nn = nb * size_t(nglE);
        std::vector<R> out(nch * nb, R(0));
        for (size_t ch = 0; ch < nch; ch++)
          for (size_t b = 0; b < nb; b++) {
            R acc = 0;
            for (int i = 0; i < nglE; i++)
              acc += wts[b * nglE + i] * in[ch * nn + b * nglE + i];
            out[ch * nb + b] = acc;
          }
        return out;
      }

      void require_params() const
      {
        if (!fHaveParams)
          throw std::logic_error("opg::Propagator: call set_params() first");
      }

      std::vector<int>                    fDevices;
      PremModel                           fEarth;
      std::unique_ptr<EngineBase<Model>>  fEngine;
      Prepared                            fPrepared{};
      Params                              fParams{};
      std::vector<std::string>            fGradNames;
      std::vector<int>                    fGradIdx;
      std::vector<int>                    fZoaTypes;  ///< layer types (zoa_<t>)
      std::vector<GradChunk<Model>>       fChunks;
      bool                                fChunksValid = false;
      bool                                fHaveParams = false;
      size_t                              fNE = 0, fNC = 0;
      size_t                              fNEb = 0, fNCb = 0;
  };

} // namespace opg

#endif
