# OscProbGPU

A CUDA port of selected [OscProb](https://github.com/joaoabcoelho/OscProb)
neutrino oscillation calculators, with a batched C++ API and Python bindings.

| OscProbGPU model | OscProb class            | Physics                                   |
|------------------|--------------------------|-------------------------------------------|
| `opg::Fast`      | `PMNS_Fast`              | standard 3-flavour, matter                |
| `opg::NSI`       | `PMNS_NSI`               | vector non-standard interactions          |
| `opg::NUNM`      | `PMNS_NUNM` (scale 0, 1) | non-unitary mixing                        |
| `opg::Sterile`   | `PMNS_Sterile(4)`        | 3+1 sterile neutrino                      |
| `opg::Decay`     | `PMNS_Decay`             | invisible decay (non-hermitian H)         |
| `opg::LIV`       | `PMNS_LIV`               | Lorentz invariance violation (SME, d = 3..8) |
| `opg::SNSI`      | `PMNS_SNSI`              | scalar non-standard interactions          |
| `opg::Deco`      | `PMNS_Deco`              | decoherence (density matrices)            |
| `opg::SiderealLIV` | `PMNS_SiderealLIV`     | direction/sidereal-time dependent SME LIV |
| `opg::OQS`       | `PMNS_OQS`               | open quantum system (Gell-Mann basis dissipator) |

Every model runs on the GPU (one or more devices) or on a multi-threaded CPU
backend that executes the *same* `__host__ __device__` physics code. Results
are validated point by point against the original OscProb (see
[Validation](#validation)).

## Quick start

### Python

```python
import numpy as np
import oscprobgpu as opg

p = opg.Fast(devices=[0])                   # devices=[] -> CPU backend
p.set_angle(2, 3, np.arcsin(np.sqrt(0.451)))  # OscProb-style setters (1-based)
p.set_delta(1, 3, np.radians(232))
p.set_dm(3, 2.507e-3)

# Oscillogram: upload the grid once, then recompute cheaply
p.set_grid(np.geomspace(1, 100, 1000), np.linspace(-1, 0, 1000))
p.calculate()                               # asynchronous launch
P = p.probs()                               # P[nubar, a, b, iC, iE] = P(a -> b)

# Bin averages (2D Gauss-Legendre in E and cos zenith)
p.set_bins(np.geomspace(1, 100, 41), np.linspace(-1, 0, 21),
           n_gl_energy=8, n_gl_cosine=8, measure="log")
p.calculate_binned()
A = p.binned()                              # A[nubar, a, b, iCbin, iEbin]

# Event lists and fixed baselines
pts = p.prob_points(E, cosZ, nubar.astype(np.uint8))      # [a, b, i]
lbl = p.prob_path(E, np.array([[1285.0, 2.84, 0.5]]))     # [a, b, iE]
avg = p.avg_path(edges, 8, np.array([[1285.0, 2.84, 0.5]]))
```

Model-specific setters: `NSI.set_eps(i, j, value, phase)`,
`NSI.set_ferm_coup(e, u, d)`, `NUNM(scale=0|1)`, `NUNM.set_alpha(i, j, value,
phase)`, `NUNM.set_frac_vnc(f)`, `Decay.set_alpha2/3(a)`,
`LIV.set_aT(i, j, dim, value, phase)` (dim 3, 5, 7), `LIV.set_cT(i, j, dim,
value, phase)` (dim 4, 6, 8), `SNSI.set_eps(i, j, value, phase)` (MeV⁻²),
`SNSI.set_ferm_coup(e, u, d)`, `SNSI.set_lowest_mass(m)` (eV),
`Deco.set_gamma(j, value)` (Γ_j1 in GeV), `Deco.set_gamma32(value)`,
`Deco.set_deco_angle(theta)`, `Deco.set_power(n)`,
`SiderealLIV.set_a(i, j, coord, v)`, `set_c(i, j, c1, c2, v)`,
`set_colatitude(chi)` / `set_latitude(deg, min, sec)`,
`set_neutrino_direction(zenith, azimuth)` (fixed direction; set the latitude
first), `set_azimuth(azimuth)` (Earth paths then use their own zenith,
arccos cos θ_z), `set_time_hours(t)`, `OQS.set_deco_element(i, value)`
(|a_i|, i = 1..8), `OQS.set_deco_angle(i, j, theta)`, `OQS.set_power(n)`.
All models start
from OscProb's PDG defaults (`set_std_pars()`). See `python/examples/`:
`oscillogram.py`, `lbl_spectrum.py`, and `gradient_fit.py` (a binned
atmospheric likelihood fit with exact Jacobians from `binned_grad()`,
optionally including the outer-core Z/A). Gradients: see
[Gradients](#gradients).

### C++

```cpp
#include "opg/propagator.h"

opg::Fast<>::Params par;
opg::set_std_pars(par.mix);
par.mix.SetAngle(2, 3, std::asin(std::sqrt(0.451)));

opg::Propagator<opg::Fast<>> prop(opg::PremModel(), /*devices=*/{0});
prop.set_grid(energies, cosines);   // once
prop.set_params(par);               // per fit iteration (host-side prepare)
prop.calculate(opg::Flavor::Both);  // asynchronous
const double* P = prop.probs();     // probs[nubar][a][b][iC][iE]
double pmue = prop.prob(1, 0, iC, iE, /*nubar=*/false);
```

Link against the `OscProbGPU::oscprobgpu` CMake target. Precision is a
template parameter (`opg::Fast<float>` is compiled for the GPU for
exploration; double is the default and recommended).

## Build

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=70   # CUDA auto-detected
cmake -S . -B build -DOPG_ENABLE_CUDA=OFF           # CPU only
cmake -S . -B build -DOPG_ENABLE_PYTHON=ON          # + Python module
cmake --build build -j8
scripts/fetch_test_data.sh                           # reference data for the tests
ctest --test-dir build                               # labels: cpu, gpu, python
```

Requirements: CMake ≥ 3.18, C++17, optionally CUDA ≥ 11 (tested with 11.8 on
V100) and Python ≥ 3.8 with numpy. nanobind (BSD) and doctest (MIT) are
vendored in `third_party/`. With the module built, use
`PYTHONPATH=build/python`, or install the package with pip
(scikit-build-core; CUDA is used when `nvcc` is found):

```sh
pip install ".[test]"
pip install . --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=70
OPG_TEST_DATA_DIR=$PWD/tests/data pytest python/tests
```

The pip build has been tested with the CPU backend (Python 3.12, numpy 2.5);
the CUDA module is tested through the CMake build.

On pre-Ampere GPUs with old drivers (no PTX JIT), set
`CMAKE_CUDA_ARCHITECTURES` to the exact SM of the card.

| CMake option          | Default | Meaning |
|-----------------------|---------|---------|
| `OPG_ENABLE_CUDA`     | auto    | build the CUDA backend |
| `OPG_ENABLE_PYTHON`   | OFF     | build the `oscprobgpu` Python module |
| `OPG_ENABLE_TESTS`    | ON      | build the test suites |
| `OPG_ENABLE_GRADIENTS`| ON      | compile the gradient code paths |
| `OPG_OSCPROB_BITWISE` | OFF     | call the 3x3 eigensolver exactly as OscProb (bit-identical on CPU, ~2x slower) |
| `OPG_OSCPROB_DIR`     | `../OscProb` | original sources, used only by tests (bitwise eigensolver check) |

## Design

* **Single-source physics.** Header-only `OPG_HD` functions on fixed-size
  `Mat<N, Real>` over a small POD `Complex<Real>`; the CPU backend (OpenMP) and
  the CUDA backend run the same code.
* **Lifecycle** (after CUDAProb3): grids and Earth tables are uploaded once into
  persistent device buffers; `calculate()` only launches; results stay on the
  device and are copied to pinned host memory on first access. Device pointers
  are available (`device_probs()`) for GPU-side consumers.
* **Parameters by value.** `Model::prepare()` builds everything that depends
  only on the parameters (e.g. `Hms = U diag(dm) U^†`, by a verbatim port of
  `PMNS_Base::BuildHms`) on the host; the resulting POD is passed by value as a
  kernel argument. There is no global device state, so several propagators can
  coexist. There is no eigensystem cache.
* **Thread mapping.** Block rows over cos zenith, threads over energy, so all
  threads of a warp walk the same layer sequence. All flavours are propagated
  together as the columns of an N x N evolution matrix.
* **Earth model.** `opg::PremModel` reproduces OscProb's `PremModel` (same
  tables, detector-layer insertion, `FillPath`). Paths are generated on the fly
  on the device from the layer table.
* **Multi-GPU.** cos-zenith rows (or bins) are dealt round-robin across devices
  and gathered with strided copies into a pinned buffer.

Result layouts are structure-of-arrays:
`probs[nubar][a][b][iC][iE]`, `binned[nubar][a][b][iCbin][iEbin]`,
`prob_points -> [a][b][i]`, `prob_path -> [a][b][iE]`, with `P(a -> b)` and
flavour indices 0 = e, 1 = μ, 2 = τ, 3 = s.

### Numerical methods

| Model | Per-segment evolution |
|-------|-----------------------|
| Fast, NSI, NUNM, SNSI, LIV | Kopp's `zheevh3` (Cardano + QL fallback), ported bit-exactly; vacuum (ρ < 1e-6) uses the analytic PMNS eigensystem as OscProb (not for LIV, whose terms persist in vacuum) |
| Sterile (4x4) | cyclic complex Jacobi with Numerical Recipes' stable update, fully unrolled (register-resident) |
| Decay | `exp(-iHL)` by scaling and squaring with Padé degree 3–13 and Eigen's thresholds (Higham 2005), LU with partial pivoting |

### Bin averaging

`set_bins()` averages over each (E, cos zenith) bin with a tensor-product
Gauss–Legendre rule:

* **Energy:** `n_gl_energy` nodes per bin, uniform in E (`linear`), log E
  (`log`) or 1/E (`invE`, uniform in L/E at fixed baseline — the analogue of
  OscProb's `AvgProbLoE`). Converges exponentially once the nodes resolve the
  oscillation phase across the bin (rule of thumb: ≥ 3 nodes per period).
* **cos zenith:** with plain GL the convergence stalls at ~1e-3, because the
  path length in each Earth layer behaves like √(cosZ − c_k) at the cosines
  c_k where trajectories graze a layer boundary (42 such points for the
  default PREM). The default `cosz_rule="layer"` splits every bin at those
  cosines and applies GL in u with cosZ = a + (b − a)(1 − cos πu)/2 on each
  piece, which restores exponential convergence. Use ≥ 6 nodes per piece.

Measured on the reference bins (`tests/test_averaging.cpp`), error vs order:

| nodes | E only | cosZ plain | cosZ layer-adapted |
|------:|-------:|-----------:|-------------------:|
| 4     | 2.9e-2 | 9.3e-4     | 2.3e-3 |
| 8     | 4.4e-3 | 1.0e-3     | 1.6e-6 |
| 12    | 6.9e-6 | 2.4e-4     | 3.1e-10 |
| 16    | 1.1e-9 | 8.1e-4     | 3.1e-14 |

(The E column is dominated by a 1–1.5 GeV core-crossing bin spanning ~2
oscillation periods.) `avg_path()` provides 1D averages for fixed baselines.

### Analytic averages (fixed paths)

`avg_path_analytic()` and `avg_path_analytic_grad()` (Fast, NSI, Sterile;
`include/opg/avg/analytic.h`) average P(a → b) over bins of E or L/E for a
fixed path without resolving the oscillations, after OscProb's
`PMNS_Maltoni` (arXiv:2308.00037). To first order in u = 1/E around a
sub-bin centre, S(u₀ + δ) = S(u₀) exp(−iKδ) with K = Σ_seg T† ∫₀^L e^{iHs} (dH/du)
e^{−iHs} ds T, so in the eigenbasis of K each pair of terms is damped by
sinc((μₙ − μₘ)h). This is exact in vacuum however many periods fit in a bin;
in matter the error is second order in the sub-bin width. Bins are split
into geometric sub-bins (u_hi/u_lo = 1 + r) with r set by the binning, the
path and `tol` only (never by the oscillation parameters), so the averages
are smooth in the parameters:

```cpp
opg::Propagator<opg::Sterile<>> prop;              // CPU (OpenMP), or GPUs with {devices}
prop.set_params(par);
prop.set_gradient_params();                          // th12 ... dm41
std::vector<opg::Segment<double>> fd{{810, 2.84, 0.5, 0}};
std::vector<double> P, dP;                           // P[a][b][bin], dP[p][a][b][bin]
prop.avg_path_analytic_grad(E_edges, fd, /*nubar=*/false, P, dP,
                            opg::EMeasure::InvE,     // uniform in L/E (or Linear, Log)
                            opg::BinVar::E);         // edges in E (or LoE, km/GeV)
```

* Measures: uniform in E (`Linear`, the default as for `avg_path`), log E or
  1/E (`InvE`, uniform in L/E); the measure's weight enters each sub-bin
  exactly to first order.
* `AnalyticAvgOptions`: `tol` (default 1e-6; sub-bin width
  r = √(tol / (0.005 Σ V L)), V the matter potential at Z/A = 1, capped at
  `max_width` = 0.1; the observed error is ≈ 0.5–1 × tol and the cost grows as
  1/√tol), `nsub` (fixed sub-bins uniform in 1/E instead), `threads` (OpenMP).
  `avg_path_analytic_subbins()` takes explicit sub-bins (e.g. OscProb's).
* Pairs whose phase sweeps more than 10⁶ rad across a sub-bin fade out
  smoothly (C² in log of the sweep) and are dropped beyond 10⁷ rad (their
  average is below 10⁻⁶ and their phases are not resolved in double
  precision), so fully fast pairs reach the incoherent limit smoothly, e.g.
  in a bin reaching L/E ~ 10⁸ km/GeV.
* Gradients are exact derivatives of the computed averages (forward mode
  through both eigen-decompositions, written with divided differences of
  smooth functions, so they stay accurate at degenerate eigenvalues), for
  the model parameters and the Earth `zoa_<t>`/`rho_<t>`. The sub-bins depend
  on the path's densities: when fitting a density scale, fix them with
  `analytic_subbins()` at the nominal path and `avg_path_analytic_subbins*()`.
* Validation (`tests/test_analytic.cpp`, `tests/validate_analytic.cpp`): with
  OscProb's own sub-bins it reproduces `PMNS_Maltoni::AvgProbLoE` to 8e-14;
  NOvA-like 3+1 configurations are in the table below.

NOvA-like 3+1 (FD: 810 km, ND: 1 km, ρ = 2.84 g/cm³), max |ΔP̄| over all 16
channels and bins, ν and ν̄, both orderings, θ₁₄ ∈ {0, 0.1}, against a
converged reference uniform in L/E (FD: 150 bins, 148 of them uniform in 1/E
between 0.3 and 44.7 GeV, plus the bin [10⁻⁵, 0.3] GeV reaching L/E =
8.1·10⁷ km/GeV; ND: 400 log bins in L/E ∈ [0.005, 5] km/GeV):

| Δm²₄₁ (eV²) | FD, tol 1e-6 | FD clamped bin | ND | FD, tol 1e-8 | FD, GL 5 nodes | FD, GL 20 nodes |
|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 9.6e-7 | 3.2e-9 | 7e-15 | 1.0e-8 | 2e-15 | 1e-15 |
| 0.1   | 8.6e-7 | 3.4e-9 | 1.8e-10 | 1.2e-8 | 4.3e-7 | 2e-14 |
| 1     | 6.2e-7 | 3.4e-9 | 1.9e-9 | 6.8e-9 | 0.18 | 4.9e-8 |
| 3–100 | 6.2e-7 | 3.4e-9 | 1.9e-9 | 6.8e-9 | 0.06–0.23 | 0.06–0.14 |

(GL: `avg_path` with nodes uniform in 1/E, excluding the clamped bin.)
Gradients agree with 4-point long-double finite differences to ≤ 1e-9
relative (≤ 3e-4 in the clamped bin at Δm²₄₁ = 100 eV², where the tiny
Δm² steps the finite differences need there are limited by round-off).
Time per call (16 channels; 151 FD bins → 1163 sub-bins, 400 ND bins → 400),
one thread / 14 threads of an Intel Core Ultra 5 225H: FD 7.9 / 1.3 ms,
with 12 gradients 68 / 8.7 ms; ND 2.5 / 0.55 ms, with gradients 24 / 3.2 ms
(FD at tol 1e-8: 73 / 8.8 ms, with gradients 650 / 73 ms). With the CUDA
backend the sub-bins run on the GPUs (one thread per sub-bin for the values,
one per sub-bin and gradient pass for the derivatives; the per-bin
reduction stays on the host, in a fixed order; `AnalyticAvgOptions::host`
forces the host): on one V100 (shared with another job) FD 0.45 ms, with
gradients 3.5 ms; ND 0.30 / 2.7 ms; FD at tol 1e-8: 2.9 / 20 ms. GPU and host
agree to ≤ 1e-10 in P̄ (phases up to 10⁶ rad at Δm²₄₁ = 100 eV²; ~1e-13 at
Δm²₄₁ ≤ 1 eV²) and 1e-9 relative in dP̄; in the clamped bin, where phases
reach 10¹⁰ rad, to 5e-11 and 1e-6.

## Absorption

`OscProb::Absorption` is available as flavour- and model-independent
transmission factors (bit-identical to `Absorption::Trans`):

```python
e = opg.PremModel()
T = e.transmission(cosZ, xsec)          # P(no interaction); xsec in cm^2/nucleon
X = e.column_depth(cosZ)                 # sum rho L in g/cm^2: T = exp(-X xsec / u)
T = opg.path_transmission(path, xsec)    # fixed path
```

C++: `opg::transmission(path, xsec)`, `opg::column_depth(path)`
(`opg/earth/absorption.h`), `Propagator::transmission(cosZ, xsec)` and
`column_depth(cosZ)`.

## Gradients

Exact derivatives dP/dp with respect to the model parameters, on CPU and GPU.
**Status:** available for all models (`Fast`, `NSI`, `NUNM`, `Sterile`,
`Decay`), for grids, event lists, fixed paths and bin averages, with respect to
the model parameters and the Earth model's Z/A per layer type (see
[docs/ROADMAP.md](docs/ROADMAP.md) for what is next). Parameters
(`parameter_names`):

| Model | Parameters |
|-------|------------|
| Fast    | `th12 th13 th23 d13 dm21 dm31` |
| NSI     | mixing, `eps_ee eps_emu eps_etau eps_mumu eps_mutau eps_tautau` (magnitudes as in `set_eps`), `ph_emu ph_etau ph_mutau`, `coup_e coup_u coup_d` |
| NUNM    | mixing, `alpha_ee alpha_mue alpha_taue alpha_mumu alpha_taumu alpha_tautau` (values as in `set_alpha`, diagonal = 1 + value), `ph_mue ph_taue ph_taumu`, `frac_vnc` |
| Sterile | `th12 th13 th23 th14 th24 th34 d13 d14 d24 dm21 dm31 dm41` |
| Decay   | mixing, `alpha2 alpha3` (eV²; at α = 0 the derivative is the one-sided one from α > 0) |
| LIV     | mixing, `aT<d>_<ab>` / `cT<d>_<ab>` magnitudes and `ph_aT<d>_<ab>` / `ph_cT<d>_<ab>` phases for d = 3..8 (60 in total; default: mixing, aT3, cT4) |
| SNSI    | NSI's, plus `mlight` (lightest mass, eV; one-sided at 0); default as NSI |
| SiderealLIV | mixing and, per flavour pair `<ab>`, `aX aY aZ cXX cYY cXY cXZ cYZ` (`aX_emu`, ...) |
| Deco    | mixing, `gamma21 gamma31` (GeV), `deco_angle`, `deco_power` (default: all but `deco_power`). Where the Γ₃₂ square-root argument vanishes (e.g. Γ₂₁ = 0) its derivative is taken as 0, so gradients stay finite at Γ = 0 |
| OQS     | mixing, `a1` .. `a8`, the angles `ang<i><j>` (1 ≤ i < j ≤ 8, e.g. `ang38`), `power` (default: mixing and `a1` .. `a8`). The 8x8 exponential is differentiated in dual arithmetic through the Padé approximant |

Earth parameters (`earth_parameter_names`, per propagator), for each layer
type t of the Earth model: `zoa_<t>`, the Z/A of all layers of that type
(`PremModel::SetLayerZoA`), and `rho_<t>`, a common relative scale of the
densities of all layers of that type: the gradient is dP/d ln ρ_t, i.e. the
change per unit fractional change of those densities (`PremModel::ScaleLayerDensity`
applies such a scale), so a 2% density uncertainty on layer type t enters a
fit as 0.02 × dP/d(rho_t). For fixed paths they apply to segments with
`layer == t`. `gradient_parameter_names` lists model and Earth parameters.

`set_gradient_params()` without arguments selects `default_gradient_params`:
all parameters except NSI's fermion couplings `coup_e coup_u coup_d`, which
are rarely fitted and must be named explicitly (as must the Earth Z/A
parameters). Phases are differentiated at
fixed magnitude, so derivatives are well defined
at zero couplings. In the NUNM high-scale scenario the derivatives include the
row normalisation of α.

Gradients are **off unless requested**: probability-only calls run the same
code as before and are unaffected, and gradient buffers are only allocated when
gradients are computed. `-DOPG_ENABLE_GRADIENTS=OFF` compiles them out.

```python
p = opg.Fast(devices=[0])
print(opg.Fast.parameter_names)   # ['th12', 'th13', 'th23', 'd13', 'dm21', 'dm31']
p.set_gradient_params(["th23", "dm31", "d13"])   # [] turns gradients off;
                                                 # no argument: defaults
p.set_grid(E, cosZ)

p.calculate_gradient()            # probabilities + gradients
P, G = p.probs(), p.grad()        # G[nubar, p, a, b, iC, iE] = dP(a->b)/dp

# Weighted mode: only sum(w * dP/dp) is formed on the device
g = p.weighted_gradient(w)        # w has the shape of probs(); g[p]
                                  # w may be a CUDA array (CuPy, PyTorch, ...)
                                  # on the propagator's GPU: no upload
d = p.device_probs()              # zero-copy DeviceArray of the GPU results

P, dP = p.prob_points_grad(E_ev, cosZ_ev, nubar_ev)   # dP[p, a, b, i]
g     = p.weighted_gradient_points(E_ev, cosZ_ev, nubar_ev, w_ev)
P, dP = p.prob_path_grad(E, path)                     # fixed baseline

# Bin averages (same Gauss-Legendre nodes and weights as binned())
p.set_bins(E_edges, cosZ_edges, 4, 4)
p.calculate_binned_gradient()
A, GA = p.binned(), p.binned_grad()   # GA[nubar, p, a, b, iCbin, iEbin]
g = p.weighted_gradient_binned(w_bins)               # w_bins: shape of binned()
A, dA = p.avg_path_grad(E_edges, 5, path)            # 1D averages, fixed path

# One gradient vector per analysis bin (e.g. dN_b/dp for a Poisson likelihood)
G = p.weighted_gradient_points_binned(E_ev, cosZ_ev, nubar_ev, w_ev, bin_ev, nbins)

p.set_gradient_params(["dm31", "zoa_0", "zoa_1"])   # inner/outer core Z/A
```

C++: `set_gradient_params()`, `calculate(Flavor, /*gradient=*/true)`, `grad()`,
`weighted_gradient()`, `prob_points_grad()`, `weighted_gradient_points()`,
`weighted_gradient_points_binned()`, `prob_path_grad()`,
`calculate_binned(Flavor, true)`, `binned_grad()`, `weighted_gradient_binned()`,
`avg_path_grad()`, `earth_parameter_names()`, and
`weighted_gradient_device()` / `weighted_gradient_binned_device()` (weights in
GPU memory, one pointer per device in the layout of `device_probs(k)` /
`device_binned(k)`) on `opg::Propagator`.

**Weighted mode.** A fit needs the gradient of a scalar such as χ², not every
dP/dp: dχ²/dp = Σ w · dP/dp with w = ∂χ²/∂P (e.g. flux × cross-section ×
exposure × ∂χ²/∂N for the bin of each point). Pass w, get one number per
parameter. For a 10⁶-point grid this avoids storing and copying
2 × N_par × 9 × 10⁶ derivatives (864 MB for 6 parameters). The sums are formed
in a fixed order, so results are reproducible run to run.

**Method.** Parameter dependence enters through the host-side prepared state,
which is computed with forward-mode dual numbers (`opg::Dual`); the value parts
are bit-identical to the plain computation. On each segment the value
eigensystem is the usual one, and the derivative of U = exp(−iHL) follows from
the Daleckii–Krein formula dU = V[(V†dH V) ∘ Γ]V†, with divided differences
Γ evaluated stably for any eigenvalue separation. Derivatives are therefore
exact (no step sizes) and well behaved at zero mixing angles and near
degeneracies. For Decay (non-hermitian H) the values come from the Padé
exponential as in the probability-only code, and the derivative from the same
formula with the complex eigensystem H = X diag(λ) X⁻¹ (cubic roots refined by
Newton steps, eigenvectors from cross products). Each segment checks that
X diag(e^{−iλL}) X⁻¹ reproduces the Padé operator to 1e-11; otherwise
(degenerate or ill-conditioned eigenvectors) it falls back to differentiating
the Padé exponential in dual arithmetic. Over 1.2M PREM segments of the test
parameter sets the fallback never triggered.

**Reverse mode (OQS).** For OQS the weighted modes (`weighted_gradient`,
`weighted_gradient_binned`, `weighted_gradient_points(_binned)`, and the
device-resident variants) run in reverse mode: a forward pass with
checkpoints every 16 segments, then a backward pass whose segment cotangent
G = L(Mᵀ, Σₐ λₐRₐᵀ) (adjoint Fréchet derivative of the 8x8 exponential, one
dual exponential per segment) is contracted with ∂M/∂p for every parameter.
The cost no longer grows with the number of parameters (43 parameters on a
500 x 500 grid, V100: 1.6 s instead of 49 s; CPU 10x). It agrees with forward
mode to ~1e-13 and is used for paths up to 256 segments and up to 96
parameters (else forward mode); `set_adjoint_gradients(false)` (Python:
`p.adjoint_gradients = False`) forces forward mode.

**Reverse mode (hermitian models).** Fast, NSI, Sterile, LIV, SNSI and
SiderealLIV use the same reverse-mode weighted gradients. The backward step
is shared: from the segment's eigensystem it gives the cotangent H̄ of the
hamiltonian (the adjoint of the eigen-decomposition derivative), and the
parameters are obtained by contracting H̄ with ∂H/∂p. Where the hamiltonian
is affine in the segment's density and density · Z/A (Fast, NSI, LIV,
SiderealLIV), the cotangents are first summed over the path into
Σ H̄, Σ ρH̄ and Σ ρ(Z/A)H̄, so the model parameters are contracted once per
point instead of once per segment. Sterile contracts per segment: the sums
lose about a digit at Δm²41 ~ 1 eV² for ~12% speed. Agreement with forward
mode is ≤ 5e-13 (Sterile ≤ 3.2e-12).

Parameters are processed in
passes of K directions (K = 2; K = 1 for Sterile and Deco, whose kernels
would otherwise spill registers; K = 8 for Decay, whose passes share the value
exponential and the eigensystem).

**Validation** (`tests/test_gradients.cpp`, `tests/gpu/test_gradients_gpu.cpp`,
`python/tests/test_gradients.py`): against 6-point central differences
computed entirely in long double, for every parameter of every model, the
maximum error relative to the largest derivative of each parameter is
≤ 1e-11 for generic parameters and ≤ 2.4e-10 at degenerate points (θ13 = 0,
θ12 = 0, α = 0, zero sterile mixing) and for Sterile at Δm²41 = 1.3 eV², where
the long-double reference itself is limited by phases of ~1e5 rad (PREM event
lists, test path and vacuum, ν and ν̄; GPU and CPU; test points include zero
NSI couplings and phases at π, unitary α in both NUNM scenarios, and
Δm²41 ≈ Δm²31). Probabilities are bit-identical with gradients on or off; GPU
and CPU gradients agree to 1e-13; grid, event-list and weighted modes are
consistent; multi-GPU results equal single-GPU results.

**Cost** (1000 × 1000 grid, ν and ν̄, all parameters, one V100, relative to
a probability-only evaluation of the same model; two GPUs halve the times):

| Model | Parameters | P + full gradients | Weighted mode | per parameter |
|-------|-----------:|-------------------:|--------------:|--------------:|
| Fast    | 6  | 10.5x (0.26 s) | 10.8x (0.28 s; 13x forward) | 1.8 |
| NSI     | 18 | 33x (0.81 s)   | 11x (0.28 s; 35x forward) | 1.8 |
| NUNM    | 16 | 32x (0.85 s)   | 36x (0.95 s)   | 2.0 |
| Sterile | 12 | 18x (3.0 s)    | 12x (2.0 s; 19x forward) | 1.5 |
| Decay   | 8  | 17x (1.15 s)   | 18x (1.24 s)   | 2.1 |
| LIV     | 60 | 123x (3.5 s)   | 20x (0.59 s; 122x forward) | 2.1 |
| SNSI    | 19 | 42x (1.1 s)    | 32x (0.87 s; 44x forward) | 2.2 |
| SiderealLIV | 54 | 104x (2.9 s) | 17x (0.49 s; 105x forward) | 1.9 |
| Deco    | 10 | 54x (2.8 s)    | 56x (2.9 s)    | 5.4 |
| OQS     | 43 | 377x (180 s)   | 13x (reverse mode; 378x forward) | 8.8 (weighted: 0.3) |

The weighted times include uploading the weights. On the CPU the factors are
9x, 28x, 28x, 17x and 6x. This is about 1.5–2 probability evaluations per
parameter: comparable to central finite differences (2 per parameter), but
exact. Deco (5.4) and OQS (8.8; 3.3 on the CPU) are the exceptions: Deco
carries three density matrices and their derivatives; OQS differentiates an
8x8 matrix exponential per segment in dual arithmetic, whose GPU kernels spill
~17 KB of registers. The weighted modes of OQS and the hermitian models use
reverse mode (see above), whose cost grows only weakly with the number of
parameters: ~13 probability evaluations in total for all 43 OQS parameters,
~20 for all 60 LIV parameters. NUNM, Decay and Deco use forward mode.

## Validation

The tests compare against reference data from the original OscProb: point
`ProbMatrix` values (cache disabled), mass matrices, PREM paths, brute-force bin
averages and `AvgProb` values (≈ 39 MB, not committed). Download it into
`tests/data/` with

```sh
scripts/fetch_test_data.sh          # release asset testdata-v2, checked against tests/data/SHA256SUMS
```

or regenerate it with `reference/make_reference.sh`, which builds OscProb
(needs ROOT; the published data is from OscProb v2.4.0-5-g5f2d719) and runs
`reference/dump_reference.cxx`. Parameter sets follow `OscProb/test/Utils.h`
plus variants with complex phases.

Maximum |ΔP| against OscProb (all channels, ν and ν̄; test path, vacuum and a
101 x 120 PREM grid):

| Model | CPU | GPU (V100) | Notes |
|-------|-----|------------|-------|
| Fast, NSI, NUNM (scale 0) | 1.8e-13 (**0** with `OPG_OSCPROB_BITWISE`) | 1.8e-13 | |
| NUNM (scale 1) | 4e-14 | 5e-14 | see note below |
| Sterile 3+1 | 1.1e-11 | 1.1e-11 | dominated by round-off amplified by phases ~1e4 rad at Δm²₄₁ = 1.3 eV²; against a long-double calculation ours is 3–4x closer than OscProb in 3 of 4 test cases and comparable in the 4th |
| Decay | 1e-14 | 1.6e-14 | |
| LIV | 7.9e-12 (**0** with `OPG_OSCPROB_BITWISE`) | 1e-11 | test values with large LIV phases (aT ~ 1e-21 GeV over the Earth) |
| SNSI | 1.2e-13 (**0** with `OPG_OSCPROB_BITWISE`) | 1.5e-13 | the default build drops the common m₁²/2E term before squaring (more accurate than OscProb's form) |
| Deco | 9e-14 (**0** with `OPG_OSCPROB_BITWISE`) | 9e-14 | |
| SiderealLIV | 1.7e-13 (**0** with `OPG_OSCPROB_BITWISE`) | 1.8e-13 | as OscProb: no sidereal terms in vacuum (ρ < 1e-6), and the cT terms enter as E[GeV]·cT without the GeV → eV factor of the aT terms |
| OQS | 2.9e-15 | 4.3e-14 | the 8x8 exponential follows Eigen's MatrixExponential (same Padé degrees and scaling), with a different summation order |

Other checks: the ported Kopp eigensolver is bit-identical to OscProb's
`MatrixDecomp`; `Hms` and the Decay effective mass matrix are bit-identical;
PREM paths are bit-identical; `expm` agrees with Eigen to 1e-15; multi-GPU
results are bit-identical to single-GPU; bin averages agree with a 400²
brute-force midpoint average to 1.5e-6 and with OscProb's `AvgProb` to 4e-4
(within its 1e-4-target approximation).

## Differences from OscProb

* **No eigensystem cache.** On the GPU recomputation is cheaper. (OscProb's
  cache also rescales cached eigenvalues by E_cached/E, which differs from a
  fresh computation at the ulp level.)
* **Rescaled 3x3 eigensolver.** OscProb calls Kopp's `zheevh3` on the raw
  Hamiltonian in eV (~1e-12); its error test, designed for O(1) matrices,
  then *always* takes the iterative QL fallback. We rescale H to O(1) first
  (2.2x faster on V100; P agrees to ~1e-13). `OPG_OSCPROB_BITWISE=ON` restores
  OscProb's exact call.
* **NUNM high-scale normalisation** is applied once when parameters are set.
  OscProb normalises α inside `PropagatePath`, after the initial state has
  already been rotated by the *unnormalised* α, so the first call on a fresh
  `PMNS_NUNM(1)` object differs from subsequent calls (and α is renormalised
  at each segment). The reference values are taken after a warm-up call.
* **Sterile** uses a Jacobi eigensolver instead of Eigen's QR, and **Decay** a
  port of Eigen's matrix exponential algorithm, so results agree to round-off
  rather than bit for bit.
* **Averaging:** `set_bins()`/`avg_path()` use the Gauss–Legendre scheme
  above. Maltoni's expansion (OscProb's default for Fast and Sterile) is
  available for fixed paths as `avg_path_analytic()`, with sub-bins chosen
  from the binning rather than OscProb's `GetSamplePointsAvgClass`, gradients,
  and fading of fully fast pairs. With OscProb's sub-bins it agrees with
  `AvgProbLoE` to 8e-14. (OscProb returns NaN from the first `AvgProbLoE` call
  after `SetPath`; later calls are fine.)
* Only spherical-shell Earth models (`PremModel`); `EarthModelBinned` and
  OscProb's other models are not ported.

## Performance

Grid of 1000 x 1000 (E, cosZ) points for ν and ν̄ (2·10⁶ points), PREM 44
layers, cosZ ∈ [−1, 0], double precision unless noted. `bench/oscillogram`
and `reference/bench_oscprob.cxx`.

| Model | OscProb, 1 thread¹ (cache on) | OscProbGPU CPU, 1 thread¹ | 1 x V100 | 2 x V100 | 1 V100 vs OscProb |
|-------|------:|------:|------:|------:|------:|
| Fast    | 5.0e4 /s | 9.6e4 /s | 7.7e7 /s | 1.5e8 /s | ~1500x |
| Fast (float) | — | — | 1.9e8 /s | 3.5e8 /s | |
| NSI     | 5.0e4 /s | 9.5e4 /s | 8.2e7 /s | 1.5e8 /s | ~1600x |
| NUNM    | 4.9e4 /s | 9.0e4 /s | 7.6e7 /s | 1.5e8 /s | ~1500x |
| Sterile | 3.2e4 /s | 1.9e4 /s | 1.2e7 /s | 2.4e7 /s | ~390x |
| Decay   | 1.4e4 /s | 6.8e4 /s | 2.9e7 /s | 5.8e7 /s | ~2100x |
| LIV     | | | 7.2e7 /s | | |
| SNSI    | | | 7.6e7 /s | | |
| SiderealLIV | | | 7.2e7 /s | | |
| Deco    | | | 3.8e7 /s | | |
| OQS     | | | 4.1e6 /s | | |

¹ Intel Core Ultra 5 225H (a different machine from the GPU host), so the last
column is indicative. GPU times exclude the one-off set-up (context creation,
uploads) and the device-to-host copy of the 144 MB result (~12 ms).

Binned: 40 x 20 bins with 8 nodes per direction (and per layer piece):
5.4 ms per evaluation on 1 V100, 3.4 ms on 2 (4.3 s single-thread CPU).

Gradients (Fast, 6 parameters, 1 V100; 2 V100 in brackets; `opg_bench ... grad*`):

| Mode | Probabilities | + gradients | Weighted |
|------|------:|------:|------:|
| Grid 1000 x 1000, ν and ν̄ | 24 ms | 0.26 s (10.5x) | 0.29 s |
| Binned 40 x 20, 8 GL nodes | 5.4 ms (3.4) | 56 ms (35) | 56 ms (35) |
| Event list, 10⁶ events | 0.12 s (0.09) | — | 0.46 s (0.28); per analysis bin (800 bins) 0.48 s (0.31) |

Event-list times include uploading the events. With weights already on the
GPU (`weighted_gradient_device`, or CUDA arrays from Python) the grid weighted
gradient takes 0.26 s on 1 V100 (0.13 s on 2) instead of 0.29 s (0.17 s). Other models: see
[Gradients](#gradients). The `gradient_fit.py` example (1200 bins, 4 GL nodes
per direction, 4 parameters including the outer-core Z/A) converges in 6
Levenberg–Marquardt iterations in 0.06 s on one V100.

## Repository layout

```
include/opg/core      Complex, Mat/Vec, constants, macros
include/opg/linalg    zheevh3 (Kopp, LGPL), Jacobi, expm, general 3x3 eigensystem
include/opg/physics   mixing (BuildHms port), propagation, gradients
include/opg/models    fast, nsi, nunm, sterile, decay
include/opg/earth     PremModel and on-device path walker
include/opg/avg       Gauss-Legendre, analytic (Maltoni) averages
include/opg/propagator.h, engine*.h   user API and CPU backend
src/                  CUDA backend
python/               nanobind module, tests, examples
docs/ROADMAP.md       status and plans
reference/            OscProb reference generator and benchmark
tests/                doctest suites (cpu, gpu) and reference data
```

## Licensing

MIT (see `LICENSE`), including portions derived from OscProb (MIT). The 3x3
Hermitian eigensolvers in `include/opg/linalg/kopp` are ported from Joachim
Kopp's code and remain under LGPL-2.1 (see the `COPYING` file there). Earth
model tables in `data/` come from OscProb. nanobind (BSD-3) and doctest (MIT)
are vendored under `third_party/`.
