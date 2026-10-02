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
phase)`, `NUNM.set_frac_vnc(f)`, `Decay.set_alpha2/3(a)`. All models start
from OscProb's PDG defaults (`set_std_pars()`). See `python/examples/`.

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
ctest --test-dir build                               # labels: cpu, gpu, python
```

Requirements: CMake ≥ 3.18, C++17, optionally CUDA ≥ 11 (tested with 11.8 on
V100) and Python ≥ 3.8 with numpy. nanobind (BSD) and doctest (MIT) are
vendored in `third_party/`. With the module built, use
`PYTHONPATH=build/python`. A `pyproject.toml` (scikit-build-core) is provided
for `pip install .` (not yet exercised: the development machines had no pip;
the CMake build of the module is what the test suite uses).

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
| Fast, NSI, NUNM | Kopp's `zheevh3` (Cardano + QL fallback), ported bit-exactly; vacuum (ρ < 1e-6) uses the analytic PMNS eigensystem as OscProb |
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

## Gradients

Exact derivatives dP/dp with respect to the model parameters, on CPU and GPU.
**Status:** available for `Fast` (θ12, θ13, θ23, δ13, Δm²21, Δm²31); NSI, NUNM,
Sterile and Decay, bin-averaged gradients and Earth Z/A parameters are planned.

Gradients are **off unless requested**: probability-only calls run the same
code as before and are unaffected, and gradient buffers are only allocated when
gradients are computed. `-DOPG_ENABLE_GRADIENTS=OFF` compiles them out.

```python
p = opg.Fast(devices=[0])
print(opg.Fast.parameter_names)   # ['th12', 'th13', 'th23', 'd13', 'dm21', 'dm31']
p.set_gradient_params(["th23", "dm31", "d13"])   # [] turns gradients off
p.set_grid(E, cosZ)

p.calculate_gradient()            # probabilities + gradients
P, G = p.probs(), p.grad()        # G[nubar, p, a, b, iC, iE] = dP(a->b)/dp

# Weighted mode: only sum(w * dP/dp) is formed on the device
g = p.weighted_gradient(w)        # w has the shape of probs(); g[p]

P, dP = p.prob_points_grad(E_ev, cosZ_ev, nubar_ev)   # dP[p, a, b, i]
g     = p.weighted_gradient_points(E_ev, cosZ_ev, nubar_ev, w_ev)
P, dP = p.prob_path_grad(E, path)                     # fixed baseline
```

C++: `set_gradient_params()`, `calculate(Flavor, /*gradient=*/true)`, `grad()`,
`weighted_gradient()`, `prob_points_grad()`, `weighted_gradient_points()`,
`prob_path_grad()` on `opg::Propagator`.

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
degeneracies. Parameters are processed in passes of K = 2 directions.

**Validation** (`tests/test_gradients.cpp`, `tests/gpu/test_gradients_gpu.cpp`,
`python/tests/test_gradients.py`): against 4-point central differences
computed entirely in long double, the maximum error relative to the largest
derivative of each parameter is ≤ 6e-12 for generic parameters and ≤ 2.4e-10
at θ13 = 0 or θ12 = 0 (PREM event lists, test path and vacuum, ν and ν̄; GPU
and CPU). Probabilities are bit-identical with gradients on or off; GPU and CPU
gradients agree to 4e-14; grid, event-list and weighted modes are consistent;
multi-GPU results equal single-GPU results.

**Cost** (1000 × 1000 grid, ν and ν̄, all 6 parameters, relative to a
probability-only evaluation): ~10.5x on one V100 (0.25 s; 0.13 s on two),
weighted mode ~12.6x including the upload of the weights (0.31 s; 0.19 s on two);
~5.7x on the CPU. This is about 1.6 probability evaluations per parameter,
cheaper than central finite differences (2 per parameter) and exact.

## Validation

`reference/make_reference.sh` builds the original OscProb (needs ROOT) and dumps
point `ProbMatrix` values (cache disabled), mass matrices, PREM paths,
brute-force bin averages and `AvgProb` values into `tests/data/` (≈ 22 MB, not
committed). Parameter sets follow `OscProb/test/Utils.h` plus variants with
complex phases.

Maximum |ΔP| against OscProb (all channels, ν and ν̄; test path, vacuum and a
101 x 120 PREM grid):

| Model | CPU | GPU (V100) | Notes |
|-------|-----|------------|-------|
| Fast, NSI, NUNM (scale 0) | 1.8e-13 (**0** with `OPG_OSCPROB_BITWISE`) | 1.8e-13 | |
| NUNM (scale 1) | 4e-14 | 5e-14 | see note below |
| Sterile 3+1 | 1.1e-11 | 1.1e-11 | dominated by round-off amplified by phases ~1e4 rad at Δm²₄₁ = 1.3 eV²; against a long-double calculation ours is 3–4x closer than OscProb in 3 of 4 test cases and comparable in the 4th |
| Decay | 1e-14 | 1.6e-14 | |

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
* **Averaging** is the Gauss–Legendre scheme above, not Maltoni's Taylor
  expansion.
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

¹ Intel Core Ultra 5 225H (a different machine from the GPU host), so the last
column is indicative. GPU times exclude the one-off set-up (context creation,
uploads) and the device-to-host copy of the 144 MB result (~12 ms).

Binned: 40 x 20 bins with 8 nodes per direction (and per layer piece):
5.4 ms per evaluation on 1 V100, 3.4 ms on 2 (4.3 s single-thread CPU).

## Repository layout

```
include/opg/core      Complex, Mat/Vec, constants, macros
include/opg/linalg    zheevh3 (Kopp, LGPL), Jacobi, expm
include/opg/physics   mixing (BuildHms port), propagation
include/opg/models    fast, nsi, nunm, sterile, decay
include/opg/earth     PremModel and on-device path walker
include/opg/avg       Gauss-Legendre
include/opg/propagator.h, engine*.h   user API and CPU backend
src/                  CUDA backend
python/               nanobind module, tests, examples
reference/            OscProb reference generator and benchmark
tests/                doctest suites (cpu, gpu) and reference data
```

## Licensing

MIT (see `LICENSE`), including portions derived from OscProb (MIT). The 3x3
Hermitian eigensolvers in `include/opg/linalg/kopp` are ported from Joachim
Kopp's code and remain under LGPL-2.1 (see the `COPYING` file there). Earth
model tables in `data/` come from OscProb. nanobind (BSD-3) and doctest (MIT)
are vendored under `third_party/`.
