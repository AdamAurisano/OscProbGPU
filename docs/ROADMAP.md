# Roadmap

Status as of 2026-10-02: all five models (Fast, NSI, NUNM, Sterile 3+1, Decay)
validated against OscProb on CPU and GPU; bin averaging; multi-GPU; Python
bindings. Gradients (G0–G5) are done for all five models, including bin
averages, Earth Z/A and per-analysis-bin weighted modes; G6 (Python docs and
benchmarks) is next.

## Gradient plan (remaining)

Design (already implemented for Fast, reuse it):

* Parameters/prepared state are templated on the scalar type at namespace
  scope (`FastParams<S>`, `FastPrepared<S>`); the model aliases
  `ParamsT<S>`/`PreparedT<S>` and provides `param_names()`, `param_ref<S>()`,
  `cast<S>()`, `prepare_generic<S>()`, `grad_chunk`, and the K-templated
  `initial_grad`, `step_grad`, `finalize_grad` (see `include/opg/models/fast.h`).
  `grad_traits<Model>` (`include/opg/physics/grad.h`) detects support; engines
  and `Propagator` need no per-model changes.
* Hermitian models: value eigensystem as usual, derivative via
  `eigen_step_grad` (Daleckii–Krein, `include/opg/physics/eigen_grad.h`), with
  dH from evaluating the model's `hamiltonian()` in `Dual` arithmetic.
* Anything that skips work for exact zeros must use `is_exact_zero()` (checks
  derivative seeds too), or derivatives at zero angles/couplings are lost.
* Value parts must stay bit-identical (tests check probabilities with gradients
  on == off).
* Validation: `tests/gradients.h` (long-double 6-point central differences,
  step 1e-4 for angles/phases/couplings and min(1e-5·|dm|, 3e-8 eV²) for
  splittings; metric = max error / max(max|ref|, floor) per parameter, floor
  1e-3, or 10 eV⁻² for splittings). New models: add a `param_points<Model>()`
  specialisation and instantiate the templated checks.
* Dual prepared states larger than the CUDA kernel-argument budget are passed
  by pointer automatically (`GradArg` in `src/engine_cuda.cu`).

### G3 — NSI, NUNM, Sterile (done)
Parameters: NSI 18 (mixing, `eps_<ab>`, `ph_<ab>`, `coup_<f>`), NUNM 16
(mixing, `alpha_<ab>`, `ph_<ab>`, `frac_vnc`), Sterile 12 (`MixingRegistry<4>`).
ε and α are stored as (magnitude, phase); NSI and NUNM scale 0 remain
bit-identical to OscProb under `OPG_OSCPROB_BITWISE`. NUNM's dual state goes to
the kernels by pointer (4 KB limit). `grad_chunk`: NSI/NUNM K = 2, Sterile
K = 1 (K = 2 spills ~5 KB on V100).

### G4 — Decay (done)
`expm` is generic over the scalar type (degree, squarings and pivots from the
values); a long-double `ExpmTraits` (Padé 13, norm ≤ 1) serves the FD
reference. `step_grad` keeps the plain `expm` for values (bit-identical) and
takes derivatives from Daleckii–Krein with the complex eigensystem of H
(`eig3_general.h`, `eigen_grad_general.h`), checked per segment by
reconstructing U; on failure it falls back to the dual `expm`
(`pade_step_grad`, not inlined on the GPU). Parameters: mixing + `alpha2`,
`alpha3` (8), tested at α = 0 and at a degenerate point (fallback). Cost on
V100: 17x a probability evaluation with K = 8 (dual-expm only: 44x).

### G5 — Binned / Earth (done)
* `calculate_binned(which, true)` / `binned_grad()`: node-grid gradients
  reduced with the GL weights; `weighted_gradient_binned()` forms node weights
  w_bin·wC·wE on the fly (nothing stored); `avg_path_grad()`.
* `zoa_<type>` parameters: model Hamiltonians take a segment whose Z/A may be
  dual (`SegmentZ`); each gradient pass (`GradPrepared`) records which
  direction seeds which layer type. Not in the defaults.
* `weighted_gradient_points_binned()`: G[bin][p] for event lists, summed per
  bin in a fixed order (per-event contractions, then one block per bin).

### G6 — Python, docs, benchmarks for the new models.

## Other improvements
* Weighted mode: accept weights as a device pointer (avoid the 144 MB upload).
* Decay register spills (probability-only kernels; the gradient kernels spill
  ~1 KB).
* Exercise `pip install .` (pyproject/scikit-build-core) on a machine with pip.
* Publish `tests/data` (OscProb references, ~22 MB) as a release asset.

## Working notes
* Local: `cmake -S . -B build -DOPG_ENABLE_CUDA=OFF -DOPG_ENABLE_PYTHON=ON`;
  references: `reference/make_reference.sh ../OscProb` (needs ROOT; set
  `ROOTSYS`/`PATH`/`LD_LIBRARY_PATH` by hand if `thisroot.sh` misbehaves).
* GPU host: `REMOTE_HOST=<host> REMOTE_DIR=<dir> scripts/remote_sync.sh`, then
  build with `CMAKE_CUDA_ARCHITECTURES` set to the card's SM and a CMake ≥ 3.18.
