# Roadmap

Status as of 2026-10-02: all five models (Fast, NSI, NUNM, Sterile 3+1, Decay)
validated against OscProb on CPU and GPU; bin averaging; multi-GPU; Python
bindings. Gradients (G0–G4 + weighted mode) are done for all five models;
G5 (binned / Earth) is next.

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
values); `step_grad` uses the plain `expm` for values and the dual one for
derivatives only, so values stay bit-identical even under FMA contraction
differences. A long-double `ExpmTraits` (Padé 13, norm ≤ 1) serves the FD
reference. Parameters: mixing + `alpha2`, `alpha3` (8), tested at α = 0.
Cost: 44x a probability evaluation on V100 (5.4 per parameter, K = 2): the
dual expm spills ~19 KB. Options: reuse the dual values instead of a second
plain expm (~15%, check GPU bit-identity); Daleckii–Krein with a complex
non-hermitian eigensystem and fall back to dual Padé when ill-conditioned.

### G5 — Binned / Earth
* Gradients of `set_bins`/`calculate_binned` and `avg_path` (linear in P: reuse
  the GL weights).
* Per-layer-type Z/A parameters (`zoa_<layer>`): seed the segment's zoa in the
  dual hamiltonian when `seg.layer` matches.
* Per-analysis-bin weighted variant (one gradient vector per analysis bin).

### G6 — Python, docs, benchmarks for the new models.

## Other improvements
* Weighted mode: accept weights as a device pointer (avoid the 144 MB upload).
* Decay register spills (also affect probability-only speed and Decay
  gradients).
* Exercise `pip install .` (pyproject/scikit-build-core) on a machine with pip.
* Publish `tests/data` (OscProb references, ~22 MB) as a release asset.

## Working notes
* Local: `cmake -S . -B build -DOPG_ENABLE_CUDA=OFF -DOPG_ENABLE_PYTHON=ON`;
  references: `reference/make_reference.sh ../OscProb` (needs ROOT; set
  `ROOTSYS`/`PATH`/`LD_LIBRARY_PATH` by hand if `thisroot.sh` misbehaves).
* GPU host: `REMOTE_HOST=<host> REMOTE_DIR=<dir> scripts/remote_sync.sh`, then
  build with `CMAKE_CUDA_ARCHITECTURES` set to the card's SM and a CMake ≥ 3.18.
