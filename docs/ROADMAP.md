# Roadmap

Status as of 2026-10-02: all five models (Fast, NSI, NUNM, Sterile 3+1, Decay)
validated against OscProb on CPU and GPU; bin averaging; multi-GPU; Python
bindings. Gradients (G0–G2 + weighted mode) are done for `Fast` only.

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
* Validation: `tests/gradients.h` (long-double 4-point central differences;
  metric = max error / max(max|ref|, 1e-3) per parameter).

### G3 — NSI, NUNM, Sterile
1. **NSI**: store ε as (magnitude, phase) in `NSIParams<S>`; build the complex ε
   in `prepare_generic`. Parameters: `eps_ee, eps_emu, eps_etau, eps_mumu,
   eps_mutau, eps_tautau`, phases `ph_emu, ph_etau, ph_mutau`, couplings
   `coup_e, coup_u, coup_d`, plus the 6 mixing ones (18 total). `hamiltonian`
   already generic-friendly; template it on S like Fast.
2. **NUNM**: α as (magnitude, phase) (diagonal entries 1 + value); α⁻¹ and the
   high-scale row normalisation in dual arithmetic in `prepare_generic`
   (inverse3 must become generic). `initial_grad` = derivative of α† (the
   initial state), `finalize_grad`: dS ← α dS + dα S (before `finalize`).
   Prepared + dual prepared exceeds the 4 KB CUDA kernel-argument limit
   (`check_param_size` static_assert): pass the dual prepared state through a
   per-engine device buffer (pointer argument) instead of by value.
   Parameters: 6 α values, 3 α phases, `frac_vnc`, + mixing (16).
3. **Sterile**: `MixingRegistry<4>` (th12..th34, d13, d14, d24, dm21, dm31,
   dm41 = 12); Jacobi stays value-only; `eigen_step_grad<4>`.
   Test at θ14 = θ24 = θ34 = 0.

### G4 — Decay
Dual numbers through `expm` (Padé + LU; pivoting/scaling decisions use values).
Parameters: mixing + `alpha2`, `alpha3`. Test at α = 0 (one-sided domain).
Decay kernels already spill registers; tune `grad_chunk`.

### G5 — Binned / Earth
* Gradients of `set_bins`/`calculate_binned` and `avg_path` (linear in P: reuse
  the GL weights).
* Per-layer-type Z/A parameters (`zoa_<layer>`): seed the segment's zoa in the
  dual hamiltonian when `seg.layer` matches.
* Per-analysis-bin weighted variant (one gradient vector per analysis bin).

### G6 — Python, docs, benchmarks for the new models.

## Other improvements
* Weighted mode: accept weights as a device pointer (avoid the 144 MB upload).
* Decay register spills (also affect probability-only speed).
* Exercise `pip install .` (pyproject/scikit-build-core) on a machine with pip.
* Publish `tests/data` (OscProb references, ~22 MB) as a release asset.

## Working notes
* Local: `cmake -S . -B build -DOPG_ENABLE_CUDA=OFF -DOPG_ENABLE_PYTHON=ON`;
  references: `reference/make_reference.sh ../OscProb` (needs ROOT; set
  `ROOTSYS`/`PATH`/`LD_LIBRARY_PATH` by hand if `thisroot.sh` misbehaves).
* GPU host: `REMOTE_HOST=<host> REMOTE_DIR=<dir> scripts/remote_sync.sh`, then
  build with `CMAKE_CUDA_ARCHITECTURES` set to the card's SM and a CMake ≥ 3.18.
