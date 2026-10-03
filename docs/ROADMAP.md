# Roadmap

Status as of 2026-10-02: all five models (Fast, NSI, NUNM, Sterile 3+1, Decay)
validated against OscProb on CPU and GPU; bin averaging; multi-GPU; Python
bindings. Gradients (G0–G6) are done for all five models, including bin
averages, Earth Z/A and per-analysis-bin weighted modes, with Python bindings,
tests, an example fit (`python/examples/gradient_fit.py`) and benchmarks.

## Gradient phases (all done)

Design (shared by all models; reuse it for new ones):

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

### G6 — Python, docs, benchmarks (done)
Python tests for every model (grid, event list, weighted, binned, default
selection), `gradient_fit.py` (binned likelihood fit with exact Jacobians,
optional core Z/A), `opg_bench` modes `grad_binned` and `grad_points`, README.

## Other improvements
* Done: weighted modes accept device-resident weights
  (`weighted_gradient_device`, `weighted_gradient_binned_device`; Python:
  CUDA arrays via DLPack, `device_probs()`/`device_binned()` views); saves the
  144 MB upload (1000 x 1000 grid: 0.29 → 0.26 s on 1 V100, 0.17 → 0.13 s on 2).
* Done: `pip install .` tested (CPU backend, Python 3.12, numpy 2.5; CUDA via
  pip untested: no pip on the GPU host).
* Investigated, no change: Decay probability kernels spill ≤ 300 bytes; their
  cost is the Padé exponential itself. Exponentials from the complex
  eigensystem (with Padé fallback) were less accurate near degeneracies
  (5e-9 vs 1e-11 in the gradient tests, 1e-13 vs 1e-14 against OscProb) and
  1.8x slower on the CPU, so Decay keeps Padé for values.
* Done: `tests/data` (OscProb references, ~39 MB) published as release
  assets (testdata-v1; testdata-v2 adds OQS); `scripts/fetch_test_data.sh`
  (default testdata-v2) checks them against
  `tests/data/SHA256SUMS`. A new data version needs a new tag and manifest.

## More OscProb calculators
* Done: LIV (`PMNS_LIV`, SME d = 3..8) and SNSI (`PMNS_SNSI`), bit-identical to
  OscProb with `OPG_OSCPROB_BITWISE`, with gradients (Z/A, binned, weighted).
* Done: absorption (`transmission`, `column_depth`; bit-identical to
  Absorption::Trans). Possible extension: apply σ(E) tables on the device
  (needed inside bin averages, where T varies across a bin).
* Done: Deco (`PMNS_Deco`): models may define their own state type
  (`State`, `store_probs`, `store_grads`, `contract_grads`); Deco propagates
  three density matrices, bit-identical to OscProb with
  `OPG_OSCPROB_BITWISE`; gradients from first-order perturbation theory of
  the eigensystem in dual arithmetic.
* Done: SiderealLIV (`PMNS_SiderealLIV`): per-path direction in a custom
  state (Earth paths: zenith from cosZ unless a fixed direction is set),
  bit-identical to OscProb with `OPG_OSCPROB_BITWISE`, with gradients. Note
  (OscProb behaviour, ported as is): the cT terms lack the GeV -> eV factor of
  the aT terms, and the sidereal terms are dropped in vacuum.
* Done (51e2387, 481803c):
  - Deco gradients from an analytic eigenbasis formula instead of dual
    arithmetic, K = 1 (done; V100 2.8 s vs 3.7 s before for 10 params on
    1000 x 1000, 5.4 P-evals/param, was 7.0). Deco sets separate_probs: in
    gradient mode the GPU engine takes the probabilities from the
    probability-only kernels, because the Deco gradient kernels (K = 1 and
    K = 2 alike) contract multiply-adds differently and broke the
    bit-identity of P with gradients on vs off (+1 P-eval per call).
  - Per-event extra inputs for event lists (Model::n_extra; engines and
    Propagator take extra[x * n + i]); SiderealLIV takes azimuth [deg] and
    sidereal time [h] per event; Python extra=None arguments and
    n_event_extra. Done and tested on CPU, GPU and Python (tests/extras.h:
    bit-identical to per-propagator settings on CPU, ~1e-14 on GPU from the
    device sin/cos).
* Done: OQS (`PMNS_OQS`): Gell-Mann 9-vectors per initial flavour, real 8x8
  exponential per segment (`opg::expm` now also takes real `RMat`); OscProb
  references `oqs` (OscProb's test values) and `oqs_full` (all dissipator
  terms, power 1, IO); 2.9e-15 CPU, 4.3e-14 GPU. Gradients: dual Padé, K = 1,
  separate_probs; mixing, a1..a8 (default), 28 angles, power. V100: P-only
  4.1e6 /s (stack 5 KB), gradients 8.8 P-evals/param (stack 17 KB) — a
  candidate for optimisation (fewer live matrices in Padé 13, or an
  eigenbasis/Fréchet formulation).
* Next: EarthModelBinned (uses the per-event azimuth).
* LIV, SNSI, SiderealLIV keep grad_chunk K = 2 (V100: best or within 5% of
  K = 1, 3); Deco uses K = 1 (2.7 s vs 3.3 s at K = 2). Rejected for Deco:
  __noinline__ value helpers shared by step() and step_grad() (bit-identical,
  but P-only 1.3-2.2x slower).
* Not planned: PMNS_Avg (obsolete), PMNS_Iter (approximate solver for Fast),
  PMNS_Maltoni (alternative bin averaging).

## Working notes
* Local: `cmake -S . -B build -DOPG_ENABLE_CUDA=OFF -DOPG_ENABLE_PYTHON=ON`;
  references: `reference/make_reference.sh ../OscProb` (needs ROOT; set
  `ROOTSYS`/`PATH`/`LD_LIBRARY_PATH` by hand if `thisroot.sh` misbehaves).
* GPU host: `REMOTE_HOST=<host> REMOTE_DIR=<dir> scripts/remote_sync.sh`, then
  build with `CMAKE_CUDA_ARCHITECTURES` set to the card's SM and a CMake ≥ 3.18.
