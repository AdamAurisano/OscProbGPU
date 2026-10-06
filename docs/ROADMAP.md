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
  4.1e6 /s (stack 5 KB), gradients 8.8 P-evals/param (stack 17 KB).
* Done: OQS GPU speed. Tried and dropped: Padé 13 with fewer live matrices
  and squarings applied to the vectors (no gain: an 8x8 product per thread
  cannot stay in 255 registers), and 8 lanes per point with warp shuffles
  (microbenchmark: <= 1.3x, 15x slower when groups in a warp diverge).
  Adopted: reverse-mode weighted gradients (physics/adjoint.h; models opt in
  with has_adjoint and adj_* hooks; engines use it for <= 256 segments and
  <= 96 parameters; Propagator::set_adjoint_gradients). V100, 43 params:
  13 P-evals in total instead of 378 (30x); CPU 10x; agrees with forward mode
  to ~1e-13. Candidates: other models (LIV with 60 params, Sterile, Decay).
* Done: Earth density derivatives per layer type, rho_<t> (after the
  zoa_<t>): a common relative scale of the densities of all layers of type t
  (dP/d ln rho_t; PremModel::ScaleLayerDensity). SegmentZ carries a dual
  density as well as Z/A; seed_segment seeds d(rho) = rho. All models, all
  gradient modes (incl. OQS reverse mode); vs long-double FD <= 3.3e-11.
* Done: shared reverse mode for the hermitian models (eigen_step_adj,
  hermitian3_adj_step, OPG_HERMITIAN3_ADJOINT): Fast, NSI, LIV, SNSI,
  SiderealLIV, Sterile. Affine models (H = H0 + rho H1 + rho zoa H2: Fast,
  NSI, LIV, SiderealLIV) sum the segment cotangents and contract the model
  parameters once per point; Sterile does not (a digit lost at dm41 ~ 1 eV^2
  for ~12%). V100 1000 x 1000 weighted, reverse vs forward: Fast 0.28/0.33 s,
  NSI 0.28/0.90, Sterile 2.0/3.2, LIV 0.59/3.6, SNSI 0.87/1.2, SiderealLIV
  0.49/3.0. Agrees with forward mode to <= 5e-13 (Sterile 3.2e-12).
* Next: EarthModelBinned (uses the per-event azimuth).
* LIV, SNSI, SiderealLIV keep grad_chunk K = 2 (V100: best or within 5% of
  K = 1, 3); Deco uses K = 1 (2.7 s vs 3.3 s at K = 2). Rejected for Deco:
  __noinline__ value helpers shared by step() and step_grad() (bit-identical,
  but P-only 1.3-2.2x slower).
* Done (requested for the PISCES NOvA 3+1 fitter): analytic bin averages for
  fixed paths after PMNS_Maltoni (`avg/analytic.h`,
  `Propagator::avg_path_analytic[_grad]`, `avg_path_analytic_subbins[_grad]`):
  first-order expansion of S in 1/E, K matrix per segment from the
  eigensystem of H, sinc damping in the eigenbasis of K, linear weight for
  E / log E measures; geometric sub-bins from the binning, the path's
  sum(V L) and `tol` (parameter-independent, so averages are smooth); C2
  fade-out of pairs sweeping > 1e6 rad per sub-bin. Exact forward-mode
  gradients (model, zoa_<t>, rho_<t>) via divided differences (no
  eigenvector derivatives). Fast, NSI, Sterile (`Model::analytic_avg`).
  Host only (OpenMP). Matches OscProb's AvgProbLoE to 8e-14 on its
  sub-bins; NOvA FD/ND accuracy ~1e-6 at tol = 1e-6 (~1e-8 at 1e-8),
  gradients ~1e-8..1e-13 vs long-double FD (`tests/validate_analytic.cpp`).
* Done: CUDA version of the analytic averages (EngineBase::analytic_subbins;
  value kernel per sub-bin with per-segment/K data in global memory,
  gradient kernel per sub-bin x pass, sub-bins split over devices, host
  reduction). V100: FD 151 bins 0.45 ms, + 12 gradients 3.5 ms.
* Done: batched parameter points (PISCES Phase 4): `analytic_batch` handles
  (sub-bins, path, per-bin lists on one device) and
  `avg_path_analytic_batch` (device output on a stream, or host output;
  host fallback on the CPU backend); kernels over points x sub-bins
  (x gradient passes), per-point fixed-order device reduction; chunks of
  points under a scratch budget; bit-identical per point for any batch.
  V100: FD 0.39 ms per point with 6 gradients, ND 0.20 ms.
* Done: faster analytic averages for one-segment paths (PISCES: NOvA FD/ND,
  `analytic::subbin_fused`): value and all gradient passes of a sub-bin in
  one thread with no per-segment/K scratch in global memory; K diagonalised
  in the eigenbasis of H; derivatives contracted through
  parameter-independent per-channel tables; divided differences reuse the
  cached F/g values; J_n by one series plus downward recurrence;
  dH/dp = u dA/dp for model parameters of models with static matter terms
  (`Model::analytic_static_matter`: Fast, Sterile).
  `AnalyticAvgOptions::rows` (initial-flavour bit mask; other rows zero).
  Agrees with the general path to ~1e-15 (P) and ~3e-14 (dP, relative).
  Multi-segment paths keep the two-kernel scheme. Tried: reverse mode per
  output channel (no per-channel tables, but more flops: no faster).
* Possible: batched fixed-node GL (not needed by PISCES now).
* Next for the analytic averages: Python bindings. Models with E-dependent matter terms are
  out of scope (the expansion needs H affine in 1/E).
* Not planned: PMNS_Avg (obsolete), PMNS_Iter (approximate solver for Fast).

## Working notes
* Local: `cmake -S . -B build -DOPG_ENABLE_CUDA=OFF -DOPG_ENABLE_PYTHON=ON`;
  references: `reference/make_reference.sh ../OscProb` (needs ROOT; set
  `ROOTSYS`/`PATH`/`LD_LIBRARY_PATH` by hand if `thisroot.sh` misbehaves).
* GPU host: `REMOTE_HOST=<host> REMOTE_DIR=<dir> scripts/remote_sync.sh`, then
  build with `CMAKE_CUDA_ARCHITECTURES` set to the card's SM and a CMake ≥ 3.18.
