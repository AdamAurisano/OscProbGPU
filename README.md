# OscProbGPU

A CUDA port of selected [OscProb](https://github.com/joaoabcoelho/OscProb) neutrino
oscillation calculators, with a batched C++ API and Python bindings.

Models: `PMNS_Fast`, `PMNS_NSI`, `PMNS_NUNM`, `PMNS_Sterile` (3+1), `PMNS_Decay`.
Results are validated point-by-point against the original OscProb.

**Status:** under development. See `docs/` (coming) and the phase plan.

## Build

```sh
cmake -S . -B build                  # CUDA backend auto-detected
cmake -S . -B build -DOPG_ENABLE_CUDA=OFF   # CPU only
cmake --build build -j8
ctest --test-dir build               # add -L gpu / -L cpu to select
```

Requirements: CMake >= 3.18, a C++17 compiler, optionally CUDA >= 11.
For pre-Ampere GPUs on old drivers (no PTX JIT), set `CMAKE_CUDA_ARCHITECTURES`
to the exact SM of the card (default `70`, V100).

## Licensing

MIT (see `LICENSE`). The 3x3 Hermitian eigensolvers in `include/opg/linalg/kopp`
are ported from Joachim Kopp's code and remain under LGPL-2.1 (see the `COPYING`
file there). Earth model tables in `data/` come from OscProb.
