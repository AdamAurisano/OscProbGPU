#!/usr/bin/env bash
# Build the original OscProb and dump reference probabilities into tests/data.
#
# Requirements: ROOT (thisroot.sh sourced or ROOTSYS set) and the OscProb
# sources (default: ../OscProb next to this repository).
#
# Usage: reference/make_reference.sh [path/to/OscProb] [nsub for bin averages]
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
OSCPROB_SRC=${1:-$HERE/../OscProb}
NSUB=${2:-400}
BUILD=$HERE/build-ref

if ! command -v root-config >/dev/null; then
  echo "root-config not found: source thisroot.sh first" >&2
  exit 1
fi

cmake -S "$OSCPROB_SRC" -B "$BUILD/oscprob" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$BUILD/oscprob-install" >/dev/null
cmake --build "$BUILD/oscprob" -j8 >/dev/null
cmake --install "$BUILD/oscprob" >/dev/null

INST=$BUILD/oscprob-install
g++ -O2 -std=c++17 "$HERE/reference/dump_reference.cxx" -o "$BUILD/dump_reference" \
  -I"$INST/include" -I"$INST/include/eigen3" $(root-config --cflags) \
  -L"$INST/lib" -Wl,-rpath,"$INST/lib" -lOscProb -lMatrixDecomp \
  $(root-config --libs)

mkdir -p "$HERE/tests/data"
"$BUILD/dump_reference" "$HERE/tests/data" "$NSUB"
