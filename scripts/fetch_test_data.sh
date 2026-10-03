#!/usr/bin/env bash
# Download the OscProb reference data used by the tests into tests/data.
#
# Usage: scripts/fetch_test_data.sh [release tag] [destination]
#        (OPG_TESTDATA_URL overrides the download URL, e.g. a mirror)
#
# The data (point probabilities, PREM grids, bin averages, ... dumped from the
# original OscProb by reference/make_reference.sh) is published as an asset of
# a GitHub release. Every file is checked against tests/data/SHA256SUMS, which
# is part of the repository. To regenerate the data instead, run
# reference/make_reference.sh (needs ROOT and the OscProb sources).
set -euo pipefail

TAG=${1:-testdata-v1}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST=${2:-$HERE/tests/data}
URL=${OPG_TESTDATA_URL:-https://github.com/AdamAurisano/OscProbGPU/releases/download/$TAG/oscprobgpu-testdata.tar.gz}

mkdir -p "$DEST"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Downloading $URL" >&2
if command -v curl >/dev/null; then
  curl -fL --retry 3 --progress-bar -o "$TMP/data.tar.gz" "$URL"
else
  wget -q -O "$TMP/data.tar.gz" "$URL"
fi
tar xzf "$TMP/data.tar.gz" -C "$TMP"

# verify against the manifest in the repository (not the one in the archive)
SUMS=$HERE/tests/data/SHA256SUMS
if command -v sha256sum >/dev/null; then
  (cd "$TMP" && sha256sum --quiet -c "$SUMS")
else
  (cd "$TMP" && shasum -a 256 --quiet -c "$SUMS")
fi

cp "$TMP"/*.npy "$DEST"/
echo "Test data ($TAG): $(ls "$TMP"/*.npy | wc -l | tr -d ' ') files in $DEST" >&2
