#!/usr/bin/env bash
# Sync the working tree to a remote build host in a single ssh call.
#
# Usage: REMOTE_HOST=<ssh host> REMOTE_DIR=<absolute path> scripts/remote_sync.sh
#
# The tree is streamed with tar (build directories and .git excluded) and all
# files are touched after unpacking, so that a remote clock lagging behind the
# local one does not make make/CMake see them as stale.
set -euo pipefail

: "${REMOTE_HOST:?set REMOTE_HOST to the ssh host (or alias) to sync to}"
: "${REMOTE_DIR:?set REMOTE_DIR to the absolute destination directory}"

cd "$(dirname "$0")/.."

tar czf - --exclude='./build*' --exclude='./.git' --exclude='./wheelhouse' \
    --exclude='__pycache__' . |
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE_HOST" \
    "mkdir -p '$REMOTE_DIR' && cd '$REMOTE_DIR' && tar xzmf - 1>&2 && \
     find . -path './build*' -prune -o -type f -exec touch {} + 1>&2 && \
     echo synced to \$(hostname):$REMOTE_DIR 1>&2"
