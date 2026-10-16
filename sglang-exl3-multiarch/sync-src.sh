#!/usr/bin/env bash
# Vendor csrc/ of sybil-solutions/sglang-exl3 (private) at one commit, so the release-image workflow builds without
# repository access. Usage: ./sync-src.sh <path-to-sglang-exl3-checkout> [commit]
set -euo pipefail
SRC=${1:?path to a sglang-exl3 checkout}; COMMIT=$(git -C "$SRC" rev-parse "${2:-HEAD}")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rm -rf "$HERE/csrc"
git -C "$SRC" archive "$COMMIT" csrc | tar -x -C "$HERE"
echo "$COMMIT" > "$HERE/VERSION"
echo "vendored sglang-exl3 csrc @ $COMMIT (tree $(git -C "$SRC" rev-parse "$COMMIT:csrc")) into $HERE/csrc"
