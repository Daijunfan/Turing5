#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

toolchain=.toolchain/lean-4.30.0-darwin_aarch64

if [[ ! -x "$toolchain/bin/lean" ]]; then
  echo "SKIP: Lean v4.30.0 project toolchain unavailable; no machine-proof claim"
  exit 0
fi

"$toolchain/bin/lean" MorphResidual.lean
"$toolchain/bin/lean" --version
