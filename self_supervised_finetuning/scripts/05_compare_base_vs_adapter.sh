#!/usr/bin/env bash
# Compute base-vs-adapter perplexity + cloze deltas for a run and write comparison.json.
# Usage: 05_compare_base_vs_adapter.sh <run_dir>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSFT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$SSFT_ROOT/src:${PYTHONPATH:-}"
PY="${PYTHON:-python3}"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run_dir>" >&2
  exit 1
fi

"$PY" -m ssft.cli compare-base-adapter --run-dir "$1"
