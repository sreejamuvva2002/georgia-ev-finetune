#!/usr/bin/env bash
# Run all evaluations (perplexity, cloze probes, instruction sanity) for a completed
# run and refresh its report.md. Usage: 04_eval_adapter.sh <run_dir>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSFT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$SSFT_ROOT/src:${PYTHONPATH:-}"
PY="${PYTHON:-python3}"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run_dir>" >&2
  exit 1
fi

"$PY" -m ssft.cli evaluate --run-dir "$1"
