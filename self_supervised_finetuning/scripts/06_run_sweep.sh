#!/usr/bin/env bash
# Run an arbitrary sweep config. Usage: 06_run_sweep.sh <sweep_config.yaml> [dry_run:true|false]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSFT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$SSFT_ROOT/src:${PYTHONPATH:-}"
PY="${PYTHON:-python3}"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <sweep_config.yaml> [dry_run:true|false]" >&2
  exit 1
fi

SWEEP_CONFIG="$1"
DRY_RUN="${2:-false}"

ARGS=(--sweep-config "$SWEEP_CONFIG")
if [ "$DRY_RUN" = "true" ]; then
  ARGS+=(--dry-run)
fi

"$PY" -m ssft.cli sweep "${ARGS[@]}"
