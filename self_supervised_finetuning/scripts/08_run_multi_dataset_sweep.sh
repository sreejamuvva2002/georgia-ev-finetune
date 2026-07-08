#!/usr/bin/env bash
# Compare Qwen2.5-14B QLoRA adaptation across KB / web / KB+web mixed datasets.
# Usage: 08_run_multi_dataset_sweep.sh [dry_run:true|false]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSFT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
exec "$SCRIPT_DIR/06_run_sweep.sh" "$SSFT_ROOT/configs/sweeps/multi_dataset_qwen14b_sweep.yaml" "${1:-false}"
