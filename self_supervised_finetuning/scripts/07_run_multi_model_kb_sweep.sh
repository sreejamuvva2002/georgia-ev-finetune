#!/usr/bin/env bash
# Compare self-supervised KB adaptation across Qwen2.5-3B/7B/14B.
# Usage: 07_run_multi_model_kb_sweep.sh [dry_run:true|false]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSFT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
exec "$SCRIPT_DIR/06_run_sweep.sh" "$SSFT_ROOT/configs/sweeps/multi_model_kb_sweep.yaml" "${1:-false}"
