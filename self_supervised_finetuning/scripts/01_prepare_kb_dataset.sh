#!/usr/bin/env bash
# Prepare both KB dataset variants (company-split generalization + memorization) from
# a kb_full.jsonl. Usage: 01_prepare_kb_dataset.sh [path/to/kb_full.jsonl]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSFT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SSFT_ROOT/.." && pwd)"
export PYTHONPATH="$SSFT_ROOT/src:${PYTHONPATH:-}"
PY="${PYTHON:-python3}"

KB_PATH="${1:-$REPO_ROOT/kb_full.jsonl}"
if [ ! -f "$KB_PATH" ]; then
  echo "KB file not found: $KB_PATH" >&2
  exit 1
fi

echo "=== Preparing kb_only_company_split ==="
"$PY" -m ssft.cli prepare-kb \
  --input "$KB_PATH" \
  --config "$SSFT_ROOT/configs/data/kb_only_company_split.yaml"

echo ""
echo "=== Preparing kb_only_memorization ==="
"$PY" -m ssft.cli prepare-kb \
  --input "$KB_PATH" \
  --config "$SSFT_ROOT/configs/data/kb_only_memorization.yaml"
