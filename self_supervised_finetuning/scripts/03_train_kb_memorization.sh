#!/usr/bin/env bash
# Qwen2.5-14B base + QLoRA default (r16) + KB memorization experiment (100% train,
# in-sample absorption only — NOT a generalization test). Usage:
# 03_train_kb_memorization.sh [path/to/kb_full.jsonl]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSFT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SSFT_ROOT/.." && pwd)"
export PYTHONPATH="$SSFT_ROOT/src:${PYTHONPATH:-}"
PY="${PYTHON:-python3}"

KB_PATH="${1:-$REPO_ROOT/kb_full.jsonl}"

"$PY" -m ssft.cli train \
  --model-config "$SSFT_ROOT/configs/models/qwen2p5_14b_base.yaml" \
  --method-config "$SSFT_ROOT/configs/methods/qlora_lora_default.yaml" \
  --data-config "$SSFT_ROOT/configs/data/kb_only_memorization.yaml" \
  --training-config "$SSFT_ROOT/configs/training/tiny_kb_memorization.yaml" \
  --input "$KB_PATH"
