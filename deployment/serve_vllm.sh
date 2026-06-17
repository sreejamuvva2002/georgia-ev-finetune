#!/usr/bin/env bash
# Serve the Georgia EV LoRA adapter on a Linux GPU box with vLLM (OpenAI-compatible API).
set -euo pipefail

BASE_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
# Path to the PEFT adapter directory (copy training_project/adapters/georgia_ev_lora to the server)
LORA_PATH="${LORA_PATH:-/opt/models/georgia_ev_lora}"
PORT="${PORT:-8000}"

vllm serve "$BASE_MODEL" \
  --enable-lora \
  --lora-modules georgia-ev="$LORA_PATH" \
  --max-lora-rank 32 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --port "$PORT"
