#!/usr/bin/env bash
# Print Python/CUDA/GPU/package versions relevant to self-supervised fine-tuning.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSFT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$SSFT_ROOT/src:${PYTHONPATH:-}"
PY="${PYTHON:-python3}"

echo "=== Python ==="
"$PY" --version

echo ""
echo "=== nvidia-smi ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found (no NVIDIA GPU or driver on this machine)"
fi

echo ""
echo "=== Key package versions ==="
"$PY" - <<'EOF'
for pkg in ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets"]:
    try:
        mod = __import__(pkg)
        print(f"{pkg}: {getattr(mod, '__version__', 'unknown')}")
    except ImportError:
        print(f"{pkg}: NOT INSTALLED")
try:
    import torch
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")
        print(f"bf16_supported: {torch.cuda.is_bf16_supported()}")
except ImportError:
    pass
EOF

echo ""
echo "=== ssft inspect-env (full JSON) ==="
"$PY" -m ssft.cli inspect-env
