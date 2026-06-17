# Georgia EV Assistant — Linux GPU Deployment (vLLM)

## What you are deploying
- **Base model:** `Qwen/Qwen2.5-Coder-7B-Instruct` (pulled automatically from Hugging Face)
- **LoRA adapter (PEFT format):** `training_project/adapters/georgia_ev_lora/`
  (contains `adapter_model.safetensors` + `adapter_config.json`; rank 16, served at runtime — no merge required)

## 1. Copy the adapter to the server

```bash
scp -r training_project/adapters/georgia_ev_lora user@gpu-server:/opt/models/georgia_ev_lora
```

## 2. Install vLLM (needs an NVIDIA GPU with ~18 GB+ VRAM for bf16; use --quantization for smaller cards)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 3. Serve

```bash
LORA_PATH=/opt/models/georgia_ev_lora ./serve_vllm.sh
```

This starts an OpenAI-compatible API on port 8000 with the adapter registered
as model name `georgia-ev`.

## 4. Query (OpenAI-compatible curl example)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "georgia-ev",
    "messages": [
      {"role": "system", "content": "You are a Georgia EV supply chain assistant. Answer only using the Georgia EV knowledge base. If the KB does not contain enough information, say so clearly."},
      {"role": "user", "content": "Which Georgia companies are classified under Battery Cell or Battery Pack roles, and what tier is each assigned?"}
    ],
    "temperature": 0,
    "max_tokens": 700
  }'
```

> Use the same system prompt as above — the adapter was trained with it.
> Use `"model": "Qwen/Qwen2.5-Coder-7B-Instruct"` in the request to hit the raw base model instead.

## Optional: merge the adapter into a standalone model

If you prefer a single merged checkpoint (no `--enable-lora`):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", dtype="bfloat16")
merged = PeftModel.from_pretrained(base, "/opt/models/georgia_ev_lora").merge_and_unload()
merged.save_pretrained("/opt/models/georgia-ev-merged")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct").save_pretrained("/opt/models/georgia-ev-merged")
```

Then: `vllm serve /opt/models/georgia-ev-merged`

## Re-training on the Linux GPU box

`training_project/scripts/train_qlora.py` auto-detects CUDA and switches to
true 4-bit NF4 QLoRA (bitsandbytes) — the same command used on macOS works
unchanged on Linux.
