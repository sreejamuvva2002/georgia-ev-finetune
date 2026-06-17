#!/usr/bin/env python3
"""Interactive chat with the Georgia EV assistant (local, MPS/CPU/CUDA).

Usage:
    python training_project/scripts/chat.py            # v2 adapter (default)
    python training_project/scripts/chat.py --model v1 # v1 adapter
    python training_project/scripts/chat.py --model base   # base model, no fine-tune
    python training_project/scripts/chat.py --temp 0.0 --max-new-tokens 700

Type a question and press Enter. Commands: /base /v1 /v2 (switch model),
/reset (clear nothing — each Q is answered fresh), /quit or Ctrl-D to exit.
"""
import os
# Must be set before torch initializes the MPS backend (avoids the invalid-watermark abort).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTERS = {"v2": ROOT / "adapters" / "georgia_ev_lora",
            "v1": ROOT / "adapters" / "georgia_ev_lora_v1"}
SYSTEM = ("You are a Georgia EV supply chain assistant. Answer only using the "
          "Georgia EV knowledge base. If the KB does not contain enough "
          "information, say so clearly.")


def device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def load(which, device, dtype, tok, base):
    """Return a model for 'base'/'v1'/'v2', reusing the loaded base weights."""
    if which == "base":
        return base
    from peft import PeftModel
    adapter = ADAPTERS[which]
    if not adapter.exists():
        print(f"[adapter {which} not found at {adapter}; staying on current model]")
        return None
    m = PeftModel.from_pretrained(base, str(adapter))
    return m.to(device).eval()


def generate(model, tok, device, question, max_new_tokens, temp):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    if temp and temp > 0:
        kw.update(do_sample=True, temperature=temp, top_p=0.9)
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        out = model.generate(**inputs, **kw)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v2", "v1", "base"], default="v2")
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy (deterministic)")
    a = ap.parse_args()

    device, dtype = device_dtype()
    print(f"Loading {BASE_MODEL} on {device} ({dtype})… (first load downloads/reads ~15GB)")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(device).eval()

    which = a.model
    model = load(which, device, dtype, tok, base) or base
    if which != "base" and model is base:
        which = "base"
    print(f"\nReady. Model: {which.upper()}  |  device: {device}  |  temp: {a.temp}")
    print("Ask a Georgia EV question. Commands: /base /v1 /v2  /quit\n")

    while True:
        try:
            q = input(f"[{which}] you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye"); break
        if not q:
            continue
        if q in ("/quit", "/exit", "/q"):
            print("bye"); break
        if q in ("/base", "/v1", "/v2"):
            target = q[1:]
            m = load(target, device, dtype, tok, base)
            if m is not None:
                model, which = m, target
                print(f"[switched to {which.upper()}]")
            continue
        try:
            ans = generate(model, tok, device, q, a.max_new_tokens, a.temp)
        except Exception as e:
            print(f"[generation error: {e}]"); continue
        print(f"\n{ans}\n")


if __name__ == "__main__":
    main()
