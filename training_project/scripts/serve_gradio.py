#!/usr/bin/env python3
"""Web chat UI for the Georgia EV assistant, with an optional public share link.

Run locally on this Mac; `--share` prints a temporary public *.gradio.live URL
(valid ~72h) you can send to someone to test. The model runs on YOUR machine, so
it must stay awake and the process must keep running while they test.

    python training_project/scripts/serve_gradio.py --share
    python training_project/scripts/serve_gradio.py --share --password hunter2   # gate it
    python training_project/scripts/serve_gradio.py --model v1 --share           # serve v1

Each message is answered fresh (single-turn, matching how the model was trained/evaluated).
"""
import os
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

_state = {"tok": None, "model": None, "device": "cpu", "which": "v2", "max_new_tokens": 700}


def _device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def load_model(which="v2"):
    device, dtype = _device_dtype()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(device).eval()
    model = base
    if which in ADAPTERS and ADAPTERS[which].exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(ADAPTERS[which])).to(device).eval()
    else:
        which = "base"
    _state.update(tok=tok, model=model, device=device, which=which)
    return device, which


def answer(message, history=None):
    """Stateless single-turn generation (history ignored, matching training)."""
    tok, model, device = _state["tok"], _state["model"], _state["device"]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=_state["max_new_tokens"],
                             do_sample=False, temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v2", "v1", "base"], default="v2")
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--share", action="store_true", help="create a public *.gradio.live link")
    ap.add_argument("--password", default=None, help="require this password (username: guest)")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    _state["max_new_tokens"] = a.max_new_tokens

    print(f"Loading {BASE_MODEL} + {a.model} adapter… (first load reads ~15GB)")
    device, which = load_model(a.model)
    print(f"Ready: model={which.upper()} device={device}")

    import gradio as gr
    demo = gr.ChatInterface(
        fn=answer,
        title="Georgia EV Supply-Chain Assistant",
        description=(f"Fine-tuned Qwen2.5-Coder-7B ({which.upper()} LoRA adapter). "
                     "Answers only from the Georgia EV knowledge base. Each question is "
                     "answered independently. Responses can take ~10–40s (7B on a Mac)."),
        examples=[
            "Which Georgia companies have a Battery Cell or Battery Pack role?",
            "Which county has the highest total employment across all companies?",
            "List the Thermal Management suppliers and their Primary OEMs.",
            "Summarize Novelis Inc. from the Georgia EV KB.",
        ],
    )
    auth = ("guest", a.password) if a.password else None
    demo.queue().launch(share=a.share, server_port=a.port, auth=auth)


if __name__ == "__main__":
    main()
