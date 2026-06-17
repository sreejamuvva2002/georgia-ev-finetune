#!/usr/bin/env python3
"""Web chat UI for the fine-tuned MLX 14B Georgia EV assistant, with a public share link.

Serves Qwen2.5-14B-Instruct-MLX-8bit + the trained LoRA adapter via mlx-lm, streaming
tokens live. `--share` prints a temporary public *.gradio.live URL (valid ~72h) you can
send to someone. The model runs on THIS Mac, so it must stay awake and the process must
keep running while they test.

    python training_project/scripts/serve_gradio_mlx.py --share --password hunter2

Each message is answered fresh (single-turn, matching how the model was trained/evaluated).
"""
import argparse
from pathlib import Path

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]
MODEL = "/Users/surya/.lmstudio/models/lmstudio-community/Qwen2.5-14B-Instruct-MLX-8bit"
ADAPTER_DEFAULT = ROOT / "adapters" / "georgia_ev_14b_mlx_v4"   # v4 (latest); override with --adapter
SYSTEM = ("You are a Georgia EV supply chain assistant. Answer only using the "
          "Georgia EV knowledge base. If the KB does not contain enough "
          "information, say so clearly.")

_state = {"model": None, "tok": None, "sampler": None, "max_tokens": 700}


def answer(message, history=None):
    """Stateless single-turn streaming generation (history ignored, matching training)."""
    tok = _state["tok"]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
    acc = ""
    for resp in stream_generate(_state["model"], tok, prompt,
                                max_tokens=_state["max_tokens"], sampler=_state["sampler"]):
        acc += resp.text
        yield acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--adapter", default=str(ADAPTER_DEFAULT), help="LoRA adapter dir (default: v4)")
    ap.add_argument("--share", action="store_true", help="create a public *.gradio.live link")
    ap.add_argument("--password", default=None, help="require this password (username: guest)")
    ap.add_argument("--port", type=int, default=7861)
    a = ap.parse_args()

    print(f"Loading MLX 14B + adapter ({a.adapter})… (reads ~15GB)", flush=True)
    model, tok = load(MODEL, adapter_path=a.adapter)
    _state.update(model=model, tok=tok, sampler=make_sampler(temp=0.0), max_tokens=a.max_tokens)
    print("Model ready.", flush=True)

    import gradio as gr
    demo = gr.ChatInterface(
        fn=answer,
        title="Georgia EV Supply-Chain Assistant (14B)",
        description=("Fine-tuned Qwen2.5-14B-Instruct (LoRA). Answers only from the Georgia EV "
                     "knowledge base. Each question is answered independently. Responses stream "
                     "and can take ~10–40s (14B on a Mac)."),
        examples=[
            "Which Georgia companies have a Battery Cell or Battery Pack role?",
            "In Gwinnett County, which company has the highest employment and what is its role?",
            "List the Thermal Management suppliers and their Primary OEMs.",
            "Which Tier 1 companies are in the Materials role?",
        ],
    )
    auth = ("guest", a.password) if a.password else None
    # prevent_thread_lock=True returns immediately with the URLs so we can print them with an
    # explicit flush (gradio's own URL print gets stuck in the pipe buffer when not a TTY),
    # then we block to keep the server alive.
    app, local_url, share_url = demo.queue().launch(
        share=a.share, server_port=a.port, auth=auth, prevent_thread_lock=True)
    print(f"LOCAL_URL: {local_url}", flush=True)
    print(f"SHARE_URL: {share_url}", flush=True)
    print(f"LOGIN: guest / {a.password}" if a.password else "LOGIN: (none)", flush=True)
    import time
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
