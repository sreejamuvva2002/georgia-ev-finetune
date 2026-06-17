#!/usr/bin/env python3
"""Answer an arbitrary question set with a given MLX adapter (greedy). Reusable for ad-hoc
goldsets (e.g. the Claude-generated 14). Usage:

  python scripts/answer_questions.py IN.json OUT.json [--adapter DIR]

IN.json: list of {"id","q",...}. OUT.json: same records with an added "v4" answer field.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_dataset as bd  # noqa: E402

MODEL = "/Users/surya/.lmstudio/models/lmstudio-community/Qwen2.5-14B-Instruct-MLX-8bit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--adapter", default=str(ROOT / "adapters" / "georgia_ev_14b_mlx_v4"))
    ap.add_argument("--field", default="v4", help="answer field name to add")
    a = ap.parse_args()

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(MODEL, adapter_path=a.adapter)
    sampler = make_sampler(temp=0.0)

    def ask(q):
        msgs = [{"role": "system", "content": bd.SYSTEM}, {"role": "user", "content": q}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
        return generate(model, tok, prompt=prompt, max_tokens=900, sampler=sampler).strip()

    recs = json.load(open(a.infile))
    for i, r in enumerate(recs):
        r[a.field] = ask(r["q"])
        print(f"[{i+1}/{len(recs)}] {r['id']}", flush=True)
    json.dump(recs, open(a.outfile, "w"), indent=2, ensure_ascii=False)
    print("saved", a.outfile, flush=True)


if __name__ == "__main__":
    main()
