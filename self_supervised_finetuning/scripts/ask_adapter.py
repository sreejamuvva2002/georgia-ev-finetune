"""Probe a trained LoRA adapter (or its base) over a fixed question set, with
reproducible, structured, checksummed output for GNEM-Bench-v1.

Loads base+adapter once via the same loader the eval harness uses
(`eval_perplexity.load_adapter_for_eval`). The base pass uses PEFT `disable_adapter()`,
so base and adapter run on identical (4-bit) base weights — the only difference is the
LoRA delta.

MODES
  --mode base     : generate BASE only (adapter disabled) -> canonical base artifact.
  --mode adapter  : generate ADAPTER only.
  --mode both     : both (ad-hoc side-by-side; also see --selfcheck).

The base is identical regardless of which run's adapter is attached-but-disabled, so it
is computed ONCE (`raw_outputs/base.json`) and reused across adapter systems; a
consistency check (score/combine step) compares each run's disabled-adapter base to it.

IMPORTANT: these are BASE (non-instruct) completion models. They complete cloze prompts
(`Company: JTEKT\nLocation:`) far better than free-form questions; the benchmark uses the
raw questions and PARSES answers from the completions.

Usage (structured, for the benchmark):
  ask_adapter.py --run-dir <run> --mode adapter --label kb_only \
     --questions-file benchmarks/gnem_bench_v1/questions.json \
     --max-new-tokens 256 --out-json outputs/question_eval/raw_outputs/kb_only.json
Ad-hoc (default demo prompts, human-readable):
  ask_adapter.py --run-dir <run>            # prints base vs adapter for a few prompts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path

DEFAULT_RUN_DIR = (
    "self_supervised_finetuning/outputs/adapters/qwen2p5-14b/"
    "qlora-lora-r64-a128-d0-rslora/kb-only-memorization/train-all/"
    "ep50-bs1-ga16-ebs16-lr2e4-seq1024/seed42/20260713_150528_712d9bf3"
)
DEFAULT_PROMPTS = [
    "Company: JTEKT\nLocation:",
    "Company: SK Battery America\nProduct or Service:",
]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolved_commit(name_or_path: str) -> str | None:
    try:
        from huggingface_hub import snapshot_download
        p = snapshot_download(name_or_path, local_files_only=True)
        return Path(p).name
    except Exception:
        return None


def _generation_config(resolved, tokenizer, max_new_tokens: int) -> dict:
    import torch
    import transformers, peft, bitsandbytes  # noqa
    mc = resolved.model_cfg
    quant = (resolved.method_cfg or {}).get("quantization", {})
    return {
        "model_name_or_path": mc.get("name_or_path"),
        "model_snapshot_commit": _resolved_commit(mc.get("name_or_path")),
        "tokenizer": mc.get("name_or_path"),
        "quantization": quant,
        "do_sample": False, "temperature": None, "top_p": None,
        "max_new_tokens": max_new_tokens,
        "seed": resolved.seed,
        "torch_dtype": str(getattr(torch, "bfloat16")),
        "pad_token_id": tokenizer.pad_token_id,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "peft_version": peft.__version__,
        "bitsandbytes_version": bitsandbytes.__version__,
    }


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    dt = time.perf_counter() - t0
    gen_ids = out[0][n_in:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    n_gen = int(gen_ids.shape[0])
    return {"output": text, "generated_tokens": n_gen, "generation_time": round(dt, 3),
            "tokens_per_sec": round(n_gen / dt, 2) if dt > 0 else None,
            "hit_max_new_tokens": n_gen >= max_new_tokens}


def _load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    qs = data["questions"] if isinstance(data, dict) else data
    return [{"question_id": q["question_id"], "section": q.get("section"), "question": q["question"]}
            for q in qs]


def _write_readonly(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(stat.S_IWUSR | stat.S_IRUSR)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    path.chmod(stat.S_IRUSR | stat.S_IRGRP)  # read-only after write (immutable raw output)


def run_structured(args) -> int:
    from ssft.eval.eval_perplexity import load_adapter_for_eval
    qpath = Path(args.questions_file)
    questions = _load_questions(qpath)
    benchmark_sha = _sha256(qpath.read_text())

    print(f"Loading base+adapter from {args.run_dir} ...", flush=True)
    resolved, _base, adapter_model, tokenizer = load_adapter_for_eval(Path(args.run_dir))
    adapter_model.eval()
    gen_cfg = _generation_config(resolved, tokenizer, args.max_new_tokens)
    gen_cfg_sha = _sha256(json.dumps(gen_cfg, sort_keys=True, default=str))

    results = []
    for i, q in enumerate(questions, 1):
        if args.mode == "base":
            with adapter_model.disable_adapter():
                g = _generate(adapter_model, tokenizer, q["question"], args.max_new_tokens)
        else:  # adapter
            g = _generate(adapter_model, tokenizer, q["question"], args.max_new_tokens)
        results.append({**q, **g})
        if i % 10 == 0 or i == len(questions):
            print(f"  [{i}/{len(questions)}]", flush=True)

    payload = {
        "label": args.label, "mode": args.mode, "run_dir": str(args.run_dir),
        "benchmark_file": str(qpath), "benchmark_sha256": benchmark_sha,
        "generation_config": gen_cfg, "generation_config_sha256": gen_cfg_sha,
        "n_questions": len(results), "results": results,
    }
    payload["raw_output_sha256"] = _sha256(json.dumps(results, sort_keys=True, ensure_ascii=False))
    _write_readonly(Path(args.out_json), payload)
    trunc = sum(r["hit_max_new_tokens"] for r in results)
    print(f"wrote {args.out_json}  (label={args.label}, mode={args.mode}, "
          f"truncated={trunc}/{len(results)})")
    return 0


def run_selfcheck(args) -> int:
    """Diagnostic: adapter -> base(disabled) -> adapter; assert run1==run3 (no adapter
    state leak). If a canonical base.json is given, also confirm disabled-adapter base
    matches it (raw-exact first, then normalized) on the sampled prompts."""
    from ssft.eval.eval_perplexity import load_adapter_for_eval
    resolved, _base, model, tok = load_adapter_for_eval(Path(args.run_dir))
    model.eval()
    prompts = DEFAULT_PROMPTS
    ok = True
    for p in prompts:
        a1 = _generate(model, tok, p, args.max_new_tokens)["output"]
        with model.disable_adapter():
            b = _generate(model, tok, p, args.max_new_tokens)["output"]
        a3 = _generate(model, tok, p, args.max_new_tokens)["output"]
        same = a1 == a3
        ok &= same
        print(f"[selfcheck] adapter-stable={same!r}  base!=adapter={(b!=a1)!r}")
    print("SELFCHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def run_adhoc(args) -> int:
    from ssft.eval.eval_perplexity import load_adapter_for_eval
    resolved, _base, model, tok = load_adapter_for_eval(Path(args.run_dir))
    model.eval()
    for p in DEFAULT_PROMPTS:
        with model.disable_adapter():
            base = _generate(model, tok, p, args.max_new_tokens)["output"]
        adap = _generate(model, tok, p, args.max_new_tokens)["output"]
        print("=" * 78); print("PROMPT:\n" + p); print("-" * 78)
        print(f"BASE    → {base!r}\nADAPTER → {adap!r}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--questions-file", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--label", default="adapter")
    ap.add_argument("--mode", choices=["base", "adapter", "both"], default="both")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return run_selfcheck(args)
    if args.questions_file and args.out_json:
        return run_structured(args)
    return run_adhoc(args)


if __name__ == "__main__":
    raise SystemExit(main())
