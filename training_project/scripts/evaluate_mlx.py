#!/usr/bin/env python3
"""v4 evaluation of the fine-tuned MLX 14B.

Generates greedy answers for BOTH eval sets and scores them:
  A. 50 held-out human Q&A  -> structured scorer (evaluate.score_answer), per-category, with
     every FAIL dumped (Q / gold / model) so misses can be attributed to a real model error
     vs a known gold inconsistency (the 50-question gold is internally inconsistent — see
     SESSION_LOG_14B.md). This is a secondary reference.
  B. 201 held-out PROBE BENCHMARK -> the PRIMARY no-compromise metric (build_probe_benchmark),
     headline = company recall / missing-company rate.

  python scripts/evaluate_mlx.py generate                 # v4 adapter, both sets
  python scripts/evaluate_mlx.py generate --adapter PATH   # a specific checkpoint
  python scripts/evaluate_mlx.py score
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_dataset as bd                 # noqa: E402
import evaluate as ev                       # noqa: E402  (score_answer, KB_PATH)
import build_probe_benchmark as pb          # noqa: E402

MODEL = "/Users/surya/.lmstudio/models/lmstudio-community/Qwen2.5-14B-Instruct-MLX-8bit"
ADAPTER_V4 = ROOT / "adapters" / "georgia_ev_14b_mlx_v4"
EVDIR = ROOT / "eval_results"
TEST_ANS = EVDIR / "answers_14b_v4.json"
PROBE_ANS = EVDIR / "probe_answers_14b_v4.json"


def _gen(adapter):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(MODEL, adapter_path=(str(adapter) if adapter else None))
    sampler = make_sampler(temp=0.0)

    def ask(question):
        msgs = [{"role": "system", "content": bd.SYSTEM}, {"role": "user", "content": question}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
        return generate(model, tok, prompt=prompt, max_tokens=900, sampler=sampler).strip()

    EVDIR.mkdir(exist_ok=True)
    tests = [json.loads(l) for l in open(ROOT / "data" / "test.jsonl")]
    out = {}
    for i, t in enumerate(tests):
        out[str(t["question_id"])] = ask(t["messages"][1]["content"])
        print(f"[test {i+1}/{len(tests)}] qid={t['question_id']}", flush=True)
    TEST_ANS.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    probes = [json.loads(l) for l in open(bd.PROBE_FILE)]
    pans = []
    for i, p in enumerate(probes):
        pans.append(ask(p["question"]))
        if (i + 1) % 20 == 0:
            print(f"[probe {i+1}/{len(probes)}]", flush=True)
    PROBE_ANS.write_text(json.dumps(pans, indent=2, ensure_ascii=False))
    print("saved", TEST_ANS, "and", PROBE_ANS)


def score():
    kb = pd.read_excel(ev.KB_PATH)
    companies = sorted(kb["Company"].astype(str).str.strip().unique())
    tests = [json.loads(l) for l in open(ROOT / "data" / "test.jsonl")]
    ans = json.loads(TEST_ANS.read_text())

    rows, fails = [], []
    for t in tests:
        qid = str(t["question_id"]); gold = t["messages"][2]["content"]
        pred = ans.get(qid, "")
        s, detail, halluc, missing = ev.score_answer(gold, pred, companies)
        rows.append({"qid": qid, "category": t.get("use_case_category", ""), "score": s})
        if s < 0.6:
            fails.append((qid, t["messages"][1]["content"], gold, pred, missing, halluc))
    tdf = pd.DataFrame(rows)
    c = int((tdf["score"] >= 0.6).sum())
    lines = ["# 14B v4 Evaluation\n", "## A. 50 held-out human Q&A (structured score >= 0.6)\n",
             f"**Accuracy: {c}/50 = {c/50:.0%}  |  mean score {tdf['score'].mean():.3f}**\n",
             "_The 50-question gold is internally inconsistent; each FAIL below is dumped for "
             "manual attribution (real model error vs gold inconsistency)._\n",
             "\n### By use-case category\n| category | n | pass |\n|---|---|---|"]
    for cat in sorted(tdf["category"].unique()):
        sub = tdf[tdf["category"] == cat]
        lines.append(f"| {cat} | {len(sub)} | {int((sub['score']>=0.6).sum())}/{len(sub)} |")
    lines.append("\n### FAILs (for attribution)\n")
    for qid, q, gold, pred, missing, halluc in fails:
        lines.append(f"**Q{qid}.** {q}\n\n- gold: {gold[:300]}\n- model: {pred[:300]}\n"
                     f"- missing companies: {missing[:8]} | hallucinated: {halluc[:8]}\n")

    EVDIR.mkdir(exist_ok=True)
    tdf.to_excel(EVDIR / "scores_50_14b_v4.xlsx", index=False)
    (EVDIR / "summary_14b_v4.md").write_text("\n".join(lines) + "\n")
    print(f"A. 50 held-out human Q&A: {c}/50 = {c/50:.0%} | mean {tdf['score'].mean():.3f}")

    # B. primary metric: probe benchmark
    print("\nB. PROBE BENCHMARK (primary):")
    pb.score(str(PROBE_ANS))
    print("\nwrote", EVDIR / "summary_14b_v4.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["generate", "score"])
    ap.add_argument("--adapter", default=str(ADAPTER_V4), help="adapter dir (default: v4)")
    ap.add_argument("--base", action="store_true", help="generate with the 14B BASE (no adapter)")
    a = ap.parse_args()
    EVDIR.mkdir(exist_ok=True)
    if a.mode == "generate":
        _gen(None if a.base else a.adapter)
    else:
        score()
