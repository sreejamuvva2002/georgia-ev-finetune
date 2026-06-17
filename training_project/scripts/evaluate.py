#!/usr/bin/env python3
"""Evaluate base vs fine-tuned model on the 50 held-out human Q&A.

Scoring (per question, against the human gold answer):
- name_recall: fraction of KB company names mentioned in gold that the model also mentions.
- number_match: 1 if the leading count/number in gold appears in the model answer.
- score = weighted composite; question counted correct if score >= 0.60.
- possible_hallucination: model mentions KB companies absent from gold, or non-KB "company-like" claims.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
KB_PATH = "/Users/surya/Downloads/GNEM - Auto Landscape Lat Long Updated (1).xlsx"


def load_test():
    rows = []
    with open(ROOT / "data" / "test.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def generate_answers(adapter: str | None, out_path: Path, max_new_tokens=700):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to(device).eval()

    tests = load_test()
    results = {}
    for i, t in enumerate(tests):
        msgs = t["messages"][:2]  # system + user only
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        results[t["question_id"]] = ans
        print(f"[{i+1}/{len(tests)}] qid={t['question_id']} -> {len(ans)} chars", flush=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("saved", out_path)


# ------------------------------------------------------------------- scoring
def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def company_mentions(text, companies):
    t = norm(text)
    found = set()
    for c in companies:
        key = norm(c).strip()
        # match on the distinctive prefix of the company name (first 2 words)
        words = [w for w in key.split() if w not in ("inc", "llc", "co", "corp", "lp", "usa", "americas", "america")]
        probe = " ".join(words[:2]) if len(words) >= 2 else (words[0] if words else key)
        if probe and probe in t:
            found.add(c)
    return found


# "Tier 1", "Tier 2/3", "Tier 1/2" contain digits that are categorical labels, not
# quantities — strip them before any numeric comparison so they don't pollute signals.
TIER_RE = re.compile(r"tier\s*\d(?:\s*/\s*\d)?", re.I)

# Genuine "headline count" phrasings (a count of companies/areas/roles), in priority order.
COUNT_PATTS = [
    r"there\s+(?:are|is|were)\s+(\d[\d,]*)",
    r"\((\d[\d,]*)\)",
    r"top\s+(\d[\d,]*)",
    r"(\d[\d,]*)\s+(?:georgia\s+)?(?:compan|supplier|area|count(?:y|ies)|role|facilit)",
    r"^\s*(\d[\d,]*)\b",
]


def numbers_in(text):
    text = TIER_RE.sub(" ", text)
    return [n.replace(",", "") for n in re.findall(r"\d[\d,]*", text)]


def lead_count(text):
    """The headline count in a gold answer (e.g. 'There are 18 ...' -> '18').

    Only fires on genuine count phrasings; returns None for answers whose key number
    is not a company count (e.g. an employment total), which are scored via number
    overlap instead.
    """
    t = TIER_RE.sub(" ", text)
    for patt in COUNT_PATTS:
        m = re.search(patt, t, re.I)
        if m:
            return m.group(1).replace(",", "")
    return None


# Fixed weights chosen a priori (not tuned to outcomes): for these KB questions the
# *substance* (which companies) matters most, then the headline count, then any other
# numbers. name_f1 captures BOTH precision (penalizes over-listing, e.g. Q12) and
# recall (penalizes missing companies). A question is "correct" at composite >= 0.60.
W_NAME_F1, W_COUNT, W_NUMS = 0.65, 0.20, 0.15
CORRECT_THRESHOLD = 0.60


def score_answer(gold, pred, companies):
    gold_names = company_mentions(gold, companies)
    pred_names = company_mentions(pred, companies)
    inter = gold_names & pred_names
    detail, signals = {}, []  # signals: list of (value, weight)

    if gold_names:
        recall = len(inter) / len(gold_names)
        precision = len(inter) / len(pred_names) if pred_names else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        detail.update(name_p=round(precision, 2), name_r=round(recall, 2), name_f1=round(f1, 2))
        signals.append((f1, W_NAME_F1))

    lead = lead_count(gold)
    if lead is not None:
        pnums = set(numbers_in(pred))
        cmatch = 1.0 if lead in pnums else 0.0
        detail["count_gold"], detail["count_match"] = lead, cmatch
        signals.append((cmatch, W_COUNT))

    gnums = set(numbers_in(gold))
    if gnums:
        pnums = set(numbers_in(pred))
        noverlap = len(gnums & pnums) / len(gnums)
        detail["num_overlap"] = round(noverlap, 2)
        signals.append((noverlap, W_NUMS))

    if not signals:  # free-form answer: token-level F1 fallback
        g, p = set(norm(gold).split()), set(norm(pred).split())
        tf1 = 2 * len(g & p) / (len(g) + len(p)) if g and p else 0.0
        detail["token_f1"] = round(tf1, 2)
        signals.append((tf1, 1.0))

    score = sum(v * w for v, w in signals) / sum(w for _, w in signals)
    halluc = sorted(pred_names - gold_names)
    missing = sorted(gold_names - pred_names)
    return score, detail, halluc, missing


def score_all():
    kb = pd.read_excel(KB_PATH)
    companies = sorted(kb["Company"].astype(str).str.strip().unique())
    tests = load_test()
    base = json.loads((ROOT / "eval_results" / "base_answers.json").read_text())
    ft = json.loads((ROOT / "eval_results" / "finetuned_answers.json").read_text())

    rows = []
    for t in tests:
        qid = t["question_id"]
        gold = t["messages"][2]["content"]
        b, f = base.get(str(qid), base.get(qid, "")), ft.get(str(qid), ft.get(qid, ""))
        bs, bcomp, bhall, bmiss = score_answer(gold, b, companies)
        fs, fcomp, fhall, fmiss = score_answer(gold, f, companies)
        rows.append({
            "question_id": qid,
            "use_case_category": t.get("use_case_category", ""),
            "question": t["messages"][1]["content"],
            "gold_answer": gold,
            "base_model_answer": b,
            "finetuned_model_answer": f,
            "base_score": round(bs, 3),
            "finetuned_score": round(fs, 3),
            "notes": f"base components: {bcomp} | ft components: {fcomp}",
            "possible_hallucination": "; ".join(fhall) if fhall else "",
            "missing_facts": "; ".join(fmiss) if fmiss else "",
        })
    df = pd.DataFrame(rows)
    thresh = CORRECT_THRESHOLD
    df["base_correct"] = df["base_score"] >= thresh
    df["finetuned_correct"] = df["finetuned_score"] >= thresh
    base_acc = df["base_correct"].mean()
    ft_acc = df["finetuned_correct"].mean()

    out = ROOT / "eval_results"
    cols = ["question_id", "question", "gold_answer", "base_model_answer",
            "finetuned_model_answer", "notes", "possible_hallucination", "missing_facts",
            "base_score", "finetuned_score", "use_case_category"]
    df[cols].to_excel(out / "evaluation.xlsx", index=False)

    by_cat = df.groupby("use_case_category").agg(
        n=("finetuned_score", "size"),
        base_correct=("base_correct", "sum"),
        ft_correct=("finetuned_correct", "sum"),
        base_mean=("base_score", "mean"),
        ft_mean=("finetuned_score", "mean")).round(3)
    by_cat["base_acc"] = (by_cat["base_correct"] / by_cat["n"]).round(2)
    by_cat["ft_acc"] = (by_cat["ft_correct"] / by_cat["n"]).round(2)

    with open(out / "summary.md", "w") as f2:
        f2.write("# Evaluation Summary (50 held-out human-validated Q&A)\n\n")
        f2.write(f"- Correctness rule: composite structured score >= {thresh}\n")
        f2.write(f"- **Base model accuracy: {base_acc:.0%}** ({df['base_correct'].sum()}/50, mean score {df['base_score'].mean():.3f})\n")
        f2.write(f"- **Fine-tuned model accuracy: {ft_acc:.0%}** ({df['finetuned_correct'].sum()}/50, mean score {df['finetuned_score'].mean():.3f})\n\n")
        f2.write("## By use-case category\n\n")
        f2.write(by_cat[["n", "base_correct", "ft_correct", "base_acc", "ft_acc", "base_mean", "ft_mean"]].to_markdown() + "\n\n")
        f2.write("## Scoring method (structured auto-score)\n\n")
        f2.write("Each answer is scored against the human gold answer with a weighted composite of "
                 "structured signals (weights fixed a priori, not tuned to results):\n\n")
        f2.write(f"- **Company-name F1** (weight {W_NAME_F1}): precision + recall of KB company names vs the gold "
                 "answer. Precision penalizes over-listing (e.g. a query whose answer is 3 companies but the model lists 40); "
                 "recall penalizes missing companies.\n")
        f2.write(f"- **Headline-count match** (weight {W_COUNT}): does the model reproduce the gold answer's leading count "
                 "(e.g. \"There are 18 ...\")?\n")
        f2.write(f"- **Number overlap** (weight {W_NUMS}): fraction of gold numbers (counts, employment figures) present in the answer.\n")
        f2.write("- **Token-F1 fallback** (weight 1.0): used only for free-form answers with no company names or numbers.\n\n")
        f2.write(f"Weights are renormalized over whichever signals apply to a question. A question counts as "
                 f"**correct** when the composite >= {thresh}.\n\n")
        f2.write("`possible_hallucination` = KB companies the fine-tuned answer named that the gold answer did not. "
                 "`missing_facts` = gold companies the answer omitted. `notes` carries the per-signal breakdown.\n")
    print(f"BASE accuracy: {base_acc:.0%} ({df['base_correct'].sum()}/50) | "
          f"FINETUNED accuracy: {ft_acc:.0%} ({df['finetuned_correct'].sum()}/50)")
    print("wrote", out / "evaluation.xlsx", "and summary.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["base", "finetuned", "score"])
    ap.add_argument("--adapter", default=str(ROOT / "adapters" / "georgia_ev_lora"))
    a = ap.parse_args()
    (ROOT / "eval_results").mkdir(exist_ok=True)
    if a.mode == "base":
        generate_answers(None, ROOT / "eval_results" / "base_answers.json")
    elif a.mode == "finetuned":
        generate_answers(a.adapter, ROOT / "eval_results" / "finetuned_answers.json")
    else:
        score_all()
