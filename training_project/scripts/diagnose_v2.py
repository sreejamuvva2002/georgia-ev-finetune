#!/usr/bin/env python3
"""Phase 1 diagnostic: find WHERE the v2 (7B) model fails, by question *shape*.

Builds a probe set deterministically from the KB (NEVER from the 50 human Q&A),
with computed gold answers, spanning many question shapes. Runs the v2 model and
auto-scores each answer, then reports accuracy per shape with concrete failing
examples. This is the evidence base for the 14B data expansion.

Usage:
  # 1. generate v2 answers for the probe set (slow on MPS; run in background)
  python scripts/diagnose_v2.py generate
  # 2. score + write logs/diagnostic_v2.md
  python scripts/diagnose_v2.py score
  # (or `all` to do both)
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_dataset as bd          # load_kb(), SYSTEM, emp_str
import evaluate as ev               # company_mentions, numbers_in, lead_count, norm, BASE_MODEL

ADAPTER = ROOT / "adapters" / "georgia_ev_lora"      # v2
ANSWERS = ROOT / "eval_results" / "diagnostic_v2_answers.json"
XLSX = ROOT / "eval_results" / "diagnostic_v2.xlsx"
REPORT = ROOT / "logs" / "diagnostic_v2.md"


# ----------------------------------------------------------------- probe set
def names(df_sub):
    return sorted(df_sub.drop_duplicates("Company")["Company"].astype(str).tolist())


def numstr(v):
    """Canonical comma-free digit string for number matching (matches numbers_in)."""
    return str(int(v))


PROBES = []


def add_probe(shape, question, kind, expected_companies=None, expected_count=None,
              expected_numbers=None, expected_company=None, note=""):
    PROBES.append({
        "id": len(PROBES) + 1,
        "shape": shape,
        "question": question,
        "kind": kind,                                   # names|breakdown|superlative|aggregate|refusal
        "expected_companies": expected_companies or [],
        "expected_count": expected_count,
        "expected_numbers": expected_numbers or [],
        "expected_company": expected_company,
        "note": note,
    })


def build_probes(df):
    ALL = sorted(df["Company"].astype(str).unique())

    # ---------- single-fact baseline (sanity) ----------
    for comp in ["Kia Georgia Inc.", "Novelis Inc.", "Hyundai Transys Georgia Powertrain"]:
        sub = df[df["Company"] == comp]
        if len(sub) == 0:
            continue
        role = sub.iloc[0].get("EV Supply Chain Role")
        loc = sub.iloc[0].get("Updated Location")
        add_probe("single_fact", f"What is {comp}'s EV supply chain role, and where is it located?",
                  "aggregate", expected_numbers=[],
                  note=f"role={role}; loc={loc}",
                  expected_company=comp)
        # store role/loc tokens as expected "numbers"? no — score via token overlap in 'value'
        PROBES[-1]["kind"] = "value"
        PROBES[-1]["expected_text"] = " ".join(str(x) for x in [role, loc] if x and str(x) != "nan")

    # ---------- COUNTS (how many) ----------
    add_probe("count", "How many companies are classified under each Category in the Georgia EV KB?",
              "breakdown",
              expected_numbers=[numstr(v) for v in df["Category"].value_counts().values] + [numstr(len(df))],
              note="category breakdown + total")
    rel = df["EV / Battery Relevant"].value_counts()
    add_probe("count", "How many companies are marked Yes, No, and Indirect for EV/Battery relevance?",
              "breakdown",
              expected_numbers=[numstr(rel.get("Yes", 0)), numstr(rel.get("No", 0)), numstr(rel.get("Indirect", 0))],
              note="Yes/No/Indirect")
    for cat in ["Tier 1", "Tier 2/3", "Tier 1/2", "OEM"]:
        n = int((df["Category"] == cat).sum())
        add_probe("count", f"How many companies are classified as {cat} in the Georgia EV KB?",
                  "breakdown", expected_numbers=[numstr(n)], note=f"{cat}={n}")
    bat = df[df["EV Supply Chain Role"].isin(["Battery Cell", "Battery Pack"])]
    add_probe("count", "How many Georgia companies have a Battery Cell or Battery Pack role?",
              "breakdown", expected_numbers=[numstr(bat["Company"].nunique())], note="battery roles count")
    add_probe("count", "How many Vehicle Assembly companies are in the Georgia EV KB?",
              "breakdown", expected_numbers=[numstr((df["EV Supply Chain Role"] == "Vehicle Assembly").sum())])

    # ---------- GROUP-BY DISTRIBUTION ----------
    t1 = df[df["Category"] == "Tier 1"]
    t1rel = t1["EV / Battery Relevant"].value_counts()
    add_probe("distribution",
              "How are Tier 1 Georgia companies distributed across the EV/Battery Relevance classifications (Yes, No, Indirect)?",
              "breakdown",
              expected_numbers=[numstr(t1rel.get("Yes", 0)), numstr(t1rel.get("No", 0)), numstr(t1rel.get("Indirect", 0))],
              note=f"Tier1 across relevance: {dict(t1rel)}")
    va = df[df["EV Supply Chain Role"] == "Vehicle Assembly"]
    add_probe("distribution",
              "Break down the Vehicle Assembly companies in Georgia by their Primary OEM.",
              "names", expected_companies=names(va), expected_count=va["Company"].nunique(),
              note="vehicle assembly by OEM")

    # ---------- MULTI-CONSTRAINT FILTERS (the fabrication zone) ----------
    def filt(mask, shape, q, note=""):
        sub = df[mask]
        add_probe(shape, q, "names", expected_companies=names(sub),
                  expected_count=sub["Company"].nunique(), note=note)

    filt((df["Category"] == "Tier 1") & (df["EV Supply Chain Role"] == "Materials"),
         "multi_filter",
         "Identify Tier 1 Georgia companies classified under the Materials EV Supply Chain Role and list their products.",
         "Tier1 ∩ Materials")
    filt((df["Category"] == "Tier 2/3") & (df["EV / Battery Relevant"] == "Yes"),
         "multi_filter",
         "Which Tier 2/3 Georgia companies are directly EV/battery relevant (marked 'Yes'), and what are their roles?",
         "Tier2/3 ∩ Yes")
    filt((df["EV Supply Chain Role"] == "Thermal Management") & (df["Employment"] < 200),
         "multi_filter",
         "Which Georgia Thermal Management companies have fewer than 200 employees?",
         "Thermal ∩ emp<200")
    filt((df["Category"] == "Tier 1/2") & (df["EV Supply Chain Role"].isin(["Battery Cell", "Battery Pack"])),
         "multi_filter",
         "List the Tier 1/2 Georgia companies that have a Battery Cell or Battery Pack role.",
         "Tier1/2 ∩ battery")
    filt((df["Category"] == "Tier 2/3") & (df["EV Supply Chain Role"] == "Materials"),
         "multi_filter",
         "Which Tier 2/3 Georgia companies are in the Materials role?",
         "Tier2/3 ∩ Materials")
    filt(df["Primary OEMs"].fillna("").str.contains("Hyundai|Kia", case=False) & (df["Category"] == "Tier 1"),
         "multi_filter",
         "Which Tier 1 Georgia suppliers are linked to Hyundai or Kia through their Primary OEMs?",
         "Tier1 ∩ Hyundai/Kia")
    filt((df["Employment"] > 1000) & (df["EV / Battery Relevant"] == "Indirect"),
         "multi_filter",
         "Which Georgia companies employ more than 1,000 workers but are only Indirectly EV-relevant?",
         "emp>1000 ∩ Indirect")

    # ---------- SUPERLATIVE / ARGMAX ----------
    top = df.dropna(subset=["Employment"]).sort_values("Employment", ascending=False).drop_duplicates("Company")
    r0 = top.iloc[0]
    add_probe("superlative", "Which company in the Georgia EV KB has the highest employment, and how many employees is that?",
              "superlative", expected_company=str(r0["Company"]),
              expected_numbers=[numstr(r0["Employment"])], note="global argmax employment")
    top5 = top.head(5)
    add_probe("superlative", "List the top 5 Georgia companies by employment size.",
              "names", expected_companies=names(top5), expected_count=5, note="top5 employment")
    for county in ["Bryan County", "Troup County"]:
        ge = df[(df["County"] == county)].dropna(subset=["Employment"])
        if len(ge) == 0:
            continue
        rr = ge.loc[ge["Employment"].idxmax()]
        add_probe("superlative",
                  f"In {county}, which company has the highest employment, and what is its EV supply chain role?",
                  "superlative", expected_company=str(rr["Company"]),
                  expected_numbers=[numstr(rr["Employment"])], note=f"argmax in {county}")

    # ---------- AGGREGATION (sums) ----------
    add_probe("aggregation", "What is the total combined employment across all companies in the Georgia EV KB?",
              "aggregate", expected_numbers=[numstr(df["Employment"].dropna().sum())], note="grand total emp")
    et = df.dropna(subset=["County", "Employment"]).groupby("County")["Employment"].sum().sort_values(ascending=False)
    add_probe("aggregation", "Which Georgia county has the highest total employment across all companies, and what is that total?",
              "superlative", expected_company=str(et.index[0]).replace(" County", ""),
              expected_numbers=[numstr(et.iloc[0])], note="top county by total emp")
    t1e = df[df["Category"] == "Tier 1"].dropna(subset=["County", "Employment"]).groupby("County")["Employment"].sum()
    if len(t1e):
        add_probe("aggregation", "What is the total employment of all Tier 1 companies in the Georgia EV KB?",
                  "aggregate", expected_numbers=[numstr(t1e.sum())], note="tier1 total emp")

    # ---------- SET OPS / NEGATION ----------
    bat_counties = set(df[df["EV Supply Chain Role"].isin(["Battery Cell", "Battery Pack"])]["County"].dropna())
    t1_counties = set(df[df["Category"] == "Tier 1"]["County"].dropna())
    gap = sorted(c.replace(" County", "") for c in (t1_counties - bat_counties))
    add_probe("setop_negation",
              "Which Georgia counties have Tier 1 automotive suppliers but no Battery Cell or Battery Pack suppliers?",
              "breakdown", expected_numbers=[numstr(len(gap))],
              note=f"{len(gap)} gap counties")
    singles = [role for role, g in df.groupby("EV Supply Chain Role") if g["Company"].nunique() == 1]
    add_probe("setop_negation",
              "Which EV supply chain roles in Georgia are served by only a single company (single-point-of-failure risk)?",
              "breakdown", expected_numbers=[numstr(len(singles))], note=f"{len(singles)} single-source roles")

    # ---------- plain LIST (role / category / county enumeration) ----------
    for role in ["Vehicle Assembly", "Materials", "Thermal Management"]:
        sub = df[df["EV Supply Chain Role"] == role]
        add_probe("list_enum", f"List every Georgia company with the EV Supply Chain Role '{role}'.",
                  "names", expected_companies=names(sub), expected_count=sub["Company"].nunique(),
                  note=f"role={role}")
    add_probe("list_enum", "List all Tier 1/2 companies in the Georgia EV KB.",
              "names", expected_companies=names(df[df["Category"] == "Tier 1/2"]),
              expected_count=df[df["Category"] == "Tier 1/2"]["Company"].nunique())

    # ---------- EXISTENCE / REFUSAL ----------
    add_probe("refusal", "Which Georgia companies manufacture solid-state batteries?",
              "refusal", note="no solid-state battery companies in KB")
    add_probe("refusal", "What is the annual revenue of Kia Georgia Inc.?",
              "refusal", note="revenue not in KB")
    add_probe("refusal", "List the EV supply chain companies located in Florida.",
              "refusal", note="KB is Georgia-only")


# ----------------------------------------------------------------- generation
def generate():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    df = bd.load_kb()
    build_probes(df)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(ev.BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(ev.BASE_MODEL, dtype=dtype)
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.to(device).eval()

    out = {}
    for i, p in enumerate(PROBES):
        msgs = [{"role": "system", "content": bd.SYSTEM}, {"role": "user", "content": p["question"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            o = model.generate(**inputs, max_new_tokens=700, do_sample=False,
                               temperature=None, top_p=None, top_k=None,
                               pad_token_id=tok.eos_token_id)
        ans = tok.decode(o[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        out[str(p["id"])] = ans
        print(f"[{i+1}/{len(PROBES)}] {p['shape']:16s} qid={p['id']} -> {len(ans)} chars", flush=True)
    ANSWERS.parent.mkdir(exist_ok=True)
    ANSWERS.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved", ANSWERS)


# ----------------------------------------------------------------- scoring
REFUSAL_MARKERS = ["does not", "doesn't", "not provide", "no information", "cannot",
                   "only covers", "only includes", "no compan", "none", "not in the",
                   "not list", "does not contain", "no data"]


def score_probe(p, pred, ALL):
    """Score one probe answer against its KB-computed gold. Returns (correct, detail).
    Shared by the v2 diagnostic and the 14B evaluation so both use identical criteria."""
    kind = p["kind"]
    pred_companies = ev.company_mentions(pred, ALL)
    pred_nums = set(ev.numbers_in(pred))
    model_count = ev.lead_count(pred)

    if kind == "names":
        exp = set(p["expected_companies"])
        inter = exp & pred_companies
        recall = len(inter) / len(exp) if exp else (1.0 if not pred_companies else 0.0)
        precision = len(inter) / len(pred_companies) if pred_companies else (1.0 if not exp else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        count_ok = (model_count == str(p["expected_count"])) if p["expected_count"] is not None else None
        consistency = (int(model_count) == len(pred_companies)) if model_count is not None else None
        correct = (f1 >= 0.6) and (count_ok is not False)
        return correct, dict(name_f1=round(f1, 2), name_p=round(precision, 2), name_r=round(recall, 2),
                             exp_n=p["expected_count"], model_count=model_count, count_ok=count_ok,
                             listed=len(pred_companies), consistency=consistency,
                             hallucinated=sorted(pred_companies - exp)[:6], missing=sorted(exp - pred_companies)[:6])

    if kind == "breakdown":
        exp_nums = p["expected_numbers"]
        present = [n for n in exp_nums if n in pred_nums]
        frac = len(present) / len(exp_nums) if exp_nums else 0.0
        return frac >= 0.8, dict(expected_numbers=exp_nums, present=present, frac=round(frac, 2), model_count=model_count)

    if kind == "superlative":
        comp = p["expected_company"]
        comp_ok = comp is not None and comp in pred_companies
        if not comp_ok and comp:
            comp_ok = ev.norm(comp).split()[0] in ev.norm(pred)
        num_ok = all(n in pred_nums for n in p["expected_numbers"]) if p["expected_numbers"] else True
        return bool(comp_ok and num_ok), dict(expected_company=comp, comp_ok=comp_ok,
                                              expected_numbers=p["expected_numbers"], num_ok=num_ok)

    if kind == "aggregate":
        num_ok = all(n in pred_nums for n in p["expected_numbers"]) if p["expected_numbers"] else False
        return bool(num_ok), dict(expected_numbers=p["expected_numbers"], num_ok=num_ok, pred_nums=sorted(pred_nums)[:10])

    if kind == "value":
        exp_text = set(ev.norm(p.get("expected_text", "")).split())
        pred_text = set(ev.norm(pred).split())
        overlap = len(exp_text & pred_text) / len(exp_text) if exp_text else 0.0
        return overlap >= 0.6, dict(token_overlap=round(overlap, 2))

    if kind == "refusal":
        declined = any(m in pred.lower() for m in REFUSAL_MARKERS)
        named = len(pred_companies)
        return (declined and named <= 1), dict(declined=declined, named_companies=named)

    return False, {}


def score_to_df(answers):
    """Score a {probe_id: answer} dict into a per-probe DataFrame."""
    df = bd.load_kb()
    build_probes(df)
    ALL = sorted(df["Company"].astype(str).unique())
    rows = []
    for p in PROBES:
        pred = answers.get(str(p["id"]), "")
        correct, detail = score_probe(p, pred, ALL)
        rows.append({"id": p["id"], "shape": p["shape"], "kind": p["kind"], "question": p["question"],
                     "note": p["note"], "correct": correct, "model_answer": pred[:1200],
                     "detail": json.dumps(detail, default=str)})
    return pd.DataFrame(rows)


def score():
    answers = json.loads(ANSWERS.read_text())
    rdf = score_to_df(answers)
    rdf.to_excel(XLSX, index=False)

    # per-shape aggregation
    by_shape = rdf.groupby("shape")["correct"].agg(["mean", "sum", "size"]).sort_values("mean")
    overall = rdf["correct"].mean()

    REPORT.parent.mkdir(exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("# v2 (7B) Diagnostic — failure map by question shape\n\n")
        f.write(f"Probe set: **{len(rdf)} questions** generated deterministically from the KB "
                f"(never from the 50 human Q&A). Each scored against the KB-computed gold.\n\n")
        f.write(f"**Overall probe accuracy: {overall:.0%}** ({int(rdf['correct'].sum())}/{len(rdf)})\n\n")
        f.write("## Accuracy by shape (worst first)\n\n")
        f.write("| shape | correct | n | acc |\n|---|---|---|---|\n")
        for shape, r in by_shape.iterrows():
            f.write(f"| {shape} | {int(r['sum'])} | {int(r['size'])} | {r['mean']:.0%} |\n")
        f.write("\n## Failing examples (concrete)\n\n")
        for _, r in rdf[~rdf["correct"]].iterrows():
            f.write(f"- **[{r['shape']}]** {r['question']}\n")
            f.write(f"  - expected: {r['note']}\n")
            f.write(f"  - score detail: `{r['detail']}`\n")
            ans1 = r["model_answer"].replace("\n", " ")[:280]
            f.write(f"  - model said: {ans1}…\n\n")
    print(f"OVERALL probe accuracy: {overall:.0%} ({int(rdf['correct'].sum())}/{len(rdf)})")
    print(by_shape)
    print("wrote", REPORT, "and", XLSX)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["generate", "score", "all"])
    a = ap.parse_args()
    if a.mode in ("generate", "all"):
        generate()
    if a.mode in ("score", "all"):
        score()
