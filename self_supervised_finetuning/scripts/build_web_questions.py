"""Draft GNEM-Web-18 from web facts PROVEN to be in the effective (sampled) training
set. Mix of numeric (investment/jobs/capacity) and NON-NUMERIC (location, parent
company, product, supplied vehicles) facts so the benchmark tests web KNOWLEDGE
absorption, not just numeric memorization. For each fact: locate its source page
(url/date/title) by an evidence phrase, confirm the phrase is in the decoded effective
training text, confirm the answer is absent from the structured KB, count exposure.
Emits a draft for the user to validate before freezing.
"""
from __future__ import annotations

import json
import re
import glob
from pathlib import Path

ROOT = Path("/home/sreeja/georgia-ev-finetune")
EFF = ROOT / "self_supervised_finetuning/outputs/question_eval/effective_web_text.txt"
OUT = ROOT / "self_supervised_finetuning/benchmarks/gnem_bench_v1/web_questions_v1.json"

# Each item: dict with evidence_phrase, question, answer_type, and either
#   numeric: gold_text + value ; or non-numeric: gold_text + match (list of required
#   normalized substrings in the model output).
ITEMS = [
    # --- non-numeric (5) -------------------------------------------------------------
    dict(phrase="SK On is working with Hyundai on a $5 billion battery joint-venture plant in Cartersville",
         q="According to the collected web corpus, which automaker is SK On working with on a battery joint-venture plant in Cartersville, Georgia?",
         answer_type="company", gold="Hyundai", match=["hyundai"]),
    dict(phrase="Anovion Technologies is a supplier of premium synthetic graphite anode materials for lithium-ion batteries",
         q="According to the collected web corpus, what battery material does Anovion Technologies manufacture at its Georgia facility?",
         answer_type="product", gold="synthetic graphite anode materials", match=["synthetic graphite"]),
    dict(phrase="SK On is the parent company of SK Battery America",
         q="According to the collected web corpus, what is the parent company of SK Battery America?",
         answer_type="company", gold="SK On", match=["sk on"]),
    dict(phrase="supplier of automotive battery modules and energy storage systems to SK Battery America",
         q="According to the collected web corpus, what products does Duckyang supply to SK Battery America?",
         answer_type="product", gold="automotive battery modules and energy storage systems",
         match=["battery modules", "energy storage"]),
    dict(phrase="including the Ford F-150 Lightning and the Volkswagen ID.4",
         q="According to the collected web corpus, the batteries SK Battery America produces in Georgia supply which two electric vehicles?",
         answer_type="product_list", gold="Ford F-150 Lightning and Volkswagen ID.4",
         match=["f-150 lightning", "id.4"]),
    # --- numeric currency (7) --------------------------------------------------------
    dict(phrase="GF Casting Solutions AG will invest over $184 million for a new facility in Augusta",
         q="According to the collected web corpus, how much will GF Casting Solutions AG invest in its new Augusta facility?",
         answer_type="currency", gold="over $184 million", value=184_000_000),
    dict(phrase="Hanon Systems will invest more than $40 million for a manufacturing facility in Bulloch County",
         q="According to the collected web corpus, how much will Hanon Systems invest in its Bulloch County manufacturing facility?",
         answer_type="currency", gold="more than $40 million", value=40_000_000),
    dict(phrase="Duracell Manufacturing officials announced last week the company will invest $25 million",
         q="According to the collected web corpus, how much will Duracell Manufacturing invest to expand its battery component manufacturing operations?",
         answer_type="currency", gold="$25 million", value=25_000_000),
    dict(phrase="Hyundai Executive Chairman Euisun Chung confirmed the company will invest $5.54 billion",
         q="According to the collected web corpus, how much did Hyundai confirm it would invest to open its facility at the Bryan County Megasite?",
         answer_type="currency", gold="$5.54 billion", value=5_540_000_000),
    dict(phrase="The battery plant is part of a $7.6 billion complex planned to eventually employ 8,500 people",
         q="According to the collected web corpus, what is the total value of the complex that the HL-GA battery plant is part of?",
         answer_type="currency", gold="$7.6 billion", value=7_600_000_000),
    dict(phrase="Rivian will invest $5 billion into building the facility",
         q="According to the collected web corpus, how much will Rivian invest into building its Georgia facility?",
         answer_type="currency", gold="$5 billion", value=5_000_000_000),
    dict(phrase="FREYR plans to invest some $1.7 billion in the first phase of Giga America",
         q="According to the collected web corpus, how much does FREYR plan to invest in the first phase of Giga America?",
         answer_type="currency", gold="some $1.7 billion", value=1_700_000_000),
    # --- numeric count (4) -----------------------------------------------------------
    dict(phrase="The project is expected to create 350 new jobs in Richmond County",
         q="According to the collected web corpus, how many new jobs is GF Casting Solutions' Richmond County project expected to create?",
         answer_type="count", gold="350 new jobs", value=350),
    dict(phrase="The $60 million project will create 600 new jobs",
         q="According to the collected web corpus, how many new jobs will Norma Precision's $60 million project create?",
         answer_type="count", gold="600 new jobs", value=600),
    dict(phrase="The project is expected to create about 8,100 new jobs",
         q="According to the collected web corpus, about how many new jobs is Hyundai Motor Group's Bryan County project expected to create?",
         answer_type="count", gold="about 8,100 new jobs", value=8100),
    dict(phrase="SK Battery is cutting nearly 1,000 jobs at its Commerce plant",
         q="According to the collected web corpus, approximately how many jobs is SK Battery cutting at its Commerce plant?",
         answer_type="count", gold="nearly 1,000 jobs", value=1000),
    # --- numeric capacity (2) --------------------------------------------------------
    dict(phrase="The first phase will be a cell production module with an annual capacity of approximately 34 GWh",
         q="According to the collected web corpus, what is the approximate annual capacity of the first-phase cell production module at Giga America?",
         answer_type="capacity_gwh", gold="approximately 34 GWh", value=34),
    dict(phrase="The company will have a combined capacity to make 22 GWh of battery cells a year",
         q="According to the collected web corpus, what combined annual battery-cell capacity will SK Battery America have in Georgia?",
         answer_type="capacity_gwh", gold="22 GWh a year", value=22),
]

STOP = {"the", "and", "for", "inc", "llc", "corp", "corporation", "company", "co", "ltd",
        "ag", "america", "american", "group", "motor", "industrial", "systems", "solutions",
        "technologies", "battery", "manufacturing", "georgia", "usa", "us", "north"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _company_tokens(title: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", (title or "").lower()) if len(t) > 2 and t not in STOP}


def _kb_hit(kb_rows, title, item) -> bool:
    """Company-aware KB-absence check. numeric: gold value in the same company's KB row.
    non-numeric: ALL match phrases present in the same company's KB row."""
    qtok = _company_tokens(title)
    if not qtok:
        return False
    for r in kb_rows:
        if not (_company_tokens(r.get("Company", "")) & qtok):
            continue
        blob = " ".join(str(v) for v in r.values()).lower().replace(",", "")
        if "value" in item:
            if str(item["value"]) in blob:
                return True
        else:
            if all(m in blob for m in item.get("match", [])):
                return True
    return False


def main() -> int:
    eff = _norm(EFF.read_text())
    eff_low = eff.lower()
    kb_rows = [json.loads(l) for l in open(ROOT / "kb_full.jsonl")]

    md_index = []
    for f in glob.glob(str(ROOT / "self_supervised_finetuning/data/raw/llm_wiki_pages/*.md")):
        txt = open(f, encoding="utf-8", errors="replace").read()
        fm = dict(re.findall(r'^(title|source_url|publication_date):\s*"?(.*?)"?\s*$', txt, re.M))
        md_index.append((_norm(txt), fm))

    questions, problems = [], []
    for i, it in enumerate(ITEMS, 1):
        np = _norm(it["phrase"])
        in_eff = np in eff
        occ = eff.count(np)
        src = next(((fm) for (t, fm) in md_index if np in t), None)
        title = (src or {}).get("title", "")
        kb_hit = _kb_hit(kb_rows, title, it)
        gold = {"answer_text": it["gold"], "value": it.get("value"), "entities": [],
                "ordered": [], "stages": None}
        if "match" in it:
            gold["match"] = it["match"]
        rec = {
            "question_id": f"web_q{i:02d}", "section": "web", "category": "web_fact",
            "operations": ["recall"], "operation_count": 1,
            "question": it["q"], "answer_type": it["answer_type"], "gold": gold,
            "source_url": (src or {}).get("source_url"),
            "source_date": (src or {}).get("publication_date"),
            "source_title": (src or {}).get("title"),
            "evidence_phrase": it["phrase"],
            "proven_in_effective_training": in_eff,
            "exact_answer_occurrences": occ,
            "exposure_over_run": occ,
            "absent_from_structured_kb": not kb_hit,
            "kb_overlap_check": {"answer_in_kb": kb_hit},
        }
        questions.append(rec)
        if not in_eff or not src or kb_hit:
            problems.append((rec["question_id"], f"in_eff={in_eff} src={'Y' if src else 'N'} kb_hit={kb_hit}"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"benchmark": "GNEM-Web-18", "version": "draft-2",
                               "n": len(questions), "questions": questions}, indent=2, ensure_ascii=False))
    from collections import Counter
    print(f"wrote {OUT}  ({len(questions)} questions)")
    print("  answer_type mix:", dict(Counter(q["answer_type"] for q in questions)))
    print(f"  proven_in_effective_training: {sum(q['proven_in_effective_training'] for q in questions)}/{len(questions)}")
    print(f"  absent_from_structured_kb:    {sum(q['absent_from_structured_kb'] for q in questions)}/{len(questions)}")
    print(f"  source resolved:              {sum(bool(q['source_url']) for q in questions)}/{len(questions)}")
    if problems:
        print("\nPROBLEMS:", problems)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
