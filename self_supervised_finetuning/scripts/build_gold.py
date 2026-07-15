"""Build the frozen GNEM-Bench-v1 gold file: GNEM-KB-42 (from the human-validated xlsx,
categorized + gold entities/values parsed) + GNEM-Web-18 (validated web questions).
Reports are kept per-section; KB is split into deterministic vs judgment subsets.

operation_count = len(operations) for deterministic questions (drives the report-time
complexity grouping); null for judgment (operation-count difficulty is not applicable to
open-ended recommendation questions).
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from bench_lib import companies_in_text, parse_count, norm_county

ROOT = Path("/home/sreeja/georgia-ev-finetune")
QE = ROOT / "self_supervised_finetuning/outputs/question_eval"
BENCH = ROOT / "self_supervised_finetuning/benchmarks/gnem_bench_v1"

# num -> (category, operations, subset). Authored from the question texts.
CAT = {
    1: ("multi_filter_list", ["filter"], "deterministic"),
    2: ("multi_filter_list", ["filter"], "deterministic"),
    3: ("multi_filter_list", ["filter"], "deterministic"),
    4: ("multi_filter_list", ["filter"], "deterministic"),
    5: ("multi_filter_list", ["filter"], "deterministic"),
    6: ("direct_lookup", ["lookup"], "deterministic"),
    7: ("max_min", ["filter", "argmax"], "deterministic"),
    8: ("geographic_aggregation", ["filter", "group_by", "sum", "argmax"], "deterministic"),
    9: ("geographic_aggregation", ["group_by", "sum", "argmax"], "deterministic"),
    10: ("multi_filter_list", ["filter"], "deterministic"),
    11: ("direct_lookup", ["lookup"], "deterministic"),
    12: ("multi_filter_list", ["filter"], "deterministic"),
    13: ("multi_filter_list", ["filter"], "deterministic"),
    14: ("multi_filter_list", ["filter"], "deterministic"),
    15: ("judgment", ["filter", "judgment"], "judgment"),
    16: ("judgment", ["filter", "judgment"], "judgment"),
    17: ("judgment", ["filter", "judgment"], "judgment"),
    18: ("multi_filter_list", ["filter"], "deterministic"),
    19: ("multi_filter_list", ["filter"], "deterministic"),
    20: ("multi_filter_list", ["filter"], "deterministic"),
    21: ("judgment", ["filter", "judgment"], "judgment"),
    22: ("multi_filter_list", ["filter"], "deterministic"),
    23: ("count", ["group_by", "count", "filter"], "deterministic"),
    24: ("judgment", ["filter", "judgment"], "judgment"),
    25: ("count", ["filter", "count"], "deterministic"),
    26: ("multi_filter_list", ["filter"], "deterministic"),
    27: ("multi_filter_list", ["filter"], "deterministic"),
    28: ("multi_filter_list", ["filter"], "deterministic"),
    29: ("multi_filter_list", ["filter"], "deterministic"),
    30: ("top_k_ranking", ["filter", "sort", "top_k"], "deterministic"),
    31: ("judgment", ["filter", "judgment"], "judgment"),
    32: ("multi_filter_list", ["filter", "text_match"], "deterministic"),
    33: ("multi_filter_list", ["filter"], "deterministic"),
    34: ("top_k_ranking", ["filter", "sort", "top_k"], "deterministic"),
    35: ("multi_filter_list", ["filter"], "deterministic"),
    36: ("multi_filter_list", ["filter"], "deterministic"),
    37: ("judgment", ["judgment"], "judgment"),
    38: ("multi_filter_list", ["filter"], "deterministic"),
    39: ("judgment", ["judgment"], "judgment"),
    40: ("judgment", ["filter", "judgment"], "judgment"),
    41: ("judgment", ["judgment"], "judgment"),
    42: ("judgment", ["judgment"], "judgment"),
}

# answer_type by category (drives the parser/scorer).
ATYPE = {
    "direct_lookup": "entity_or_text", "multi_filter_list": "entity_list",
    "count": "count", "sum_aggregate": "number", "max_min": "entity_value",
    "top_k_ranking": "entity_list_ordered", "comparison": "text",
    "geographic_aggregation": "county_value", "not_found_abstention": "abstain",
    "judgment": "entity_list",
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    kb_raw = json.loads((QE / "kb_gold_raw.json").read_text())
    assert len(kb_raw) == 42, f"expected 42 KB gold, got {len(kb_raw)}"
    web = json.loads((BENCH / "web_questions_v1.json").read_text())["questions"]
    assert len(web) == 18, f"expected 18 web questions, got {len(web)}"

    questions = []
    for r in kb_raw:
        num = int(r["num"])
        cat, ops, subset = CAT[num]
        gold_text = r["gold_answer"]
        ents = sorted(companies_in_text(gold_text))
        atype = ATYPE[cat]
        gold = {"answer_text": gold_text, "entities": ents, "value": None,
                "county": None, "ordered": [], "answer_type": atype, "stages": None}
        if cat in ("count",) or (cat == "geographic_aggregation"):
            gold["value"] = parse_count(gold_text)
        if cat == "geographic_aggregation":
            gold["county"] = norm_county(gold_text)
        if cat == "top_k_ranking":
            gold["ordered"] = ents  # order approximate; ranking scored by set overlap + rank-1
        questions.append({
            "question_id": f"kb_q{num:02d}", "section": "kb", "subset": subset,
            "use_case_category": r["use_case_category"], "question": r["question"],
            "category": cat, "operations": ops,
            "operation_count": (len(ops) if subset == "deterministic" else None),
            "requires_authoritative_kb": True,
            "gold": gold, "source_url": None,
        })

    for w in web:
        questions.append({
            "question_id": w["question_id"], "section": "web", "subset": "web_absorption",
            "use_case_category": "web_fact", "question": w["question"],
            "category": "web_fact", "operations": ["recall"], "operation_count": 1,
            "requires_authoritative_kb": False,
            "gold": {"answer_text": w["gold"]["answer_text"], "value": w["gold"]["value"],
                     "entities": [], "county": None, "ordered": [],
                     "match": w["gold"].get("match", []),
                     "answer_type": w["answer_type"], "stages": None},
            "source_url": w["source_url"], "source_date": w["source_date"],
            "exact_answer_occurrences": w["exact_answer_occurrences"],
            "absent_from_structured_kb": w["absent_from_structured_kb"],
        })

    payload = {
        "benchmark": "GNEM-Bench-v1",
        "sections": {"GNEM-KB-42": 42, "GNEM-Web-18": 18},
        "version": "1.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "kb_snapshot": f"kb_full.jsonl@{_sha(ROOT / 'kb_full.jsonl')[:12]}",
        "web_snapshot": f"web_corpus.jsonl@{_sha(ROOT / 'self_supervised_finetuning/data/raw/web_corpus.jsonl')[:12]}",
        "freeze": {
            "kb_gold_frozen_before_kb_web_training": True,
            "web_gold_frozen_before_kb_web_inference": True,
            "gold_frozen_before_scoring": True,
            "gold_frozen_before_output_review": True,
        },
        "n_questions": len(questions),
        "questions": questions,
    }
    out = BENCH / "questions_gold_v1.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # census
    from collections import Counter
    det = [q for q in questions if q["subset"] == "deterministic"]
    jud = [q for q in questions if q["subset"] == "judgment"]
    print(f"wrote {out}")
    print(f"  KB deterministic: {len(det)}  | KB judgment: {len(jud)}  | web: 18  | total {len(questions)}")
    print("  KB categories:", dict(Counter(q["category"] for q in questions if q["section"] == "kb")))
    print("  KB gold entity-set sizes (det):",
          {q['question_id']: len(q['gold']['entities']) for q in det if q['category'] in ('multi_filter_list','top_k_ranking')})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
