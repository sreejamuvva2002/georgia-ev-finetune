"""Parse + score one system's raw outputs against the frozen GNEM-Bench-v1 gold.

Per-type parsers (point 6) turn a raw completion into a normalized answer; the raw
outputs are never mutated (read from the immutable raw_outputs/ file) — scored results
are a NEW file. Metrics per category/subset/section; hallucination Type-A/B automated,
Type-C reserved. Section-level Wilson CIs.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from bench_lib import (companies_in_text, kb_companies, parse_count, parse_currency,
                       parse_gwh, parse_year, norm_county, set_prf, wilson_ci, _canon_company)

# ---- per-type parsers --------------------------------------------------------------
def entity_list_parser(text): return companies_in_text(text)
def count_parser(text): return parse_count(text)
def currency_parser(text): return parse_currency(text)
def gwh_parser(text): return parse_gwh(text)
def county_parser(text): return norm_county(text)

_CAND = re.compile(r"\b([A-Z][A-Za-z&.\-]+(?:\s+[A-Z][A-Za-z&.\-]+){0,4}"
                   r"(?:\s+(?:Inc|LLC|Corp|Corporation|Co|Ltd|LP|GmbH|AG))\.?)")


def hallucinated_typeA(text: str) -> list[str]:
    """Heuristic: company-like names (…Inc/LLC/Corp) in the output whose canonical form
    matches NO KB company -> not-in-KB (Type-A). Approximate; flagged as heuristic."""
    kb = {c["canon"] for c in kb_companies()}
    out = []
    for m in _CAND.finditer(text or ""):
        canon = _canon_company(m.group(1))
        if canon and len(canon) >= 3 and canon not in kb and not any(canon in k or k in canon for k in kb):
            out.append(m.group(1).strip())
    return sorted(set(out))


def _num_eq(a, b, rel=0.001):
    if a is None or b is None:
        return False
    return abs(a - b) <= max(1, abs(b) * rel)


def score_one(q: dict, output: str) -> dict:
    cat, atype = q["category"], q["gold"]["answer_type"]
    gold = q["gold"]
    parsed, scores = {}, {}

    if q["section"] == "web":
        if atype == "currency":
            pv = currency_parser(output); parsed["value"] = pv
            scores["correct"] = _num_eq(pv, gold["value"])
        elif atype == "count":
            pv = count_parser(output); parsed["value"] = pv
            scores["correct"] = _num_eq(pv, gold["value"])
        elif atype == "capacity_gwh":
            pv = gwh_parser(output); parsed["value"] = pv
            scores["correct"] = _num_eq(pv, gold["value"])
        else:
            # non-numeric (location/company/product/product_list): ALL gold `match`
            # phrases must appear (normalized-substring) in the output.
            low = re.sub(r"\s+", " ", (output or "").lower())
            phrases = gold.get("match", [])
            hits = [m for m in phrases if m in low]
            parsed["match_hits"] = hits
            scores["correct"] = bool(phrases) and len(hits) == len(phrases)
            if len(phrases) > 1:
                scores["partial"] = round(len(hits) / len(phrases), 3)
        parsed["hallucinated_typeA"] = hallucinated_typeA(output)
        return {"parsed": parsed, "scores": scores}

    # KB section
    if atype in ("entity_list", "entity_list_ordered"):
        pred = entity_list_parser(output)
        parsed["entities"] = sorted(pred)
        scores.update(set_prf(pred, set(gold["entities"])))
        if cat in ("top_k_ranking", "max_min") and gold["entities"]:
            scores["rank1_correct"] = bool(pred) and (sorted(gold["entities"])[0] in pred)
    elif atype == "count":
        pv = count_parser(output); parsed["value"] = pv
        scores["exact"] = _num_eq(pv, gold["value"])
    elif atype == "county_value":
        pc = county_parser(output); parsed["county"] = pc
        scores["county_correct"] = (pc == gold["county"]) if gold["county"] else None
        pv = count_parser(output); parsed["value"] = pv
        scores["value_correct"] = _num_eq(pv, gold["value"]) if gold["value"] else None
    elif atype == "entity_value":  # max_min single company + value
        pred = entity_list_parser(output); parsed["entities"] = sorted(pred)
        gold_top = sorted(gold["entities"])[0] if gold["entities"] else None
        scores["entity_correct"] = (gold_top in pred) if gold_top else None
    elif atype == "entity_or_text":  # direct_lookup
        pred = entity_list_parser(output); parsed["entities"] = sorted(pred)
        scores.update(set_prf(pred, set(gold["entities"])) if gold["entities"] else {})
    parsed["hallucinated_typeA"] = hallucinated_typeA(output)
    return {"parsed": parsed, "scores": scores}


def _primary_metric(q, s):
    """A single 0/1 (or None) 'correct' per question for section accuracy + CI."""
    sc = s["scores"]
    if q["section"] == "web":
        return 1 if sc.get("correct") else 0
    cat = q["category"]
    if cat in ("multi_filter_list", "top_k_ranking", "judgment", "direct_lookup"):
        f1 = sc.get("f1")
        return None if f1 is None else (1 if f1 >= 0.5 else 0)  # F1>=.5 as a binary proxy
    if cat == "count":
        return 1 if sc.get("exact") else 0
    if cat == "geographic_aggregation":
        return 1 if sc.get("county_correct") else 0
    if cat == "max_min":
        return 1 if sc.get("entity_correct") else 0
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw_outputs/<label>.json")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = json.loads(Path(args.raw).read_text())
    gold = {q["question_id"]: q for q in json.loads(Path(args.gold).read_text())["questions"]}

    rows = []
    for r in raw["results"]:
        q = gold[r["question_id"]]
        res = score_one(q, r["output"])
        rows.append({"question_id": q["question_id"], "section": q["section"], "subset": q["subset"],
                     "category": q["category"], "operation_count": q.get("operation_count"),
                     "raw_output_ref": {"file": args.raw, "sha": raw.get("raw_output_sha256")},
                     "output": r["output"], **res, "primary_correct": _primary_metric(q, res)})

    # aggregates
    def agg(subrows):
        vals = [x["primary_correct"] for x in subrows if x["primary_correct"] is not None]
        k, n = sum(vals), len(vals)
        acc = (k / n) if n else None
        return {"n_scored": n, "correct": k, "accuracy": round(acc, 4) if acc is not None else None,
                "wilson95": wilson_ci(k, n) if n else None}
    by_section = {sec: agg([x for x in rows if x["section"] == sec]) for sec in ("kb", "web")}
    by_subset = {sub: agg([x for x in rows if x["subset"] == sub])
                 for sub in ("deterministic", "judgment", "web_absorption")}
    by_cat = {c: agg([x for x in rows if x["category"] == c])
              for c in sorted({x["category"] for x in rows})}
    by_opcount = {}
    for oc in sorted({x["operation_count"] for x in rows if x["operation_count"] is not None}):
        by_opcount[oc] = agg([x for x in rows if x["operation_count"] == oc])

    payload = {"label": raw["label"], "gold_file": args.gold,
               "benchmark_sha256": raw.get("benchmark_sha256"),
               "aggregates": {"by_section": by_section, "by_subset": by_subset,
                              "by_category": by_cat, "by_operation_count": by_opcount},
               "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    print(f"  KB-42 acc: {by_section['kb']['accuracy']}  {by_section['kb']['wilson95']}  "
          f"(det {by_subset['deterministic']['accuracy']}, judg {by_subset['judgment']['accuracy']})")
    print(f"  Web-18 acc: {by_section['web']['accuracy']}  {by_section['web']['wilson95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
