"""Combine scored systems into the canonical comparison.json + a readable per-section
Markdown. Sections (GNEM-KB-42, GNEM-Web-18) are reported INDEPENDENTLY — never a
combined score. RAG column reserved. Web section shows Delta vs base and vs KB-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["base", "kb_only", "kb_web", "rag"]
LABELFMT = {"base": "Base", "kb_only": "KB-only", "kb_web": "KB+web", "rag": "RAG"}


def _fmt(a):
    return "—" if a is None else f"{100*a:.1f}%"


def _ci(agg):
    if not agg or agg.get("wilson95") is None:
        return ""
    lo, hi = agg["wilson95"]
    return f" [{100*lo:.0f}–{100*hi:.0f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", nargs="+", required=True, help="label=path ...")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    systems = {}
    for spec in args.scored:
        label, path = spec.split("=", 1)
        systems[label] = json.loads(Path(path).read_text())
    gold = {q["question_id"]: q for q in json.loads(Path(args.gold).read_text())["questions"]}
    present = [l for l in ORDER if l in systems]

    # canonical json
    comparison = {"benchmark": "GNEM-Bench-v1", "systems": present,
                  "sections": {}, "per_question": []}
    for sec_key, sec_name in (("kb", "GNEM-KB-42"), ("web", "GNEM-Web-18")):
        comparison["sections"][sec_name] = {
            l: systems[l]["aggregates"]["by_section"][sec_key] for l in present}

    for qid, q in gold.items():
        row = {"question_id": qid, "section": q["section"], "category": q["category"],
               "subset": q["subset"], "gold": q["gold"]["answer_text"], "systems": {}}
        for l in present:
            r = next((x for x in systems[l]["rows"] if x["question_id"] == qid), None)
            if r:
                row["systems"][l] = {"output": r["output"], "scores": r["scores"],
                                     "primary_correct": r["primary_correct"]}
        comparison["per_question"].append(row)
    Path(args.out_json).write_text(json.dumps(comparison, indent=2, ensure_ascii=False))

    # markdown
    L = ["# GNEM-Bench-v1 — system comparison", "",
         "Sections reported **independently** (no combined score). Accuracy = primary "
         "per-question correctness (list Qs: F1≥0.5; value Qs: exact); [..] = 95% Wilson CI.", ""]

    L += ["## GNEM-KB-42 — structured-KB reasoning & retention", "",
          "| System | KB-42 | Deterministic | Judgment |", "|---|---|---|---|"]
    for l in present:
        a = systems[l]["aggregates"]
        L.append(f"| {LABELFMT[l]} | {_fmt(a['by_section']['kb']['accuracy'])}{_ci(a['by_section']['kb'])} "
                 f"| {_fmt(a['by_subset']['deterministic']['accuracy'])} "
                 f"| {_fmt(a['by_subset']['judgment']['accuracy'])} |")
    L.append("")

    # by category (KB)
    kb_cats = sorted({q["category"] for q in gold.values() if q["section"] == "kb"})
    L += ["### KB-42 by category", "", "| Category | " + " | ".join(LABELFMT[l] for l in present) + " |",
          "|---|" + "---|" * len(present)]
    for c in kb_cats:
        cells = []
        for l in present:
            a = systems[l]["aggregates"]["by_category"].get(c)
            cells.append(_fmt(a["accuracy"]) if a else "—")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("")

    # by operation count
    ocs = sorted({q["operation_count"] for q in gold.values() if q.get("operation_count")})
    if ocs:
        L += ["### KB-42 by operation count (complexity)", "",
              "| #ops | " + " | ".join(LABELFMT[l] for l in present) + " |", "|---|" + "---|" * len(present)]
        for oc in ocs:
            cells = [(_fmt(systems[l]["aggregates"]["by_operation_count"].get(str(oc),
                     systems[l]["aggregates"]["by_operation_count"].get(oc, {})).get("accuracy"))
                      if (str(oc) in systems[l]["aggregates"]["by_operation_count"] or
                          oc in systems[l]["aggregates"]["by_operation_count"]) else "—") for l in present]
            L.append(f"| {oc} | " + " | ".join(cells) + " |")
        L.append("")

    # Web section with deltas
    L += ["## GNEM-Web-18 — training-corpus web-fact absorption", "",
          "| System | Web-18 | Δ vs Base | Δ vs KB-only |", "|---|---|---|---|"]
    base_acc = systems.get("base", {}).get("aggregates", {}).get("by_section", {}).get("web", {}).get("accuracy")
    kbonly_acc = systems.get("kb_only", {}).get("aggregates", {}).get("by_section", {}).get("web", {}).get("accuracy")
    for l in present:
        a = systems[l]["aggregates"]["by_section"]["web"]
        acc = a["accuracy"]
        dvb = f"{100*(acc-base_acc):+.1f}" if (acc is not None and base_acc is not None) else "—"
        dvk = f"{100*(acc-kbonly_acc):+.1f}" if (acc is not None and kbonly_acc is not None) else "—"
        L.append(f"| {LABELFMT[l]} | {_fmt(acc)}{_ci(a)} | {dvb} | {dvk} |")
    L.append("")

    # per-question appendix
    L += ["## Per-question detail", ""]
    for row in comparison["per_question"]:
        L.append(f"### {row['question_id']} ({row['category']}/{row['subset']})")
        L.append(f"**Gold:** {row['gold'][:300]}")
        for l in present:
            s = row["systems"].get(l)
            if s:
                pc = {1: "✓", 0: "✗", None: "·"}[s["primary_correct"]]
                L.append(f"- **{LABELFMT[l]}** {pc}: {s['output'][:220]!r}")
        if "rag" not in present:
            L.append("- **RAG** ·: _pending_")
        L.append("")

    Path(args.out_md).write_text("\n".join(L))
    print(f"wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
