"""Merge my 3 PARAMETRIC systems (base/kb_only/kb_web, graded fresh with the repo's
Deterministic-V2 + DeepEval) with the LLM_Context repo's 7 published CONTEXT systems
into one 10-system comparison.

Comparability rule (user requirement): DeepEval's 6-metric mean is NOT compared directly
across parametric vs context. We report:
  - CORE answer-quality mean (ALL systems): completeness, correctness, company_grounding,
    usefulness.
  - TRACEABILITY (context/KB-evidence systems ONLY): factual_faithfulness, evidence_grounding.
Deterministic-V2 composite is the repo's standard (format/row_id are context-format-specific;
parametric answers have no Evidence table, so those read ~0 — flagged, not blended away).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path("/home/sreeja/georgia-ev-finetune/self_supervised_finetuning")
REPO = ROOT / "external/LLM_Context"
CVP = ROOT / "outputs/context_vs_parametric"
PARAM_MODELS = {"base", "kb_only", "kb_web"}

CORE = ["mean_completeness", "mean_correctness", "mean_company_grounding", "mean_usefulness"]
TRACE = ["mean_factual_faithfulness", "mean_evidence_grounding"]


def _relabel(df):
    df = df.copy()
    df.loc[df["model_name"].isin(PARAM_MODELS), "arm"] = "parametric"
    return df


def main() -> int:
    # ---- Deterministic (10 systems) ----
    det_param = pd.read_csv(CVP / "deterministic_parametric/scores_by_model.csv")
    det_ctx = pd.read_csv(REPO / "evaluation_v2/scores_by_model.csv")
    det = _relabel(pd.concat([det_param, det_ctx], ignore_index=True))

    # ---- DeepEval (10 systems) ----
    de_param = pd.read_csv(CVP / "deepeval_parametric/deepeval_scores_by_model.csv")
    de_ctx = pd.read_csv(REPO / "evaluation_v2/deepeval/deepeval_scores_by_model.csv")
    de = _relabel(pd.concat([de_param, de_ctx], ignore_index=True))
    de["deepeval_core_mean"] = de[CORE].mean(axis=1).round(4)
    de["deepeval_traceability"] = de[TRACE].mean(axis=1).round(4)
    # traceability only meaningful for context systems
    de.loc[de["arm"] == "parametric", "deepeval_traceability"] = None

    # ---- merge on (arm, model_name) ----
    key = ["arm", "model_name"]
    detc = ["reliability", "mean_entity_f1", "count_accuracy_rate", "field_value_accuracy",
            "format_core_ok_rate", "true_hallucination_answer_rate", "research_score_deterministic"]
    dec = CORE + TRACE + ["mean_deepeval_mean", "deepeval_core_mean", "deepeval_traceability"]
    merged = det[key + [c for c in detc if c in det.columns]].merge(
        de[key + [c for c in dec if c in de.columns]], on=key, how="outer")

    # order: parametric first, then context by det score
    merged["_is_param"] = (merged["arm"] == "parametric").astype(int)
    merged = merged.sort_values(["_is_param", "research_score_deterministic"],
                                ascending=[False, False]).drop(columns="_is_param")
    merged.to_csv(CVP / "comparison_by_system.csv", index=False)

    # ---- failure-mode panel for parametric (grounded/readable but incomplete?) ----
    fm = None
    mc_path = CVP / "deterministic_parametric/mention_classification.csv"
    if mc_path.exists():
        mc = pd.read_csv(mc_path)
        fm = mc.groupby("model_name")["classification"].value_counts().unstack(fill_value=0)

    # ---- markdown ----
    def fmt(v, pct=False):
        if pd.isna(v) or v == "":
            return "—"
        return f"{100*float(v):.1f}%" if pct else f"{float(v):.3f}"

    L = ["# Context vs Parametric — GNEM 42Q (LLM_Context Deterministic-V2 + DeepEval)", "",
         "10 systems: **3 parametric** (no KB context — my Qwen2.5-14B base/KB-only/KB+web) + "
         "**7 context** (KB in prompt — repo). Sections reported per the comparability rule: "
         "DeepEval **core** mean = completeness/correctness/company-grounding/usefulness (all systems); "
         "**traceability** = faithfulness/evidence-grounding (context only — parametric cite no evidence rows).",
         ""]

    L += ["## Deterministic-V2 (repo composite)", "",
          "| arm | model | research_score | entity_f1 | count_acc | field_value | true_halluc | format_ok |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in merged.iterrows():
        L.append(f"| {r['arm']} | {r['model_name']} | {fmt(r.get('research_score_deterministic'))} "
                 f"| {fmt(r.get('mean_entity_f1'))} | {fmt(r.get('count_accuracy_rate'),1)} "
                 f"| {fmt(r.get('field_value_accuracy'),1)} | {fmt(r.get('true_hallucination_answer_rate'),1)} "
                 f"| {fmt(r.get('format_core_ok_rate'),1)} |")
    L.append("")

    L += ["## DeepEval (gpt-oss:120b judge) — core vs traceability", "",
          "| arm | model | **core mean** | traceability | completeness | correctness | company_grounding | usefulness |",
          "|---|---|---|---|---|---|---|---|"]
    dd = de.set_index(["arm", "model_name"])
    for _, r in merged.iterrows():
        k = (r["arm"], r["model_name"])
        row = dd.loc[k] if k in dd.index else {}
        g = (lambda c: row.get(c) if hasattr(row, "get") else None)
        L.append(f"| {r['arm']} | {r['model_name']} | **{fmt(r.get('deepeval_core_mean'))}** "
                 f"| {fmt(r.get('deepeval_traceability'))} | {fmt(g('mean_completeness'))} "
                 f"| {fmt(g('mean_correctness'))} | {fmt(g('mean_company_grounding'))} | {fmt(g('mean_usefulness'))} |")
    L.append("")

    if fm is not None:
        L += ["## Parametric failure-mode panel (mention classification)", "",
              "Tests the claim: *grounded & readable but incomplete — missed records / wrong counts, "
              "not hallucinated companies.*", "",
              "| model | known_kb | true_hallucination | misspelled_kb |", "|---|---|---|---|"]
        for m in ["base", "kb_only", "kb_web"]:
            if m in fm.index:
                row = fm.loc[m]
                L.append(f"| {m} | {int(row.get('known_kb_company',0))} "
                         f"| {int(row.get('true_out_of_kb_hallucination',0))} "
                         f"| {int(row.get('misspelled_kb_company',0))} |")
        L.append("")

    (CVP / "comparison.md").write_text("\n".join(L))
    merged.to_json(CVP / "comparison.json", orient="records", indent=2)
    print(f"wrote {CVP/'comparison.md'}, comparison.json, comparison_by_system.csv")
    print(merged[["arm", "model_name", "research_score_deterministic",
                  "deepeval_core_mean", "deepeval_traceability"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
