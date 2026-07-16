# GNEM-Bench-v1 — results (Qwen2.5-14B, 4-bit QLoRA r64+rsLoRA)

Three systems, three **independent** evaluation suites (never blended into one score).
Models are base (non-instruct) completion models; answers parsed from raw greedy
completions (`max_new_tokens=256`). Web+KB run = full web corpus (all 7,760 train pages,
1×) + each of 205 KB records 50× (exposure-matched to the KB-only baseline), 1 epoch,
12,447 train examples.

## Headline table

| Metric (what it measures) | Base | KB-only | KB+web |
|---|---|---|---|
| **KB cloze recall** — exact, seen-company (KB memorization / *retention*) | 0.027 | **0.782** | **0.608** |
| **GNEM-KB-42** — analytical KB questions (filter/rank/aggregate) | 5.0% [1–16] | 2.5% [0–13] | 7.5% [3–20] |
| **GNEM-Web-18** — training-corpus web-fact *absorption* | 22.2% [9–45] | 5.6% [1–26] | **38.9% [20–61]** |

(train perplexity base 13.05 → KB+web adapter 1.29; [..] = 95% Wilson CI.)

## Three findings

1. **Web absorption: YES.** KB+web Web-18 = 38.9% (+33.3 over KB-only, +16.7 over base).
   Won 7/18, incl. non-numeric relationship facts base/KB-only both missed (SK On→Hyundai
   Cartersville JV; SK On = parent of SKBA) and **correcting a wrong KB fact** (Anovion
   makes *synthetic graphite*, not the KB's mis-entered "floor mats"). Single-pass web
   exposure didn't cement every dollar amount, but the corpus knowledge clearly landed.

2. **Analytical KB reasoning: NO — for any memorization model.** Base/KB-only/KB+web are all
   near-zero on GNEM-KB-42 with fully overlapping CIs. By category only `direct_lookup`
   scores; `count / multi_filter_list / top_k_ranking / geographic_aggregation / judgment`
   are ~0 everywhere. Memorization stores facts; it cannot filter/rank/count/aggregate over
   205 rows. → this is what the reserved **RAG (retrieval + deterministic computation)**
   column is for.

3. **Retention: mostly preserved, mildly diluted.** KB+web cloze recall 0.608 vs KB-only
   0.782 — interleaving the full web corpus (at matched 50× KB exposure) cost ~0.17 exact
   recall (~22% relative), but KB knowledge is still strongly retained (0.61 ≫ 0.03 base).
   Separately, KB-only *forgot* pretrained web knowledge (Web-18 22.2%→5.6%); adding the web
   corpus reversed that (→38.9%).

## Interpretation
Fine-tuning improves **parametric fact retention/absorption** (KB cloze 0.78; web +33), but
**analytical business questions require authoritative structured access + deterministic
computation (RAG/tool), not more facts in the weights.** The KB and web capabilities trade
off slightly (KB recall 0.78→0.61 when the web corpus is added) but do not collapse.

## Provenance
- Gold: `questions_gold_v1.json` (frozen before any output review); KB gold =
  `Human validated questions.xlsx` (42, asserted); Web-18 authored from
  effective-training-set pages, KB-absent (company-aware).
- Systems: base = KB-only run with adapter disabled (canonical `raw_outputs/base.json`);
  KB+web run `…/kb-web-mixed-full-corpus-kb50-web1/…/20260714_195323_749ad79f`.
- Reproduce (run from `self_supervised_finetuning/`; `bench_lib` imports `ssft`, so
  `PYTHONPATH=src` is required): `outputs/question_eval/raw_outputs/*` (immutable,
  checksummed) → `benchmarks/gnem_bench_v1/scripts/score_questions.py` →
  `outputs/question_eval/scored/*` → `.../combine_question_eval.py` →
  `outputs/question_eval/comparison.{json,md}`.

  ```bash
  PYTHONPATH=src python benchmarks/gnem_bench_v1/scripts/score_questions.py \
    --raw outputs/question_eval/raw_outputs/base.json \
    --gold benchmarks/gnem_bench_v1/questions_gold_v1.json \
    --out outputs/question_eval/scored/base.json
  ```

  The `provenance.gold_file` / `systems[].file` strings in `scored/*.json` are recorded
  relative to the invocation directory, so they vary with CWD; scores do not.

## Reserved / future
RAG column (retrieval + SQL/pandas/DuckDB computation) — the path to the analytical
questions; held-out web *generalization* set; KB↔web conflict set.
