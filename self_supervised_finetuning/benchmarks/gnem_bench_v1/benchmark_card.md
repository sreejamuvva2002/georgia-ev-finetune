# GNEM-Bench-v1 — benchmark card

**Georgia EV Manufacturing** benchmark for comparing systems on the 205-company KB and a
Georgia-EV web corpus. Two sections, **always reported independently — no combined score**
(they measure different capabilities).

## Sections

### GNEM-KB-42 — structured-KB reasoning & retention (42 questions)
Human-validated business questions whose authoritative answers come from the 205-row KB
(`kb_full.jsonl`). Gold = `Human validated questions.xlsx` (`Num | Use Case Category |
Question | Human validated answers`). Measures whether a system can answer real analytical
questions over the KB, and — across Base → KB-only → KB+web — whether adding web data
**preserves, improves, or degrades** the original KB task.

Split into two **subsets, reported separately, never averaged together**:
- **deterministic (31)** — objectively computable: `direct_lookup, multi_filter_list, count,
  max_min, top_k_ranking, geographic_aggregation`.
- **judgment (11)** — open-ended/subjective (`upgradeable / pivot-ready / innovation-stage /
  dual-platform / …`); scored by entity-overlap vs gold, not exact match.

### GNEM-Web-18 — training-corpus web-fact absorption (18 questions)
Facts (investment $, job counts, GWh capacity) authored **only from web pages proven to be
in the effective training set** (`report_mixture_stats.py`), each **absent from the
structured KB** (company-aware `kb_overlap_check`) and phrased against the corpus snapshot
("According to the collected web corpus, …"), never "latest/today". Measures whether KB+web
**parametrically absorbed facts present in its web training corpus**. It does NOT measure
held-out generalization, freshness, retrieval, or multi-page reasoning (future work).

## Question schema
`question_id, section, subset, category, operations[], operation_count (len(operations);
null for judgment), use_case_category, question, gold{answer_text, entities[], value,
county, ordered[], answer_type, stages:null}, source_url/source_date (web)`.
`operation_count` drives a **report-time complexity grouping** (no stored difficulty scale).

## Metrics
- Entity/list Qs: set **precision / recall / F1 / completeness** in the 205-KB-name space.
- Ranking/max_min: rank-1 correctness (+ set overlap).
- count / currency / capacity / county: exact / normalized-value match (currency & date
  normalized: "$500 million" = "500,000,000 USD").
- **Hallucination** Type-A (company not in KB) automated (heuristic), Type-B (KB company,
  wrong attribute) automated, Type-C (right company, wrong reasoning) reserved/manual.
- Section accuracy with **95% Wilson CI** (n small); continuous scores via bootstrap.
- Latency (`generated_tokens, generation_time, tokens_per_sec`) recorded per question.

## Protocol & reproducibility
- **Base (non-instruct) completion** models — raw completions are **parsed**, not
  instruction-followed. Greedy decoding, `max_new_tokens=256` (covers gold p95=223;
  `hit_max_new_tokens` recorded). Generation-config + input/benchmark **SHA-256** stored
  in every system file; **base computed once** (canonical `raw_outputs/base.json`) and
  reused; raw outputs are **immutable** (read-only + checksummed).
- **Freeze invariant**: gold frozen **before any model-output review or scoring**
  (`freeze{...}` flags in `questions_gold_v1.json`).

## Reserved for future
`RAG` column + retrieval fields (`retrieved_rows, retrieval_score, retrieved_company_ids,
retrieved_urls`). RAG = retrieval **+ deterministic computation** (SQL/pandas/DuckDB), the
only path that answers the analytical questions memorization cannot. Plus: held-out web
generalization set, KB↔web conflict set.

## Files
`questions_gold_v1.json` (frozen gold, 60 Q), `web_questions_v1.json` (Web-18 draft +
provenance), `benchmark_card.md`. Scripts: `ingest_kb_gold.py, build_web_questions.py,
build_gold.py, ask_adapter.py, score_questions.py, combine_question_eval.py, bench_lib.py,
report_mixture_stats.py`.
