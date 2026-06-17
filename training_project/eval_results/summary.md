# Evaluation Summary (50 held-out human-validated Q&A)

- Correctness rule: composite structured score >= 0.6
- **Base model accuracy: 4%** (2/50, mean score 0.201)
- **Fine-tuned model accuracy (v2): 54%** (27/50, mean score 0.584)

## v1 → v2 (data-centric retrain)

| Model | Accuracy | Mean | Notes |
|---|---|---|---|
| Base Qwen2.5-Coder-7B | 4% (2/50) | 0.201 | no fine-tuning |
| Fine-tuned **v1** (r=16) | 26% (13/50) | 0.363 | 8 test questions had leaked into training |
| Fine-tuned **v2** (r=32) | **54% (27/50)** | 0.584 | **contamination-free** benchmark (0 test-question/gold-answer leakage) |

v2 doubled accuracy on a *cleaner* benchmark. The biggest gains came from binding facts the model used to fabricate: county-employment aggregates (Q8 0.00→1.00, Q9 0.00→1.00), Tier-1/2 and battery-role lists (Q1, Q2 → 1.00), and OEM-linkage/footprint queries (Q21, Q33 → pass). Still below the 75–80% target — table-wide aggregation and exact long-list enumeration remain the residual failure mode for a pure fine-tune (e.g. Q12 still over-lists: 10 vs gold 3).

## By use-case category

| use_case_category                   |   n |   base_correct |   ft_correct |   base_acc |   ft_acc |   base_mean |   ft_mean |
|:------------------------------------|----:|---------------:|-------------:|-----------:|---------:|------------:|----------:|
| Product & Technology Trends         |  11 |              0 |            5 |       0    |     0.45 |       0.172 |     0.547 |
| Site Selection & Expansion Planning |   5 |              0 |            4 |       0    |     0.8  |       0.14  |     0.725 |
| Supplier Discovery & Matchmaking    |  11 |              0 |            5 |       0    |     0.45 |       0.236 |     0.51  |
| Supply Chain Mapping & Visibility   |  16 |              2 |            8 |       0.12 |     0.5  |       0.266 |     0.614 |
| Supply Chain Risk & Resilience      |   7 |              0 |            5 |       0    |     0.71 |       0.084 |     0.591 |

## Scoring method (structured auto-score)

Each answer is scored against the human gold answer with a weighted composite of structured signals (weights fixed a priori, not tuned to results):

- **Company-name F1** (weight 0.65): precision + recall of KB company names vs the gold answer. Precision penalizes over-listing (e.g. a query whose answer is 3 companies but the model lists 40); recall penalizes missing companies.
- **Headline-count match** (weight 0.2): does the model reproduce the gold answer's leading count (e.g. "There are 18 ...")?
- **Number overlap** (weight 0.15): fraction of gold numbers (counts, employment figures) present in the answer.
- **Token-F1 fallback** (weight 1.0): used only for free-form answers with no company names or numbers.

Weights are renormalized over whichever signals apply to a question. A question counts as **correct** when the composite >= 0.6.

`possible_hallucination` = KB companies the fine-tuned answer named that the gold answer did not. `missing_facts` = gold companies the answer omitted. `notes` carries the per-signal breakdown.
