# Georgia EV Supply-Chain — LLM Fine-Tuning

Fine-tuning a local LLM on Georgia's electric-vehicle supply chain (companies, supply-chain
roles, OEM relationships, counties, employment) and measuring what fine-tuning can and cannot
teach it.

The active study separates two learning goals that are usually conflated:

- **Continued pretraining (CPT)** on raw documents — teaches *domain knowledge*.
- **Analytical SFT** on deterministically computed examples — teaches *task behaviour*
  (count, sum, filter, group, rank, list).

Seven experiments (E0–E6) measure each stage independently before evaluating the combination.
The full design is in **[`gnem_finetuning_experiment_plan.md`](gnem_finetuning_experiment_plan.md)**.

- **Base model:** `Qwen2.5-14B` (base, non-instruct — answers are parsed from raw completions)
- **Method:** full-parameter CPT and SFT, DeepSpeed ZeRO-3 + 8-bit AdamW
- **Scope:** fine-tuning only; RAG and tool-use are explicitly out of scope
- **Data:** 205-company KB (`kb_full.jsonl`) + ~9.7k LLM-generated wiki pages
- **Evaluation:** [GNEM-Bench-v1](self_supervised_finetuning/benchmarks/gnem_bench_v1/) —
  deterministic scoring first, DeepEval as a secondary layer that never overrides it

## Repository layout

```
kb_full.jsonl                      # the 205-company knowledge base (authoritative)
Human validated questions.xlsx     # the 42 human-validated questions (frozen gold)
gnem_finetuning_experiment_plan.md # the E0–E6 experiment design
self_supervised_finetuning/        # the framework ("ssft")
  src/ssft/                        #   data/ train/ eval/ utils/
  configs/                         #   data, methods, models, training, experiments
  benchmarks/gnem_bench_v1/        #   frozen gold, benchmark card, scorer, results
  outputs/question_eval/           #   immutable raw outputs -> scored -> comparison
  reports/                         #   thesis_paper.md, course_project_paper.md
  external/LLM_Context/            #   vendored DeepEval / direct-context harness
chats/                             # archived session transcripts (untracked)
```

> Model weights, adapters, and the web corpus are **not** tracked in git (size); they are
> reproduced from the configs and `kb_full.jsonl`.

## Quick start

```bash
cd self_supervised_finetuning
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PY=.venv/bin/python

$PY -m ssft.cli inspect-repo    # KB schema + row/company counts
$PY -m ssft.cli inspect-env     # GPU / environment snapshot
$PY -m ssft.cli prepare-kb      # serialize the KB into raw documents
$PY -m ssft.cli train-experiment --config configs/experiments/<experiment>.yaml
```

## Results so far

`benchmarks/gnem_bench_v1/RESULTS_v1.md` reports the QLoRA arm (base / KB-only / KB+web):
fine-tuning reliably improves **parametric fact retention** (KB cloze recall 0.03 → 0.78) and
**web-fact absorption** (+33 points over KB-only), but **analytical questions stay near zero for
every memorization model** — filtering, ranking, counting, and aggregating over 205 rows is not
something more facts in the weights can buy. That negative result is what motivates the
analytical-SFT arm (E4–E6) of the current plan.

## Reproducing the benchmark

```bash
cd self_supervised_finetuning
PYTHONPATH=src .venv/bin/python benchmarks/gnem_bench_v1/scripts/score_questions.py \
  --raw outputs/question_eval/raw_outputs/base.json \
  --gold benchmarks/gnem_bench_v1/questions_gold_v1.json \
  --out outputs/question_eval/scored/base.json
```
