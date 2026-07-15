# Context vs Parametric — GNEM 42Q (LLM_Context Deterministic-V2 + DeepEval)

10 systems: **3 parametric** (no KB context — my Qwen2.5-14B base/KB-only/KB+web) + **7 context** (KB in prompt — repo). Sections reported per the comparability rule: DeepEval **core** mean = completeness/correctness/company-grounding/usefulness (all systems); **traceability** = faithfulness/evidence-grounding (context only — parametric cite no evidence rows).

## Deterministic-V2 (repo composite)

| arm | model | research_score | entity_f1 | count_acc | field_value | true_halluc | format_ok |
|---|---|---|---|---|---|---|---|
| parametric | kb_web | 0.311 | 0.104 | 40.0% | 29.5% | 9.5% | 0.0% |
| parametric | base | 0.210 | 0.073 | 0.0% | 32.2% | 21.4% | 0.0% |
| parametric | kb_only | 0.203 | 0.051 | 0.0% | 28.8% | 11.9% | 0.0% |
| direct_context | qwen3.6:35b-a3b | 0.846 | 0.813 | 77.8% | 82.7% | 0.0% | 100.0% |
| direct_context | qwen3:30b | 0.762 | 0.766 | 50.0% | 78.1% | 0.0% | 100.0% |
| batchwise_map_reduce | qwen2.5:14b | 0.588 | 0.529 | 22.2% | 70.4% | 0.0% | 100.0% |
| batchwise_map_reduce | gemma3:12b | 0.587 | 0.504 | 22.2% | 77.0% | 0.0% | 100.0% |
| direct_context | deepseek-r1:32b | 0.529 | 0.442 | 19.4% | 62.5% | 2.4% | 100.0% |
| direct_context | mistral-small3.2:24b | 0.461 | 0.319 | 19.4% | 52.2% | 0.0% | 100.0% |
| direct_context | gemma3:27b | 0.435 | 0.263 | 11.1% | 67.7% | 7.1% | 100.0% |

## DeepEval (gpt-oss:120b judge) — core vs traceability

| arm | model | **core mean** | traceability | completeness | correctness | company_grounding | usefulness |
|---|---|---|---|---|---|---|---|
| parametric | kb_web | **0.171** | — | 0.000 | 0.007 | 0.559 | 0.117 |
| parametric | base | **0.148** | — | 0.000 | 0.000 | 0.452 | 0.138 |
| parametric | kb_only | **0.134** | — | 0.005 | 0.009 | 0.452 | 0.069 |
| direct_context | qwen3.6:35b-a3b | **0.788** | 0.880 | 0.571 | 0.695 | 0.971 | 0.914 |
| direct_context | qwen3:30b | **0.688** | 0.838 | 0.336 | 0.486 | 0.993 | 0.938 |
| batchwise_map_reduce | qwen2.5:14b | **0.607** | 0.767 | 0.243 | 0.307 | 0.952 | 0.926 |
| batchwise_map_reduce | gemma3:12b | **0.514** | 0.552 | 0.164 | 0.219 | 0.938 | 0.733 |
| direct_context | deepseek-r1:32b | **0.507** | 0.477 | 0.124 | 0.145 | 0.907 | 0.852 |
| direct_context | mistral-small3.2:24b | **0.477** | 0.383 | 0.079 | 0.100 | 0.981 | 0.750 |
| direct_context | gemma3:27b | **0.360** | 0.184 | 0.007 | 0.043 | 0.761 | 0.629 |

## Parametric failure-mode panel (mention classification)

Tests the claim: *grounded & readable but incomplete — missed records / wrong counts, not hallucinated companies.*

| model | known_kb | true_hallucination | misspelled_kb |
|---|---|---|---|
| base | 5 | 16 | 4 |
| kb_only | 19 | 5 | 1 |
| kb_web | 20 | 8 | 0 |
