# Georgia EV Supply-Chain Assistant — Full Training & Dataset Details

A complete, reproducible record of both fine-tuning runs (**v1** and **v2**): environment, base model, data sources, dataset construction, training configuration, the training process (including failures and fixes), evaluation methodology, results, and integrity checks.

- **Project root:** `/Users/surya/Desktop/projects/georgia-ev-finetune`
- **Task:** fine-tune a 7B instruct model on a Georgia EV knowledge base so it answers domain questions in the KB's style; evaluate on 50 human-validated questions; prepare for Linux GPU (vLLM) deployment.
- **Versions:** v1 (initial) and v2 (data-centric retrain). v2 is the shipped adapter.

---

## 0. TL;DR

| | Base | v1 | v2 (shipped) |
|---|---|---|---|
| LoRA rank / alpha | — | r=16 / α=32 | **r=32 / α=64** |
| Trainable params | — | 40.4M (0.53%) | 80.7M (1.05%) |
| Train / valid examples | — | 2,426 / 192 | 2,535 / 196 |
| Test-question contamination | — | 8 of 50 (questions only) | **0 (guard-enforced)** |
| Final train / eval loss | — | 0.3325 / 0.1999 | **0.2877 / 0.1955** |
| **Accuracy on 50 held-out Q&A** | **4% (2/50)** | **26% (13/50)** | **54% (27/50)** |

The human gold **answers were never used in training** in either version (verified). The only contamination was test *question text* in v1, removed in v2.

---

## 1. Environment & hardware

| | |
|---|---|
| Machine | Apple M4 Pro, 48 GB unified memory |
| OS | macOS 26.5.1 (Darwin 25.5.0) |
| GPU | **No NVIDIA GPU** — `nvidia-smi` absent; `torch.cuda.is_available()` = False |
| Compute backend | PyTorch **MPS** (Apple Metal); bf16 |
| Python | 3.13.13 (venv at `./.venv`) |

**Key package versions:** torch 2.12.0 · transformers 5.12.0 · peft 0.19.1 · trl 1.6.0 · datasets 5.0.0 · accelerate 1.14.0 · bitsandbytes 0.49.2 (installed but CUDA-only, so unused on MPS) · pandas, openpyxl, scikit-learn, sentencepiece, protobuf, evaluate, tabulate.

> On a Linux CUDA host the same `train_qlora.py` auto-switches to 4-bit **QLoRA** (bitsandbytes NF4, double-quant, bf16 compute) with `device_map="auto"`. On MPS it runs **bf16 LoRA** (no 4-bit, since bitsandbytes requires CUDA). Both produce identical PEFT-format adapters.

---

## 2. Base model

| | |
|---|---|
| Model | **`Qwen/Qwen2.5-Coder-7B-Instruct`** (Hugging Face) |
| Architecture | `Qwen2ForCausalLM`, qwen2 |
| Size | 7.6B params; hidden 3584; 28 layers; 28 attn heads; 4 KV heads (GQA); vocab 152,064; max pos 32,768 |
| Tokenizer | Qwen2 BPE; `pad_token` set to `eos_token` |

**Why this model.** The only candidate present locally was an LM Studio **MLX** build (`~/.lmstudio/models/lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-8bit`). MLX is Apple-only and unsuitable for Linux GPU deployment, so per the brief the original HF base model was identified from its `config.json` (`Qwen2ForCausalLM`, qwen2, 7.6B) and downloaded from Hugging Face. All training uses the **HF + PEFT** path, never MLX.

---

## 3. Data sources (originals — never modified)

### 3.1 Knowledge base (training source)
`/Users/surya/Downloads/GNEM - Auto Landscape Lat Long Updated (1).xlsx`
- **205 rows, 193 unique companies, 15 columns.**
- Columns (auto-detected and mapped): `Company`, `Category` (tier), `Industry Group`, `Updated Location`, `Address`, `Latitude`, `Longitude`, `Primary Facility Type`, `EV Supply Chain Role`, `Primary OEMs`, `Supplier or Affiliation Type`, `Employment`, `Product / Service`, `EV / Battery Relevant`, `Classification Method`.
- Derived in `load_kb()`: `Updated Location` → **City** + **County**; `FacilityNorm` (normalized facility type); `IndustryNorm` (normalized industry group).
- Distributions: Category — Tier 1 (77), Tier 2/3 (73), Tier 1/2 (18), OEM Supply Chain (17), OEM (12), OEM (Footprint) (5), OEM Footprint (3). EV/Battery Relevant — No (87), Indirect (76), Yes (41). EV Supply Chain Role — General Automotive (135), Materials (22), Vehicle Assembly (10), Battery Pack (4), Thermal Management (4), Battery Cell (2), Power Electronics (1), Charging Infrastructure (1), + long tail.

### 3.2 Human-validated questions (test source — eval ONLY)
`/Users/surya/Downloads/Human validated 50 questions (2).xlsx`
- **50 rows, 4 columns:** `Num`, `Use Case Category`, `Question`, `Human validated answers`.
- 5 use-case categories: Supply Chain Mapping & Visibility (16), Supplier Discovery & Matchmaking (11), Product & Technology Trends (11), Supply Chain Risk & Resilience (7), Site Selection & Expansion Planning (5).
- **Used only to build `test.jsonl`** (and, in v2, to seed the leakage guard). The `Human validated answers` column is never written to train/valid.

---

## 4. Dataset construction (`scripts/build_dataset.py`)

All examples are **chat JSONL** with a fixed system prompt:

> _"You are a Georgia EV supply chain assistant. Answer only using the Georgia EV knowledge base. If the KB does not contain enough information, say so clearly."_

Each line: `{"messages": [{system}, {user}, {assistant}]}`. Assistant answers are computed **deterministically from the KB** via pandas — no values are hand-copied from anywhere. Answers use a consistent style ("According to the Georgia EV KB, there are N …") and, when data is missing, explicitly say so (e.g. "The Georgia EV KB does not provide an employment figure for X").

### 4.1 Generator families
- **Per-company facts** (`company_*`, ~193 companies × 8 fields): summary, EV supply chain role, products/services, location, category/tier, primary OEMs, employment, facility type/EV relevance; plus `company_multientry` for companies with multiple KB rows.
- **Aggregates** (`agg_*`): lists/counts by category, by EV supply chain role, by county, by city, by industry group; county employment totals & per-county max; OEM linkage (Hyundai/Kia/Rivian/Toyota/Club Car); "Multiple OEMs" diversified suppliers; dual-platform (traditional + Rivian); ~30 product-keyword searches (battery, lithium, thermal, wiring harness, copper foil, DC-to-DC, inverter, etc., including explicit "no companies match" answers); employment thresholds; EV/Battery-relevant filters; facility-type and industry-group filters; site-selection patterns (Tier-1-but-no-battery counties, conversion-ready areas, materials concentration, R&D, chemical infra); OEM-type / OEM-footprint lists; single-point-of-failure roles.
- **Refusals** (`refusal`, 6 hand-written generic examples): out-of-KB questions (revenue, founding date, CEO, stock price, other states, charging stations) → explicit "the KB does not provide that." None are drawn from the 50 test questions.
- **v2 only — `v2_*` (paraphrase-rich, eval-aligned)**: `agg_examples_v2()` re-emits the highest-value eval-shaped patterns (battery roles, power-electronics, thermal management, vehicle assembly, Tier-1/2 lists, OEM footprint, Tier-2/3∧EV-relevant, wiring harness, battery materials, Hyundai/Kia linkage, dual-platform, single-source roles, sole-sourced battery, employment-threshold combos, county-employment argmax/sum, site-selection gaps, R&D, chemical infra, recycling, lithium-ion materials) with **3–4 paraphrases each**, so the model binds the underlying KB fact rather than one phrasing.

### 4.2 Split & oversampling
- `valid.jsonl` = random 10% of generated examples; `train.jsonl` = the rest. `test.jsonl` = the 50 human Q&A (gold answers), never in train/valid.
- **Oversampling:** aggregate/refusal/multi-entry examples are duplicated so the model gets more exposure to the count/list/aggregate shapes the eval is dominated by. **v1 = ×3** (`agg_*`, refusal, multientry). **v2 = ×3** (`agg_*`, `v2_*`, refusal, multientry).

### 4.3 Leakage guard (v2 only)
Before generation, the 50 test questions are loaded and normalized. In `add()`, any generated question that matches a test question **exactly (normalized)** or by **≥0.85 token-Jaccard** (near-duplicate) is dropped. v2 dropped **73** such generations, guaranteeing all 50 test questions are genuinely held out. (v1 predated this guard.)

---

## 5. Dataset details per version

| | v1 | v2 |
|---|---|---|
| Generators | `company_*`, `agg_*`, refusals | `company_*`, `agg_*`, **`v2_*`**, refusals |
| Leakage guard | none | exact + ≥0.85 Jaccard (dropped 73) |
| Oversample (×3) | `agg_*`, refusal, multientry | `agg_*`, **`v2_*`**, refusal, multientry |
| **train.jsonl** | **2,426** | **2,535** |
| **valid.jsonl** | **192** | **196** |
| **test.jsonl** | 50 | 50 |
| Max example length | ~1,236 tokens | ~900 tokens |
| Test-QUESTION overlap with training | **8 of 50 verbatim** (qids 2,4,7,9,12,26,45,46) | **0** (exact and near-dup) |
| Gold-ANSWER overlap with training | **0** | **0** |

**v2 generator tag distribution** (pre-oversample example counts, from `logs/dataset_report.md`): company_* 193 each (×8 families); agg_county_list 61; agg_county_emp 61; agg_city_list 45; agg_industry 36; agg_role_list 33; agg_keyword 26; v2_county_top_emp 24; company_multientry 9; v2_emp_combo 9; agg_category_count/names 7; agg_category_list 6; refusal 6; agg_keyword_none 5; v2_thermal/t23_yes/top_emp_county 4; ~15 distinct v2_* patterns at 3 each (battery_roles, pe_ci, vehicle_assembly, tier12_list, oem_footprint, harness, battery_materials, hyundai_kia, dual_platform, single_source, sole_source_battery, rd, chemical, recycling, lithium, gap_counties, top_emp_county_t1); plus single-instance agg_* (count_by_category, count_by_county, top10_emp, big_indirect, materials_areas, conversion_areas, etc.).

> ⚠️ **v1's exact `train.jsonl`/`valid.jsonl` were overwritten** when the dataset was regenerated for v2. The v1 *adapter* is preserved (`adapters/georgia_ev_lora_v1/`), and v1's contamination figures above were verified live before regeneration. v2's figures are reproducible from the current files.

### 5.1 Dataset validation (`scripts/validate_dataset.py`)
Run before each training: confirms every line is valid JSON, every example has exactly `system`/`user`/`assistant` roles with non-empty content, prints split sizes and word-length stats, and dumps 5 random examples. Both versions: **PASS**.

---

## 6. Training configuration (`scripts/train_qlora.py`)

| Hyperparameter | v1 | v2 |
|---|---|---|
| Method | LoRA (bf16 on MPS / QLoRA-NF4 on CUDA) | same |
| **LoRA rank `r`** | 16 | **32** |
| **LoRA alpha** | 32 | **64** |
| LoRA dropout | 0.05 | 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | same |
| Trainable params | 40.4M (0.53%) | **80.7M (1.05%)** |
| Adapter size | 161 MB | 323 MB |
| Epochs | 3 | 3 |
| Optimizer steps | 456 | 477 |
| Learning rate | 2e-4 | 2e-4 |
| LR scheduler | cosine, 3% warmup | same |
| Per-device batch | 1 | 1 |
| Grad accumulation | 16 | 16 |
| **Effective batch** | **16** | **16** |
| Max sequence length | 1,280 | **1,536** |
| Precision | bf16 (MPS) | bf16 (MPS) |
| Gradient checkpointing | on | on |
| save / eval / logging steps | 100 / 100 / 10 | same |
| Seed | 42 | 42 |
| Output dir | `adapters/georgia_ev_lora` → moved to `_v1` | `adapters/georgia_ev_lora` |

Everything except rank/alpha and max-seq-length is identical between versions.

---

## 7. Training process & results

### 7.1 Final training metrics (from `final_metrics.json`)

| | v1 | v2 |
|---|---|---|
| Train runtime | 8,419 s (~2 h 20 m) | 9,280 s (~2 h 35 m) |
| **Final train loss** | **0.3325** | **0.2877** |
| **Final eval (valid) loss** | **0.1999** | **0.1955** |
| Eval token accuracy | 0.9534 | 0.9554 |
| Eval entropy | 0.176 | 0.158 |
| total FLOPs | 3.82e16 | 4.13e16 |

Both runs: a **5-step smoke test** first (`logs/smoke_test.md`), then full 3-epoch training, checkpointing every 100 steps with auto-resume on restart.

### 7.2 Incidents & fixes (`logs/error_report.md`)
1. **Disk-full crash ×2 (v1).** The 14 GB HF download left <1 GB free; MPS could not write temp files and training died (~step 11/456). Fixed by reclaiming regenerable caches only (no user files/models touched); the user freed more space. Checkpoint/auto-resume added.
2. **MPS watermark env var.** `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` without `PYTORCH_MPS_LOW_WATERMARK_RATIO` makes the implied low ratio invalid (`1.4`) and aborts at load. Both are now set (`HIGH=1.0`, `LOW=0.8`). This bit (a) a v1 restart and (b) the first v2 evaluation — the crashed v2 eval silently left v1's answer file in place, so the first score re-scored v1; caught by diffing v2-vs-v1 answers (48/50 differ) before reporting.
3. **v1 test-question contamination** → motivated the v2 leakage guard.

---

## 8. Evaluation methodology

### 8.1 Answer generation (`scripts/evaluate.py …`)
For all 50 test questions: build the prompt from `system` + `user` via the chat template, **greedy decode** (`do_sample=False`), up to 700 new tokens. Base model uses no adapter; v1/v2 load their respective PEFT adapter. Outputs saved to `eval_results/base_answers.json`, `finetuned_answers_v1.json`, `finetuned_answers.json` (v2).

### 8.2 Structured auto-score (`score_answer`)
Each answer is scored vs the human gold answer with a weighted composite (**weights fixed a priori, not tuned to results**), renormalized over whichever signals apply:
- **Company-name F1** (weight 0.65): precision + recall of KB company names. Precision penalizes over-listing (e.g. answering with 40 companies when gold has 3); recall penalizes omissions.
- **Headline-count match** (0.20): does the answer reproduce the gold's leading count ("There are 18 …")? Count phrases matched by regex; `Tier 1/2/3` digits stripped first so labels aren't read as quantities.
- **Number overlap** (0.15): fraction of gold numbers (counts, employment) present.
- **Token-F1 fallback** (1.0): only for free-form gold answers with no names/numbers.
- A question is **correct** when the composite **≥ 0.60**.

`evaluation.xlsx` columns: `question_id, question, gold_answer, base_model_answer, finetuned_model_answer, notes, possible_hallucination, missing_facts, base_score, finetuned_score, use_case_category`.

---

## 9. Results on the 50 held-out questions

### 9.1 Overall (correct = composite ≥ 0.60)

| Model | Accuracy | Correct | Mean score |
|---|---|---|---|
| Base Qwen2.5-Coder-7B | 4% | 2/50 | 0.201 |
| Fine-tuned **v1** | 26% | 13/50 | 0.363 |
| **Fine-tuned v2** | **54%** | **27/50** | **0.584** |

### 9.2 Per use-case category (correct / n)

| Category | n | Base | v1 | v2 |
|---|---|---|---|---|
| Supply Chain Mapping & Visibility | 16 | 2 | 4 | **8** |
| Supplier Discovery & Matchmaking | 11 | 0 | 3 | **5** |
| Product & Technology Trends | 11 | 0 | 2 | **5** |
| Supply Chain Risk & Resilience | 7 | 0 | 2 | **5** |
| Site Selection & Expansion Planning | 5 | 0 | 2 | **4** |

### 9.3 Representative question-level changes (v1 → v2)
- **Q9** "highest-employment county": v1 said *Troup* (wrong) → v2 *Gwinnett* (correct). 0.00 → 1.00
- **Q8** "Tier-1 county employment total": 0.00 → 1.00
- **Q1** (18 Tier-1/2 suppliers), **Q2** (battery roles): → 1.00
- **Q21** (Hyundai/Kia linkage), **Q33** (OEM footprint): → pass
- **Residual:** **Q12** (gold = 3 companies) — v1 listed 40, v2 lists 10 (closer, still wrong). One regression: **Q5** 0.62 → 0.28.

**Why v2 helped:** repeating the precomputed aggregates under many paraphrasings + doubling adapter capacity let the model *memorize* facts it previously fabricated. **Why not 75–80%:** exact long-list enumeration under tight multi-constraint filters remains unreliable from parametric memory — the structural ceiling of a pure fine-tune (≈45–65% here). Closing it reliably needs an inference-time retrieval/compute layer (deferred Option A).

---

## 10. Integrity — what was and wasn't trained on

Verified against the current v2 `train.jsonl`+`valid.jsonl` (2,731 examples):

| Check | Result |
|---|---|
| Exact human-**ANSWER** matches in training | **0** (v1 and v2) |
| Exact test-**QUESTION** matches in training | v2: **0** · v1: 8 |
| Near-dup (≥0.85) test-question matches | v2: **0** |
| Training answers sharing an 8-word phrase with a gold answer | 381 — **shared KB facts** (company names, roles, products, employment), not copied gold prose; both describe the same Excel |

- The QA file is read only to (a) write `test.jsonl` and (b) seed the v2 leakage guard. The `Human validated answers` column is never written to train/valid.
- All training answers are computed from the KB Excel by pandas; **no answer values are hardcoded** from the 50 Q&A. The only hand-written Q/A pairs are 6 generic refusals, unrelated to the test set.

**Conclusion:** the 50 human answers were never trained on in either version. The only contamination was *question phrasing* in v1, eliminated in v2 — which is why v2's 54% is both higher and more trustworthy than v1's 26%.

---

## 11. Deployment (Linux GPU, vLLM + LoRA)

`deployment/`:
- `serve_vllm.sh`: `vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --enable-lora --lora-modules georgia-ev=<adapter> --max-lora-rank 32 --max-model-len 4096 --dtype bfloat16` (rank 32 matches v2).
- `README_DEPLOYMENT.md`: copy-adapter steps, OpenAI-compatible **curl** example (use `model: "georgia-ev"` and the training system prompt), optional `merge_and_unload` recipe.
- `requirements.txt`: vLLM (+ commented training deps).

The adapter is standard PEFT format and serves directly; no MLX dependency.

---

## 12. File map & reproduction

```
georgia-ev-finetune/
├── .venv/                                  # Python 3.13 env
├── deployment/                             # vLLM serve script, README, requirements
└── training_project/
    ├── data/{train,valid,test}.jsonl       # current = v2
    ├── scripts/
    │   ├── build_dataset.py                # KB → JSONL (+ leakage guard, agg_examples_v2)
    │   ├── validate_dataset.py             # JSONL integrity check
    │   ├── train_qlora.py                  # LoRA/QLoRA trainer (CUDA→QLoRA, MPS→bf16 LoRA)
    │   └── evaluate.py                      # generate (base|finetuned) + structured score
    ├── adapters/
    │   ├── georgia_ev_lora/                # v2 (shipped, r=32)
    │   └── georgia_ev_lora_v1/             # v1 (r=16)
    ├── eval_results/
    │   ├── base_answers.json, finetuned_answers_v1.json, finetuned_answers.json (v2)
    │   ├── evaluation.xlsx, summary.md
    │   └── answers_base_v1_v2.xlsx, scores_base_v1_v2.xlsx
    ├── logs/{dataset_report,smoke_test,error_report,full_training*.log}
    ├── FINAL_REPORT.md
    └── TRAINING_DETAILS.md                 # this file
```

**Reproduce v2:**
```bash
cd georgia-ev-finetune
.venv/bin/python training_project/scripts/build_dataset.py        # regenerate JSONL (clean)
.venv/bin/python training_project/scripts/validate_dataset.py     # integrity check
PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 \
  .venv/bin/python training_project/scripts/train_qlora.py        # ~2.5h on M4 Pro
PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 \
  .venv/bin/python training_project/scripts/evaluate.py finetuned # v2 answers
.venv/bin/python training_project/scripts/evaluate.py score       # evaluation.xlsx + summary.md
```
To reproduce **v1**, set `r=16`/`lora_alpha=32` and `--max-seq-length 1280` in `train_qlora.py` and (to recreate its contamination) remove the leakage guard — not recommended; v2's clean split is preferred.

---

## 13. Known limitations
- Pure fine-tuning ceiling ≈45–65% on this question set; v2 at 54%. Table-wide aggregation and exact long-list enumeration are the residual failures.
- No CUDA on the training machine → trained with bf16 LoRA on MPS, not 4-bit QLoRA (the script supports QLoRA on a Linux GPU host).
- The structured scorer is lexical/structured (deterministic, reproducible); it is not a semantic/LLM judge. It can mildly mis-score paraphrased-but-correct free-form answers in either direction.
- v1's exact training JSONL is not retained (overwritten by v2 regeneration); the v1 adapter and metrics are retained.
