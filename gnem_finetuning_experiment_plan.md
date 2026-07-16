# Fine-Tuning Experiment Plan

**Project:** Georgia EV Supply-Chain Domain Model  
**Base model:** `Qwen2.5-14B` (base, non-instruct)  
**Primary goal:** Compare raw-document continued pretraining with analytical supervised fine-tuning  
**Scope:** Fine-tuning only; RAG and tool-use experiments are excluded  
**Training approach:** Full-parameter fine-tuning on 4 GPUs (~190 GB VRAM), DeepSpeed ZeRO-3 with 8-bit AdamW  
**Current status:** Planned — blocked on hardware (see §0)

---

## 0. Preconditions

Three things must be true before E1 can start. None are true today.

1. **GPU driver.** `nvidia-smi` fails with `Driver/library version mismatch` (NVML 580.126).
   The GPUs are not visible, so the 4×48 GB assumption is unverified.
2. **Disk.** The filesystem is at 99% (~54 GB free). Full-parameter 14B writes ~29 GB per
   checkpoint in bf16 weights alone; E1–E6 is ~174 GB of final weights before optimizer state,
   intermediate checkpoints, or the ~29 GB base download. Provision ≥250 GB.
3. **DeepSpeed.** Not installed. Required for the ZeRO-3 path in §9.

### Why 8-bit AdamW, not plain AdamW

Full-parameter 14.7B with a standard fp32 AdamW does not fit in 190 GB:

| Component | Bytes/param | Total |
|---|---:|---:|
| bf16 weights | 2 | 29 GB |
| bf16 gradients | 2 | 29 GB |
| AdamW fp32 master + m + v | 12 | 176 GB |
| **Total** | **16** | **~235 GB** |

That is 235 GB against a 190 GB budget, before activations. ZeRO-3 shards it but does not
reduce the total. Using bitsandbytes `AdamW8bit` drops optimizer states from 176 GB to ~88 GB
(total ~147 GB), which fits with headroom. The alternative — ZeRO-3 with CPU optimizer offload —
also fits (251 GB host RAM) but makes every step several times slower across six runs.

**This is a deviation from a plain fp32 AdamW and must be recorded in §9.** It applies
identically to every experiment, so cross-experiment comparability holds.

### Why base, not Instruct

GNEM-Bench-v1's protocol scores **parsed raw completions** from a base (non-instruct)
completion model, and its canonical frozen `raw_outputs/base.json` was generated that way.
Staying on `Qwen2.5-14B` base keeps that protocol and lets E0 reuse those frozen outputs
directly. Every experiment below reads `Qwen2.5-14B` base where an earlier draft said
`-Instruct`.

### Relationship to the existing QLoRA results

`benchmarks/gnem_bench_v1/RESULTS_v1.md` reports base / KB-only / KB+web under **4-bit QLoRA
r64**. Those runs cannot serve as E1/E3 for a full-parameter study — the training method
differs. They are retained as a **separate QLoRA arm, reported in an appendix**, and E1–E6 are
trained fresh. The one genuine carry-over is E0: the base model is untouched and decoding is
identical, so `outputs/question_eval/raw_outputs/base.json` is reused as-is.

---

## 1. Research Objective

This study separates two different learning goals:

1. **Domain knowledge learning**
   - Train the model on raw Georgia EV supply-chain documents.
   - Use continued pretraining with a causal language-modeling objective.

2. **Analytical task learning**
   - Train the model to answer count, sum, filter, grouping, ranking, and list questions.
   - Use supervised fine-tuning on analytical examples generated deterministically from the company JSON.

The main question is:

> Does raw-document continued pretraining improve domain knowledge, and does analytical SFT make that knowledge more usable for structured questions?

---

## 2. Available Data

### 2.1 Structured KB data

- Approximately 205 company records.
- Available in Excel and JSON.
- Possible fields:
  - company name,
  - canonical company ID,
  - supplier category,
  - EV role,
  - products and services,
  - city,
  - county,
  - latitude and longitude,
  - employment,
  - investment,
  - OEM relationships,
  - source information.

### 2.2 Web data

- Approximately 9,700 LLM-generated wiki pages.
- Derived from collected web pages.
- May contain:
  - company history,
  - facilities,
  - employment,
  - investments,
  - products,
  - partnerships,
  - OEM relationships,
  - expansions,
  - closures,
  - layoffs,
  - dated announcements.

The web pages must be cleaned, deduplicated, dated where possible, and linked to original source URLs before training.

### 2.3 Existing validated questions

- 42 human-validated questions based on the Excel knowledge base.
- These questions are the frozen final KB test set.
- They must not be used for training, validation, early stopping, prompt tuning, or paraphrase generation.

### 2.4 Analytical SFT data

This dataset does not exist yet.

It will be generated automatically from the 205-company JSON using deterministic Python or SQL calculations.

### 2.5 Web evaluation data

Create a new human-validated web test set from held-out web documents.

**This replaces GNEM-Web-18, and the replacement is not a refinement — it changes the
question being asked.** Web-18 is built deliberately from pages *proven to be in the training
set*, and measures **absorption**: did facts present in the corpus land in the weights?
`RESULTS_v1.md` finding #1 (38.9% vs 5.6% KB-only) rests entirely on it. `Web-Gold-42` is built
from **held-out** pages and measures **generalization** instead — a different and harder
question. Consequences:

- Finding #1 becomes historical. The papers must say so explicitly rather than silently
  dropping it; the Web-18 gold and its scored outputs stay in the repo as the QLoRA-arm record.
- Held-out pages must be excluded from the corpus **before** E2/E3 train, so the web split is
  redefined for every run in this plan. The existing KB+web run used all 7,760 train pages and
  therefore has no held-out partition to score against.

Recommended evaluation sets:

| Test set | Size | Purpose |
|---|---:|---|
| `KB-Gold-42` | 42 | Existing Excel-grounded final test |
| `Web-Gold-42` | 42 | Held-out web knowledge evaluation |
| `Mixed-Gold-20` | 20 | Questions requiring KB and web knowledge |
| `Analytical-Synthetic-Test` | 100+ | Held-out count, sum, filter, and ranking tasks |
| `General-Capability-Test` | Small fixed set | Catastrophic-forgetting check |

---

## 3. Data Preparation

### 3.1 KB raw-document corpus

Convert each JSON company record into a consistent textual document.

Example:

```text
<document>
<source_type>structured_kb</source_type>
<record_id>KB-00125</record_id>
<entity_id>COMPANY-0082</entity_id>
<company_name>Example Company</company_name>
<supplier_category>Tier 1</supplier_category>
<ev_role>Battery Pack</ev_role>
<products_services>Battery enclosures</products_services>
<county>Jackson</county>
<state>Georgia</state>
<employment>250</employment>
<primary_oems>Example OEM</primary_oems>
<as_of_date>2026-05-01</as_of_date>

Example Company is a Tier 1 supplier in Georgia's electric-vehicle
manufacturing ecosystem. It produces battery enclosures and operates
in Jackson County, Georgia. The record reports 250 employees.
</document>
```

Rules:

- Do not invent missing values.
- Preserve record IDs and canonical entity IDs.
- Preserve dates for time-sensitive facts.
- Distinguish companies, facilities, projects, and relationships.
- Do not repeat the same fact unnecessarily.
- Keep the 42 gold questions out of the corpus.

### 3.2 Web raw-document corpus

Treat each wiki page as one logical source document.

Example:

```text
<document>
<source_type>web_wiki</source_type>
<page_id>WEB-0051685</page_id>
<canonical_entity_id>COMPANY-0017</canonical_entity_id>
<title>Company manufacturing announcement</title>
<source_url>...</source_url>
<published_date>...</published_date>
<retrieved_date>...</retrieved_date>

Cleaned source-grounded page text...
</document>
```

Rules:

- Use the whole page when it fits within the selected sequence length.
- Split long pages by Markdown headings, sections, or paragraphs.
- Pack multiple short pages only with clear end-of-document separators.
- Split train and validation data at the document level before chunking.
- Keep duplicate or near-duplicate pages in the same split.
- Remove boilerplate, menus, repeated navigation, and empty sections.
- Preserve original source URLs and dates.
- Do not trust generated entity names without validation.
- Leave uncertain entity links unresolved.

### 3.3 Analytical SFT dataset

Generate the analytical training examples from the JSON using deterministic code.

Include:

- count rows,
- count distinct companies,
- count facilities,
- single-field filtering,
- multi-field filtering,
- sum employment,
- average employment,
- minimum and maximum,
- group by county,
- group by supplier category,
- top-k and ranking,
- exhaustive lists,
- no-result questions,
- missing-value handling,
- duplicate-company handling,
- comparison questions,
- percentage and ratio questions.

Example:

```json
{
  "instruction": "How many Tier 1/2 companies are located in Georgia?",
  "operation": "filter_then_count_distinct",
  "filters": {
    "supplier_category": "Tier 1/2",
    "state": "Georgia"
  },
  "target_field": "canonical_company_id",
  "answer": 18
}
```

The answer must be calculated by Python or SQL, not generated by an LLM.

Recommended analytical split:

- 80% training,
- 10% development,
- 10% synthetic internal test.

The split should hold out operation combinations, filters, entities, and question templates where possible.

---

## 4. Fine-Tuning Techniques

### 4.1 Continued pretraining

**Correct name:** Full-parameter continued pretraining or domain-adaptive pretraining.

**Objective:**

```text
Predict the next token in raw domain documents.
```

Use it for:

- raw KB documents,
- raw web documents,
- combined KB and web documents.

It teaches:

- domain vocabulary,
- company names,
- company-product associations,
- locations,
- supplier relationships,
- historical and web-derived facts.

It does not guarantee exact count, sum, filtering, or aggregation.

### 4.2 Analytical supervised fine-tuning

**Correct name:** Full-parameter analytical SFT.

**Objective:**

```text
Instruction + optional structured input → validated analytical answer
```

Use it for:

- counts,
- sums,
- grouping,
- filtering,
- exhaustive lists,
- ranking,
- missing values,
- duplicate handling,
- no-result questions.

### 4.3 Sequential training

The main sequential pipeline is:

```text
Qwen2.5-14B
        ↓
Full raw-document continued pretraining
        ↓
Full analytical supervised fine-tuning
```

The raw-document checkpoint must be evaluated before analytical SFT so that the contribution of each stage can be measured.

---

## 5. Core Experiments

### Experiment E0 — Base Model

| Field | Description |
|---|---|
| Experiment ID | `E0-BASE` |
| Starting checkpoint | `Qwen2.5-14B` |
| Training data | None |
| Fine-tuning technique | None |
| Purpose | Establish the unchanged baseline |
| Evaluate on | KB-Gold-42, Web-Gold-42, Mixed-Gold-20, Analytical-Synthetic-Test, General-Capability-Test |
| Expected result | Baseline factual, analytical, and general performance |
| Status | Planned |

### Experiment E1 — KB Raw-Document Continued Pretraining

| Field | Description |
|---|---|
| Experiment ID | `E1-KB-CPT` |
| Starting checkpoint | E0 base model |
| Training data | Serialized raw documents from the 205-company JSON |
| Fine-tuning technique | Full-parameter continued pretraining |
| Training objective | Causal language modeling |
| Purpose | Measure knowledge learned from the structured KB alone |
| Important limitation | The KB corpus is small and may encourage memorization |
| Evaluate on | KB-Gold-42, Web-Gold-42, Analytical-Synthetic-Test, General-Capability-Test |
| Main comparison | E0 vs E1 |
| Status | Planned |

Questions answered:

- Does the model learn company facts from serialized KB records?
- Does KB-only CPT improve KB-Gold-42?
- Does it improve structured analytics without analytical SFT?
- Does it cause catastrophic forgetting?

### Experiment E2 — Web Raw-Document Continued Pretraining

| Field | Description |
|---|---|
| Experiment ID | `E2-WEB-CPT` |
| Starting checkpoint | E0 base model |
| Training data | Cleaned, deduplicated, held-in web wiki pages |
| Fine-tuning technique | Full-parameter continued pretraining |
| Training objective | Causal language modeling |
| Purpose | Isolate the knowledge contribution of the web corpus |
| Evaluate on | Web-Gold-42, KB-Gold-42, Mixed-Gold-20, General-Capability-Test |
| Main comparison | E0 vs E2 |
| Status | Planned |

Questions answered:

- Does web-only training improve web factual recall?
- Does web training damage structured KB performance?
- Does the model learn temporal and historical facts?
- Are web gains caused by useful information or noisy repetition?

### Experiment E3 — KB + Web Raw-Document Continued Pretraining

| Field | Description |
|---|---|
| Experiment ID | `E3-KBW-CPT` |
| Starting checkpoint | E0 base model |
| Training data | Serialized KB documents plus cleaned web wiki pages |
| Fine-tuning technique | Full-parameter continued pretraining |
| Training objective | Causal language modeling |
| Purpose | Build the primary raw-document domain model |
| Evaluate on | All test sets |
| Main comparisons | E0 vs E3, E1 vs E3, E2 vs E3 |
| Status | Planned |

Questions answered:

- Does adding web data improve beyond KB-only training?
- Does adding the structured KB improve beyond web-only training?
- Does the combined corpus improve mixed-source questions?
- Does the larger corpus increase hallucinations or outdated answers?

### Experiment E4 — Analytical SFT Only

| Field | Description |
|---|---|
| Experiment ID | `E4-SFT-ONLY` |
| Starting checkpoint | E0 base model |
| Training data | Deterministically generated analytical SFT dataset from the company JSON |
| Fine-tuning technique | Full-parameter supervised fine-tuning |
| Training objective | Instruction and analytical-answer learning |
| Purpose | Measure what analytical SFT provides without raw-document CPT |
| Evaluate on | KB-Gold-42, Analytical-Synthetic-Test, Web-Gold-42, General-Capability-Test |
| Main comparison | E0 vs E4 |
| Status | Planned |

Questions answered:

- Does analytical SFT improve count, filter, sum, list, and ranking questions?
- Does it mostly memorize the fixed dataset?
- Does it improve compositional analytical questions?
- Does it harm general or web knowledge?

### Experiment E5 — KB CPT Followed by Analytical SFT

| Field | Description |
|---|---|
| Experiment ID | `E5-KB-CPT-SFT` |
| Starting checkpoint | E1-KB-CPT |
| Stage 1 data | Raw KB documents |
| Stage 1 technique | Full-parameter continued pretraining |
| Stage 2 data | Analytical SFT dataset generated from JSON |
| Stage 2 technique | Full-parameter supervised fine-tuning |
| Purpose | Test whether KB CPT improves analytical SFT |
| Evaluate on | KB-Gold-42, Analytical-Synthetic-Test, Web-Gold-42, General-Capability-Test |
| Main comparisons | E1 vs E5, E4 vs E5 |
| Status | Planned |

Questions answered:

- Does domain knowledge learned from raw KB documents improve analytical SFT?
- Does CPT before SFT improve factual recall?
- Does SFT improve exact operations over the KB?

### Experiment E6 — KB + Web CPT Followed by Analytical SFT

| Field | Description |
|---|---|
| Experiment ID | `E6-KBW-CPT-SFT` |
| Starting checkpoint | E3-KBW-CPT |
| Stage 1 data | Raw KB plus cleaned web documents |
| Stage 1 technique | Full-parameter continued pretraining |
| Stage 2 data | Analytical SFT dataset generated from the KB JSON |
| Stage 2 technique | Full-parameter supervised fine-tuning |
| Purpose | Build and evaluate the main final fine-tuned model |
| Evaluate on | All test sets |
| Main comparisons | E3 vs E6, E4 vs E6, E5 vs E6 |
| Status | Planned |

Questions answered:

- Does KB+Web CPT improve analytical SFT beyond SFT alone?
- Does analytical SFT preserve web knowledge learned during CPT?
- Does the final model improve factual and analytical performance together?
- Does the SFT stage cause web-knowledge forgetting?

---

## 6. Main Comparisons

| Comparison | Research meaning |
|---|---|
| E0 vs E1 | Effect of KB raw-document CPT |
| E0 vs E2 | Effect of web raw-document CPT |
| E0 vs E3 | Effect of combined raw-document CPT |
| E1 vs E3 | Marginal contribution of web data |
| E2 vs E3 | Marginal contribution of structured KB data |
| E0 vs E4 | Effect of analytical SFT alone |
| E1 vs E5 | Effect of analytical SFT after KB CPT |
| E3 vs E6 | Effect of analytical SFT after KB+Web CPT |
| E4 vs E5 | Does KB CPT improve SFT? |
| E4 vs E6 | Does KB+Web CPT improve SFT? |
| E5 vs E6 | Does web pretraining add value before analytical SFT? |

---

## 7. Web Test-Set Construction

Create `Web-Gold-42` only from documents excluded from training.

Recommended distribution:

| Category | Questions |
|---|---:|
| Company and facility facts | 10 |
| Products and services | 7 |
| Investment and employment | 6 |
| OEM and supplier relationships | 6 |
| Temporal or historical facts | 6 |
| Location questions | 4 |
| Insufficient-evidence questions | 3 |
| **Total** | **42** |

Each question record should contain:

```json
{
  "question_id": "WEB-Q-001",
  "question": "...",
  "validated_answer": "...",
  "acceptable_aliases": [],
  "source_url": "...",
  "supporting_passage": "...",
  "published_date": "...",
  "canonical_entity_id": "...",
  "category": "...",
  "answer_type": "...",
  "time_sensitive": true
}
```

Requirements:

- Validate against the original source, not only the generated wiki page.
- Hold out the full source page and all duplicate versions.
- Freeze the test set before training.
- Store a version number and checksum.
- Do not revise answers after viewing final model outputs.

---

## 8. Evaluation Framework

### 8.1 Primary deterministic metrics

Use deterministic metrics as the main research scores.

#### Fact and field questions

- normalized exact match,
- field-value accuracy,
- entity accuracy,
- temporal accuracy.

#### Count questions

- exact count accuracy,
- absolute error,
- relative error,
- correct count basis:
  - distinct companies,
  - rows,
  - facilities.

#### List questions

- precision,
- recall,
- entity F1,
- exact-set match,
- missing-entity count,
- extra-entity count.

#### Sum and aggregation questions

- exact numeric accuracy,
- tolerance accuracy for decimal results,
- absolute error,
- relative error,
- missing-value handling correctness.

#### Structured outputs

- JSON parse success,
- schema validation,
- required-field accuracy,
- operation-label accuracy,
- filter-field accuracy.

### 8.2 DeepEval metrics

Use DeepEval as a secondary layer for:

- correctness of open-ended answers,
- completeness,
- usefulness,
- answer relevance,
- hallucination against curated evidence,
- explanation quality.

Do not allow DeepEval scores to override deterministic errors.

Example:

- A fluent answer with the wrong count is incorrect even if usefulness is high.
- A complete-looking company list with extra companies must be penalized by entity precision.

### 8.3 Human evaluation

Manually audit:

- all 42 KB-Gold answers for the main models,
- all 42 Web-Gold answers for E0, E2, E3, and E6,
- a stratified sample from the analytical synthetic test,
- all major disagreements between deterministic and DeepEval scores.

### 8.4 General-capability evaluation

Because all model parameters are updated, evaluate:

- instruction following,
- summarization,
- basic reasoning,
- basic coding,
- general factual behavior,
- refusal behavior.

Purpose:

- detect catastrophic forgetting,
- detect loss of instruction-following ability,
- measure the trade-off between domain gain and general degradation.

---

## 9. Training Controls

All comparable experiments should use:

- the same exact base checkpoint (`Qwen2.5-14B`, base, non-instruct),
- the same tokenizer,
- the same random-seed policy,
- BF16 where supported,
- distributed full-parameter training,
- **DeepSpeed ZeRO-3** (not FSDP — one strategy, applied uniformly),
- **bitsandbytes `AdamW8bit`** — a deliberate deviation from fp32 AdamW, forced by the VRAM
  arithmetic in §0 and applied identically to every experiment so comparisons stay valid,
- gradient checkpointing,
- the same sequence length where possible,
- identical evaluation prompts,
- identical decoding parameters,
- identical test sets,
- validation-only checkpoint selection,
- no test-set tuning.

Record:

- total tokens seen,
- optimizer steps,
- effective batch tokens,
- corpus passes,
- learning rate,
- scheduler,
- warmup,
- sequence length,
- GPU memory,
- training time,
- checkpoint size.

---

## 10. Recommended Training Order

```text
0. Clear the §0 preconditions: GPU driver, >=250 GB disk, DeepSpeed installed.
1. Audit and clean the KB data.
2. Clean and deduplicate the web wiki corpus; carve out the held-out web partition.
3. Freeze KB-Gold-42.
4. Create and freeze Web-Gold-42 (held-out pages only).
5. Create and freeze Mixed-Gold-20.
6. Generate analytical SFT train/dev/test data deterministically.
7. Evaluate E0-BASE (reuses the frozen raw_outputs/base.json).
8. Run a small full-training smoke test; log peak VRAM and confirm it is under budget.
9. Train and evaluate E1-KB-CPT.
10. Train and evaluate E2-WEB-CPT.
11. Train and evaluate E3-KBW-CPT.
12. Train and evaluate E4-SFT-ONLY.
13. Continue E1 into E5-KB-CPT-SFT.
14. Continue E3 into E6-KBW-CPT-SFT.
15. Compare factual learning, analytical improvement, web contribution, and forgetting.
```

Steps 2 and 4 are ordered before any training for a reason: once E2/E3 have trained on a page,
that page can never re-enter `Web-Gold-42`. The held-out partition is a one-way decision.

### Implementation gaps this order assumes are closed

| Gap | Where |
|---|---|
| Full-parameter ZeRO-3 path (only LoRA/QLoRA exists today; `configs/methods/full_finetune_placeholder.yaml` is a stub that raises `NotImplementedError`) | `src/ssft/train/{trainer_factory,model_loader}.py` |
| Per-stage SFT gate — `ssft.data.schemas.assert_no_qa_fields()` currently bans QA fields globally; CPT stages must keep the ban, E4–E6 must not | `src/ssft/data/schemas.py`, `tests/test_no_qa_format.py` |
| Analytical SFT generator (does not exist) | new `src/ssft/data/analytical_sft.py` |
| General-Capability-Test (does not exist) | new |

---

## 11. Required Result Table

| Model | KB-Gold-42 | Web-Gold-42 | Mixed-Gold-20 | Analytical test | Count acc. | Sum acc. | List F1 | General capability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 Base | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| E1 KB CPT | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| E2 Web CPT | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| E3 KB+Web CPT | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| E4 SFT only | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| E5 KB CPT → SFT | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| E6 KB+Web CPT → SFT | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

---

## 12. Experiment Record Template

```markdown
## Experiment ID

**Status:**  
**Starting checkpoint:**  
**Training data:**  
**Dataset version:**  
**Dataset checksum:**  
**Training technique:**  
**Objective:**  
**Train/validation split:**  
**Sequence length:**  
**Total training tokens:**  
**Optimizer:**  
**Learning rate:**  
**Scheduler:**  
**Warmup:**  
**Microbatch:**  
**Gradient accumulation:**  
**Effective batch tokens:**  
**Random seed:**  
**GPU configuration:**  
**Distributed strategy:**  
**Training time:**  
**Peak GPU memory:**  
**Selected checkpoint:**  
**Selection reason:**  
**KB-Gold results:**  
**Web-Gold results:**  
**Mixed-Gold results:**  
**Analytical results:**  
**General-capability results:**  
**Failures or errors:**  
**Interpretation:**  
**Limitations:**  
**Decision:**  
**Next action:**  
```

---

## 13. Final Research Framing

The project should be described as:

> We perform full-parameter continued pretraining of Qwen2.5-14B (base) on raw Georgia EV supply-chain documents, comparing structured-KB-only, web-only, and combined KB+Web corpora. We then perform full-parameter analytical supervised fine-tuning using count, sum, filter, list, grouping, and ranking examples computed deterministically from the structured company JSON. The unchanged base model, the analytical-SFT-only model, and the sequential CPT-to-SFT models are evaluated on separate frozen KB, held-out web, mixed-source, analytical, and general-capability test sets. All runs use DeepSpeed ZeRO-3 with an 8-bit AdamW optimizer, applied uniformly.

This plan is motivated by a negative result the QLoRA arm already established
(`benchmarks/gnem_bench_v1/RESULTS_v1.md`): fine-tuning reliably improves parametric fact
retention (KB cloze recall 0.03 → 0.78) and web-fact absorption (+33 points), yet **analytical
accuracy stays near zero for every memorization model, with fully overlapping confidence
intervals**. Storing more facts in the weights did not buy the ability to filter, rank, count,
or aggregate over 205 rows. E4–E6 test whether analytical SFT is what closes that gap — and a
null result there is a publishable finding, since it would localize the capability to
retrieval plus deterministic computation rather than to fine-tuning at all.

The key distinction is:

> **Continued pretraining teaches domain knowledge. Analytical SFT teaches task behavior. The experiments must measure both separately before evaluating the combined model.**
