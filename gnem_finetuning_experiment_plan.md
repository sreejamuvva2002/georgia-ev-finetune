# Fine-Tuning Experiment Plan

**Project:** Georgia EV Supply-Chain Domain Model  
**Base model:** `Qwen2.5-14B` (base, non-instruct)  
**Primary goal:** Compare raw-document continued pretraining with analytical supervised fine-tuning  
**Scope:** Fine-tuning only; RAG and tool-use experiments are excluded  
**Training approach:** Full-parameter fine-tuning on 4 × RTX A6000 (192 GiB total, PCIe-only), DeepSpeed ZeRO-3 with 8-bit AdamW  
**Execution:** Two phases — E0 + E4 pilot, then 5 more runs only if the §10.1 gate opens  
**Current status:** Planned — Phase A unblocked; §0.1 driver repoint required before the first run

---

## 0. Preconditions

Three things must be true before any training can start (E4 is the first run — see §10).

1. **GPU driver.** `nvidia-smi` fails with `Driver/library version mismatch`. **Root-caused and
   locally fixable — see below.** Not a sysadmin blocker.
2. **Storage routing.** Not a "free up space" problem — a *which disk* problem. See below.
3. **DeepSpeed.** Not installed. Required for the ZeRO-3 path in §9.

Nothing here blocks Phase A once the driver symlinks are repointed. The plan's one remaining
*external* dependency is archival storage, and it is a **Phase B** precondition only (§0.4).

### 0.1 GPU driver — six symlinks, not a broken driver

The host kernel module is **580.95.05**. Six loader symlinks in `/usr/lib/x86_64-linux-gnu/`
point at **580.126.20**, whose module is not loaded — hence the mismatch. The correct 580.95.05
libraries are already present, bind-mounted read-only from the host, and the symlinks sit on the
container's writable overlay. Repoint them:

```bash
cd /usr/lib/x86_64-linux-gnu
for l in libnvidia-ml libcuda libnvidia-opencl libnvidia-opticalflow \
         libnvidia-ptxjitcompiler libnvidia-sandboxutils; do
  sudo ln -sf ${l}.so.580.95.05 ${l}.so.1
done
nvidia-smi   # verify before every run; never assume
```

Requires root. `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05` is a proven
per-process fallback for `nvidia-smi`, but it will not carry a training run — PyTorch needs
`libcuda.so.1` to resolve correctly too.

**This is a workaround to reapply, not a one-shot fix.** The symlinks are dated 2026-04-12 (image
build) while the container started 2026-07-11: the mismatch is baked into the *image*, and the
overlay is discarded on container recreation, which silently reintroduces the blocker. The
durable fix belongs to infra — either the host upgrades its module to 580.126.20 to match the
image, or the image ships 580.95.05 symlinks to match the host.

### 0.2 GPU inventory (verified, not assumed)

| Property | Value |
|---|---|
| GPUs | **4 × NVIDIA RTX A6000**, 48 GiB each — **192 GiB total** |
| Architecture | Ampere (sm_86) — bf16 supported, so §9's bf16 control holds |
| Topology (`nvidia-smi topo -m`) | all pairs `NODE` — **no NVLink; every GPU-to-GPU hop is PCIe** |
| Driver / CUDA | 580.95.05 / 13.0 |

The ~190 GB budget is **confirmed**. But the topology is a finding, not a formality:

**ZeRO-3 is forced onto the least communication-friendly fabric available.** ZeRO-3 all-gathers
sharded parameters on every forward *and* backward — ~600 GB/s over NVLink, PCIe here, one to two
orders of magnitude slower. And ZeRO-3 is not optional: ZeRO-2 would need 29 (replicated params)
+ 7 (sharded grads) + 22 (sharded 8-bit optimizer) ≈ **58 GiB per GPU against 48 GiB**, so it
does not fit. **The memory arithmetic was never the risk; step time is.**

Consequently the §10 smoke test gates on **throughput, not just peak VRAM** — see §0.7.

### 0.3 Storage — two disks, and the repo is on the small one

This box has two separate filesystems, and the repo sits on the one without room:

| Path | Device | Size | Free |
|---|---|---:|---:|
| `/` — the repo, `~/.cache/huggingface` | `overlay` (nvme2n1) | 3.7 T | **54 GB** |
| `/data/sreeja` — ollama store | `/dev/nvme0n1` | 3.7 T | **211 GB** |

No amount of pruning helps `/`: the large artifacts (ollama models) were never on it.
**Training artifacts must be routed to `/data/sreeja`.** `/dev/nvme1n1p2` shows 860 GB free but
is the **host's root filesystem**, bind-mounted **read-only** at 18 paths to inject the NVIDIA
driver. It is not spare storage and cannot be written to.

Budget against the 211 GB on `/data`. **Every figure below is provisional** until the §10 14B
smoke test measures the real ones — ~29 GB is a bf16 weights estimate that ignores ZeRO shard
size and consolidation scratch:

| Item | Size (provisional) |
|---|---:|
| Base model download (`Qwen2.5-14B`, bf16) | ~29 GB |
| **Phase A** — E4 only, 1 × ~29 GB | **~29 GB** |
| **Phase A total** | **~58 GB against 211 GB — comfortable** |

**Phase A needs no archiving and has no storage risk.** Two rules keep it that way, and they
apply to every run:

- **Keep one checkpoint per run** (`save_total_limit=1`). Intermediate checkpoints at ~29 GB
  each will blow the budget within a single experiment otherwise.
- **Do not persist optimizer state.** ZeRO-3 resumable state is ~88 GB per run at 8-bit and
  would not fit even once. Nothing in this plan needs it: E5 starts from E1's *weights* and E6
  from E3's, not from a resumable optimizer.

The 14B smoke test must establish, and §0's numbers must then be replaced by: peak ZeRO shard
size, export size, temporary consolidation size, total filesystem peak, and a **minimum
free-space threshold to start a run**.

### 0.4 Archive — a Phase B precondition, never a Phase A blocker

Phase B retains five more models. Under any accumulate-locally policy that lands near ~203 GB
against 211 GB, and **~8 GB of headroom cannot absorb ZeRO-3 checkpoint consolidation**, which
gathers a full ~29 GB bf16 model before writing. So Phase B is not runnable by accumulation.

**Policy: score, then archive, then start the next run.** Peak usage then stays near Phase A
levels regardless of how many experiments run. Per-run ordering:

```text
train → select checkpoint → export → score all required evaluations
      → manifest + checksums → archive → verify archive
      → delete local shards/export → confirm free space → next run
```

**The destination does not exist yet, and this box cannot provide one.** `/` is full, the host
root is read-only, and there is no NFS, CIFS, S3, or rclone — only `scp`. Note that
`/data/gnem_archive/` **frees no space at all**: it is the same disk as the checkpoints. Name it
anyway and rehearse the checklist against it, so the procedure is proven before it matters —
but treat it as a **rehearsal target, not a working archive**.

Phase B's real precondition is genuine external storage, verified by: named location; test
transfer; source/destination checksums match; test restore; permissions verified; and **the model
loads from the restored copy**. With the driver now fixable locally, **this is the plan's only
remaining external dependency** — worth requesting on its own.

The seven context-eval ollama models (~112 GB) are *not* the answer: under an archive policy the
space is unnecessary, and they back committed results in `outputs/context_vs_parametric/`.
**`gpt-oss:120b` (65 GB) must stay regardless** — it is the DeepEval judge §8.2 depends on.

### 0.5 How to route the artifacts

`ssft.utils.paths` hard-wires `OUTPUTS_ROOT = SSFT_ROOT / "outputs"` with no environment
override, and states the invariant that *"nothing this package writes ever lands outside its own
top-level folder."* Note that `outputs/question_eval/` and `outputs/context_vs_parametric/` are
**tracked in git** and must stay on `/` — so redirecting `outputs/` wholesale is wrong.

Route via symlink rather than a code change — the package still only writes under `SSFT_ROOT`,
so the invariant holds as written. Use a **new** `outputs/checkpoints` root for the
full-parameter weights:

```bash
mkdir -p /data/sreeja/ssft/{checkpoints,hf_cache}
ln -s /data/sreeja/ssft/checkpoints self_supervised_finetuning/outputs/checkpoints
echo 'outputs/checkpoints' >> self_supervised_finetuning/.gitignore
export HF_HOME=/data/sreeja/ssft/hf_cache   # moves the ~29 GB base download off /
```

Do **not** symlink `outputs/adapters/`: it tracks a `.gitkeep`, and it already holds ~17 GB of
real QLoRA checkpoints on `/` that back the appendix arm. Leave it where it is. A fresh
`checkpoints/` root avoids the git conflict entirely and gives the full-parameter path a name
that isn't a LoRA-era misnomer — `ssft.utils.paths` should gain a `CHECKPOINTS_ROOT` beside the
existing `ADAPTERS_ROOT`.

Verify with `df -h /data /` before and after the §10 step-8 smoke test: `/data` should drop and
**`/` should not move**. If `/` moves, the routing is wrong and the run will fill the 54 GB disk.

### 0.6 Why 8-bit AdamW, not plain AdamW

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

### 0.7 Throughput floor — fixed in advance, as a number

Because ZeRO-3 is forced onto a PCIe-only fabric (no NVLink), training is expected to be
**communication-bound**. The §10 14B smoke test must report, and gate on:

- **seconds/step and tokens/second** — not just peak VRAM;
- the all-gather / communication share of step time, if DeepSpeed's profiler exposes it;
- extrapolated wall-clock for E4, and separately for the web-corpus runs — E2/E3 cover ~9.7k
  pages and will hurt far more than E1's 205 records.

**Pick N before the smoke test runs and record it here:**

> Projected E4 wall-clock **> N days → do not start.** First reduce sequence length, tune
> gradient accumulation to amortise all-gather, or reconsider ZeRO CPU offload.

"Unacceptable" without a number invites rationalising a bad run after the fact. The same
discipline §10.1 applies to the gate applies here.

### 0.8 Why base, not Instruct

GNEM-Bench-v1's protocol scores **parsed raw completions** from a base (non-instruct)
completion model, and its canonical frozen `raw_outputs/base.json` was generated that way.
Staying on `Qwen2.5-14B` base keeps that protocol and lets E0 reuse those frozen outputs
directly. Every experiment below reads `Qwen2.5-14B` base where an earlier draft said
`-Instruct`.

### 0.9 Relationship to the existing QLoRA results

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
| `KB-Gold-42` | 42 | Existing Excel-grounded final test (closed-book) |
| `Web-Gold-42` | 42 | Held-out web knowledge evaluation |
| `Mixed-Gold-20` | 20 | Questions requiring KB and web knowledge |
| `Analytical-Synthetic-Test` | 100+ | Held-out count, sum, filter, and ranking tasks (closed-book) |
| **`Analytical-Counterfactual-Test`** | 100+ | **The §10.1 gate.** Same operations over *supplied* record subsets; answers recomputed per subset; scored native-chat only |
| `General-Capability-Test` | Small fixed set | Catastrophic-forgetting check |

### 2.6 `Analytical-Counterfactual-Test` — freeze it like gold

This is a new frozen artifact, not a variant of `KB-Gold-42` (whose questions are closed-book).
It is the gate's primary instrument, so it gets the gold-freeze discipline:

- **Freeze before any training.**
- **The SFT generator must not be able to reach it:** disjoint subset seeds, disjoint
  answer-value buckets, disjoint templates. Record the partition in the manifest.
- Store a version number and checksum, as §7 requires of the other gold sets.

This is the subtlest leakage vector in the design: no question string need be shared for the test
to leak, because both sets are *generated* from the same 205 rows. §9's leakage report checks
documents, not generator state, so it will not catch this on its own.

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

#### Emit context-grounded examples too — not only closed-book

The example above is **closed-book**: no records supplied, the answer recalled from weights. §4.2
already specifies the objective as `Instruction + optional structured input → validated
analytical answer`, and that optional input is now mandatory for part of the set.

**If every training example were closed-book, the §10.1 counterfactual gate would be
out-of-distribution and its failures uninterpretable** — the model would never have been taught
to read a table from context. Emit both forms in a **fixed, recorded proportion**:

```json
{
  "instruction": "Using only the supplied records, calculate the total employment of Tier 1 companies.",
  "records": [
    {"company_id": "C001", "supplier_category": "Tier 1", "employment": 120},
    {"company_id": "C002", "supplier_category": "OEM",    "employment": 400},
    {"company_id": "C003", "supplier_category": "Tier 1", "employment": 80}
  ],
  "operation": "filter_then_sum",
  "answer": 200
}
```

The answer is computed deterministically **over the supplied subset**, not over the full KB.

#### Vary the supplied records

Never repeatedly ask the same operation over the complete 205 — that teaches "this question → 18"
rather than the operation. Generate from: random company subsets; county subsets;
supplier-category subsets; varied subset sizes; modified numeric values; records with missing
values; duplicate-company scenarios; empty-result scenarios.

#### Hold out answer values, not just templates and entities

Template and entity holdout still permits memorizing a small answer set if the same values recur.
Track and record the distributions of: answer-value frequency, count distribution, sum
distribution, subset-size distribution, and operation frequency. The analytical and counterfactual
tests should favour **answer values uncommon or absent in training** wherever feasible.

Recommended analytical split:

- 80% training,
- 10% development,
- 10% synthetic internal test.

The split should hold out operation combinations, filters, entities, question templates, **answer
values, and subset seeds** where possible.

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
| Status | Planned — Phase B, conditional on the §10.1 gate |

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
| Status | Planned — Phase B, conditional on the §10.1 gate |

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
| Status | Planned — Phase B, conditional on the §10.1 gate |

Questions answered:

- Does adding web data improve beyond KB-only training?
- Does adding the structured KB improve beyond web-only training?
- Does the combined corpus improve mixed-source questions?
- Does the larger corpus increase hallucinations or outdated answers?

### Experiment E4 — Analytical SFT Only — **the pilot**

Runs **second**, immediately after E0, not fourth. It is the gate that decides whether E1/E2/E3/
E5/E6 run at all — see §10.1.

| Field | Description |
|---|---|
| Experiment ID | `E4-SFT-ONLY` |
| Starting checkpoint | E0 base model |
| Training data | Deterministically generated analytical SFT dataset from the company JSON |
| Fine-tuning technique | Full-parameter supervised fine-tuning |
| Training objective | Instruction and analytical-answer learning |
| Purpose | Measure what analytical SFT provides without raw-document CPT — and decide whether the analytical hypothesis is live before spending five more runs |
| Evaluate on | **Analytical-Counterfactual-Test** (gate, native chat), Analytical-Synthetic-Test, KB-Gold-42, Web-Gold-42, General-Capability-Test — all SFT scoring dual-format per §9.1 |
| Main comparison | E0 vs E4 — **reported but confounded**; E0 is a format floor, so the margin is dominated by format acquisition. Not the gate. |
| Gate | §10.1 — counterfactual-context, native chat, thresholds pre-registered before scoring |
| Expected outcome | **A2** — E4 receives no CPT, so weak closed-book knowledge is near-guaranteed |
| Status | Planned — Phase A |

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
| Status | Planned — Phase B, conditional on the §10.1 gate |

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
| Status | Planned — Phase B, conditional on the §10.1 gate |

Questions answered:

- Does KB+Web CPT improve analytical SFT beyond SFT alone?
- Does analytical SFT preserve web knowledge learned during CPT?
- Does the final model improve factual and analytical performance together?
- Does the SFT stage cause web-knowledge forgetting?

---

## 6. Main Comparisons

**Any comparison spanning a CPT-only and an SFT model carries a format confound** (§9.1): the
CPT arm is a completion model, the SFT arm a chat model. Report the SFT side's
`completion_bridge_score` for those rows, and state the confound. Only the §10.1 counterfactual
gate is confound-free, being within-model and within-format.

| Comparison | Research meaning |
|---|---|
| E0 vs E4 | **Confounded — reported, not decisive.** E0 is a format floor (it continues prompts rather than answering), so the margin is dominated by format acquisition, not analytics. |
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

### 8.5 Protocol sanity test — a §10.1 gate input

E4 starts from `Qwen2.5-14B` **base**, which has had no instruction post-training. So E4's SFT
set must teach the interaction format itself: user/assistant turns, output structure, operation
labels, refusal behaviour, analytical answer formatting. If it does not, an E4 failure means
*"the base model never learned the protocol"* — not *"analytical SFT cannot teach the task."*
Those must not be confused, so a **frozen, GNEM-free** protocol test gates the analytical
numbers.

`src/ssft/eval/eval_instruction_sanity.py` already exists and is the module to extend — but its
five prompts are **completion-format** (`"Q: What is 12 + 7?\nA:"`), written as a CPT damage
check. They cannot verify that an SFT'd model learned its chat template. Add an **SFT-format
variant** and keep the existing prompts for E1/E2/E3.

The SFT-format prompts must require no GNEM knowledge:

- return the number 19;
- extract the county from a supplied record;
- count three supplied records;
- return valid JSON with the requested fields;
- **follow the instruction rather than continuing the prompt** — aimed squarely at E0's observed
  failure mode.

**Failure routes to A1 (repair), never to Fail or A2.** A model that did not learn the protocol
has produced uninterpretable analytical numbers, not a finding about analytical SFT.

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
- identical evaluation **questions** and identical decoding — but **model-native prompt
  formatting**, recorded per run (see below); "identical prompts" is unachievable once SFT
  stages exist,
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
- checkpoint size,
- **seconds/step and tokens/second** (§0.7 — the PCIe-only topology makes throughput the risk),
- **communication share of step time**, where the profiler exposes it.

### 9.1 Dual-format evaluation for SFT stages

CPT-only models (E1/E2/E3) are completion-format; SFT models (E4/E5/E6) are chat-format. One
prompt cannot serve both without putting an arm outside its trained format. **Score every SFT
model twice:**

| Pass | Prompt | Role |
|---|---|---|
| **Native chat** — primary | the exact chat serialization used during SFT | **The official capability result.** What gets reported. |
| **Completion bridge** — diagnostic | the frozen `Q: …\nA:` wrapper E0–E3 receive | Quantifies prompt-format sensitivity; comparable to `raw_outputs/base.json`. |

Record per SFT run:

| Field | Meaning |
|---|---|
| `native_chat_score` | Performance in its trained format — the reported result |
| `completion_bridge_score` | Performance under the E0–E3 wrapper |
| `format_delta` | native − completion; the format-acquisition effect, measured not assumed |
| `prompt_template_hash` | Exact template used |
| `answer_parser_version` | Parser used for scoring |
| `stop_conditions` | Generation termination rules |

**Reading `format_delta`:** a large gap *with a strong native score* is **format acquisition, not
capability failure** — it is the expected signature of a model that learned its chat template,
and it is precisely the quantity this design sets out to measure. Completion-side collapse bears
only on the E0–E3 comparability narrative. It is never evidence against analytical capability,
and it has no bearing on the §10.1 gate, which is **native-chat only** — being within-model, the
gate needs no bridge and admits no format confound.

---

## 10. Recommended Training Order

**The six training runs are not committed to up front.** They run in two phases with a decision
gate between them, because the risk in this plan is concentrated in a single assumption and
that assumption can be tested with one run instead of six. See §10.1.

### Phase A — setup and pilot (2 runs)

```text
0. Clear the §0 preconditions: driver symlinks (verify nvidia-smi), artifacts routed to
   /data/sreeja, DeepSpeed installed. Pick and record N for the §0.7 throughput floor.
1. Audit and clean the KB data.
2. Clean and deduplicate the web wiki corpus; carve out the held-out web partition.
3. Freeze KB-Gold-42.
4. Create and freeze Web-Gold-42 (held-out pages only).
5. Create and freeze Mixed-Gold-20.
6. Generate analytical SFT train/dev/test — closed-book AND context-grounded (§3.3).
7. Build and freeze Analytical-Counterfactual-Test (§2.6); verify generator disjointness.
8. Pre-register every §10.1 threshold. Before any scoring.
9. Evaluate E0-BASE (reuses the frozen raw_outputs/base.json — no training).
10. Infrastructure smoke test: small model, ZeRO-3 + AdamW8bit. Proves the plumbing only.
11. **14B smoke test** — the one that validates the memory AND throughput arithmetic.
12. Train and evaluate E4-SFT-ONLY (dual-format, §9.1).   <-- the pilot
13. DECISION GATE (§10.1): A1 / Fail / A2 / Pass.
```

**Step 10 does not substitute for step 11.** A small-model run proves ZeRO-3 + `AdamW8bit` is
wired correctly; it says nothing about a 14B model's memory or step time. Do not mark the
configuration confirmed until the **14B** run:

- trains all parameters;
- stays under the per-GPU limit (48 GiB);
- **saves, reloads, resumes, and exports a usable checkpoint**;
- reports seconds/step, tokens/second, and a projected E4 wall-clock under the §0.7 floor;
- yields the real storage numbers that replace §0's provisional estimates.

### Phase B — the full matrix (5 runs), conditional on the gate

Ordered so each parent feeds its child immediately — under archive-after-each (§0.4), training
E1 and consuming it three runs later would force an archive/restore round-trip per lineage:

```text
14. Train and evaluate E1-KB-CPT.
15. Continue E1 into E5-KB-CPT-SFT.      (E1 consumed immediately; archive both)
16. Train and evaluate E3-KBW-CPT.
17. Continue E3 into E6-KBW-CPT-SFT.     (E3 consumed immediately; archive both)
18. Train and evaluate E2-WEB-CPT.       (no child; archive)
19. Compare factual learning, analytical improvement, web contribution, and forgetting.
```

Between every Phase B run, follow §0.4's ordering: score → manifest + checksums → archive →
verify → delete local → confirm free space → next run.

Steps 2 and 4 are ordered before **any** training for a reason: once E2/E3 have trained on a
page, that page can never re-enter `Web-Gold-42`. The held-out partition is a one-way decision,
and it must be made in Phase A even though the runs that consume it are in Phase B.

## 10.1 The decision gate

E4 is SFT-only from base. It is the cheapest direct test of this plan's central hypothesis —
**can analytical SFT teach count / sum / filter / rank at all?** — and E0 vs E4 is already a
stated main comparison in §6. Everything downstream (E5, E6) assumes the answer is yes.

Why the assumption deserves a gate rather than trust:

- §4.1 already concedes that CPT "does not guarantee exact count, sum, filtering, or
  aggregation." The whole analytical story rests on the SFT stage carrying that weight alone.
- The QLoRA arm found analytical accuracy near zero across base / KB-only / KB+web with fully
  overlapping CIs (`benchmarks/gnem_bench_v1/RESULTS_v1.md`). Nothing yet suggests more
  training on the same 205 rows changes that.
- E1/E2 mostly re-establish structure the QLoRA arm already characterized (KB-only forgets web;
  KB+web retains both). They are ablations, not the risk.

**E0 vs E4 cannot be the gate criterion.** E0 does not answer questions — it *continues* them.
Sampled from the frozen `outputs/question_eval/raw_outputs/base.json`:

| Question | E0's actual output |
|---|---|
| Show all "Tier 1/2" suppliers in Georgia… | `"Sort by EV Supply Chain Role. Show only the first 10 results."` |
| Which Georgia companies are classified under Battery Cell… | `"Please provide the company name, role, and tier in the following format:…"` |

E0's 5.0% is a **format floor, not a knowledge measurement**. E4 is SFT'd and will answer, so
`E4 > E0` is near-guaranteed and its margin is dominated by *format acquisition*. It stays a
reported number (§6) with that confound stated — it is not the gate.

**The gate is the counterfactual-context test**, because it is the only instrument that holds
format constant and varies only the data. Supply a *subset* of records and ask for the same
operation over that subset: the correct answer moves with the records (full KB → 18; subset A →
7; subset B → 11). A memorizing model answers 18 regardless. This is **not RAG** — no retrieval,
no ranking, no index — so it does not breach the scope exclusion in the header.

### The four outcomes

Evaluated in this order; first match wins.

```text
Is the E4 run technically interpretable?
├── No  → A1 Repair
└── Yes → Does E4 generalize on counterfactual context-grounded analytics?
          ├── No  → Fail
          └── Yes → Is closed-book GNEM knowledge still clearly weak?
                    ├── Yes → A2 Knowledge Probe
                    └── No  → Pass
```

| # | Outcome | Exact condition | Action |
|---|---|---|---|
| 1 | **A1 — Repair** | Training unstable; SFT protocol sanity fails; leakage; malformed or severely imbalanced dataset; gains only on dev | Uninterpretable. Repair dataset or training config, **re-run E4**. |
| 2 | **Fail — analytical capability** | Run clean, but answers do not track supplied records, collapse to fixed values, or fail held-out counterfactual operations | **Stop at two runs.** Report: analytical SFT over the fixed GNEM data did not produce generalizing aggregation. |
| 3 | **A2 — Knowledge probe** | Context-grounded analytics generalize, **but** closed-book KB/web factual performance is weak | Short KB+Web CPT → analytical SFT pilot; **one** re-gate. |
| 4 | **Pass** | Context-grounded analytics generalize **and** closed-book performance is adequate enough that missing knowledge is not the dominant limitation | Open Phase B in full. |

The cut that makes this coherent:

- **Cannot compute from supplied records → analytical-operation failure** (Fail). CPT is not the
  remedy: the records were already in the prompt, so missing parametric knowledge is *proven not
  to be* the bottleneck. CPT only adds knowledge to weights.
- **Can compute from supplied records but cannot answer without them** → the operation exists,
  parametric knowledge is missing (A2). CPT is exactly this intervention.
- **Both** → Pass.

Order matters and so does Pass's second clause. A1 precedes everything: an uninterpretable run
cannot be a Pass, a Fail, or evidence for A2. And **Pass must explicitly require adequate
closed-book performance** — without that clause it subsumes every case A2 exists to catch, and
the knowledge-probe branch becomes unreachable.

### Pre-registration — every boundary, before E4 is scored

§9 forbids test-set tuning; a threshold chosen after seeing the numbers breaches that in spirit.
Fix all of these here, in this section, beside the frozen gold:

- **Fail vs Pass** — numeric floors on the counterfactual primary: count, sum, filter, exact-set
  and operation-selection accuracy, plus answer-value generalization. Without floors, "does not
  track supplied records" is a judgement call and Fail-vs-weak-Pass becomes post-hoc.
- **A2 vs Pass** — the closed-book threshold, stated **relative to the frozen base**: cloze
  recall or KB-Gold entity F1 clearly above E0 (0.027), with a stated absolute floor. **Do not
  use the QLoRA arm's 0.782**: that came from KB-only *CPT* under *QLoRA* — a different method
  and a different stage. E4 receives no CPT, so a CPT-derived ceiling would make Pass unreachable
  by construction. That ceiling is the yardstick for E5/E6, not for SFT-only.

### A2 is the expected outcome, and the loop is bounded

E4 is SFT-only, so weak closed-book knowledge is near-guaranteed — whatever KB facts it absorbs
arrive incidentally through the analytical SFT data. **If the operation is learned at all, expect
A2.** Pass would be the surprise: it would mean the analytical SFT set incidentally taught enough
KB facts to matter, itself worth reporting.

That is a feature — A2 lands exactly on *"SFT supplies the operation, CPT supplies the
knowledge"*, which is E5/E6's hypothesis. Two things to state plainly:

- **A2's pilot is a scaled-down E6.** It is justified by the same cost-control logic as Phase A
  (one short pilot de-risks five full runs), not because it is a distinct experiment.
- **One pilot only.** Allow one A2 pilot → one re-gate → then either open Phase B or stop, with
  the decision written down either way. "Re-read this gate" must not license piloting forever.

### Gate measurements

**Primary — E4, native chat, on counterfactual record sets.** The same question run over
multiple supplied datasets with different correct answers. Score: count, sum, filter, exact-set,
and operation-selection accuracy, plus answer-value generalization.

**Secondary generalization checks:** held-out templates; held-out entities; held-out
operation/filter compositions; uncommon or unseen answer values; varied subset sizes; duplicates
and missing values; empty-result cases.

**Closed-book diagnostic:** the frozen KB and web questions. This separates **A2 from Pass
only**. It must never decide whether E4 learned aggregation — that is the primary measurement's
job, and conflating the two is exactly what makes the branches overlap.

### Phase B entry paths

Pass is rare by construction, so Phase B must not be read as though Pass is the normal door:

| Gate outcome | Route into Phase B |
|---|---|
| **Fail** | Stop at two runs. No Phase B. |
| **A2** | Short KB+Web CPT → analytical SFT pilot → **one** re-gate → Phase B or stop. **The expected path.** |
| **Pass** | Phase B in full, directly. |
| **A1** | Not an exit — repair, re-run E4, re-enter the gate. |

A null result is not a failed project — it is the finding, and §13 frames it as such. The gate
exists so that finding costs two runs instead of six.

### Implementation gaps this order assumes are closed

| Gap | Where |
|---|---|
| Full-parameter ZeRO-3 path (only LoRA/QLoRA exists today; `configs/methods/full_finetune_placeholder.yaml` is a stub that raises `NotImplementedError`) | `src/ssft/train/{trainer_factory,model_loader}.py` |
| `CHECKPOINTS_ROOT` for full-parameter weights, routed to `/data` per §0 | `src/ssft/utils/paths.py` |
| Per-stage SFT gate — `ssft.data.schemas.assert_no_qa_fields()` currently bans QA fields globally; CPT stages must keep the ban, E4–E6 must not | `src/ssft/data/schemas.py`, `tests/test_no_qa_format.py` |
| Analytical SFT generator, emitting **both** closed-book and context-grounded examples (does not exist) | new `src/ssft/data/analytical_sft.py` |
| `Analytical-Counterfactual-Test` builder + freeze (does not exist) | new |
| SFT-format protocol sanity prompts — the existing five are completion-format (`"Q: …\nA:"`), a CPT damage check, and cannot verify an SFT'd model learned its chat template | `src/ssft/eval/eval_instruction_sanity.py` |
| Dual-format scoring (native chat + completion bridge) for SFT stages | `src/ssft/eval/`, benchmark scripts |
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
**Seconds/step:**  
**Tokens/second:**  
**Communication share of step time:**  
**Peak GPU memory:**  
**Peak filesystem usage (/data):**  
**Selected checkpoint:**  
**Selection reason:**  
**Prompt format (native chat / completion bridge):**  
**Prompt template hash:**  
**Answer parser version:**  
**Stop conditions:**  
**KB-Gold results:**  
**Web-Gold results:**  
**Mixed-Gold results:**  
**Analytical results (closed-book):**  
**Analytical-Counterfactual results (native chat):**  
**native_chat_score / completion_bridge_score / format_delta:**  
**Protocol sanity result:**  
**General-capability results:**  
**Archive location + checksum + restore verified:**  
**Failures or errors:**  
**Interpretation:**  
**Limitations:**  
**Decision:**  
**Next action:**  
```

---

## 13. Final Research Framing

The project should be described as:

> We perform full-parameter continued pretraining of Qwen2.5-14B (base) on raw Georgia EV supply-chain documents, comparing structured-KB-only, web-only, and combined KB+Web corpora. We then perform full-parameter analytical supervised fine-tuning using count, sum, filter, list, grouping, and ranking examples computed deterministically from the structured company JSON. The unchanged base model, the analytical-SFT-only model, and the sequential CPT-to-SFT models are evaluated on separate frozen KB, held-out web, mixed-source, analytical, and general-capability test sets. **We distinguish memorization from acquired capability with a counterfactual-context test: the same operation is posed over varying supplied record subsets, so a model that has memorized a fixed answer is separated from one that computes over the records in front of it.** All runs use DeepSpeed ZeRO-3 with an 8-bit AdamW optimizer on 4 × RTX A6000, applied uniformly.

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
