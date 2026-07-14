# Plan: KB-only memorization + KB+web mixed experiment

Status doc for the self-supervised fine-tuning experiments on the Georgia EV knowledge base.
Persisted in-repo so it survives across sessions.

**Order (per user):** (1) Qwen2.5-14B KB-only memorization — *built + running*; (2) evaluate
those 14B adapters; (3) **Qwen3.6-35B-A3B-Base** (MoE) KB-only memorization; (4) **KB + web-wiki
mixed on the 35B model ONLY**. The 14B is the fast dense baseline; all 35B work is disk-gated
(35B bf16 ≈ 70 GB, so the 28 GB 14B cache is freed first) and runs in 4-bit/8-bit QLoRA. The
KB+web mixed experiment was originally scoped on the 14B but is now **on the 35B only**.

---

## Phase 1 — KB-only memorization (BUILT + RUNNING)

**Goal:** push the KB *memorization* run's `train_loss` below 0.1 using researched techniques,
on Qwen2.5-14B via LoRA/QLoRA over `kb_full.jsonl` (205 rows, `train_all`, no held-out split).

**Honest framing:** `train_loss < 0.1` is a *memorization* target (loss < 0.2 already signals
overfitting). It measures verbatim absorption of the 205 rows — **not** generalization. The
resulting adapter is not a QA model and does not replace RAG.

**Researched levers (cited):** high rank (r=64) + zero dropout; **rsLoRA** (`use_rslora`,
`α/√r` scaling — the key lever for lowering loss at high rank); quantization precision (4-bit
NF4 noise floor vs 8-bit vs bf16); aggressive over-training (~50 epochs, no early stopping,
`wd=0`, `lr=2e-4`, `max_grad_norm=1.0`). Sources: Unsloth LoRA guide; HF rsLoRA blog; rsLoRA
paper (arXiv:2312.03732); QuAILoRA (arXiv:2410.14713).

**Code changes (additive, existing behavior preserved):**
- `src/ssft/train/lora_factory.py` — pass `use_rslora`/`use_dora` (default False) to `LoraConfig`.
- `src/ssft/train/model_loader.py` — 8-bit `BitsAndBytesConfig` branch; and **bug fix**: do not
  call `prepare_model_for_kbit_training` on non-quantized (bf16/fp16) models (it upcast the whole
  14B base to fp32 → ~56 GB), just enable gradient checkpointing directly.

**New configs:**
- Methods: `qlora_lora_r64_rslora_memorize.yaml` (4-bit), `lora_bf16_r64_rslora_memorize.yaml`
  (bf16, no quant floor), `lora8bit_r64_rslora_memorize.yaml` (8-bit),
  `qlora_lora_r16_d0_memorize.yaml` (r16 control).
- Training: `kb_memorize_aggressive.yaml` (50 epochs, lr 2e-4, no early stopping, wd 0).

**Runs (all 650 steps = 50 epochs on the 205 rows), one per GPU:**
| Config | Quant | Rank | Trainable |
|--------|-------|------|-----------|
| bf16 r64 + rsLoRA  | none (bf16) | 64 | 275 M |
| 4-bit r64 + rsLoRA | 4-bit NF4   | 64 | 275 M |
| 8-bit r64 + rsLoRA | 8-bit       | 64 | 275 M |
| 4-bit r16 control  | 4-bit NF4   | 16 | 68.8 M |

**Environment note:** hardware is 4× RTX A6000 (~51 GB each). The `nvidia-smi` failure is an
NVML-only mismatch; CUDA compute works. Runs use a fresh `.venv` and
`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` (avoids the NVML path in the caching allocator).

**Verification:** per run `_SUCCESS` + `metrics.json["status"]=="completed"`; tabulate final
`train_loss`/`train_perplexity`, note which cross < 0.1 and at what epoch; flag instability
(loss→0 in a few steps + grad-norm spike) from `train_log.jsonl`.

---

## Step A — Evaluate the KB-only adapters ("test once done")

For each completed run dir:
```
python -m ssft.cli compare-base-adapter --run-dir <run_dir>
```
Runs base-vs-adapter perplexity (train/val/test) + the 6 KB cloze probes + a keep/discard
verdict written to `report.md`. Produce a side-by-side table of the four adapters.

---

## Phase 2 — KB + web-wiki mixed (DEFERRED)

**Decisions locked with the user:** corpus is **a file the user will provide a path to**
(format auto-detected); mixing is **KB-dominant** so the 205 companies stay a strong signal and
the comparison to KB-only stays meaningful.

### Step B — Materialize the web corpus (needs the user's file)
Write `scripts/convert_web_corpus.py` (new) to convert whatever the user provides into the
schema `data/web_converter.py` already expects — one JSONL line per page:
```json
{"source_url": "<page url or stable id>", "text": "<page body>"}
```
Handles: a JSONL/CSV (map url/title→`source_url`, body/content/markdown→`text`) or a directory
of `.md` pages (filename/frontmatter→`source_url`, file text→`text`). Drop empty pages. Output:
`self_supervised_finetuning/data/raw/web_corpus.jsonl`. Sanity-check row count (~9000),
non-empty text, unique `source_url` count.

### Step C — KB-dominant mixed configs
The framework already mixes KB+web leak-free via `data/kb_web_mixed.yaml` +
`data/dataset_mixer.py::mix_split` (weights = each source's share of mixed-**train** examples;
val/test stay unweighted).
- `configs/data/kb_web_mixed_kbdominant.yaml` — copy of `kb_web_mixed.yaml` with
  `sources.kb.input_path: <repo>/kb_full.jsonl`, `sources.web.input_path:
  .../data/raw/web_corpus.jsonl`, `sampling_weights: {kb: 0.7, web: 0.3}`. With ~164 KB train
  examples vs thousands of packed web sequences this takes all 205 KB rows and oversamples them
  ~10–13× while web contributes ~30% — "205 strong + web a bit."
- `configs/training/kb_web_mixed_kbdominant.yaml` — from `mixed_default.yaml` but 5 epochs,
  lr 1e-4, cosine, warmup 0.03, wd 0, no early stopping, `eval_strategy: epoch`.
- **Method:** reuse the *winning* KB-only recipe unchanged so only the data differs.

### Step D — Run KB+web and compare
```
python -m ssft.cli train \
  --model-config    configs/models/qwen2p5_14b_base.yaml \
  --method-config   configs/methods/<winning_r64>.yaml \
  --data-config     configs/data/kb_web_mixed_kbdominant.yaml \
  --training-config configs/training/kb_web_mixed_kbdominant.yaml
```
Then `compare-base-adapter --run-dir <mixed_run_dir>`. Compare **KB-only vs KB+web** on
**KB cloze accuracy** (seen + held-out company) and KB-split perplexity — raw mixed train-loss
is not comparable to KB-only train-loss (different corpus). Caveat: KB-only *memorization*
(train_all) and KB+web *held-out split* are different setups.

### What the user needs to provide
Just the ~9000 wiki pages. Per page, the minimum is:
- the **text body**, and
- a **stable id per page** — ideally the source **URL** (else a title/entity/filename), used as
  `source_url` for leak-free splitting.

Optional-only: a company mapping per page (enables company-aware splitting later). Everything
else (KB, model, venv, mixing pipeline) is ready.

---

## Phase 3 — Qwen3.6-35B-A3B-Base (MoE) as a second base model (QUEUED)

Run the KB memorization experiment (and, after it, the KB+web mixed run from Phase 2) on
**Qwen3.6-35B-A3B-Base** — a 35B-total / 3B-active MoE. Sequenced after the 14B runs finish and
are evaluated.

**Constraints:**
- **Disk:** 35B bf16 ≈ 70 GB vs ~61 GB free → delete the 28 GB Qwen2.5-14B HF cache first (only
  after all 14B evals are done; re-downloadable). Verify headroom before the pull.
- **Precision:** 35B bf16 (~70 GB) won't fit one 51 GB A6000 → **4-bit/8-bit QLoRA only**. Use
  the bf16 `-Base` weights, not NVFP4/GGUF prebuilt quant repos.
- **Version:** verify `Qwen/Qwen3.6-35B-A3B-Base` exists at download; if 3.6 ships only
  instruct/quantized, confirm fallback (3.5-35B-A3B-Base) with the user first.
- **MoE target modules:** introspect module names post-download; attention q/k/v/o_proj match,
  MLP is per-expert gate/up/down_proj (router `gate` excluded). Add a MoE method config if the
  target modules differ from the dense configs; `validate_target_modules` guards mismatches.
  Possible transformers bump for a 3.6-specific arch.

**Steps:** free 14B cache → download 35B-A3B-Base → `configs/models/qwen35b_a3b_base.yaml` +
MoE method config → KB-only memorization (reuse `kb_only_memorization.yaml` +
`kb_memorize_aggressive.yaml`) → `compare-base-adapter` → then Phase 2 Steps B–D with
`--model-config .../qwen35b_a3b_base.yaml`.
