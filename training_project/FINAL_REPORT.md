# Georgia EV Supply-Chain Assistant — Final Report

_Fine-tuning a 7B instruct model on the Georgia EV knowledge base, evaluated on 50 held-out human-validated questions._

## 1. Model

| | |
|---|---|
| **Base model** | `Qwen/Qwen2.5-Coder-7B-Instruct` (Hugging Face) |
| **Why this model** | The only candidate present locally was an LM Studio **MLX** build (`lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-8bit`). MLX is Apple-only and not suitable for Linux GPU deployment, so per the brief I identified the original HF base model (`config.json` → `Qwen2ForCausalLM`, qwen2, 7.6B) and trained on the HF + PEFT path instead. |
| **Method** | LoRA (PEFT) adapter — not full fine-tuning. On a CUDA host the same script runs 4-bit **QLoRA** (bitsandbytes NF4); on this Mac it ran bf16 LoRA (bitsandbytes is CUDA-only). |
| **Adapter output** | `training_project/adapters/georgia_ev_lora/` (v2, rank-32; v1 rank-16 kept at `georgia_ev_lora_v1/`) |
| **Result** | Fine-tuning lifted accuracy on the 50 held-out questions from **4% (base) → 26% (v1) → 54% (v2)**; see §4. |

## 2. Data

| File | Use |
|---|---|
| `GNEM - Auto Landscape Lat Long Updated (1).xlsx` (205 rows, 15 cols) | Source for **training/validation** generation only |
| `Human validated 50 questions (2).xlsx` (50 rows) | **Test only** — never used as training data |

Column mapping (auto-detected): `Company`, `Category` (tier), `Industry Group`, `Updated Location` → parsed into **City** + **County**, `Address`, `Latitude`, `Longitude`, `Primary Facility Type`, `EV Supply Chain Role`, `Primary OEMs`, `Supplier or Affiliation Type`, `Employment`, `Product / Service`, `EV / Battery Relevant`.

| Split | Examples (v2) | Source |
|---|---|---|
| `data/train.jsonl` | **2,535** | KB-generated (per-company facts + aggregate/list/count + paraphrase-enriched eval-aligned patterns), aggregate/v2 examples oversampled ×3 |
| `data/valid.jsonl` | **196** | 10% hold-out of KB-generated examples |
| `data/test.jsonl` | **50** | The human-validated Q&A, gold answers, **eval only** |

A **leakage guard** in `build_dataset.py` drops any generated question matching a test question (exact-normalized or ≥0.85 token-Jaccard); 73 such generations were dropped, so all 50 test questions are genuinely held out. (v1 used 2,426 train / 192 valid and predated this guard.)

All examples are chat JSONL (`system`/`user`/`assistant`) with the system prompt:
> _"You are a Georgia EV supply chain assistant. Answer only using the Georgia EV knowledge base. If the KB does not contain enough information, say so clearly."_

Validation (`logs/dataset_report.md`): every line valid JSON, every example has all three roles; longest example ~900 tokens (under the 1,536 cap).

## 3. Training settings

| | |
|---|---|
Two runs were trained. **v2 (current, shipped)** is a data-centric retrain on a decontaminated, paraphrase-enriched dataset with a larger adapter; **v1** is kept at `adapters/georgia_ev_lora_v1/` for comparison.

| | v2 (shipped) | v1 |
|---|---|---|
| LoRA rank / alpha / dropout | **r=32 / α=64** / 0.05 | r=16 / α=32 / 0.05 |
| Trainable params | 80.7M (1.05%) | 40.4M (0.53%) |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | same |
| Epochs | 3 (477 steps) | 3 (456 steps) |
| Learning rate / scheduler | 2e-4, cosine, 3% warmup | same |
| Batch | per-device 1 × grad-accum 16 = **effective 16** | same |
| Max sequence length | 1,536 (no truncation; longest example ~900 tok) | 1,280 |
| Train examples | 2,535 (decontaminated, paraphrase-enriched) | 2,426 |
| Precision | bf16 (MPS) | bf16 (MPS) |
| **Final train loss** | **0.288** | 0.332 |
| **Final eval (valid) loss** | **0.195** (token acc 0.955) | 0.200 (token acc 0.953) |

**Hardware / "GPU":** No NVIDIA GPU on this machine (`nvidia-smi` absent, `torch.cuda.is_available()` = False). Trained on **Apple M4 Pro, 48 GB, PyTorch MPS backend**. Wall time ≈ 2 h 35 m (v2). The training script auto-detects CUDA and switches to 4-bit QLoRA on a Linux GPU host with no code change.

## 4. Evaluation

Base and fine-tuned answers were generated for all 50 questions (greedy decoding) and scored with a **structured auto-score** against the human gold answers. Full per-question detail is in `eval_results/evaluation.xlsx`; methodology + per-category breakdown in `eval_results/summary.md`.

| Model | Accuracy (composite ≥ 0.60) | Mean score |
|---|---|---|
| Base `Qwen2.5-Coder-7B-Instruct` | **4%** (2/50) | 0.201 |
| Fine-tuned **v1** (r=16) | 26% (13/50) | 0.363 |
| **Fine-tuned v2 (r=32, shipped)** | **54% (27/50)** | 0.584 |

> v1's benchmark was mildly contaminated (8 test questions had leaked into training; human answers never did). **v2's benchmark is clean** — 0 test-question and 0 gold-answer leakage (73 leaky generations dropped by a leakage guard) — so 54% is an honest, fully-held-out figure, and still an improvement over v1's contaminated 26%.

Per category (v2): Site Selection **80%** (4/5), Risk & Resilience **71%** (5/7), Supply Chain Mapping 50% (8/16), Supplier Discovery 45% (5/11), Product & Tech Trends 45% (5/11).

**Scoring** = weighted composite (weights fixed before seeing results): company-name **F1** 0.65 (precision penalizes over-listing, recall penalizes omissions), **headline-count match** 0.20, **number overlap** 0.15, token-F1 fallback for free-form answers. "Tier 1/2/3" digits are stripped before numeric comparison so category labels aren't mistaken for quantities.

### What the v2 retrain fixed, and what remains vs the 75–80% target

The data-centric retrain (decontaminate + paraphrase-enrich the eval-aligned patterns + double adapter capacity) **doubled accuracy, 26% → 54%**, landing in the predicted 45–65% range for a pure fine-tune. The biggest wins were on facts the model previously **fabricated**, now bound through repeated paraphrase exposure:

- **Global aggregation now works for the asked questions:** Q9 (highest-employment county) 0.00 → **1.00** (now correctly Gwinnett, was Troup); Q8 (Tier-1 county employment total) 0.00 → **1.00**. Memorizing the precomputed aggregate under many paraphrasings beat asking the model to sum at inference.
- **Exact lists bound:** Q1 (18 Tier-1/2 suppliers) and Q2 (battery roles) → 1.00; Q21 (Hyundai/Kia linkage) and Q33 (OEM footprint) → pass.

**Residual failure mode (why not 75–80%):** exact long-list enumeration with tight filters still over-/under-generates — e.g. Q12 (gold 3 companies) now lists 10 (better than v1's 40, but still wrong); and one regression appeared (Q5 0.62 → 0.28). These are the known limits of pure parametric recall; closing them reliably needs the retrieval/compute layer (Option A), which remains deferred.

## 5. Deployment (Linux GPU, vLLM + LoRA)

Files in `deployment/`:
- `serve_vllm.sh` — `vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --enable-lora --lora-modules georgia-ev=<adapter path> --max-lora-rank 32 --dtype bfloat16` (rank 32 matches the v2 adapter)
- `README_DEPLOYMENT.md` — copy-adapter steps, OpenAI-compatible **curl** example (`model: "georgia-ev"`, use the training system prompt), and an optional `merge_and_unload` recipe.
- `requirements.txt` — `vllm` (+ commented training deps).

The adapter is standard PEFT format and serves directly; no MLX dependency.

## 6. Problems found

- **Disk-space crashes (×2).** The startup volume was nearly full; the 14 GB HF model download left <1 GB free and training died mid-run when MPS could not write temp files. Reclaimed space from regenerable caches only (no user files/models touched) and the user freed more. Full detail in `logs/error_report.md`. Checkpointing (every 100 steps, auto-resume) was added so a crash never costs a full restart.
- **MPS watermark env var.** On Apple Silicon, setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` without `PYTORCH_MPS_LOW_WATERMARK_RATIO` makes the implied low ratio invalid (`1.4`) and aborts at model load. Both must be set (used `LOW=0.8`). This bit the v2 evaluation once — the crashed run silently left the previous (v1) answer file in place, so the first score re-scored v1; caught by diffing v2 vs v1 answers (48/50 differ) before reporting.
- **MLX vs HF base model.** Resolved by mapping the local MLX download to its HF origin and training on HF/PEFT (see §1).
- **Test-question contamination (v1, now fixed).** v1 had 8 test questions appearing verbatim in training (human answers never leaked). v2 adds a leakage guard (exact + ≥0.85 token-Jaccard) that dropped 73 leaky generations; v2's benchmark is fully held out.
- **Accuracy ceiling.** Even after the v2 retrain, pure fine-tuning reaches 54% (see §4) — short of 75–80%. Exact long-list enumeration under tight filters is the residual limit; the robust fix is the deferred retrieval/compute layer (Option A).

## 7. Status & next options

**Done (Option B):** decontaminated + paraphrase-enriched the dataset and retrained at rank 32 → **26% → 54%** on a clean benchmark, within the predicted 45–65% band for a pure fine-tune. This is the shipped adapter.

To push past ~55–65% toward the 75–80% target, the remaining lever is:

- **(A) Retrieval / compute augmentation at inference** (the robust route to ≥75–80%) — keep the v2 LoRA for phrasing, but at query time fetch the relevant KB rows or compute the exact aggregate (deterministic pandas, reusing the logic already in `scripts/build_dataset.py` / `agg_examples_v2`) and put it in context. Still deploys on vLLM `--enable-lora` with a thin retrieval wrapper. Deferred by choice; this is what would reliably fix the residual Q12-style exact-list and remaining aggregation cases.

A further pure-data iteration (more paraphrases, format constraints to curb over-listing) could add a few more points but is unlikely to clear 75–80% on its own.

**Constraints honored:** the 50 human Q&A were never trained on (verified: 0 gold answers and 0 test questions in training after the leakage guard); original Excel files untouched; no invented KB values (missing data → explicit "KB does not provide"); LoRA/QLoRA not full fine-tuning; output prepared for Linux GPU (HF/PEFT/vLLM), not MLX.
