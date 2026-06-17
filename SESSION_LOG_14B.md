# Georgia EV — 14B Session Log & Handoff

Continuation of `SESSION_LOG.md` (the 7B v1/v2 work). This file captures the **14B session**:
diagnosing v2, expanding the data, training a 14B on MLX, evaluating it (KB-verified), and
standing up a shareable demo. Read this + `SESSION_LOG.md` to fully resume after a `/clear`.

- **Project root:** `/Users/surya/Desktop/projects/georgia-ev-finetune`
- **Machine:** Apple M4 Pro, 48 GB RAM, ~36 GB free disk, macOS. No NVIDIA GPU.
- **Date:** 2026-06-13 → 06-14.

---

## TL;DR / current state
- **Trained a 14B model** (`Qwen2.5-14B-Instruct-MLX-8bit` + LoRA via mlx-lm). Final train loss
  **0.056**, held-out val loss **0.158**. Adapter: `training_project/adapters/georgia_ev_14b_mlx/`
  (iter-1200, `adapters.safetensors`, 275 MB; checkpoints at 300/600/900/1200).
- **Honest result: the 14B is ~on par with v2, NOT a clear win.** Auto-score on the 50 held-out
  human Q&A: **14B 46% (23/50), mean 0.605** vs **v2 54% (27/50), mean 0.584**. Probes: 14B 21/36
  vs v2 20/36. (Higher mean, lower pass-rate, 14 near-misses.)
- **Why it didn't beat v2 = 3 fixable issues I introduced, not a model ceiling** (details below).
- **A public demo of the 14B is RUNNING** (gradio share link) — see "Demo server". Remember to
  stop it / keep the Mac awake.
- **v2 7B remains the higher-pass-rate model to ship for now.**

---

## 1. Model + toolchain decision
- New model downloaded from LM Studio: **`Qwen2.5-14B-Instruct-MLX-8bit`** at
  `~/.lmstudio/models/lmstudio-community/Qwen2.5-14B-Instruct-MLX-8bit` (15 GB, genuinely MLX
  8-bit quantized: config has `"quantization": {"bits": 8, "group_size": 64}`).
- **HF/PEFT cannot train MLX-quantized weights** → must use Apple's **`mlx-lm`** (`mlx_lm.lora`).
  Installed `mlx 0.31.2` + `mlx-lm 0.31.3` into `.venv`.
- **User decisions (confirmed):** train on MLX/this-Mac (not Linux-portable HF); **fine-tune only,
  no retrieval**. Hard rule preserved: **never train on the 50 human Q&A** (held-out test only).
- The training data is model-agnostic chat JSONL, so an HF/cloud build is possible later from the
  same data.

## 2. Phase 1 — Diagnostic (NEW `scripts/diagnose_v2.py`)
- Built a **36-question probe set from the KB** (never the 50 Q&A), 9 shapes, KB-computed gold.
- **v2 (7B) = 56% (20/36).** Worst shapes: **superlative 25%, aggregation 33%, count 38%,
  multi_filter 43%, distribution 50%**. Fine: refusal/setop/single_fact = 100%.
- Core failure = **quantitative precision** (exact counts, multi-constraint filters, argmax, sums)
  → the model fabricates. Confirmed the screenshots (Tier 1∩Materials → alphabetical Tier-2/3
  fabrication; miscounts; can't sum).
- Outputs: `logs/diagnostic_v2.md`, `eval_results/diagnostic_v2.xlsx`,
  `eval_results/diagnostic_v2_answers.json`. Reusable scorer: `diagnose_v2.score_probe()`.

## 3. Phase 2 — Data expansion (`scripts/build_dataset.py` → `agg_examples_v3()`)
- Added exhaustive **Category×Role, Category×Relevance, Role×Relevance, Category×OEM** filtered
  lists (count == list by construction), distribution cross-tabs, argmax, sums — paraphrase-rich.
- `train 2535→3206`, `valid 221`, `test 50`. Verified **0 exact leakage** of the 50 questions/answers.
- **⚠️ BUG I introduced:** v3 generators count **unique companies** (`drop_duplicates("Company")`)
  while v2's generators, the probe golds, and the human gold count **rows**. KB has 205 rows /
  193 unique companies (Tier 1 = 77 rows / 71 unique). The 14B learned the *unique* counts and
  scored "wrong" against row-count gold. **This is the #1 thing to fix in v4 — pick ONE convention.**

## 4. Phase 3 — Training (NEW `scripts/train_mlx.py`, `scripts/lora_14b.yaml`)
- `mlx_lm.lora`: rank 16, **scale 10**, all 48 layers (default keys = q/k/v/o/gate/up/down),
  `mask_prompt` (completion-only loss), grad_checkpoint, max_seq_length 1024, batch 4,
  **cosine LR 5e-5 with 60-step warmup → 5e-6**, **1200 iters (~1.5 epochs)**.
- **First config DIVERGED** (lr 1e-4 + scale 20, no warmup → loss 2.95→6.5 by iter 75). Fixed by
  halving scale + lower LR + warmup; a 90-iter **stability probe** confirmed clean descent
  (3.26→0.78), then the full run.
- **Memory incident:** leftover **gradio 7B demo + chat REPL** (each ~15 GB bf16) ran alongside
  the 14B → **39 GB swap thrash**, dragging it to ~40–90 s/iter. Killed them → clean (~24 GB peak).
- **Hardware reality:** ~10 s/iter cool → **~40 s/iter thermally throttled**; full 1200-iter run
  took **~12.5 h**. This is the practical ceiling on this Mac.
- iter-300 fallback preserved at `adapters/georgia_ev_14b_mlx_iter300_fallback.safetensors`.

## 5. Phase 4 — Evaluation (NEW `scripts/evaluate_mlx.py`)
- Generated 14B greedy answers for 50 Q&A + 36 probes; auto-scored + **manually KB-verified**.
- **GAINS (KB-verified):** superlative/argmax **fully fixed** (probes 1→4; Q7 "Gwinnett highest
  employment" → WIKA USA 250,000, v2 fabricated); Q12 over-listing fixed (3 not 10); some
  power-electronics filters.
- **REGRESSIONS (3 root causes):**
  1. **Count-convention bug** (unique vs rows) — see Phase 2. Data bug, fixable.
  2. **Refusal regression** — over-trained on "answer from KB" → now answers "solid-state
     batteries" / "Florida" / Q20 (gold = none). Fix: keep refusal examples, don't over-oversample.
  3. **Over-listing on narrow filters still present** (Q45 dropped Rivian constraint → 13 vs 5;
     Q25/Q29 over-included; Q46 gap counties 7 vs 39 undercounted). Fix: **list-then-count** training.
- Outputs: `eval_results/summary_14b.md`, **`eval_results/manual_eval_14b.md`** (the KB-verified
  verdict), `answers_14b.json`, `diagnostic_14b_answers.json`, `scores_14b_vs_v2.xlsx`,
  `diagnostic_14b.xlsx`.
- **Bottom line:** more parameters + more memorization data did NOT reliably beat v2; the shortfall
  is fixable data/training, not capacity. The 14B DID solve the argmax class v2 couldn't.

## 6. Demo server (NEW `scripts/serve_gradio_mlx.py`)
- MLX gradio web server: streams tokens, `--share`, `--password`. Loads the 14B + adapter.
- **Must run unbuffered** (`PYTHONUNBUFFERED=1 python -u …`) or the gradio URL stays stuck in the
  pipe buffer.
- **Currently RUNNING.** Public link: `https://b46c6a069ff213d94b.gradio.live`
  (login `guest` / `evtest123`). Valid ~72 h; Mac must stay awake; process must keep running.
  To stop: `pkill -f serve_gradio_mlx`.
- The 7B server (`scripts/serve_gradio.py`, `--model v2/v1/base`) still works for v2.

## 7. Key conceptual decisions discussed (so we don't re-derive)
- **5 accuracy levers (ranked):** (1) **list-then-count** outputs *[biggest, untapped]*,
  (2) exhaustive enumeration *[partly done]*, (3) heavy paraphrasing *[partly]*, (4) **KB cleanup**
  *[untapped]*, (5) more capacity/epochs *[done]*.
- **What pure-FT optimization sacrifices:** freshness (breaks on any KB change → full retrain),
  robustness to novel questions, auditability (silent plausible errors), general assistant
  flexibility (overfit risk), iteration agility.
- **In-context KB updates:** help freshness for simple lookups but (a) conflict with memorized
  weights and (b) don't fix the computation weakness (it still miscounts in-context rows). *Promised
  but NOT yet run: an empirical test feeding a modified value in the prompt to see if the 14B uses
  it vs reverts to memory.*
- **"DeepSeek-with-Opus" = distillation.** Use a strong teacher (Opus) for question variety +
  reasoning format (list-then-count), but **facts MUST come from deterministic KB computation** (the
  teacher would hallucinate them).
- **Hardware/time:** no-compromise pipeline ≈ **~1 week on this Mac** vs **~1–3 h on a cloud A100**
  (RTX 4090 ~3–6 h, H100 ~30–60 min). 120 GB RAM barely affects training speed — the **GPU** is the
  lever. `train_qlora.py` already auto-switches to 4-bit QLoRA on CUDA.
- **KB normalization issues (not fixed):** `OEM (Footprint)` vs `OEM Footprint` dup labels;
  `EV Supply Chain Role` mixes ~6 real categories with **28 free-text one-offs**; facility-type
  case dupes (handled via `FacilityNorm`); Industry leading-space dupes (handled via `IndustryNorm`);
  **Employment mixes global corporate vs Georgia-site headcounts** (WIKA 250k, Yazaki 230k);
  205 rows / 193 companies (12 multi-row companies).

## 8. Open items / next steps
1. **v4 corrected data pass (highest value):** (a) pick ONE count convention (rows OR unique) and
   use it everywhere; (b) restore refusal examples + stop over-oversampling; (c) add **list-then-count**
   answer format; (d) optional KB cleanup. Then retrain — **ideally on a cloud GPU** (~1 h vs ~12 h).
2. **Run the in-context KB-update test** (promised).
3. **Snapshot the 14B** via `scripts/save_version.py` (already extended for the MLX adapter):
   `…/save_version.py v3_14b --adapter training_project/adapters/georgia_ev_14b_mlx
   --base-model "/Users/surya/.lmstudio/models/lmstudio-community/Qwen2.5-14B-Instruct-MLX-8bit"
   --notes "14B MLX LoRA, 1.5 epochs, ~on par with v2"` — NOT yet done.
4. Optional: reclaim ~1.9 GB by deleting `adapters/georgia_ev_lora*/checkpoint-*` (was blocked
   earlier — needs explicit OK).

## 9. New/changed files this session
```
training_project/scripts/
  diagnose_v2.py        # NEW — KB probe set + scorer (score_probe/score_to_df)
  build_dataset.py      # EDITED — added agg_examples_v3() + v3 oversampling (⚠ unique-count bug)
  train_mlx.py          # NEW — mlx_lm.lora wrapper (smoke/full/resume)
  lora_14b.yaml         # NEW — mlx LoRA config (rank16/scale10/cosine warmup)
  evaluate_mlx.py       # NEW — generate+score 14B on 50 Q&A + 36 probes
  serve_gradio_mlx.py   # NEW — MLX streaming web demo (--share/--password)
  save_version.py       # EDITED — now snapshots MLX adapters too (--base-model arg)
training_project/adapters/
  georgia_ev_14b_mlx/                         # the 14B adapter (iter-1200 final + ckpts)
  georgia_ev_14b_mlx_iter300_fallback.safetensors
training_project/eval_results/
  diagnostic_v2.{md→logs, xlsx}, diagnostic_v2_answers.json
  summary_14b.md, manual_eval_14b.md, answers_14b.json, diagnostic_14b_answers.json,
  scores_14b_vs_v2.xlsx, diagnostic_14b.xlsx
training_project/logs/  diagnostic_v2.md, train_mlx_*.log, eval_14b_generate.log, serve_14b.log
```

## 10. How to run (cheat sheet)
```bash
cd /Users/surya/Desktop/projects/georgia-ev-finetune
PY=.venv/bin/python

# rebuild data / validate
$PY training_project/scripts/build_dataset.py
$PY training_project/scripts/validate_dataset.py

# diagnose v2 (7B)  — needs HF 7B + adapters/georgia_ev_lora
$PY training_project/scripts/diagnose_v2.py all

# train the 14B (mlx) — smoke first, then full (~12h on this Mac)
$PY training_project/scripts/train_mlx.py --smoke
.venv/bin/python -m mlx_lm lora -c training_project/scripts/lora_14b.yaml   # full run

# evaluate the 14B
$PY training_project/scripts/evaluate_mlx.py generate
$PY training_project/scripts/evaluate_mlx.py score

# serve the 14B demo (unbuffered is required for the URL to print!)
PYTHONUNBUFFERED=1 .venv/bin/python -u training_project/scripts/serve_gradio_mlx.py --share --password evtest123
```
```
