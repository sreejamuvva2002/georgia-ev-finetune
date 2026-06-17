# Georgia EV — v4 Session Log & Handoff

Continuation of `SESSION_LOG.md` (7B v1/v2) and `SESSION_LOG_14B.md` (the first 14B = "v3").
This file is the **v4** record: a no-compromise retrain of the 14B for maximal accuracy + full
company coverage, the data/eval rewrite behind it, the results, and the live demo. Read this +
the two prior logs to fully resume after a `/clear`.

- **Project root:** `/Users/surya/Desktop/Desktop - Unknown/projects/georgia-ev-finetune`
  (note: the project MOVED here from `/Users/surya/Desktop/projects/...`; old hardcoded paths in
  `lora_14b.yaml` were stale and are now fixed.)
- **Machine:** Apple M4 Pro, 48 GB RAM, macOS. No NVIDIA GPU. Trained on MLX (Metal).
- **Dates:** 2026-06-15 → 06-16.

---

## TL;DR / current state
- **Trained v4** = 14B (`Qwen2.5-14B-Instruct-MLX-8bit` + LoRA via mlx-lm), rank 24 / scale 10,
  3300 iters (~2.5 epochs). Adapter: `training_project/adapters/georgia_ev_14b_mlx_v4/`
  (6 checkpoints 550…3300 + final `adapters.safetensors`). Final val loss **0.039** (best
  checkpoint by val loss = iter-2200 @ 0.031); train loss ~0.003; peak mem 29 GB.
- **v4 is the best model yet:** 50 human Q&A **54%** (27/50, = v2, > v3's 46%); **company recall
  97.5%** on the held-out probe benchmark (the "no missing companies" metric — the Rivian-style
  drops are GONE); **refusals 100%** (v3's regression fixed).
- **Probe benchmark (primary metric): 85.1%** (171/201). Strong on names (90%), refusal (100%),
  superlative (94%); weak on **count (58%, off-by-one)** and **none-match (0/3, under-trained)**.
- **A public demo of v4 is RUNNING:** `https://eb66e792004358ce31.gradio.live` (login
  `guest` / `evtest123`). `caffeinate` keeps the Mac awake; valid ~72 h. Stop: `pkill -f serve_gradio_mlx`.
- **OPEN DECISION (pending user):** freeze v4 as-is, OR do one **v5 data pass + retrain** (~6–8 h)
  to fix none-match + the OEM-footprint naming bug + ground the counts. NOT yet frozen via save_version.

---

## 1. The request + user decisions (confirmed 2026-06-15)
Goal: implement "v4" = no-compromise on accuracy AND company coverage, able to answer any KB
question, then **freeze the 14B**. Decisions made via AskUserQuestion:
- **Train on this Mac (MLX)** — not cloud HF. Frozen artifact = MLX adapter.
- **Offline teacher paraphrase bank** — author rich phrasings in the generator; **no Anthropic API**.
- **Pure fine-tuning only** — no retrieval (the inference-time KB-compute layer remains deferred).
- Hard rule preserved: **never train on the 50 human Q&A** (held-out test only).

---

## 2. Critical findings — three places the prior session logs were WRONG
Verified against the raw KB during the build (see memory `v4-corrections-to-session-logs`):

1. **Count convention was documented backwards.** `SESSION_LOG_14B.md §3` says the 50 gold counts
   *rows* and v3's bug was *unique companies*. The raw gold proves the **opposite** — the gold counts
   **unique companies** (Q4 lists 4 companies counting ZF, which has 3 KB rows, **once**; Q3 thermal=5
   unique). → v4 standardizes on **UNIQUE COMPANIES everywhere**.
2. **The real under-listing cause = role normalization, not counting.** `EV Supply Chain Role` has
   8 clean roles + **26 free-text one-offs**; the gold groups them **semantically**. v3 filtered on
   exact role strings (`RoleBucket = role if role in MAIN_ROLES else None` → dropped all 26 one-offs)
   → that is why it under-listed (Rivian 2/6, etc.). v4 adds a substring `RoleNorm` (`role_tags`) that
   reproduces the gold (thermal→5, power-electronics/charging→4, harness→2) + an `oem_tags` splitter
   so **Rivian = all 6 companies** (`Hyundai Kia Rivian`×5 + `Rivian Automotive`×1).
3. **The 50-question benchmark is internally inconsistent → cannot be the target.** Q3 vs Q42 ask the
   same thing, gold says **5 vs 4**; Q41 gold=6 where KB has 11. So 100% on the 50 is impossible. The
   v4 primary metric is a held-out **probe benchmark**; the 50 are a secondary reference.

Also discovered: KB has **2 spelling-duplicate company pairs** (`Jefferson Southern Corp.`/`Corporation`;
`Trenton Pressing`/`Trenton Pressing Inc.` — same site) → 193 names = **191 real companies**. `load_kb`
now canonicalizes them (longest spelling wins).

---

## 3. The canonical conventions (defined once in `build_dataset.py`, reused in probes + scorer)
- **Counting = unique companies** (`uniq = drop_duplicates("Company")`).
- **`role_tags(role)`** — substring buckets, multi-tag: `Battery Cell/Pack`, `Thermal Management`
  (any "thermal"), `Power Electronics` (any "power electronics"), `Charging Infrastructure`,
  `Wiring Harness` (any "harness"), `Vehicle Assembly`, `OEM Footprint` (any "footprint"),
  `Materials`, `General Automotive`. The 26 free-text one-offs that match no keyword get NO bucket
  (covered by per-company facts + an exact-role generator).
- **`oem_tags(primary_oems)`** — split "Hyundai Kia Rivian" → {Hyundai, Kia, Rivian}; "Multiple
  OEMs"/missing → empty.
- **`CategoryNorm`** merges `OEM (Footprint)`→`OEM Footprint`. **`IndustryNorm`** strips leading spaces.
- **Employment = raw KB value as-is** (the gold uses the global-vs-site mix, e.g. WIKA 250k; don't "fix").
- **List-then-count answers:** headline count + fully enumerated unique list, count == len(list) by
  construction. Mega-lists (>50, e.g. General Automotive=135) capped at 50 with an explicit
  "(listing the first 50 of N)" so truncation is never taught as completeness.

---

## 4. What was built / changed (all in `training_project/scripts/`)
- **`build_dataset.py`** — REWRITTEN. Canonical layer (`role_tags`/`oem_tags`/`CategoryNorm`/name
  canonicalization/`uniq`), robust null-cleaning (`_clean` — pandas 3.0 `.replace` was leaving float
  `nan` that rendered as literal "nan"). Generators: `company_examples` (kept), `list_examples`
  (exhaustive 1-way: category, role bucket, exact one-off roles, county, city, industry, relevance,
  OEM, facility), `cross_examples` (2-way + key 3-way), `agg_examples` (counts/dists/sums/argmax/topk/
  single-source/gap/footprint/affiliation), `keyword_examples` (product search), `refusal_examples`
  (33 diverse refusals + none-match). Offline paraphrase bank `vary(P_*, k, x=…)`. Balanced
  oversample (refusal/none ×8, list/cross/agg/kw ×2). Decontaminates vs the 50 **and** the probes.
  Output: **train 5,289 / valid 229 / test 50**; refusals 270 (5.1%); leakage-dropped 39.
- **`build_probe_benchmark.py`** — NEW. **201 held-out probes** (123 names / 24 count / 20 aggregate /
  16 superlative / 15 refusal / 3 none), KB-computed gold under the canonical convention, RESERVED
  phrasings (held out from training). Precise company `detect()` (full-name, longest-first masking;
  strips "Primary OEMs:" clauses so OEM values aren't miscounted). **Scorer ceiling verified 100%** on
  perfect answers (so any real shortfall = genuine model error). `score(answers.json)` reports pass
  rate per kind + **company recall / missing-company rate** (the coverage gate).
- **`validate_dataset.py`** — gates: 0 leakage (vs 50 + 201), count==listed, refusal fraction,
  KB-consistency (role/OEM listed counts == KB unique), seq-length safety. **PASSES.**
- **`lora_14b.yaml`** — stale paths fixed; rank 24, scale 10 (kept stable — NOT 12; divergence
  history), max_seq 1536, iters 3300, cosine 5e-5→5e-6 warmup 80, save_every 550, adapter →
  `georgia_ev_14b_mlx_v4`.
- **`train_mlx.py`** — adds `--stability` (90-iter probe); always passes `--data`/`--adapter-path`.
- **`evaluate_mlx.py`** — rewired to the v4 adapter; scores the 50 (ev.score_answer + FAIL dump for
  attribution) and the 201 probes (build_probe_benchmark.score). Retired the old 36-probe diagnose_v2 set.
- **`serve_gradio_mlx.py`** — `--adapter` arg (default v4), max_tokens 900, and `prevent_thread_lock=True`
  + explicit `SHARE_URL:` print (gradio's own URL print gets stuck in the pipe buffer / the process
  was exiting before binding).

---

## 5. Training (MLX, this Mac)
- **Smoke** (8 iters): clean, val 2.90→1.80, peak mem 21.6 GB, 103M trainable (0.699%).
- **Stability probe** (90 iters): clean descent val 2.90→**0.69**, NO divergence past iter 75 (where
  v3's first config blew up to 6.5); peak mem 28.6 GB. Config committed.
- **Full run**: 3300 iters, ran faster than the worst-case estimate (Mac stayed cool). Val loss
  trajectory 2.90 → 0.231(275) → 0.072(1100) → 0.031(2200) → **0.027(3025)** → 0.039(3300). **Not
  overfitting** (val never rose much) despite train loss ~0.003 — ~6× better val than v3's 0.158.
  6 checkpoints saved.

---

## 6. v4 results
| Metric | v4 |
|---|---|
| 50 human Q&A (structured score ≥0.6) | **54%** (27/50), mean 0.598 |
| Probe benchmark overall | **85.1%** (171/201) |
| · names | 90% (111/123) |
| · count | **58%** (14/24) ← off-by-one |
| · none-match | **0/3** ← under-trained |
| · aggregate | 80% (16/20) |
| · refusal | **100%** (15/15) |
| · superlative | 94% (15/16) |
| **Company recall (coverage gate)** | **0.975** (missing-company rate 2.5%) |
| precision | 0.954 |

**Model comparison (50 held-out human Q&A):** Base 4% → v1 26% → **v2 54%** → v3(14B) 46% →
**v4(14B) 54%**. v4 ties v2 on the human eval but adds 97.5% coverage recall + fixed refusals.

**Scorer note:** the first probe score read 80.1%; +5 pts were a scorer artifact — the model
correctly listed companies WITH their OEMs, and the OEM values (which are also KB company names) were
miscounted as hallucinations. Fixed by stripping "Primary OEMs:" clauses before detection → 85.1%.

**Failure diagnosis (from `eval_results/probe_benchmark_scores.json`):**
- names fails: mostly cross-products (2-way filters) — 5 miss ≥1 company (recall<1), ~12 over-list.
- count fails: mostly **off-by-one** (Tier 1: model 70 vs 71) + OEM-footprint **naming collision**
  (a role bucket AND a category are both named "OEM Footprint" → exp 5 vs 8). Single-source roles
  badly off (27 vs 37).
- none fails: model ignores the 2nd filter and lists the unfiltered set (e.g. "Vehicle Assembly in
  Chatham County" → lists all 10) — only 9 none-match training examples.

What's **fixable** (→ v5): none-match (expand), OEM-footprint role/category naming, list-then-count
grounding for counts. What's **structural** (pure-FT ceiling): exact off-by-one counts on big sets &
exact sums — the only true fix is the deferred inference-time KB-compute layer.

---

## 7. Live demo
`scripts/serve_gradio_mlx.py --share --password evtest123` (running under `caffeinate -i`):
- **URL:** `https://eb66e792004358ce31.gradio.live` · **login** `guest` / `evtest123`
- Serves v4 final checkpoint. Single-turn, streams, ~10–40 s/answer, max 900 tokens.
- Mac must stay awake + process alive. Stop: `pkill -f serve_gradio_mlx`. Log: `logs/serve_v4.log`.

---

## 8. Open decision / next steps
1. **Freeze v4** (if satisfied): `save_version.py v4_14b --adapter
   training_project/adapters/georgia_ev_14b_mlx_v4 --base-model
   "/Users/surya/.lmstudio/models/lmstudio-community/Qwen2.5-14B-Instruct-MLX-8bit"
   --notes "14B MLX LoRA v4, 54% human eval, 97.5% probe recall, refusals fixed"`. — NOT yet done.
2. **OR v5 data pass + retrain** (~6–8 h): (a) expand none-match for empty cross-filters; (b) rename
   the "OEM Footprint" role bucket to remove the category collision; (c) list-then-count grounding for
   count questions (≤50). Expected probes ~85%→~90%+; won't fully clear off-by-one (pure-FT ceiling).
3. Optional cheap check: evaluate the **iter-2200 checkpoint** (lower val loss) on the probes before
   deciding — unlikely to fix none-match/counting.

---

## 9. File map (v4 additions/changes)
```
training_project/scripts/
  build_dataset.py          # REWRITTEN (canonical conventions + exhaustive coverage)
  build_probe_benchmark.py  # NEW — 201 held-out probes + precise scorer (primary metric)
  validate_dataset.py       # extended gates
  lora_14b.yaml             # v4 config, stale paths fixed
  train_mlx.py              # + --stability
  evaluate_mlx.py           # rewired to v4 + probe benchmark
  serve_gradio_mlx.py       # --adapter, explicit SHARE_URL print
training_project/data/
  train.jsonl / valid.jsonl / test.jsonl   # v4 (5289/229/50)
  probe_benchmark.jsonl                     # NEW — 201 held-out probes
training_project/adapters/
  georgia_ev_14b_mlx_v4/    # the v4 adapter (550..3300 + final)
  georgia_ev_14b_mlx/       # v3 (kept; frozen in releases/v3_14b_2026-06-14)
training_project/eval_results/
  answers_14b_v4.json, probe_answers_14b_v4.json, probe_benchmark_scores.json,
  scores_50_14b_v4.xlsx, summary_14b_v4.md
training_project/logs/  train_mlx_v4.log, eval_v4_generate.log, serve_v4.log, dataset_report.md
```

## 10. How to run (cheat sheet)
```bash
cd "/Users/surya/Desktop/Desktop - Unknown/projects/georgia-ev-finetune"; PY=.venv/bin/python
$PY training_project/scripts/build_probe_benchmark.py        # build probes (do BEFORE build_dataset)
$PY training_project/scripts/build_dataset.py                # build train/valid (decontaminates vs probes)
$PY training_project/scripts/validate_dataset.py             # integrity gates
$PY training_project/scripts/train_mlx.py --stability        # 90-iter stability probe
$PY training_project/scripts/train_mlx.py                    # full run (yaml iters 3300, ~1 day)
# eval (greedy generate, then score):
PYTHONUNBUFFERED=1 $PY training_project/scripts/evaluate_mlx.py generate
$PY training_project/scripts/evaluate_mlx.py score
# demo:
PYTHONUNBUFFERED=1 caffeinate -i $PY -u training_project/scripts/serve_gradio_mlx.py --share --password evtest123
```
Hard reminders: no other large model processes during training (v3 swap-thrash); keep both MPS
watermark vars only matter on the torch/7B path (MLX uses `clear_cache_threshold` in the yaml).
```
