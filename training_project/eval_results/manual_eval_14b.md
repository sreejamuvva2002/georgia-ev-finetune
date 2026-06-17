# 14B — manual, KB-verified evaluation (not just the auto-scorer)

Final 14B model: Qwen2.5-14B-Instruct-MLX-8bit + LoRA (rank 16/scale 10, 1.5 epochs,
train loss 0.056, held-out val loss 0.158).

## Headline (auto-scorer, 50 held-out human Q&A, correct = composite ≥ 0.6)

| model | accuracy | mean score |
|---|---|---|
| Base 7B | 4% (2/50) | 0.201 |
| Fine-tuned v2 (7B) | **54% (27/50)** | 0.584 |
| Fine-tuned 14B | **46% (23/50)** | **0.605** |

Probe set (KB-computed gold, 36 Q): v2 20/36 → 14B 21/36.

**The 14B's mean score is higher (0.605 vs 0.584) but its pass-rate is lower (46% vs 54%),
with 14 near-misses (0.4–0.6).** So it's "closer on average, crosses the line less often" —
i.e. roughly on par with v2, NOT the clear win we hoped for. Manual KB checking explains why.

## What the 14B genuinely FIXED (KB-verified gains)
- **Superlative / argmax — fully fixed.** Probe superlative 1→4 (perfect); Q7 Gwinnett highest
  employment → WIKA USA 250,000 ✓ (v2 fabricated "Toyota 18,100"; v2 even gave the *same* fake
  answer for two different counties). This was v2's worst category.
- **Q12 over-listing trap fixed:** "Tier 2/3 directly EV-relevant" → exactly the 3 correct
  companies (v2 listed 10). 0.33→1.00.
- **Several power-electronics / role filters** (Q22 DC-DC 0.30→0.93, multi_filter probes 3→5).

## What REGRESSED (KB-verified) — and the 3 root causes

**Cause 1 — count-convention bug I introduced (data inconsistency, fixable).**
My v3 generators counted **unique companies** (`drop_duplicates`), but v2's generators, the
probe golds, and the human gold use **rows**. KB: Tier 1 = 77 rows / 71 unique; total 205 rows /
193 unique. The 14B learned the *unique* counts (Tier 1→71, total→193) and gets marked wrong
against row-count gold. The model is arguably *more* correct ("how many companies" → count
companies), but the conflicting training signal also confused it. **This is a data bug, not a
model limit.**

**Cause 2 — refusal regression (over-training to "always answer").**
Exhaustive "answer from the KB" data made the 14B unwilling to say "none / not in KB":
- Probe refusals 3→1 — it now answers "solid-state batteries" and "Florida companies."
- Q20 ("Tier 1/2 engineered plastics" — gold: *none*) → 14B listed 3 companies that are actually
  **Tier 1**, not Tier 1/2. Should have refused.

**Cause 3 — over-listing on narrow multi-constraint queries still present (sometimes worse).**
The exhaustive data fixed *some* filters but the 14B still drops constraints on others:
- Q45 dual-platform (gold 5: Hyundai/Kia **AND** Rivian) → 14B listed 13 (all Hyundai/Kia,
  dropped the Rivian constraint).
- Q25 Tier 2/3 GA emp>300 (gold 8) → 14B gave 3 and even included a <300 company.
- Q29 Metaplant suppliers <200 (gold 6) → 14B gave 11 (over-included).
- Q19 copper-foil (gold 1: Duckyang) → 14B added a recycler (2).
- Large free-text aggregates undercounted: Q27 single-source roles 28→14B said 18; Q46 gap
  counties 39→14B said 7.

## Bottom line
Bigger model + exhaustive memorization data **did not reliably beat v2.** The shortfall is
mostly **three fixable issues** — (1) the unique-vs-row count bug, (2) over-trained-away refusals,
(3) persistent over-listing — NOT a ceiling of the 14B. This validates the earlier analysis:
the next gains come from **fixing the data/training (list-then-count, refusal examples, one count
convention, KB cleanup)**, not from more parameters. The 14B *did* prove it can fix the
argmax/superlative class that v2 couldn't — so the capacity helps where the data is clean and
consistent.
