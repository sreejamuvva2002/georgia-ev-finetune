# Self-Supervised Fine-Tuning (`ssft`)

An **experiment framework** for self-supervised continued pretraining / domain-adaptive
fine-tuning of causal language models (Qwen2.5 family by default) on the Georgia EV
supply-chain knowledge base.

> **Scope note.** This framework was built CPT-only, deliberately: the CPT stages must never
> see instruction/answer pairs. The E4–E6 analytical-SFT stages of
> [the experiment plan](../gnem_finetuning_experiment_plan.md) add a supervised path
> alongside it, gated per stage — §2 below describes the invariant that guard enforces and
> which stages it applies to.

## 1. What self-supervised continued pretraining is

Standard causal language modeling: given the tokens seen so far, predict the next
token. There is no "question" and no "answer" — every training example is just a
document (a formatted KB record, or a chunk of web text) that the model reads
left-to-right, predicting each token from the ones before it. The loss is computed
by shifting `input_ids` by one position internally (`labels = input_ids`, with
padded positions masked to `-100` so they don't contribute to the loss) — exactly
what `transformers.DataCollatorForLanguageModeling(mlm=False)` does. This is the
same objective GPT-style pretraining uses, just continued on domain-specific text
instead of started from scratch.

## 2. Why this is not Q&A fine-tuning

Supervised fine-tuning builds instruction/answer pairs and trains on them with a chat
template — that's how you make a model follow instructions or answer specific questions.
The continued-pretraining path here does the opposite: it
never constructs a prompt/response pair, never applies a chat template, and never
computes loss only over an "answer" span. Every module in `src/ssft/data/` enforces
this — `ssft.data.schemas.assert_no_qa_fields()` is called on every example this
framework produces and raises immediately if it ever sees a `role`, `messages`,
`prompt`, `completion`, `instruction`, or `answer` key. `tests/test_no_qa_format.py`
checks this holds for every forbidden key.

The practical implication: this framework teaches a model the *distribution* of
Georgia EV supply-chain text (structured record patterns, vocabulary, entity
co-occurrence) — it does not teach the model to answer a specific question format.
It is a research tool for studying domain adaptation, not a drop-in replacement for
the existing Q&A-tuned model.

## 3. KB company-split vs KB memorization — two different experiments

- **`kb_only_company_split`** (`configs/data/kb_only_company_split.yaml`): splits the
  193 unique companies ~80/10/10 into train/validation/test, with **every row for a
  given company kept in the same split** (`split.py::group_by_company`). This is the
  only mode that can say anything about *generalization* — whether the model learned
  reusable structured-record patterns, not just memorized specific companies. A
  `leakage_report.json` proves zero company overlap across splits for every run.
- **`kb_only_memorization`** (`configs/data/kb_only_memorization.yaml`): trains on
  **100% of the 205 KB rows**, no held-out split. This can only measure in-sample
  absorption ("did the model learn to reproduce facts it was shown"), and every
  report generated from this config says so explicitly — it is never allowed to
  claim generalization.

Conflating these two would be a real methodology error given how tiny this KB is
(205 rows ≈ 28k tokens) — memorization is nearly guaranteed at that scale, so a
memorization result says nothing about whether the model would generalize to a
new, unseen Georgia EV supplier.

## 4. Adapter folder organization

Every run's full artifact bundle lives at:

```
outputs/adapters/{model_id_slug}/{method_id_slug}/{dataset_variant_slug}/{split_strategy_slug}/{training_slug}/seed{seed}/{timestamp}_{short_hash}/
  adapter/              # peft LoRA adapter (adapter_config.json + weights)
  tokenizer/             # tokenizer files (only relevant if it ever diverges from base)
  resolved_config.yaml   # the fully-merged model+method+data+training config actually used
  hyperparams.json       # flat summary of the run's identity + key hyperparameters
  data_manifest.json     # input/processed data SHA256 hashes
  split_manifest.json    # per-split group-key (company/source) membership
  leakage_report.json    # proof of zero cross-split group overlap
  metrics.json           # status + train/eval/test loss & perplexity
  train_log.jsonl        # per-step training log (crash-safe, flushed every line)
  environment.json       # python/torch/transformers/peft/... versions + git commit
  gpu.json               # GPU snapshot (nvidia-smi + torch.cuda stats)
  command.txt            # exact CLI invocation
  timestamps.json        # start/end wall-clock time
  report.md              # human-readable summary + keep/discard recommendation
  eval/                  # written by `ssft evaluate` / `compare-base-adapter`
```

Only the six identity fields (model / method / dataset / split / training_slug /
seed) plus a `{timestamp}_{short_hash}` leaf go into the path — everything else
(the full hyperparameter set) lives inside `resolved_config.yaml` /
`hyperparams.json` so the path never becomes unreadably long. `training_slug` is
**always computed live** from the resolved hyperparameters
(`ep{epochs}-bs{batch}-ga{grad_accum}-ebs{effective_batch}-lr{lr_slug}-seq{seq_len}`)
— never trusted from a config file's literal field — so a sweep that overrides
`training.num_train_epochs` correctly lands in a different directory than the base
config's label would suggest.

## 5. Preparing the KB data

```bash
python -m ssft.cli prepare-kb \
  --input kb_full.jsonl \
  --config self_supervised_finetuning/configs/data/kb_only_company_split.yaml
```

This loads only a tokenizer (for its `eos_token`, defaulting to the Qwen2.5-14B base
config — override with `--model-config`), converts every KB row to the canonical
`<record>...</record>` text format, splits by Company, and writes
`data/processed/<variant>/{kb}/{train,validation,test}.jsonl` plus the manifests —
no model weights are downloaded. Run it a second time with
`--config configs/data/kb_only_memorization.yaml` to prepare the memorization
variant, or use `scripts/01_prepare_kb_dataset.sh <kb_full.jsonl>` to do both.

## 6. Training Qwen2.5-14B with QLoRA

```bash
python -m ssft.cli train \
  --model-config self_supervised_finetuning/configs/models/qwen2p5_14b_base.yaml \
  --method-config self_supervised_finetuning/configs/methods/qlora_lora_default.yaml \
  --data-config self_supervised_finetuning/configs/data/kb_only_company_split.yaml \
  --training-config self_supervised_finetuning/configs/training/tiny_kb_conservative.yaml \
  --input kb_full.jsonl
```

Or the equivalent bundled experiment config:
`python -m ssft.cli train-experiment --experiment-config self_supervised_finetuning/configs/experiments/qwen14b_qlora_kb_company_split.yaml --input kb_full.jsonl`,
or `scripts/02_train_kb_company_split.sh kb_full.jsonl`. For the memorization variant,
use `scripts/03_train_kb_memorization.sh kb_full.jsonl` (or the matching
`--data-config`/`--training-config` pair).

`Qwen/Qwen2.5-14B` (base, not Instruct) is the default primary model — continued
pretraining is a base-model operation; fine-tuning the Instruct variant risks
damaging instruction-following behavior (see the warning in
`configs/models/qwen2p5_14b_instruct_optional.yaml`, kept only for comparison).

## 7. Training a different model

Swap `--model-config`: `configs/models/qwen2p5_3b_base.yaml` or `qwen2p5_7b_base.yaml`
for smaller/faster Qwen models, or `configs/models/mistral_7b_optional.yaml`. Every
model config declares its own `model_id_slug`, which is what appears in the output
path, so different models never collide.

## 8. Testing different epochs

`configs/sweeps/kb_epoch_sweep.yaml` sweeps `training.num_train_epochs` over
`[1, 3, 5, 8, 12]` on the company-split KB experiment:

```bash
python -m ssft.cli sweep --sweep-config self_supervised_finetuning/configs/sweeps/kb_epoch_sweep.yaml --dry-run
python -m ssft.cli sweep --sweep-config self_supervised_finetuning/configs/sweeps/kb_epoch_sweep.yaml
```

## 9. Testing different batch sizes and effective batch sizes

`configs/sweeps/kb_batch_sweep.yaml` sweeps `per_device_train_batch_size ∈ {1, 2}` x
`gradient_accumulation_steps ∈ {8, 16, 32}` — six effective batch sizes (8, 16, 32,
16, 32, 64) on one GPU. `effective_batch_size = per_device_train_batch_size *
gradient_accumulation_steps * world_size`, computed in
`ssft.train.run_naming.effective_batch_size` and recorded in every run's
`hyperparams.json` and the sweep summary. If `bs2` OOMs on the 14B model, the sweep
runner catches it (`ssft.utils.gpu.oom_safe`), records `status="oom"` in
`sweep_summary.csv`, and continues to the next point.

```bash
python -m ssft.cli sweep --sweep-config self_supervised_finetuning/configs/sweeps/kb_batch_sweep.yaml
```

## 10. Testing different LoRA ranks

`configs/sweeps/kb_lora_rank_sweep.yaml` sweeps across the three method configs —
`qlora_lora_low_rank.yaml` (r=8), `qlora_lora_default.yaml` (r=16),
`qlora_lora_high_rank.yaml` (r=32), each with `lora_alpha = 2r`:

```bash
python -m ssft.cli sweep --sweep-config self_supervised_finetuning/configs/sweeps/kb_lora_rank_sweep.yaml
```

Before wrapping the model in LoRA, `ssft.train.lora_factory.validate_target_modules`
checks every `target_modules` entry against `model.named_modules()` and raises
immediately (listing the available Linear-layer suffixes) if any target module
doesn't exist — this framework never silently trains on the wrong layers.

## 11. Testing different learning rates

`configs/sweeps/kb_learning_rate_sweep.yaml` sweeps `[3e-5, 5e-5, 8e-5, 1e-4, 1.5e-4]`:

```bash
python -m ssft.cli sweep --sweep-config self_supervised_finetuning/configs/sweeps/kb_learning_rate_sweep.yaml
```

## 12. Running sweeps in general

```bash
python -m ssft.cli sweep --sweep-config <path/to/sweep.yaml> --dry-run          # print the planned run table only
python -m ssft.cli sweep --sweep-config <path/to/sweep.yaml>                     # run it
python -m ssft.cli sweep --sweep-config <path/to/sweep.yaml> --max-runs 3        # cap it
python -m ssft.cli sweep --sweep-config <path/to/sweep.yaml> --skip-existing false
python -m ssft.cli sweep --sweep-config <path/to/sweep.yaml> --resume-failed true
python -m ssft.cli sweep --sweep-config <path/to/sweep.yaml> --fail-fast
```

A sweep config lists `models` / `methods` / `data` (each a list of config paths), a
`base_training_config` (or `training_config_by_dataset`, mapping
`dataset_variant_slug -> training config path`, for `multi_dataset_qwen14b_sweep.yaml`),
an `overrides` block of dotted-path -> value-list pairs applied as a cartesian
product, and a `seeds` list. Every expanded run gets its own adapter folder and full
resolved config; `outputs/sweeps/<sweep_name>/sweep_summary.{csv,json}` collects one
row per run with every column listed in the framework spec (run_id, status, all
hyperparameters, all loss/perplexity metrics, `failure_reason`). See also
`scripts/06_run_sweep.sh`, `07_run_multi_model_kb_sweep.sh`,
`08_run_multi_dataset_sweep.sh`.

## 13. Evaluating base vs adapter

```bash
python -m ssft.cli evaluate --run-dir <path/to/run>              # full eval: perplexity + cloze + instruction sanity
python -m ssft.cli compare-base-adapter --run-dir <path/to/run>  # perplexity + cloze only
```

Both reload the base model plus the saved adapter, compute train/validation/test
loss & perplexity for each, run the six deterministic cloze probes
(`Company: {X}\nLocation:`, `\nCategory:`, `\nEV or Battery Relevant:`,
`\nEmployment:`, `\nPrimary OEMs:`, `\nProduct or Service:` — generated from KB
fields, never used for training), score them (exact/normalized-exact match for
short fields, token-F1/containment for the long Product-or-Service field), split
results into seen-company (memorization signal) vs held-out-company (generalization
signal) using the run's `split_manifest.json`, and rewrite `report.md` with the
full comparison plus a keep/discard/needs-more-data recommendation.
`scripts/04_eval_adapter.sh` / `05_compare_base_vs_adapter.sh` wrap these.

## 14. Adding future dataset variants

Add a new `configs/data/<name>.yaml` with a `dataset_variant_slug`,
`split_strategy`/`split_strategy_slug`, `source_type` (`kb_jsonl` | `web_corpus` |
`mixed`), and `text_format`. The four already present —
`web_only_source_split` / `raw_web_only` / `cleaned_web_only` (source_type:
`web_corpus`, split by `source_url` before chunking) and `structured_kb_only`
(source_type: `kb_jsonl`, split by `Company`) — are templates; `kb_web_mixed`
additionally shows how to nest two sources under `sources:` with
`sampling_weights`. `ssft.data.dataset_builder._normalize_sources` dispatches on
`source_type`, so a new source type only needs a converter
(`ssft/data/<name>_converter.py`, following `kb_converter.py`/`web_converter.py`)
plus a branch in `build_examples_for_source`.

## 15. Adding future fine-tuning methods

`configs/methods/` already has three implemented QLoRA ranks plus
`lora_fp16_optional.yaml` (no quantization), and three intentional placeholders —
`adalora_placeholder.yaml`, `ia3_placeholder.yaml`, `full_finetune_placeholder.yaml`
— each with `status: not_implemented`, so `ssft.train.hyperparams.resolve_run_config`
and `ssft.train.lora_factory.build_lora_config` raise a clear `NotImplementedError`
if one is selected. To implement one: remove `status: not_implemented`, add the
real hyperparameter block (e.g. an `adalora:` section mirroring peft's
`AdaLoraConfig`), and add a branch in `lora_factory.py` that builds the
corresponding peft config instead of `LoraConfig`.

## 16. Why tiny-KB experiments overfit

The KB has 205 rows / 193 unique companies / ~28k tokens total — several orders of
magnitude smaller than what LLM pretraining or even typical LoRA fine-tuning
corpora look like. A 14B model has enormous capacity relative to that: even with a
rank-16 LoRA adapter (a small fraction of the base model's parameters), it's
entirely plausible for training loss to approach zero while held-out (validation/
test) loss stays flat or worsens — classic overfitting. This is expected, not a
bug, and is exactly why:
- the company-split experiment exists (to *measure* whether anything generalizes,
  rather than assume it does),
- `tiny_kb_conservative.yaml` uses early stopping (`patience: 2` on `eval_loss`),
- `report.md`'s "Overfitting analysis" section flags when `eval_loss` diverges
  from `train_loss` by more than 1.5x, and
- the recommendation logic (`ssft.eval.reports.recommend_outcome`) requires
  held-out perplexity to actually improve over the base model before recommending
  "keep adapter" — training loss decreasing alone is not sufficient evidence.

## 17. This does not replace RAG

A Q&A-tuned pipeline and any RAG-based approach both directly optimize for answering
questions correctly. Continued pretraining does not: it has no notion of "correct answer,"
no retrieval, and no
evaluation against a held-out Q&A set — its cloze probes measure whether specific
field values got absorbed into the weights, not whether the model can converse
about the KB. It exists to answer a narrower research question (does self-supervised
continued pretraining on structured Georgia EV records change what a base LLM
knows, and does that change generalize) — not to produce a deployable
question-answering system. Every generated report says this explicitly and the
recommendation logic never suggests otherwise.

---

## Requirements

```bash
pip install -r self_supervised_finetuning/requirements.txt
pip install -e self_supervised_finetuning   # makes `ssft` importable / python -m ssft.cli work
```

Real (non-`debug`) training configs require a CUDA GPU — `ssft.train.model_loader`
fails loudly rather than silently falling back to slow/incorrect CPU training.
`configs/training/debug_cpu_or_small_gpu.yaml` (`training.debug: true`) is the one
exception, meant for pipeline smoke-testing, not real results.

## Environment inspection

```bash
bash self_supervised_finetuning/scripts/00_inspect_environment.sh
python -m ssft.cli inspect-repo    # KB schema check, git commit, row/company counts
python -m ssft.cli inspect-env     # python/torch/transformers/peft/accelerate/bitsandbytes + GPU snapshot
```

## Tests

```bash
pytest self_supervised_finetuning/tests/ -v
```

Covers: KB loading + 16-column validation, canonical text formatting (every column
present, missing -> "Not specified"), company-split leakage (zero overlap, ratios
≈80/10/10), output-path/`training_slug`/`run_id`/`effective_batch_size` generation
against the spec's own worked examples, LoRA target-module validation, sweep
expansion counts, and — most importantly — that no example anywhere contains a
`role`/`messages`/`prompt`/`completion`/`instruction`/`answer` key.
