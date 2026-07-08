"""Assemble a run's report.md and a sweep's sweep_summary.csv/json.

`recommend_outcome` turns the spec's qualitative keep/discard/needs-more-data
criteria into a reproducible rule using named constants below. These are heuristics
for a tiny-KB research framework, not a scientific claim.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Optional

VAL_LOSS_EXPLOSION_RATIO = 1.5    # eval_loss > this * train_loss => "exploded"
MIN_PERPLEXITY_IMPROVEMENT = 0.0  # adapter_ppl < base_ppl - this => "improved"
MIN_CLOZE_IMPROVEMENT = 0.0       # adapter_score > base_score + this => "improved"

SWEEP_SUMMARY_COLUMNS = [
    "run_id", "status", "model_id_slug", "method_id_slug", "dataset_variant_slug",
    "split_strategy_slug", "training_slug", "seed", "num_train_epochs",
    "per_device_train_batch_size", "gradient_accumulation_steps", "effective_batch_size",
    "learning_rate", "max_seq_length", "lora_r", "lora_alpha", "lora_dropout",
    "output_dir", "train_loss", "eval_loss", "test_loss", "train_perplexity",
    "eval_perplexity", "test_perplexity", "failure_reason",
]


def recommend_outcome(comparison: Optional[dict], split_strategy_slug: str) -> str:
    if split_strategy_slug == "train-all":
        return (
            "MEMORIZATION RUN — this does not evaluate generalization. Judge only on "
            "in-sample (train) loss decrease and in-sample (seen-company) cloze recall."
        )
    if comparison is None:
        return "NEEDS EVALUATION — run `ssft compare-base-adapter --run-dir <dir>` before deciding keep/discard."

    ppl = comparison.get("perplexity", {}) or {}
    base, adapter = ppl.get("base", {}) or {}, ppl.get("adapter", {}) or {}
    reasons = []
    held_out_improved = None
    for split_name in ("eval", "test"):
        b, a = base.get(split_name) or {}, adapter.get(split_name) or {}
        if b.get("perplexity") is not None and a.get("perplexity") is not None:
            improved = a["perplexity"] < b["perplexity"] - MIN_PERPLEXITY_IMPROVEMENT
            held_out_improved = improved if held_out_improved is None else (held_out_improved or improved)
            reasons.append(
                f"{split_name} perplexity {'improved' if improved else 'did not improve'} "
                f"(base={b['perplexity']:.2f} -> adapter={a['perplexity']:.2f})"
            )

    cloze_summary = (comparison.get("cloze", {}) or {}).get("summary", {}) or {}
    held_out_cloze = cloze_summary.get("held_out_company", {}) or {}
    ha, hb = held_out_cloze.get("adapter", {}) or {}, held_out_cloze.get("base", {}) or {}
    cloze_improved = None
    if ha and hb:
        key = "normalized_exact_match" if "normalized_exact_match" in ha else "token_f1"
        cloze_improved = ha.get(key, 0) > hb.get(key, 0) + MIN_CLOZE_IMPROVEMENT
        reasons.append(
            f"held-out cloze {key} {'improved' if cloze_improved else 'did not improve'} "
            f"(base={hb.get(key):.3f} -> adapter={ha.get(key):.3f})"
        )

    if held_out_improved and cloze_improved is not False:
        verdict = "KEEP ADAPTER"
    elif held_out_improved is False and cloze_improved is False:
        verdict = "DISCARD ADAPTER"
    else:
        verdict = "NEEDS MORE DATA — mixed signal, treat as inconclusive given the tiny KB"
    return verdict + (". " + " ".join(reasons) if reasons else "")


def build_run_report(
    *,
    run_dir: Path,
    resolved: dict,
    metrics: dict,
    data_manifest_info: dict,
    env: dict,
    gpu: dict,
    git_commit: Optional[str],
    command: str,
    start_time: str,
    end_time: str,
    run_id: str,
    comparison: Optional[dict] = None,
    sanity: Optional[dict] = None,
) -> str:
    from ssft.train import run_naming
    from ssft.train.hyperparams import ResolvedConfig

    rc = ResolvedConfig(
        resolved["model"], resolved["method"], resolved["data"], resolved["training"],
        resolved["seed"], resolved.get("provenance", {}),
    )
    split_strategy_slug = run_naming.get_split_strategy_slug(rc)

    lines = [f"# Run report: {run_id}", ""]
    lines += [
        f"- model: `{rc.model_cfg.get('name_or_path')}` ({run_naming.get_model_id_slug(rc)})",
        f"- method: {run_naming.get_method_id_slug(rc)}",
        f"- dataset: {run_naming.get_dataset_variant_slug(rc)}",
        f"- split strategy: {split_strategy_slug}",
        f"- seed: {rc.seed}",
        f"- git commit: {git_commit or 'unknown'}",
        f"- command: `{command}`",
        f"- start: {start_time} / end: {end_time}",
        "",
    ]

    if split_strategy_slug == "train-all":
        lines += [
            "> **WARNING**: This run trains on all KB rows. It can test in-sample "
            "absorption/memorization but cannot prove generalization.",
            "",
        ]

    t = rc.training_cfg
    lora = rc.method_cfg.get("lora") or {}
    lines += [
        "## Hyperparameters",
        f"- epochs: {t.get('num_train_epochs')}",
        f"- per_device_train_batch_size: {t.get('per_device_train_batch_size')}",
        f"- gradient_accumulation_steps: {t.get('gradient_accumulation_steps')}",
        f"- effective_batch_size: {run_naming.effective_batch_size(t.get('per_device_train_batch_size', 1), t.get('gradient_accumulation_steps', 1))}",
        f"- learning_rate: {t.get('learning_rate')}",
        f"- max_seq_length: {rc.data_cfg.get('max_seq_length')}",
        f"- lora_r: {lora.get('r')}, lora_alpha: {lora.get('lora_alpha')}, lora_dropout: {lora.get('lora_dropout')}",
        "",
        "## Data",
    ]
    for name, sizes in (data_manifest_info.get("per_source_split_sizes") or {}).items():
        lines.append(f"- {name}: train={sizes.get('train', 0)} validation={sizes.get('validation', 0)} test={sizes.get('test', 0)}")
    lines.append("")

    lines += [
        "## Training result",
        f"- train_loss: {metrics.get('train_loss')}",
        f"- eval_loss: {metrics.get('eval_loss')}",
        f"- test_loss: {metrics.get('test_loss')}",
        f"- train_perplexity: {metrics.get('train_perplexity')}",
        f"- eval_perplexity: {metrics.get('eval_perplexity')}",
        f"- test_perplexity: {metrics.get('test_perplexity')}",
        f"- adapter path: `{run_dir}/adapter/`",
        "",
        "## Overfitting analysis",
    ]
    tl, el = metrics.get("train_loss"), metrics.get("eval_loss")
    if tl is not None and el is not None:
        if el > tl * VAL_LOSS_EXPLOSION_RATIO:
            lines.append(f"- eval_loss ({el:.3f}) is notably higher than train_loss ({tl:.3f}) — signs of overfitting, expected given the tiny KB.")
        else:
            lines.append(f"- eval_loss ({el:.3f}) tracks train_loss ({tl:.3f}) reasonably closely.")
    else:
        lines.append("- insufficient data to assess.")
    lines.append("")

    lines.append("## Base vs adapter comparison")
    if comparison:
        ppl = comparison.get("perplexity", {}) or {}
        for split in ("train", "eval", "test"):
            b = (ppl.get("base") or {}).get(split, {}) or {}
            a = (ppl.get("adapter") or {}).get(split, {}) or {}
            lines.append(f"- {split}: base_ppl={b.get('perplexity')} adapter_ppl={a.get('perplexity')}")
        cloze = (comparison.get("cloze", {}) or {}).get("summary", {}) or {}
        lines.append(f"- cloze (seen-company, memorization signal): {cloze.get('seen_company')}")
        lines.append(f"- cloze (held-out-company, generalization signal): {cloze.get('held_out_company')}")
    else:
        lines.append("- Not yet computed. Run `ssft compare-base-adapter --run-dir <this dir>`.")
    lines.append("")

    lines.append("## Instruction sanity check")
    if sanity:
        n_degraded = sum(1 for r in sanity.get("rows", []) if r.get("degraded"))
        lines.append(f"- {n_degraded} / {len(sanity.get('rows', []))} prompts flagged as degraded vs base (damage check only, not the main benchmark).")
    else:
        lines.append("- Not yet computed. Run `ssft evaluate --run-dir <this dir>`.")
    lines.append("")

    lines.append("## Recommendation")
    lines.append(recommend_outcome(comparison, split_strategy_slug))
    lines.append("")

    pkgs = env.get("packages", {})
    lines += [
        "## Environment",
        f"- python: {env.get('python_version', '?').split()[0]}",
        f"- torch: {pkgs.get('torch')}",
        f"- transformers: {pkgs.get('transformers')}",
        f"- peft: {pkgs.get('peft')}",
        f"- bitsandbytes: {pkgs.get('bitsandbytes')}",
    ]
    if gpu.get("torch_cuda"):
        lines.append(f"- gpu: {gpu['torch_cuda']}")
    lines += [
        "",
        "_This adapter is a research artifact for studying domain-adaptive continued "
        "pretraining on a tiny KB — it does not replace RAG._",
    ]

    report = "\n".join(lines)
    with open(Path(run_dir) / "report.md", "w") as f:
        f.write(report)
    return report


def build_sweep_summary(rows: list[dict], sweep_dir: Path) -> None:
    sweep_dir = Path(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    with open(sweep_dir / "sweep_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in SWEEP_SUMMARY_COLUMNS})

    with open(sweep_dir / "sweep_summary.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
