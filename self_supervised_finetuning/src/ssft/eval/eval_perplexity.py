"""Base-vs-adapter loss/perplexity evaluation on train/validation/test splits.

Evaluation-only: nothing here trains anything. `validation` is reported under the
key "eval" throughout (matching HF Trainer / the sweep-summary column names
eval_loss / eval_perplexity).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional


def compute_split_losses(model, tokenizer, datasets: dict, batch_size: int = 1) -> dict:
    import torch
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    was_training = model.training
    model.eval()

    results = {}
    for split_name, out_key in (("train", "train"), ("validation", "eval"), ("test", "test")):
        examples = datasets.get(split_name) or []
        if not examples:
            results[out_key] = {"loss": None, "perplexity": None, "n_examples": 0}
            continue
        ds = Dataset.from_list(examples)
        loader = DataLoader(ds, batch_size=batch_size, collate_fn=collator)
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                out = model(**batch)
                total_loss += out.loss.item()
                n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        results[out_key] = {
            "loss": avg_loss,
            "perplexity": math.exp(avg_loss) if avg_loss < 20 else float("inf"),
            "n_examples": len(examples),
        }
    if was_training:
        model.train()
    return results


def load_adapter_for_eval(run_dir: Path):
    """Reload base model + adapter for post-hoc evaluation from a completed run's
    resolved_config.yaml and adapter/ subfolder."""
    from peft import PeftModel

    from ssft.train.hyperparams import ResolvedConfig
    from ssft.train.model_loader import load_model_and_tokenizer
    from ssft.utils.yaml_utils import load_yaml

    run_dir = Path(run_dir)
    raw = load_yaml(run_dir / "resolved_config.yaml")
    resolved = ResolvedConfig(
        model_cfg=raw["model"], method_cfg=raw["method"], data_cfg=raw["data"],
        training_cfg=raw["training"], seed=raw["seed"], provenance=raw.get("provenance", {}),
    )
    base_model, tokenizer = load_model_and_tokenizer(resolved.model_cfg, resolved.method_cfg, resolved.training_cfg)
    adapter_model = PeftModel.from_pretrained(base_model, str(run_dir / "adapter"))
    return resolved, base_model, adapter_model, tokenizer


def run_perplexity_eval(run_dir: Path) -> dict:
    """Rebuild the run's datasets deterministically (same seed/config) and compute
    base vs adapter loss/perplexity on every split. Writes eval/perplexity_eval.json."""
    from ssft.data.dataset_builder import build_datasets

    run_dir = Path(run_dir)
    resolved, base_model, adapter_model, tokenizer = load_adapter_for_eval(run_dir)
    datasets, _ = build_datasets(resolved.data_cfg, tokenizer)
    if resolved.data_cfg.get("split_strategy") == "train_all" and not datasets.get("validation"):
        datasets["validation"] = list(datasets["train"])

    # PeftModel.from_pretrained wraps base_model IN PLACE, so base_model and adapter_model are
    # the same underlying module with the adapter attached. Measure the true base with the
    # adapter disabled, not by trusting the (now adapter-carrying) base_model reference.
    with adapter_model.disable_adapter():
        base_metrics = compute_split_losses(adapter_model, tokenizer, datasets)
    adapter_metrics = compute_split_losses(adapter_model, tokenizer, datasets)

    delta = {}
    for split in ("train", "eval", "test"):
        b, a = base_metrics[split], adapter_metrics[split]
        if b["loss"] is None or a["loss"] is None:
            delta[split] = {"delta_loss": None, "delta_perplexity": None}
        else:
            both_finite = math.isfinite(a["perplexity"]) and math.isfinite(b["perplexity"])
            delta[split] = {
                "delta_loss": a["loss"] - b["loss"],
                "delta_perplexity": (a["perplexity"] - b["perplexity"]) if both_finite else None,
            }

    result = {"base": base_metrics, "adapter": adapter_metrics, "delta": delta}
    out_path = run_dir / "eval" / "perplexity_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result
