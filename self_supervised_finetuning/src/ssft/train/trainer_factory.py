"""Build the transformers.Trainer used for CLM training.

DataCollatorForLanguageModeling(mlm=False) is what actually implements "labels =
input_ids shifted internally by the causal LM loss, pad positions masked as -100":
it dynamically pads input_ids/attention_mask, clones input_ids into labels, and sets
label positions at pad_token_id to -100. No trl.SFTTrainer anywhere — this is CLM,
not SFT.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def build_data_collator(tokenizer):
    from transformers import DataCollatorForLanguageModeling
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


def _bool_or_auto(value, auto_default: bool) -> bool:
    if value == "auto":
        return auto_default
    return bool(value)


def build_training_arguments(resolved, output_dir: Path, has_eval: bool = True):
    import torch
    from transformers import TrainingArguments

    t = resolved.training_cfg

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    bf16 = _bool_or_auto(t.get("bf16", "auto"), bf16_ok)
    fp16 = _bool_or_auto(t.get("fp16", "auto"), torch.cuda.is_available() and not bf16_ok)

    optim = t.get("optim", "adamw_torch")
    if "8bit" in optim or optim.startswith("paged_"):
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as e:
            raise ImportError(f"training.optim={optim!r} requires bitsandbytes, which is not installed.") from e

    load_best = bool(t.get("load_best_model_at_end", False)) and has_eval
    eval_strategy = t.get("eval_strategy", "epoch") if has_eval else "no"

    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=t.get("num_train_epochs", 1),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        learning_rate=t.get("learning_rate", 5e-5),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.0),
        weight_decay=t.get("weight_decay", 0.0),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        optim=optim,
        logging_strategy=t.get("logging_strategy", "steps"),
        logging_steps=t.get("logging_steps", 10),
        eval_strategy=eval_strategy,
        save_strategy=t.get("save_strategy", "epoch"),
        save_total_limit=t.get("save_total_limit", 3),
        load_best_model_at_end=load_best,
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        bf16=bf16,
        fp16=fp16,
        report_to=t.get("report_to", []) or [],
        seed=t.get("seed", 42),
        data_seed=t.get("data_seed", t.get("seed", 42)),
        dataloader_num_workers=t.get("dataloader_num_workers", 0),
        remove_unused_columns=t.get("remove_unused_columns", False),
    )
    if eval_strategy == "steps":
        kwargs["eval_steps"] = t.get("eval_steps", 200)
    if t.get("save_strategy", "epoch") == "steps":
        kwargs["save_steps"] = t.get("save_steps", 200)
    if load_best:
        kwargs["metric_for_best_model"] = t.get("metric_for_best_model", "eval_loss")
        kwargs["greater_is_better"] = t.get("greater_is_better", False)

    return TrainingArguments(**kwargs)


def build_callbacks(resolved, output_dir: Path, has_eval: bool = True):
    from ssft.train.callbacks import GpuMemoryCallback, JsonlLoggerCallback

    callbacks = [
        JsonlLoggerCallback(output_dir / "train_log.jsonl"),
        GpuMemoryCallback(output_dir / "train_log.jsonl"),
    ]
    early_stopping = resolved.training_cfg.get("early_stopping") or {}
    if early_stopping.get("enabled") and has_eval:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping.get("patience", 1),
            early_stopping_threshold=early_stopping.get("threshold", 0.0),
        ))
    return callbacks


def build_trainer(model, tokenizer, datasets: dict, resolved, output_dir: Path, resume: bool = False):
    from datasets import Dataset
    from transformers import Trainer

    train_examples = datasets.get("train") or []
    if not train_examples:
        raise ValueError("train split is empty — nothing to train on")
    train_dataset = Dataset.from_list(train_examples)

    eval_examples = datasets.get("validation") or []
    eval_dataset = Dataset.from_list(eval_examples) if eval_examples else None
    has_eval = eval_dataset is not None

    args = build_training_arguments(resolved, output_dir, has_eval=has_eval)
    collator = build_data_collator(tokenizer)
    callbacks = build_callbacks(resolved, output_dir, has_eval=has_eval)

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
