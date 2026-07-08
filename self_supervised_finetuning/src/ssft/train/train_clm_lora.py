"""Main training entrypoint: compose model_loader + lora_factory + dataset_builder +
trainer_factory + run_naming into one causal-LM LoRA/QLoRA training run, and write
the full required artifact bundle.

`train`, `train-experiment`, and `sweep` all call `run_training()` — no duplicated
training logic. On failure this function writes best-effort partial artifacts
(resolved_config.yaml, metrics.json with status="failed") and then RE-RAISES, so a
direct `ssft train` invocation shows a full traceback; the sweep runner is
responsible for catching failures itself and continuing to the next run.
"""
from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ssft.data.dataset_builder import build_datasets
from ssft.train import run_naming
from ssft.train.hyperparams import ResolvedConfig, validate_resolved_config
from ssft.train.lora_factory import build_lora_model
from ssft.train.model_loader import load_model_and_tokenizer
from ssft.train.trainer_factory import build_trainer
from ssft.utils import env as env_utils
from ssft.utils import git as git_utils
from ssft.utils import gpu as gpu_utils
from ssft.utils.hashing import sha256_json, short_hash
from ssft.utils.logging_utils import get_run_logger
from ssft.utils.paths import ensure_dir
from ssft.utils.seed import set_all_seeds
from ssft.utils.yaml_utils import dump_yaml


@dataclass
class RunResult:
    status: str
    output_dir: Optional[Path]
    run_id: Optional[str]
    metrics: dict = field(default_factory=dict)


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_hyperparams_summary(resolved: ResolvedConfig, run_id: str, output_dir: Path, data_manifest_info: dict) -> dict:
    t, d, lora = resolved.training_cfg, resolved.data_cfg, (resolved.method_cfg.get("lora") or {})
    return {
        "run_id": run_id,
        "model_id_slug": run_naming.get_model_id_slug(resolved),
        "method_id_slug": run_naming.get_method_id_slug(resolved),
        "dataset_variant_slug": run_naming.get_dataset_variant_slug(resolved),
        "split_strategy_slug": run_naming.get_split_strategy_slug(resolved),
        "training_slug": run_naming.build_training_slug(resolved),
        "seed": resolved.seed,
        "num_train_epochs": t.get("num_train_epochs"),
        "per_device_train_batch_size": t.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": t.get("gradient_accumulation_steps"),
        "effective_batch_size": run_naming.effective_batch_size(
            t.get("per_device_train_batch_size", 1), t.get("gradient_accumulation_steps", 1),
        ),
        "learning_rate": t.get("learning_rate"),
        "max_seq_length": d.get("max_seq_length"),
        "lora_r": lora.get("r"),
        "lora_alpha": lora.get("lora_alpha"),
        "lora_dropout": lora.get("lora_dropout"),
        "output_dir": str(output_dir),
        "data_provenance": data_manifest_info,
    }


def run_training(
    resolved: ResolvedConfig,
    *,
    adapters_root: Path,
    command: str,
    output_dir_override: Optional[Path] = None,
    resume: bool = False,
    run_eval: bool = True,
) -> RunResult:
    """`resolved` must already have any --input/data.input_path override baked in
    (via hyperparams.resolve_run_config(..., input_path_override=...)) — this
    function never mutates the ResolvedConfig it's given."""
    validate_resolved_config(resolved)
    set_all_seeds(resolved.seed)
    logger = get_run_logger(adapters_root)

    timestamp = _timestamp()
    hsh = short_hash(sha256_json(resolved.as_dict()) + timestamp)
    output_dir = ensure_dir(Path(output_dir_override) if output_dir_override else run_naming.build_output_dir(resolved, adapters_root, timestamp, hsh))
    run_id = run_naming.build_run_id(resolved, timestamp, hsh)
    start_time = _timestamp()
    logger.info(f"run_id={run_id} output_dir={output_dir}")

    resolved_dict = resolved.as_dict()
    resolved_dict["provenance"]["run_id"] = run_id
    dump_yaml(resolved_dict, output_dir / "resolved_config.yaml")

    try:
        model, tokenizer = load_model_and_tokenizer(resolved.model_cfg, resolved.method_cfg, resolved.training_cfg)
        peft_model = build_lora_model(model, resolved.method_cfg)
        peft_model.print_trainable_parameters()

        datasets, data_manifest_info = build_datasets(
            resolved.data_cfg, tokenizer, processed_dir=output_dir,
        )

        # KB memorization mode (train_all): eval on the SAME train data so
        # eval_strategy: epoch still tracks in-sample absorption without fabricating a
        # held-out split. Never treat this as a generalization measurement.
        if resolved.data_cfg.get("split_strategy") == "train_all" and not datasets.get("validation"):
            datasets["validation"] = list(datasets["train"])

        trainer = build_trainer(peft_model, tokenizer, datasets, resolved, output_dir, resume=resume)

        resume_from_checkpoint = None
        if resume:
            checkpoints = sorted(output_dir.glob("checkpoint-*"))
            resume_from_checkpoint = str(checkpoints[-1]) if checkpoints else None

        train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        adapter_dir = ensure_dir(output_dir / "adapter")
        tokenizer_dir = ensure_dir(output_dir / "tokenizer")
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(tokenizer_dir))

        from ssft.eval.eval_perplexity import compute_split_losses
        split_metrics = compute_split_losses(trainer.model, tokenizer, datasets) if run_eval else {}

        train_metrics = getattr(train_output, "metrics", {}) or {}
        metrics = {
            "status": "completed",
            "run_id": run_id,
            "train_runtime_sec": train_metrics.get("train_runtime"),
            "n_train_examples": len(datasets.get("train") or []),
            "n_eval_examples": len(datasets.get("validation") or []),
            "n_test_examples": len(datasets.get("test") or []),
        }
        for split_name, out_key in (("train", "train"), ("eval", "eval"), ("test", "test")):
            m = split_metrics.get(out_key, {})
            metrics[f"{split_name}_loss"] = m.get("loss")
            metrics[f"{split_name}_perplexity"] = m.get("perplexity")
        if metrics["train_loss"] is None:
            metrics["train_loss"] = train_metrics.get("train_loss")

        end_time = _timestamp()

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        hyperparams = _build_hyperparams_summary(resolved, run_id, output_dir, data_manifest_info)
        with open(output_dir / "hyperparams.json", "w") as f:
            json.dump(hyperparams, f, indent=2, default=str)

        env_snapshot = env_utils.snapshot()
        env_snapshot["git_commit"] = git_utils.current_commit()
        with open(output_dir / "environment.json", "w") as f:
            json.dump(env_snapshot, f, indent=2, default=str)

        gpu_snapshot = gpu_utils.snapshot()
        with open(output_dir / "gpu.json", "w") as f:
            json.dump(gpu_snapshot, f, indent=2, default=str)

        with open(output_dir / "command.txt", "w") as f:
            f.write(command + "\n")
        with open(output_dir / "timestamps.json", "w") as f:
            json.dump({"start": start_time, "end": end_time}, f, indent=2)

        from ssft.eval.reports import build_run_report
        build_run_report(
            run_dir=output_dir, resolved=resolved_dict, metrics=metrics,
            data_manifest_info=data_manifest_info, env=env_snapshot, gpu=gpu_snapshot,
            git_commit=env_snapshot["git_commit"], command=command,
            start_time=start_time, end_time=end_time, run_id=run_id,
        )

        (output_dir / "_SUCCESS").touch()
        logger.info(f"run_id={run_id} completed: train_loss={metrics.get('train_loss')} eval_loss={metrics.get('eval_loss')}")
        return RunResult(status="completed", output_dir=output_dir, run_id=run_id, metrics=metrics)

    except Exception as e:
        failure_reason = f"{type(e).__name__}: {e}"
        logger.exception(f"run {run_id} failed: {failure_reason}")
        try:
            with open(output_dir / "metrics.json", "w") as f:
                json.dump({"status": "failed", "run_id": run_id, "failure_reason": failure_reason}, f, indent=2, default=str)
        except OSError:
            pass
        raise
