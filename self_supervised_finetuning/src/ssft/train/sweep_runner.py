"""Expand a sweep config into concrete runs and execute them.

Deterministic cartesian product over models x methods x data x (sorted override-key
combinations) x seeds. `train`, `train-experiment`, and `sweep` all funnel into the
same `hyperparams.resolve_run_config` + `train_clm_lora.run_training` — nothing here
duplicates training logic, only orchestrates many calls to it.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ssft.train import run_naming
from ssft.train.hyperparams import load_data_config, resolve_run_config
from ssft.train.train_clm_lora import run_training
from ssft.utils import gpu as gpu_utils
from ssft.utils.paths import ADAPTERS_ROOT, SWEEPS_ROOT, ensure_dir, resolve_repo_relative
from ssft.utils.yaml_utils import load_yaml


@dataclass
class SweepPoint:
    model_config: str
    method_config: str
    data_config: str
    training_config: str
    overrides: dict = field(default_factory=dict)
    seed: int = 42
    sweep_index: int = 0


def load_sweep_config(path) -> dict:
    raw = load_yaml(resolve_repo_relative(str(path)))
    if "sweep" not in raw:
        raise ValueError(f"{path}: expected a top-level 'sweep:' key")
    return raw["sweep"]


def _dataset_variant_slug_for(data_config_path: str) -> str:
    data_cfg = load_data_config(data_config_path)
    return data_cfg.get("dataset_variant_slug") or run_naming.slugify(data_cfg.get("source_type", "dataset"))


def expand_sweep(sweep_cfg: dict) -> list[SweepPoint]:
    models = sweep_cfg.get("models") or []
    methods = sweep_cfg.get("methods") or []
    data_list = sweep_cfg.get("data") or []
    base_training_config = sweep_cfg.get("base_training_config")
    training_config_by_dataset = sweep_cfg.get("training_config_by_dataset")
    overrides_spec = sweep_cfg.get("overrides") or {}
    seeds = sweep_cfg.get("seeds") or [42]
    max_runs = sweep_cfg.get("max_runs")

    if not models or not methods or not data_list:
        raise ValueError("sweep config must specify at least one model, method, and data config")
    if not base_training_config and not training_config_by_dataset:
        raise ValueError("sweep config must specify either base_training_config or training_config_by_dataset")

    override_keys = sorted(overrides_spec.keys())
    override_value_lists = [overrides_spec[k] for k in override_keys]
    override_combos = list(itertools.product(*override_value_lists)) if override_value_lists else [()]

    points: list[SweepPoint] = []
    idx = 0
    for model_config in models:
        for method_config in methods:
            for data_config in data_list:
                if training_config_by_dataset:
                    variant_slug = _dataset_variant_slug_for(data_config)
                    training_config = training_config_by_dataset.get(variant_slug)
                    if not training_config:
                        raise ValueError(
                            f"data config {data_config} resolves to dataset_variant_slug="
                            f"{variant_slug!r}, which has no entry in sweep.training_config_by_dataset "
                            f"(keys present: {sorted(training_config_by_dataset.keys())})"
                        )
                else:
                    training_config = base_training_config

                for combo in override_combos:
                    overrides = dict(zip(override_keys, combo))
                    for seed in seeds:
                        points.append(SweepPoint(
                            model_config=model_config, method_config=method_config,
                            data_config=data_config, training_config=training_config,
                            overrides=overrides, seed=seed, sweep_index=idx,
                        ))
                        idx += 1

    if max_runs is not None:
        points = points[:max_runs]
    return points


def print_dry_run_table(points: list[SweepPoint], resolved_list: list) -> None:
    header = f"{'idx':>4}  {'model':<16} {'method':<26} {'dataset':<22} {'training_slug':<34} {'seed':>5}"
    print(header)
    print("-" * len(header))
    for point, resolved in zip(points, resolved_list):
        print(
            f"{point.sweep_index:>4}  {run_naming.get_model_id_slug(resolved):<16} "
            f"{run_naming.get_method_id_slug(resolved):<26} {run_naming.get_dataset_variant_slug(resolved):<22} "
            f"{run_naming.build_training_slug(resolved):<34} {resolved.seed:>5}"
        )


def _row_from_result(resolved, run_id, status, output_dir, metrics: dict, failure_reason: Optional[str]) -> dict:
    t, d = resolved.training_cfg, resolved.data_cfg
    lora = resolved.method_cfg.get("lora") or {}
    return {
        "run_id": run_id,
        "status": status,
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
        "output_dir": str(output_dir) if output_dir else None,
        "train_loss": metrics.get("train_loss"),
        "eval_loss": metrics.get("eval_loss"),
        "test_loss": metrics.get("test_loss"),
        "train_perplexity": metrics.get("train_perplexity"),
        "eval_perplexity": metrics.get("eval_perplexity"),
        "test_perplexity": metrics.get("test_perplexity"),
        "failure_reason": failure_reason,
    }


def run_sweep(
    sweep_config_path,
    *,
    dry_run: bool = False,
    max_runs: Optional[int] = None,
    skip_existing: bool = True,
    resume_failed: bool = False,
    fail_fast: bool = False,
    adapters_root: Optional[Path] = None,
    sweeps_root: Optional[Path] = None,
    command: str = "",
) -> list[dict]:
    adapters_root = adapters_root or ADAPTERS_ROOT
    sweeps_root = sweeps_root or SWEEPS_ROOT

    sweep_cfg = load_sweep_config(sweep_config_path)
    sweep_name = sweep_cfg.get("sweep_name", "sweep")
    points = expand_sweep(sweep_cfg)
    if max_runs is not None:
        points = points[:max_runs]

    resolved_list = [
        resolve_run_config(p.model_config, p.method_config, p.data_config, p.training_config,
                            seed=p.seed, overrides=p.overrides)
        for p in points
    ]

    print(f"Sweep '{sweep_name}': {len(points)} planned run(s)")
    print_dry_run_table(points, resolved_list)

    sweep_dir = ensure_dir(Path(sweeps_root) / sweep_name)
    if dry_run:
        return []

    rows = []
    for point, resolved in zip(points, resolved_list):
        attempts = run_naming.find_existing_attempts(resolved, adapters_root)

        if skip_existing and attempts and attempts[0][1] == "completed":
            existing_dir = attempts[0][0]
            metrics = {}
            metrics_path = existing_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
            run_id = metrics.get("run_id", existing_dir.name)
            rows.append(_row_from_result(resolved, run_id, "skipped", existing_dir, metrics, None))
            print(f"[skip] idx={point.sweep_index} already completed at {existing_dir}")
            continue

        output_dir_override = None
        resume = False
        if resume_failed and attempts and attempts[0][1] in ("failed", "oom"):
            output_dir_override, resume = attempts[0][0], True

        run_cmd = command or f"ssft sweep --sweep-config {sweep_config_path}"
        try:
            result = gpu_utils.oom_safe(run_training)(
                resolved, adapters_root=adapters_root, command=run_cmd,
                output_dir_override=output_dir_override, resume=resume,
            )
            rows.append(_row_from_result(resolved, result.run_id, result.status, result.output_dir, result.metrics, None))
        except gpu_utils.OomError as e:
            rows.append(_row_from_result(resolved, None, "oom", output_dir_override, {}, str(e)))
            print(f"[oom] idx={point.sweep_index} failed: {e}")
            if fail_fast:
                break
        except Exception as e:
            rows.append(_row_from_result(resolved, None, "failed", output_dir_override, {}, f"{type(e).__name__}: {e}"))
            print(f"[fail] idx={point.sweep_index} failed: {e}")
            if fail_fast:
                break

    from ssft.eval.reports import build_sweep_summary
    build_sweep_summary(rows, sweep_dir)
    print(f"Sweep '{sweep_name}' finished: {len(rows)} run(s) attempted. Summary: {sweep_dir}/sweep_summary.csv")
    return rows
