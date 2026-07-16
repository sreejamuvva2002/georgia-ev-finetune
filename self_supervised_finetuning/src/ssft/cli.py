"""ssft CLI — self-supervised (causal-LM) continued-pretraining experiment framework.

Runnable as `python -m ssft.cli <subcommand> ...`. Heavy imports (torch/transformers/
peft) happen inside each cmd_* function, not at module import time, so
`inspect-repo`/`inspect-env`/`sweep --dry-run` stay fast even without a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {v!r}")


def _default_model_config() -> str:
    from ssft.utils.paths import CONFIGS_ROOT
    return str(CONFIGS_ROOT / "models" / "qwen2p5_14b_base.yaml")


def cmd_inspect_repo(args) -> int:
    from ssft.data.schemas import KB_EXPECTED_COLUMNS
    from ssft.utils import git as git_utils
    from ssft.utils.paths import REPO_ROOT, SSFT_ROOT

    print(f"repo_root: {REPO_ROOT}")
    print(f"ssft_root: {SSFT_ROOT}")
    print(f"git_commit: {git_utils.current_commit()}")
    print(f"git_dirty: {git_utils.is_dirty()}")

    kb_path = Path(args.input) if args.input else (REPO_ROOT / "kb_full.jsonl")
    print(f"kb_path: {kb_path} exists={kb_path.exists()}")
    if kb_path.exists():
        from ssft.data.kb_converter import load_kb_jsonl
        rows = load_kb_jsonl(kb_path)
        companies = {r.get("Company") for r in rows}
        missing_cols = [c for c in KB_EXPECTED_COLUMNS if rows and c not in rows[0]]
        print(f"kb_rows: {len(rows)}")
        print(f"kb_unique_companies: {len(companies)}")
        print(f"kb_expected_columns_present: {not missing_cols}")
        if missing_cols:
            print(f"kb_missing_columns: {missing_cols}")

    return 0


def cmd_inspect_env(args) -> int:
    from ssft.utils import env as env_utils
    from ssft.utils import gpu as gpu_utils

    result = {"env": env_utils.snapshot(), "gpu": gpu_utils.snapshot()}
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_prepare_kb(args) -> int:
    from ssft.data.dataset_builder import prepare_text_splits
    from ssft.train.hyperparams import load_data_config, load_model_config
    from ssft.train.model_loader import load_tokenizer
    from ssft.utils.paths import PROCESSED_DATA_ROOT

    data_cfg = load_data_config(args.config)
    model_cfg = load_model_config(args.model_config)
    tokenizer = load_tokenizer(model_cfg)

    variant_slug = data_cfg.get("dataset_variant_slug", "dataset")
    processed_dir = Path(args.output_dir) if args.output_dir else (PROCESSED_DATA_ROOT / variant_slug)
    input_override = Path(args.input) if args.input else None

    result = prepare_text_splits(data_cfg, tokenizer.eos_token, processed_dir, input_path_override=input_override)
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_train(args) -> int:
    from ssft.train.hyperparams import resolve_run_config
    from ssft.train.train_clm_lora import run_training
    from ssft.utils.paths import ADAPTERS_ROOT

    resolved = resolve_run_config(
        args.model_config, args.method_config, args.data_config, args.training_config,
        seed=args.seed, input_path_override=args.input,
    )
    result = run_training(resolved, adapters_root=ADAPTERS_ROOT, command=" ".join(sys.argv))
    print(f"status={result.status} run_id={result.run_id} output_dir={result.output_dir}")
    return 0 if result.status == "completed" else 1


def cmd_train_experiment(args) -> int:
    from ssft.train.hyperparams import resolve_experiment_config
    from ssft.train.train_clm_lora import run_training
    from ssft.utils.paths import ADAPTERS_ROOT

    resolved = resolve_experiment_config(args.experiment_config, seed=args.seed, input_path_override=args.input)
    result = run_training(resolved, adapters_root=ADAPTERS_ROOT, command=" ".join(sys.argv))
    print(f"status={result.status} run_id={result.run_id} output_dir={result.output_dir}")
    return 0 if result.status == "completed" else 1


def cmd_sweep(args) -> int:
    from ssft.train.sweep_runner import run_sweep

    rows = run_sweep(
        args.sweep_config, dry_run=args.dry_run, max_runs=args.max_runs,
        skip_existing=args.skip_existing, resume_failed=args.resume_failed,
        fail_fast=args.fail_fast, command=" ".join(sys.argv),
    )
    if args.dry_run:
        return 0
    n_failed = sum(1 for r in rows if r["status"] in ("failed", "oom"))
    print(f"{len(rows)} run(s) attempted, {n_failed} failed/oom.")
    return 1 if n_failed and args.fail_fast else 0


def _load_run_context(run_dir: Path) -> dict:
    from ssft.utils.yaml_utils import load_yaml

    resolved = load_yaml(run_dir / "resolved_config.yaml")
    metrics, split_manifest, env, gpu, timestamps = {}, {}, {}, {}, {}
    if (run_dir / "metrics.json").exists():
        metrics = json.loads((run_dir / "metrics.json").read_text())
    if (run_dir / "split_manifest.json").exists():
        split_manifest = json.loads((run_dir / "split_manifest.json").read_text())
    if (run_dir / "environment.json").exists():
        env = json.loads((run_dir / "environment.json").read_text())
    if (run_dir / "gpu.json").exists():
        gpu = json.loads((run_dir / "gpu.json").read_text())
    if (run_dir / "timestamps.json").exists():
        timestamps = json.loads((run_dir / "timestamps.json").read_text())
    command = (run_dir / "command.txt").read_text().strip() if (run_dir / "command.txt").exists() else ""
    return {
        "resolved": resolved,
        "metrics": metrics,
        "data_manifest_info": {"per_source_split_sizes": split_manifest.get("sizes", {})},
        "env": env,
        "gpu": gpu,
        "command": command,
        "timestamps": timestamps,
    }


def cmd_evaluate(args) -> int:
    from ssft.eval.compare_base_adapter import run_comparison
    from ssft.eval.eval_instruction_sanity import run_instruction_sanity
    from ssft.eval.reports import build_run_report

    run_dir = Path(args.run_dir)
    comparison = run_comparison(run_dir)
    sanity = run_instruction_sanity(run_dir)
    ctx = _load_run_context(run_dir)
    build_run_report(
        run_dir=run_dir, resolved=ctx["resolved"], metrics=ctx["metrics"],
        data_manifest_info=ctx["data_manifest_info"], env=ctx["env"], gpu=ctx["gpu"],
        git_commit=ctx["env"].get("git_commit"), command=ctx["command"],
        start_time=ctx["timestamps"].get("start"), end_time=ctx["timestamps"].get("end"),
        run_id=ctx["metrics"].get("run_id", run_dir.name), comparison=comparison, sanity=sanity,
    )
    print(f"Evaluation complete. report.md updated at {run_dir / 'report.md'}")
    return 0


def cmd_compare_base_adapter(args) -> int:
    from ssft.eval.compare_base_adapter import run_comparison
    from ssft.eval.reports import build_run_report

    run_dir = Path(args.run_dir)
    comparison = run_comparison(run_dir)
    ctx = _load_run_context(run_dir)
    build_run_report(
        run_dir=run_dir, resolved=ctx["resolved"], metrics=ctx["metrics"],
        data_manifest_info=ctx["data_manifest_info"], env=ctx["env"], gpu=ctx["gpu"],
        git_commit=ctx["env"].get("git_commit"), command=ctx["command"],
        start_time=ctx["timestamps"].get("start"), end_time=ctx["timestamps"].get("end"),
        run_id=ctx["metrics"].get("run_id", run_dir.name), comparison=comparison,
    )
    print(f"Comparison complete. report.md updated at {run_dir / 'report.md'}")
    return 0


def cmd_summarize_sweep(args) -> int:
    from ssft.eval.reports import build_sweep_summary

    sweep_dir = Path(args.sweep_dir)
    rows = []
    for hp_path in sorted(sweep_dir.rglob("hyperparams.json")):
        run_dir = hp_path.parent
        hp = json.loads(hp_path.read_text())
        metrics = {}
        if (run_dir / "metrics.json").exists():
            metrics = json.loads((run_dir / "metrics.json").read_text())
        rows.append({
            "run_id": hp.get("run_id"), "status": metrics.get("status", "unknown"),
            "model_id_slug": hp.get("model_id_slug"), "method_id_slug": hp.get("method_id_slug"),
            "dataset_variant_slug": hp.get("dataset_variant_slug"),
            "split_strategy_slug": hp.get("split_strategy_slug"), "training_slug": hp.get("training_slug"),
            "seed": hp.get("seed"), "num_train_epochs": hp.get("num_train_epochs"),
            "per_device_train_batch_size": hp.get("per_device_train_batch_size"),
            "gradient_accumulation_steps": hp.get("gradient_accumulation_steps"),
            "effective_batch_size": hp.get("effective_batch_size"), "learning_rate": hp.get("learning_rate"),
            "max_seq_length": hp.get("max_seq_length"), "lora_r": hp.get("lora_r"),
            "lora_alpha": hp.get("lora_alpha"), "lora_dropout": hp.get("lora_dropout"),
            "output_dir": hp.get("output_dir"), "train_loss": metrics.get("train_loss"),
            "eval_loss": metrics.get("eval_loss"), "test_loss": metrics.get("test_loss"),
            "train_perplexity": metrics.get("train_perplexity"),
            "eval_perplexity": metrics.get("eval_perplexity"), "test_perplexity": metrics.get("test_perplexity"),
            "failure_reason": metrics.get("failure_reason"),
        })
    build_sweep_summary(rows, sweep_dir)
    print(f"Rebuilt sweep summary for {len(rows)} run(s) under {sweep_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ssft", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect-repo", help="Inspect the repo + KB without loading any model.")
    p.add_argument("--input", default=None, help="Path to kb_full.jsonl (defaults to <repo_root>/kb_full.jsonl).")
    p.set_defaults(func=cmd_inspect_repo)

    p = sub.add_parser("inspect-env", help="Print Python/torch/transformers/peft/GPU environment info.")
    p.set_defaults(func=cmd_inspect_env)

    p = sub.add_parser("prepare-kb", help="Convert + split the KB into canonical text (no model weights loaded).")
    p.add_argument("--input", required=True, help="Path to kb_full.jsonl")
    p.add_argument("--config", required=True, help="Path to a configs/data/*.yaml file")
    p.add_argument("--model-config", default=None, help="Model config to source eos_token from (default: Qwen2.5-14B base)")
    p.add_argument("--output-dir", default=None, help="Override the processed-data output directory")
    p.set_defaults(func=cmd_prepare_kb)

    p = sub.add_parser("train", help="Run a single training run from 4 explicit config files.")
    p.add_argument("--model-config", required=True)
    p.add_argument("--method-config", required=True)
    p.add_argument("--data-config", required=True)
    p.add_argument("--training-config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--input", default=None, help="Override data.input_path (e.g. path to kb_full.jsonl)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("train-experiment", help="Run a single training run from one experiment config bundle.")
    p.add_argument("--experiment-config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--input", default=None)
    p.set_defaults(func=cmd_train_experiment)

    p = sub.add_parser("sweep", help="Expand + run a sweep config (many model/method/data/hyperparameter combinations).")
    p.add_argument("--sweep-config", required=True)
    p.add_argument("--dry-run", action="store_true", help="Only print the planned run table; run nothing.")
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--skip-existing", type=_str2bool, default=True)
    p.add_argument("--resume-failed", type=_str2bool, default=False)
    p.add_argument("--fail-fast", action="store_true")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("evaluate", help="Run perplexity + cloze + instruction-sanity eval for a run, refresh report.md.")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("compare-base-adapter", help="Run perplexity + cloze base-vs-adapter comparison for a run.")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_compare_base_adapter)

    p = sub.add_parser("summarize-sweep", help="Rebuild sweep_summary.csv/json from a sweep's run directories.")
    p.add_argument("--sweep-dir", required=True)
    p.set_defaults(func=cmd_summarize_sweep)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare-kb" and args.model_config is None:
        args.model_config = _default_model_config()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
