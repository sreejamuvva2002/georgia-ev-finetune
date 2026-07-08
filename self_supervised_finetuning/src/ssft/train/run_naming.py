"""Pure functions over a ResolvedConfig: slugs, training_slug, run_id, output paths.

training_slug is ALWAYS computed live from the resolved hyperparameters (epochs,
batch size, grad-accum, learning rate, max_seq_length), never trusted from a
config's literal `training_slug:` field — that field is documentation. This is
required for correctness under sweep overrides: a sweep that overrides
training.num_train_epochs must produce a different training_slug/output path than
the base config's label would suggest, or every override in an epoch/LR/batch sweep
would collide into the same run directory. Verified against every example the spec
gives (tiny_kb_conservative/tiny_kb_memorization/web_default all reproduce their
literal training_slug exactly by this formula; mixed_default's "seqmixed" is
reproduced via the mixed-source special case below).
"""
from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ssft.train.hyperparams import ResolvedConfig


def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unnamed"


def get_model_id_slug(resolved: "ResolvedConfig") -> str:
    return resolved.model_cfg.get("model_id_slug") or slugify(resolved.model_cfg.get("name_or_path", "model"))


def get_method_id_slug(resolved: "ResolvedConfig") -> str:
    return resolved.method_cfg.get("method_id_slug") or slugify(resolved.method_cfg.get("name", "method"))


def get_dataset_variant_slug(resolved: "ResolvedConfig") -> str:
    return resolved.data_cfg.get("dataset_variant_slug") or slugify(resolved.data_cfg.get("source_type", "dataset"))


def get_split_strategy_slug(resolved: "ResolvedConfig") -> str:
    return resolved.data_cfg.get("split_strategy_slug") or slugify(resolved.data_cfg.get("split_strategy", "split"))


def format_lr_slug(lr: float) -> str:
    """0.00008 -> lr8e5 | 0.00005 -> lr5e5 | 0.00010 -> lr1e4 | 0.00015 -> lr15e5."""
    d = Decimal(str(lr))
    exp = -d.as_tuple().exponent
    mantissa = int(d.scaleb(exp))
    while mantissa % 10 == 0 and exp > 0:
        mantissa //= 10
        exp -= 1
    return f"lr{mantissa}e{exp}"


def effective_batch_size(per_device_train_batch_size: int, gradient_accumulation_steps: int, world_size: int = 1) -> int:
    return int(per_device_train_batch_size) * int(gradient_accumulation_steps) * int(world_size)


def _format_epochs(num_train_epochs) -> str:
    f = float(num_train_epochs)
    if f == int(f):
        return str(int(f))
    return str(f).replace(".", "p")


def _seq_slug_component(data_cfg: dict) -> str:
    if data_cfg.get("source_type") == "mixed":
        return "mixed"
    return str(data_cfg.get("max_seq_length"))


def build_training_slug(resolved: "ResolvedConfig") -> str:
    t, d = resolved.training_cfg, resolved.data_cfg
    epochs = _format_epochs(t.get("num_train_epochs", 1))
    bs = int(t.get("per_device_train_batch_size", 1))
    ga = int(t.get("gradient_accumulation_steps", 1))
    ebs = effective_batch_size(bs, ga, t.get("world_size", 1))
    lr = t.get("learning_rate")
    lr_slug = format_lr_slug(lr) if lr is not None else "lrNA"
    seq = _seq_slug_component(d)
    slug = f"ep{epochs}-bs{bs}-ga{ga}-ebs{ebs}-{lr_slug}-seq{seq}"
    if t.get("debug"):
        slug += "-debug"
    return slug


def build_run_id(resolved: "ResolvedConfig", timestamp: str, short_hash: str) -> str:
    parts = [
        get_model_id_slug(resolved),
        get_method_id_slug(resolved),
        get_dataset_variant_slug(resolved),
        get_split_strategy_slug(resolved),
        build_training_slug(resolved),
        f"seed{resolved.seed}",
        f"{timestamp}_{short_hash}",
    ]
    return "__".join(parts)


def run_dir_parent(resolved: "ResolvedConfig", adapters_root: Path) -> Path:
    """Everything up to (excluding) the timestamp_hash leaf — this is what
    find_existing_attempts scans and build_output_dir appends a leaf to."""
    return (
        Path(adapters_root)
        / get_model_id_slug(resolved)
        / get_method_id_slug(resolved)
        / get_dataset_variant_slug(resolved)
        / get_split_strategy_slug(resolved)
        / build_training_slug(resolved)
        / f"seed{resolved.seed}"
    )


def build_output_dir(resolved: "ResolvedConfig", adapters_root: Path, timestamp: str, short_hash: str) -> Path:
    return run_dir_parent(resolved, adapters_root) / f"{timestamp}_{short_hash}"


def find_existing_attempts(resolved: "ResolvedConfig", adapters_root: Path) -> list[tuple[Path, str]]:
    """Newest-first list of (attempt_dir, status) for this exact
    model/method/dataset/split/training_slug/seed combination."""
    parent = run_dir_parent(resolved, adapters_root)
    if not parent.exists():
        return []
    attempts = []
    for child in sorted(parent.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        if (child / "_SUCCESS").exists():
            status = "completed"
        elif (child / "metrics.json").exists():
            try:
                with open(child / "metrics.json") as f:
                    status = json.load(f).get("status", "failed")
            except (json.JSONDecodeError, OSError):
                status = "failed"
        else:
            status = "failed"
        attempts.append((child, status))
    return attempts
