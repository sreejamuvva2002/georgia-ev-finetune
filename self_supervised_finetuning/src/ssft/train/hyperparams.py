"""Load + merge the 4 independent config files (model/method/data/training) into one
ResolvedConfig, and apply sweep-style dotted-path overrides on top.

This is the single funnel `train`, `train-experiment`, and `sweep` all pass through —
no config-resolution logic is duplicated across those three entry points.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from ssft.utils.paths import resolve_repo_relative
from ssft.utils.yaml_utils import load_yaml, set_by_dotted_path

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ResolvedConfig:
    model_cfg: dict
    method_cfg: dict
    data_cfg: dict
    training_cfg: dict
    seed: int
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "model": self.model_cfg,
            "method": self.method_cfg,
            "data": self.data_cfg,
            "training": self.training_cfg,
            "seed": self.seed,
            "provenance": self.provenance,
        }


def _load_section(path: PathLike, section_key: str) -> dict:
    resolved_path = resolve_repo_relative(str(path))
    raw = load_yaml(resolved_path)
    if section_key not in raw:
        raise ValueError(f"{resolved_path}: expected a top-level '{section_key}:' key, got {sorted(raw.keys())}")
    return raw[section_key]


def load_model_config(path: PathLike) -> dict:
    return _load_section(path, "model")


def load_method_config(path: PathLike) -> dict:
    return _load_section(path, "method")


def load_data_config(path: PathLike) -> dict:
    return _load_section(path, "data")


def load_training_config(path: PathLike) -> dict:
    return _load_section(path, "training")


def apply_overrides(sections: dict, overrides: Optional[dict]) -> dict:
    """sections: {"model":..,"method":..,"data":..,"training":..}. `overrides` uses
    dotted keys like "training.num_train_epochs" or "data.max_seq_length" — each is
    applied against whichever top-level section its first path component names."""
    if not overrides:
        return sections
    result = dict(sections)
    for dotted_key, value in overrides.items():
        result = set_by_dotted_path(result, dotted_key, value)
    return result


def validate_resolved_sections(sections: dict) -> None:
    required = {
        "model": ["model_id_slug", "name_or_path"],
        "method": ["method_id_slug", "name"],
        "data": ["dataset_variant_slug", "split_strategy_slug", "split_strategy"],
        "training": [],
    }
    for section, keys in required.items():
        if section not in sections:
            raise ValueError(f"resolved config is missing the '{section}' section")
        for key in keys:
            if key not in sections[section]:
                raise ValueError(f"config section '{section}' is missing required key '{key}'")
    if sections["method"].get("status") == "not_implemented":
        raise NotImplementedError(
            f"method '{sections['method'].get('name')}' "
            f"({sections['method'].get('method_id_slug')}) is a placeholder and is not "
            "implemented yet — choose a different --method-config."
        )


def validate_resolved_config(resolved: ResolvedConfig) -> None:
    validate_resolved_sections({
        "model": resolved.model_cfg, "method": resolved.method_cfg,
        "data": resolved.data_cfg, "training": resolved.training_cfg,
    })


def _apply_input_path_override(data_cfg: dict, input_path_override: Optional[str]) -> dict:
    if not input_path_override:
        return data_cfg
    data_cfg = dict(data_cfg)
    if data_cfg.get("source_type") == "mixed":
        sources = dict(data_cfg.get("sources", {}))
        if "kb" in sources:
            sources["kb"] = dict(sources["kb"])
            sources["kb"]["input_path"] = input_path_override
        data_cfg["sources"] = sources
    else:
        data_cfg["input_path"] = input_path_override
    return data_cfg


def resolve_run_config(
    model_config: PathLike,
    method_config: PathLike,
    data_config: PathLike,
    training_config: PathLike,
    *,
    seed: Optional[int] = None,
    overrides: Optional[dict] = None,
    input_path_override: Optional[str] = None,
) -> ResolvedConfig:
    sections = {
        "model": load_model_config(model_config),
        "method": load_method_config(method_config),
        "data": load_data_config(data_config),
        "training": load_training_config(training_config),
    }
    sections = apply_overrides(sections, overrides)
    sections["data"] = _apply_input_path_override(sections["data"], input_path_override)

    validate_resolved_sections(sections)

    resolved_seed = (
        seed if seed is not None
        else sections["training"].get("seed", sections["data"].get("seed", 42))
    )
    provenance = {
        "model_config_path": str(model_config),
        "method_config_path": str(method_config),
        "data_config_path": str(data_config),
        "training_config_path": str(training_config),
        "overrides": overrides or {},
    }
    return ResolvedConfig(
        model_cfg=sections["model"], method_cfg=sections["method"],
        data_cfg=sections["data"], training_cfg=sections["training"],
        seed=resolved_seed, provenance=provenance,
    )


def resolve_experiment_config(
    experiment_config_path: PathLike,
    *,
    overrides: Optional[dict] = None,
    seed: Optional[int] = None,
    input_path_override: Optional[str] = None,
) -> ResolvedConfig:
    path = resolve_repo_relative(str(experiment_config_path))
    raw = load_yaml(path)
    if "experiment" not in raw:
        raise ValueError(f"{path}: expected a top-level 'experiment:' key")
    exp = raw["experiment"]
    resolved = resolve_run_config(
        exp["model_config"], exp["method_config"], exp["data_config"], exp["training_config"],
        seed=seed if seed is not None else exp.get("seed"),
        overrides=overrides,
        input_path_override=input_path_override,
    )
    provenance = dict(resolved.provenance)
    provenance["experiment_config_path"] = str(path)
    provenance["experiment_name"] = exp.get("experiment_name")
    return ResolvedConfig(
        resolved.model_cfg, resolved.method_cfg, resolved.data_cfg, resolved.training_cfg,
        resolved.seed, provenance,
    )
