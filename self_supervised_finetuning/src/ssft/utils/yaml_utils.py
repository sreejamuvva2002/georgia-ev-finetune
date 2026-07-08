"""YAML load/dump + dict merge/dotted-path helpers used by config resolution."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path | str) -> dict:
    path = Path(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(data)}")
    return data


def dump_yaml(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, default_flow_style=False)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`. Values in `override` win;
    nested dicts are merged key-by-key, everything else (incl. lists) is replaced."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_by_dotted_path(cfg: dict, dotted_key: str) -> Any:
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"dotted path '{dotted_key}' not found (failed at '{part}')")
        node = node[part]
    return node


def set_by_dotted_path(cfg: dict, dotted_key: str, value: Any) -> dict:
    """Set a dotted-path key on a *copy* of cfg, creating intermediate dicts as needed,
    and return the copy. `cfg` here is the top-level resolved dict with section keys
    like "training"/"data", e.g. set_by_dotted_path(resolved, "training.learning_rate", 8e-5)."""
    result = copy.deepcopy(cfg)
    node = result
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value
    return result
