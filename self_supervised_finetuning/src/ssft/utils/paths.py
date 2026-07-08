"""Path constants and helpers for the ssft framework.

All output/data paths are rooted at SSFT_ROOT (self_supervised_finetuning/) so nothing this
package writes ever lands outside its own top-level folder.
"""
from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk parents from `start` looking for a `.git` directory. Falls back to the
    ssft package's own repo-relative ancestor if no `.git` is found (e.g. the
    package was copied out of its git repo)."""
    here = (start or Path(__file__).resolve()).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").exists():
            return parent
    # src/ssft/utils/paths.py -> self_supervised_finetuning/ -> repo root
    return Path(__file__).resolve().parents[4]


REPO_ROOT = find_repo_root()
SSFT_ROOT = Path(__file__).resolve().parents[3]
assert SSFT_ROOT.name == "self_supervised_finetuning", SSFT_ROOT

CONFIGS_ROOT = SSFT_ROOT / "configs"
DATA_ROOT = SSFT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw"
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
SPLITS_ROOT = DATA_ROOT / "splits"
MANIFESTS_ROOT = DATA_ROOT / "manifests"
PROBES_ROOT = DATA_ROOT / "probes"

OUTPUTS_ROOT = SSFT_ROOT / "outputs"
ADAPTERS_ROOT = OUTPUTS_ROOT / "adapters"
RUNS_ROOT = OUTPUTS_ROOT / "runs"
METRICS_ROOT = OUTPUTS_ROOT / "metrics"
REPORTS_ROOT = OUTPUTS_ROOT / "reports"
SWEEPS_ROOT = OUTPUTS_ROOT / "sweeps"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_repo_relative(path_str: str) -> Path:
    """Resolve a path given in a config file. Absolute paths pass through; relative
    paths are tried against the current working directory first (the convention used
    by every example command in the spec, e.g. `self_supervised_finetuning/configs/...`
    run from the repo root), then against REPO_ROOT as a fallback."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    repo_candidate = REPO_ROOT / p
    if repo_candidate.exists():
        return repo_candidate
    return cwd_candidate
