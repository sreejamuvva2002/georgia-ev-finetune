"""Best-effort git metadata for the OUTER repo (ssft has no .git of its own)."""
from __future__ import annotations

import subprocess

from ssft.utils.paths import REPO_ROOT


def current_commit(repo_root=None) -> str | None:
    root = repo_root or REPO_ROOT
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def is_dirty(repo_root=None) -> bool | None:
    root = repo_root or REPO_ROOT
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())
