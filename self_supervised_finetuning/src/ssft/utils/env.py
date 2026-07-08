"""environment.json snapshot: interpreter, platform, and key package versions."""
from __future__ import annotations

import platform
import sys
from typing import Any


def _version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
    except ImportError:
        return None
    return getattr(mod, "__version__", "unknown")


def snapshot() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "peft": _version("peft"),
            "accelerate": _version("accelerate"),
            "bitsandbytes": _version("bitsandbytes"),
            "datasets": _version("datasets"),
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
        },
    }
