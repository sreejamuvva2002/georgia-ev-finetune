"""gpu.json snapshot + an OOM-safe wrapper used by the sweep runner."""
from __future__ import annotations

import functools
import subprocess
from typing import Any, Callable


class OomError(RuntimeError):
    """Raised (in place of the original CUDA OOM) so callers can distinguish OOM
    from other training failures without importing torch."""


def _nvidia_smi_snapshot() -> dict | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        idx, name, mem_total, mem_used, mem_free, util, temp = parts
        gpus.append({
            "index": int(idx),
            "name": name,
            "memory_total_mib": int(mem_total),
            "memory_used_mib": int(mem_used),
            "memory_free_mib": int(mem_free),
            "utilization_pct": int(util),
            "temperature_c": int(temp),
        })
    return {"gpus": gpus}


def snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"nvidia_smi": _nvidia_smi_snapshot(), "torch_cuda": None}
    try:
        import torch
        if torch.cuda.is_available():
            result["torch_cuda"] = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_bytes": torch.cuda.memory_allocated(),
                "memory_reserved_bytes": torch.cuda.memory_reserved(),
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        else:
            result["torch_cuda"] = {"available": False}
    except ImportError:
        pass
    return result


def oom_safe(fn: Callable) -> Callable:
    """Decorator: catches CUDA OOM, empties the cache, and re-raises as OomError so
    the sweep runner can record status="oom" instead of a generic failure."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            is_oom = "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"
            if is_oom:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                raise OomError(str(e)) from e
            raise

    return wrapper
