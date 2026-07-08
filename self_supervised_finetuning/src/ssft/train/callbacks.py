"""Trainer callbacks: crash-safe JSONL step logger + periodic GPU memory snapshot."""
from __future__ import annotations

from pathlib import Path

from transformers import TrainerCallback

from ssft.utils.gpu import snapshot as gpu_snapshot
from ssft.utils.logging_utils import JsonlWriter


class JsonlLoggerCallback(TrainerCallback):
    def __init__(self, log_path: Path):
        self.writer = JsonlWriter(log_path)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        record = dict(logs)
        record["step"] = state.global_step
        record["epoch"] = state.epoch
        self.writer.write(record)


class GpuMemoryCallback(TrainerCallback):
    def __init__(self, log_path: Path, every_n_steps: int = 50):
        self.writer = JsonlWriter(log_path)
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_n_steps > 0 and state.global_step % self.every_n_steps == 0:
            self.writer.write({"step": state.global_step, "event": "gpu_snapshot", "gpu": gpu_snapshot()})
