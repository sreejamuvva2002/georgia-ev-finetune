"""Crash-safe JSONL writer (train_log.jsonl) + console logging setup."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


class JsonlWriter:
    """Appends one JSON object per line, flushing on every write so a crash mid-run
    never loses already-logged steps."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()


def configure_console_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_run_logger(output_dir: Path, name: str = "ssft") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        configure_console_logging()
    return logger
