"""Schema constants + the canonical no-Q&A guard.

This module is the single source of truth both for what a valid KB row looks like
and for what a training example must NEVER look like. `assert_no_qa_fields` is the
one place `tests/test_no_qa_format.py` needs to import to prove the whole pipeline
never produces supervised/chat-style examples.
"""
from __future__ import annotations

from typing import Any

KB_EXPECTED_COLUMNS = [
    "row_id",
    "Company",
    "Category",
    "Industry Group",
    "Location",
    "Address",
    "Latitude",
    "Longitude",
    "Primary Facility Type",
    "EV Supply Chain Role",
    "Primary OEMs",
    "Supplier or Affiliation Type",
    "Employment",
    "Product / Service",
    "EV / Battery Relevant",
    "Classification Method",
]

# Generic expectation for a not-yet-existing web corpus JSONL: one text document per
# row, grouped by source so a whole page/document never splits across train/val/test.
WEB_EXPECTED_COLUMNS = [
    "document_id",
    "source_url",
    "text",
]

# Any of these keys appearing anywhere in a built example means something regressed
# toward supervised/instruction/chat formatting — this framework is CLM-only.
FORBIDDEN_QA_KEYS = {
    "role", "messages", "prompt", "completion", "instruction",
    "answer", "response", "chosen", "rejected", "system",
}


def validate_columns(row: dict, expected: list[str], record_desc: str = "row") -> None:
    missing = [c for c in expected if c not in row]
    if missing:
        raise ValueError(
            f"{record_desc} is missing expected column(s) {missing}. "
            f"Present columns: {sorted(row.keys())}"
        )


def assert_no_qa_fields(obj: Any, _path: str = "$") -> None:
    """Recursively walk `obj` (dict/list/str/...) and raise if any dict key matches
    FORBIDDEN_QA_KEYS. Called on every built training example."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and key.lower() in FORBIDDEN_QA_KEYS:
                raise ValueError(
                    f"Q&A/chat-style key '{key}' found at {_path}.{key} — this framework is "
                    "CLM-only (self-supervised continued pretraining), not supervised "
                    "fine-tuning. No instruction/Q&A/chat fields are allowed anywhere in the "
                    "data pipeline."
                )
            assert_no_qa_fields(value, f"{_path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            assert_no_qa_fields(item, f"{_path}[{i}]")
