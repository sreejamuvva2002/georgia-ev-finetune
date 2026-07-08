"""Shared low-level text-rendering engine used by both kb_converter and web_converter.

No source-specific knowledge lives here — just whitespace normalization, missing-value
coalescing, template rendering, and eos-token appending, plus the TextRecord type both
converters produce.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


def normalize_whitespace(s: str) -> str:
    """Collapse runs of whitespace (incl. newlines) to a single space and strip ends.
    Does not alter the characters themselves — only spacing — so values are preserved
    exactly, per the "do not paraphrase values" rule."""
    return re.sub(r"\s+", " ", s).strip()


def coalesce_missing(value, missing_token: str = "Not specified") -> str:
    if value is None:
        return missing_token
    if isinstance(value, float):
        # NaN check without importing pandas/numpy here
        if value != value:
            return missing_token
    s = str(value)
    if s.strip() == "":
        return missing_token
    return normalize_whitespace(s)


def render_template(template: str, fields: dict) -> str:
    """Render `template` (a str.format-style template, e.g. "{Company}") against
    `fields`. Missing keys are a programming error (callers must pre-populate every
    placeholder via coalesce_missing) so we let KeyError propagate loudly rather
    than silently drop a field."""
    return template.format_map(fields)


def append_eos(text: str, eos_token: str) -> str:
    if not eos_token:
        raise ValueError("append_eos requires a non-empty eos_token")
    return f"{text}\n{eos_token}"


@dataclass
class TextRecord:
    record_id: str
    source_type: str  # "kb" | "web"
    text: str
    group_key: str
    metadata: dict = field(default_factory=dict)
