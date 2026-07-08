"""Convert a generic web-corpus JSONL into CLM training records.

No real web corpus ships with this repo yet (all web data configs have
`input_path: null`) — this module is deliberately generic/defensive: it expects one
JSON object per line with `source_url` (or `document_id`) and a `text` field, and
fails loudly if no input path is supplied rather than silently doing nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

from ssft.data.text_formatters import TextRecord, append_eos, normalize_whitespace


def load_web_jsonl(path: Path | None) -> list[dict]:
    if path is None:
        raise ValueError(
            "web data.input_path is not set — pass --web-input-path or set data.input_path "
            "(or data.sources.web.input_path for kb_web_mixed) before running; no web corpus "
            "ships with this repo."
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Web corpus file not found: {path}")
    rows = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON — {e}") from e
    if not rows:
        raise ValueError(f"Web corpus file {path} contained no rows")
    return rows


def row_to_text_record(
    row: dict,
    eos_token: str,
    text_format: str = "cleaned_web_text",
    min_chars_per_doc: int = 0,
) -> TextRecord | None:
    doc_id = row.get("document_id") or row.get("source_url")
    if not doc_id:
        raise ValueError(f"web row missing both document_id and source_url: {row!r}")
    group_key = row.get("source_url") or row.get("document_id")

    if text_format == "raw_json":
        body = json.dumps(row, sort_keys=True)
    else:
        text = row.get("text", "")
        body = normalize_whitespace(text)

    if len(body) < min_chars_per_doc:
        return None

    text = append_eos(body, eos_token)
    return TextRecord(
        record_id=f"web-doc-{doc_id}",
        source_type="web",
        text=text,
        group_key=group_key,
        metadata={"document_id": row.get("document_id"), "source_url": row.get("source_url")},
    )


def convert_web_file(
    path: Path | None,
    eos_token: str,
    text_format: str = "cleaned_web_text",
    min_chars_per_doc: int = 0,
    deduplicate_exact_text: bool = True,
) -> list[TextRecord]:
    rows = load_web_jsonl(path)
    records = []
    seen_texts: set[str] = set()
    for row in rows:
        record = row_to_text_record(row, eos_token, text_format, min_chars_per_doc)
        if record is None:
            continue
        if deduplicate_exact_text:
            if record.text in seen_texts:
                continue
            seen_texts.add(record.text)
        records.append(record)
    return records
