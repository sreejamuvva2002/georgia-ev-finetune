"""Convert kb_full.jsonl rows into canonical structured-text CLM training records.

Every column is included, values are preserved exactly (only whitespace is
normalized), missing/blank/null values become "Not specified", and no facts are
added or paraphrased. This is plain text formatting for next-token prediction —
never Q&A, never chat.
"""
from __future__ import annotations

import json
from pathlib import Path

from ssft.data.schemas import KB_EXPECTED_COLUMNS, validate_columns
from ssft.data.text_formatters import TextRecord, append_eos, coalesce_missing, render_template

KB_TEMPLATE = """<record>
Dataset: Georgia EV KB
Row ID: {row_id}
Company: {Company}
Category: {Category}
Industry Group: {Industry Group}
Location: {Location}
Address: {Address}
Latitude: {Latitude}
Longitude: {Longitude}
Primary Facility Type: {Primary Facility Type}
EV Supply Chain Role: {EV Supply Chain Role}
Primary OEMs: {Primary OEMs}
Supplier or Affiliation Type: {Supplier or Affiliation Type}
Employment: {Employment}
Product or Service: {Product / Service}
EV or Battery Relevant: {EV / Battery Relevant}
Classification Method: {Classification Method}
</record>"""


def load_kb_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"KB input file not found: {path}")
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
        raise ValueError(f"KB input file {path} contained no rows")
    return rows


def row_to_text_record(row: dict, eos_token: str, missing_value_token: str = "Not specified") -> TextRecord:
    validate_columns(row, KB_EXPECTED_COLUMNS, record_desc=f"KB row_id={row.get('row_id')!r}")
    fields = {col: coalesce_missing(row[col], missing_value_token) for col in KB_EXPECTED_COLUMNS}
    text = render_template(KB_TEMPLATE, fields)
    text = append_eos(text, eos_token)
    company = coalesce_missing(row["Company"], missing_value_token)
    return TextRecord(
        record_id=f"kb-row-{row.get('row_id')}",
        source_type="kb",
        text=text,
        group_key=company,
        metadata={col: row.get(col) for col in KB_EXPECTED_COLUMNS},
    )


def convert_kb_file(path: Path, eos_token: str, missing_value_token: str = "Not specified") -> list[TextRecord]:
    rows = load_kb_jsonl(path)
    return [row_to_text_record(row, eos_token, missing_value_token) for row in rows]
