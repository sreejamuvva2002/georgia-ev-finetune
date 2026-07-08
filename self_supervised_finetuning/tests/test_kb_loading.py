"""KB JSONL loads correctly and all 16 expected columns are present."""
import json

import pytest

from ssft.data.kb_converter import load_kb_jsonl
from ssft.data.schemas import KB_EXPECTED_COLUMNS, validate_columns

SAMPLE_ROW = {
    "row_id": 1, "Company": "Acme Corp", "Category": "Tier 1",
    "Industry Group": "Transportation Equipment", "Location": "Atlanta, Fulton County",
    "Address": "123 Main St", "Latitude": 33.7, "Longitude": -84.4,
    "Primary Facility Type": "Manufacturing Plant", "EV Supply Chain Role": "General Automotive",
    "Primary OEMs": "Multiple OEMs", "Supplier or Affiliation Type": "Automotive supply chain participant",
    "Employment": 100, "Product / Service": "Widgets", "EV / Battery Relevant": "Indirect",
    "Classification Method": "Supplier",
}


def _write_kb(tmp_path, rows):
    path = tmp_path / "kb.jsonl"
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def test_load_kb_jsonl_reads_all_rows(tmp_path):
    path = _write_kb(tmp_path, [SAMPLE_ROW, {**SAMPLE_ROW, "row_id": 2, "Company": "Beta Inc"}])
    rows = load_kb_jsonl(path)
    assert len(rows) == 2
    assert rows[0]["Company"] == "Acme Corp"
    assert rows[1]["Company"] == "Beta Inc"


def test_load_kb_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "kb.jsonl"
    path.write_text(json.dumps(SAMPLE_ROW) + "\n\n\n" + json.dumps({**SAMPLE_ROW, "row_id": 2}) + "\n")
    rows = load_kb_jsonl(path)
    assert len(rows) == 2


def test_load_kb_jsonl_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_kb_jsonl(tmp_path / "missing.jsonl")


def test_all_16_expected_columns_defined():
    assert len(KB_EXPECTED_COLUMNS) == 16
    assert "row_id" in KB_EXPECTED_COLUMNS
    assert "Company" in KB_EXPECTED_COLUMNS
    assert "Product / Service" in KB_EXPECTED_COLUMNS
    assert "EV / Battery Relevant" in KB_EXPECTED_COLUMNS


def test_validate_columns_passes_for_complete_row():
    validate_columns(SAMPLE_ROW, KB_EXPECTED_COLUMNS)  # should not raise


def test_validate_columns_raises_for_missing_column():
    row = dict(SAMPLE_ROW)
    del row["Employment"]
    with pytest.raises(ValueError, match="Employment"):
        validate_columns(row, KB_EXPECTED_COLUMNS)
