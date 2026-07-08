"""Canonical KB text formatting includes every column; missing values become
"Not specified"; values are preserved exactly (only whitespace is normalized)."""
from ssft.data.kb_converter import row_to_text_record
from ssft.data.text_formatters import coalesce_missing, normalize_whitespace

EOS = "<|endoftext|>"

SAMPLE_ROW = {
    "row_id": 1, "Company": "Acme  Corp", "Category": "Tier 1",
    "Industry Group": "Transportation Equipment", "Location": "Atlanta, Fulton County",
    "Address": "123 Main St", "Latitude": 33.7, "Longitude": -84.4,
    "Primary Facility Type": "Manufacturing Plant", "EV Supply Chain Role": "General Automotive",
    "Primary OEMs": "Multiple OEMs", "Supplier or Affiliation Type": "Automotive supply chain participant",
    "Employment": 100, "Product / Service": "Widgets and gadgets", "EV / Battery Relevant": None,
    "Classification Method": "Supplier",
}

EXPECTED_LABELS = {
    "row_id": "Row ID:", "Company": "Company:", "Category": "Category:",
    "Industry Group": "Industry Group:", "Location": "Location:", "Address": "Address:",
    "Latitude": "Latitude:", "Longitude": "Longitude:",
    "Primary Facility Type": "Primary Facility Type:", "EV Supply Chain Role": "EV Supply Chain Role:",
    "Primary OEMs": "Primary OEMs:", "Supplier or Affiliation Type": "Supplier or Affiliation Type:",
    "Employment": "Employment:", "Product / Service": "Product or Service:",
    "EV / Battery Relevant": "EV or Battery Relevant:", "Classification Method": "Classification Method:",
}


def test_record_includes_every_column():
    record = row_to_text_record(SAMPLE_ROW, EOS)
    for label in EXPECTED_LABELS.values():
        assert label in record.text, f"missing label {label!r} in:\n{record.text}"


def test_missing_value_becomes_not_specified():
    record = row_to_text_record(SAMPLE_ROW, EOS)
    assert "EV or Battery Relevant: Not specified" in record.text


def test_values_preserved_exactly_only_whitespace_normalized():
    record = row_to_text_record(SAMPLE_ROW, EOS)
    assert "Company: Acme Corp" in record.text  # double space collapsed to one
    assert "Widgets and gadgets" in record.text  # value not paraphrased/altered


def test_eos_appended_after_record():
    record = row_to_text_record(SAMPLE_ROW, EOS)
    assert record.text.endswith(EOS)


def test_group_key_is_company():
    record = row_to_text_record(SAMPLE_ROW, EOS)
    assert record.group_key == "Acme Corp"


def test_coalesce_missing():
    assert coalesce_missing(None) == "Not specified"
    assert coalesce_missing("") == "Not specified"
    assert coalesce_missing("   ") == "Not specified"
    assert coalesce_missing(float("nan")) == "Not specified"
    assert coalesce_missing("hello  world") == "hello world"
    assert coalesce_missing(100) == "100"


def test_normalize_whitespace():
    assert normalize_whitespace("a\n\nb   c") == "a b c"
    assert normalize_whitespace("  leading and trailing  ") == "leading and trailing"
