"""No training example anywhere contains Q&A/chat-style keys. This is CLM-only:
next-token prediction on plain text, never instruction/answer/message pairs."""
import pytest

from ssft.data.kb_converter import row_to_text_record
from ssft.data.schemas import FORBIDDEN_QA_KEYS, assert_no_qa_fields

SAMPLE_ROW = {
    "row_id": 1, "Company": "Acme Corp", "Category": "Tier 1",
    "Industry Group": "Transportation Equipment", "Location": "Atlanta, Fulton County",
    "Address": "123 Main St", "Latitude": 33.7, "Longitude": -84.4,
    "Primary Facility Type": "Manufacturing Plant", "EV Supply Chain Role": "General Automotive",
    "Primary OEMs": "Multiple OEMs", "Supplier or Affiliation Type": "Automotive supply chain participant",
    "Employment": 100, "Product / Service": "Widgets", "EV / Battery Relevant": "Indirect",
    "Classification Method": "Supplier",
}


def test_assert_no_qa_fields_passes_for_clean_tokenized_example():
    assert_no_qa_fields({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]})


@pytest.mark.parametrize("key", sorted(FORBIDDEN_QA_KEYS))
def test_assert_no_qa_fields_rejects_every_forbidden_key(key):
    with pytest.raises(ValueError):
        assert_no_qa_fields({key: "value"})


def test_assert_no_qa_fields_recurses_into_nested_structures():
    with pytest.raises(ValueError):
        assert_no_qa_fields({"outer": {"messages": [{"role": "user", "content": "hi"}]}})


def test_assert_no_qa_fields_is_case_insensitive():
    with pytest.raises(ValueError):
        assert_no_qa_fields({"Instruction": "do something"})


def test_kb_text_record_metadata_has_no_qa_fields():
    record = row_to_text_record(SAMPLE_ROW, eos_token="<|endoftext|>")
    assert_no_qa_fields(record.metadata)


def test_kb_record_text_is_plain_structured_text_not_chat():
    # "Role" legitimately appears as part of the "EV Supply Chain Role" KB column
    # label, so check for actual chat-format markers, not the bare substring.
    record = row_to_text_record(SAMPLE_ROW, eos_token="<|endoftext|>")
    assert record.text.startswith("<record>")
    assert '"role":' not in record.text
    assert '"messages"' not in record.text
    assert not record.text.lstrip().startswith(("system:", "user:", "assistant:"))
