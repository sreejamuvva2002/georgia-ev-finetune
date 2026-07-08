"""Company-level split has no leakage; split ratios are approximately correct."""
from ssft.data.splitters import group_by_company, train_all
from ssft.data.text_formatters import TextRecord


def _make_records(n_companies, extra_dupes=0):
    records = []
    for i in range(n_companies):
        company = f"Company{i}"
        records.append(TextRecord(
            record_id=f"kb-row-{i}", source_type="kb", text=f"record {i}", group_key=company,
            metadata={
                "EV / Battery Relevant": "Yes" if i % 2 == 0 else "No",
                "Category": "Tier 1" if i % 3 == 0 else "Tier 2/3",
                "Classification Method": "Supplier",
            },
        ))
    # simulate the real KB's duplicate-company rows: extra rows for the same group_key
    for i in range(extra_dupes):
        records.append(TextRecord(
            record_id=f"kb-dup-{i}", source_type="kb", text="dup", group_key=f"Company{i}",
            metadata={"EV / Battery Relevant": "Yes", "Category": "Tier 1", "Classification Method": "Supplier"},
        ))
    return records


def test_no_company_appears_in_more_than_one_split():
    records = _make_records(193, extra_dupes=9)  # mirrors the real KB: 205 rows / 193 companies
    split_records, report = group_by_company(
        records, seed=42, stratify_fields=["EV / Battery Relevant", "Category", "Classification Method"],
    )
    assert report.overlap_train_validation == []
    assert report.overlap_train_test == []
    assert report.overlap_validation_test == []

    train_companies = {r.group_key for r in split_records["train"]}
    val_companies = {r.group_key for r in split_records["validation"]}
    test_companies = {r.group_key for r in split_records["test"]}
    assert train_companies.isdisjoint(val_companies)
    assert train_companies.isdisjoint(test_companies)
    assert val_companies.isdisjoint(test_companies)

    # every duplicate-company row must land in the SAME split as its sibling rows
    for i in range(9):
        company = f"Company{i}"
        splits_for_company = {
            split for split, recs in split_records.items() if any(r.group_key == company for r in recs)
        }
        assert len(splits_for_company) == 1


def test_split_ratios_approximately_correct():
    records = _make_records(193)
    _, report = group_by_company(records, seed=42)
    assert report.n_groups_train + report.n_groups_validation + report.n_groups_test == 193
    assert 140 <= report.n_groups_train <= 170  # ~80% of 193 = 154.4
    assert 8 <= report.n_groups_validation <= 30
    assert 8 <= report.n_groups_test <= 30


def test_deterministic_given_seed():
    records = _make_records(50)
    split1, _ = group_by_company(records, seed=7)
    split2, _ = group_by_company(records, seed=7)
    assert {r.group_key for r in split1["train"]} == {r.group_key for r in split2["train"]}
    assert {r.group_key for r in split1["test"]} == {r.group_key for r in split2["test"]}


def test_different_seeds_can_produce_different_splits():
    records = _make_records(50)
    split1, _ = group_by_company(records, seed=1)
    split2, _ = group_by_company(records, seed=2)
    assert {r.group_key for r in split1["train"]} != {r.group_key for r in split2["train"]}


def test_train_all_puts_everything_in_train_and_warns():
    records = _make_records(10)
    split_records, report = train_all(records)
    assert len(split_records["train"]) == 10
    assert split_records["validation"] == []
    assert split_records["test"] == []
    assert report.warning is not None
    assert "cannot prove generalization" in report.warning
