"""Group-based (leakage-free) split strategies.

KB records are split by Company, web records by source_url/document_id — always
whole-group, never row/chunk-level, so no group can ever straddle train/val/test.
Zero overlap is true by construction; `_compute_leakage_report` still explicitly
verifies and persists the proof rather than trusting construction alone.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ssft.data.text_formatters import TextRecord

SPLIT_NAMES = ["train", "validation", "test"]


@dataclass
class LeakageReport:
    strategy: str
    group_key_field: str
    n_groups_train: int
    n_groups_validation: int
    n_groups_test: int
    n_records_train: int
    n_records_validation: int
    n_records_test: int
    overlap_train_validation: list
    overlap_train_test: list
    overlap_validation_test: list
    stratification_summary: dict
    warning: Optional[str] = None

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def _allocate_counts(n: int, ratios: dict) -> dict:
    """Largest-remainder allocation of n items across SPLIT_NAMES per `ratios`."""
    raw = {name: n * ratios.get(name, 0.0) for name in SPLIT_NAMES}
    counts = {name: int(raw[name]) for name in SPLIT_NAMES}
    remainder = n - sum(counts.values())
    order = sorted(SPLIT_NAMES, key=lambda name: (-(raw[name] - counts[name]), SPLIT_NAMES.index(name)))
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _stratify_label(records: list[TextRecord], stratify_fields: Optional[list[str]]) -> tuple:
    if not stratify_fields:
        return ("__all__",)
    first = records[0]
    return tuple(str(first.metadata.get(f, "")) for f in stratify_fields)


def _compute_leakage_report(
    strategy: str,
    group_key_field: str,
    assignment: dict[str, str],
    groups: dict[str, list[TextRecord]],
    stratify_fields: Optional[list[str]],
    warning: Optional[str] = None,
) -> LeakageReport:
    split_group_sets = {name: set() for name in SPLIT_NAMES}
    for key, split_name in assignment.items():
        split_group_sets[split_name].add(key)

    overlap_tv = sorted(split_group_sets["train"] & split_group_sets["validation"])
    overlap_tt = sorted(split_group_sets["train"] & split_group_sets["test"])
    overlap_vt = sorted(split_group_sets["validation"] & split_group_sets["test"])
    if overlap_tv or overlap_tt or overlap_vt:
        raise AssertionError(
            f"Group leakage detected under strategy={strategy}: "
            f"train∩validation={overlap_tv} train∩test={overlap_tt} validation∩test={overlap_vt}"
        )

    strat_summary: dict = defaultdict(lambda: {name: 0 for name in SPLIT_NAMES})
    for key, split_name in assignment.items():
        label = _stratify_label(groups[key], stratify_fields)
        strat_summary["|".join(label)][split_name] += 1

    return LeakageReport(
        strategy=strategy,
        group_key_field=group_key_field,
        n_groups_train=len(split_group_sets["train"]),
        n_groups_validation=len(split_group_sets["validation"]),
        n_groups_test=len(split_group_sets["test"]),
        n_records_train=sum(len(groups[k]) for k in split_group_sets["train"]),
        n_records_validation=sum(len(groups[k]) for k in split_group_sets["validation"]),
        n_records_test=sum(len(groups[k]) for k in split_group_sets["test"]),
        overlap_train_validation=overlap_tv,
        overlap_train_test=overlap_tt,
        overlap_validation_test=overlap_vt,
        stratification_summary=dict(strat_summary),
        warning=warning,
    )


def _group_split(
    records: list[TextRecord],
    seed: int,
    ratios: dict,
    stratify_fields: Optional[list[str]] = None,
    strategy_name: str = "group_split",
    group_key_field: str = "group_key",
) -> tuple[dict[str, list[TextRecord]], LeakageReport]:
    if abs(sum(ratios.get(n, 0.0) for n in SPLIT_NAMES) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0, got {ratios}")

    groups: dict[str, list[TextRecord]] = defaultdict(list)
    for r in records:
        groups[r.group_key].append(r)
    group_keys = sorted(groups.keys())

    buckets: dict[tuple, list[str]] = defaultdict(list)
    for key in group_keys:
        buckets[_stratify_label(groups[key], stratify_fields)].append(key)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for label in sorted(buckets.keys()):
        bucket_keys = list(buckets[label])
        rng.shuffle(bucket_keys)
        counts = _allocate_counts(len(bucket_keys), ratios)
        idx = 0
        for split_name in SPLIT_NAMES:
            n = counts[split_name]
            for key in bucket_keys[idx: idx + n]:
                assignment[key] = split_name
            idx += n

    split_records: dict[str, list[TextRecord]] = {name: [] for name in SPLIT_NAMES}
    for key in group_keys:
        split_records[assignment[key]].extend(groups[key])

    report = _compute_leakage_report(strategy_name, group_key_field, assignment, groups, stratify_fields)
    return split_records, report


def group_by_company(
    records: list[TextRecord],
    seed: int = 42,
    ratios: Optional[dict] = None,
    stratify_fields: Optional[list[str]] = None,
) -> tuple[dict[str, list[TextRecord]], LeakageReport]:
    ratios = ratios or {"train": 0.80, "validation": 0.10, "test": 0.10}
    return _group_split(
        records, seed, ratios, stratify_fields,
        strategy_name="group_by_company", group_key_field="Company",
    )


def group_by_source(
    records: list[TextRecord],
    seed: int = 42,
    ratios: Optional[dict] = None,
    stratify_fields: Optional[list[str]] = None,
) -> tuple[dict[str, list[TextRecord]], LeakageReport]:
    ratios = ratios or {"train": 0.80, "validation": 0.10, "test": 0.10}
    return _group_split(
        records, seed, ratios, stratify_fields,
        strategy_name="group_by_source", group_key_field="source_url",
    )


def train_all(records: list[TextRecord]) -> tuple[dict[str, list[TextRecord]], LeakageReport]:
    """100% train, no held-out data. eval_dataset should be set to the SAME train
    dataset by the caller (in-sample loss only) — this function never fabricates a
    held-out split. Report carries the mandatory non-generalization warning."""
    groups: dict[str, list[TextRecord]] = defaultdict(list)
    for r in records:
        groups[r.group_key].append(r)
    assignment = {key: "train" for key in groups}
    warning = (
        "This run trains on all KB rows. It can test in-sample absorption/memorization "
        "but cannot prove generalization."
    )
    report = _compute_leakage_report("train_all", "Company", assignment, groups, None, warning=warning)
    split_records = {name: [] for name in SPLIT_NAMES}
    for r in records:
        split_records["train"].append(r)
    return split_records, report
