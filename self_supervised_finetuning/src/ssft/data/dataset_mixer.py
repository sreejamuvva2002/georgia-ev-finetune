"""Mix already-split, already-tokenized per-source examples at configurable weights.

Mixing happens strictly AFTER each source has been independently split (KB by
Company, web by source_url) and tokenized (respecting each source's own packing
flag) — never on raw text — so mixing can never introduce cross-source leakage.
Only the "train" split is weighted/resampled; validation/test are the union of every
source's own held-out examples, since eval should reflect the true unweighted data.
"""
from __future__ import annotations

import random

SPLIT_NAMES = ["train", "validation", "test"]


def mix_split(
    per_source_examples: dict[str, list],
    sampling_weights: dict[str, float],
    seed: int,
    target_size: int | None = None,
) -> list:
    present = {k: v for k, v in per_source_examples.items() if v}
    if not present:
        return []
    weights = {k: sampling_weights.get(k, 0.0) for k in present}
    if sum(weights.values()) <= 0:
        weights = {k: 1.0 for k in present}
    weight_sum = sum(weights.values())

    total = target_size if target_size is not None else sum(len(v) for v in present.values())
    rng = random.Random(seed)

    mixed = []
    for name, pool in present.items():
        n = round(total * weights[name] / weight_sum)
        if n <= len(pool):
            mixed.extend(rng.sample(pool, n))
        else:
            mixed.extend(pool)
            mixed.extend(rng.choices(pool, k=n - len(pool)))
    rng.shuffle(mixed)
    return mixed


def mix_by_presentations(
    per_source_train: dict[str, list],
    presentations: dict[str, int],
    seed: int,
) -> list:
    """Deterministic exposure control: include EVERY example of each source exactly
    `presentations[source]` times (default 1), then shuffle. Unlike weighted sampling
    this never drops any source's examples and never depends on a fixed epoch size —
    it lets you present, e.g., all web chunks once while presenting each KB record 50x
    (matching the KB-only baseline). Sources absent from `presentations` default to 1.
    """
    rng = random.Random(seed)
    combined: list = []
    for src in sorted(per_source_train.keys()):
        reps = int(presentations.get(src, 1))
        pool = per_source_train[src]
        for _ in range(max(0, reps)):
            combined.extend(pool)
    rng.shuffle(combined)
    return combined


def mix_datasets(
    per_source_split_datasets: dict[str, dict[str, list]],
    sampling_weights: dict[str, float],
    seed: int,
    train_presentations: dict[str, int] | None = None,
) -> dict[str, list]:
    mixed: dict[str, list] = {}
    for split_name in SPLIT_NAMES:
        per_source = {src: sds.get(split_name, []) for src, sds in per_source_split_datasets.items()}
        if split_name == "train":
            if train_presentations:
                mixed[split_name] = mix_by_presentations(per_source, train_presentations, seed)
            else:
                mixed[split_name] = mix_split(per_source, sampling_weights, seed)
        else:
            combined = []
            for src in sorted(per_source.keys()):
                combined.extend(per_source[src])
            mixed[split_name] = combined
    return mixed
