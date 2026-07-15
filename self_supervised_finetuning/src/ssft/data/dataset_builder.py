"""The single CLM tokenization engine — one code path for KB and web data.

`tokenize_split` branches only on the `packing` flag (KB: pad-via-collator, one
example per record; web: concatenate + fixed-size chunks). `build_datasets` composes
per-source conversion -> split -> tokenize -> (if >1 source) mix, and never builds
anything resembling a Q&A/chat example — `schemas.assert_no_qa_fields` is called on
every example this module produces.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ssft.data import kb_converter, manifests, schemas, splitters, web_converter
from ssft.data.dataset_mixer import mix_datasets
from ssft.data.splitters import LeakageReport
from ssft.data.text_formatters import TextRecord

DEFAULT_RATIOS = {"train": 0.80, "validation": 0.10, "test": 0.10}


def _normalize_sources(data_cfg: dict) -> dict[str, dict]:
    """Return {source_name: source_cfg} regardless of whether data_cfg describes a
    single kb_jsonl/web_corpus source or a "mixed" bundle of sources.*."""
    if data_cfg.get("source_type") == "mixed":
        top_seed = data_cfg.get("seed", 42)
        top_ratios = data_cfg.get("split_ratios", DEFAULT_RATIOS)
        normalized = {}
        for name, src in data_cfg.get("sources", {}).items():
            merged = dict(src)
            merged.setdefault("seed", top_seed)
            merged.setdefault("split_ratios", top_ratios)
            normalized[name] = merged
        return normalized
    name = "kb" if data_cfg.get("source_type") == "kb_jsonl" else "web"
    return {name: dict(data_cfg)}


def build_examples_for_source(
    source_cfg: dict,
    eos_token: str,
    input_path_override: Optional[Path] = None,
) -> tuple[dict[str, list[TextRecord]], LeakageReport]:
    source_type = source_cfg["source_type"]
    input_path = input_path_override or source_cfg.get("input_path")
    missing_value_token = source_cfg.get("missing_value_token", "Not specified")
    seed = source_cfg.get("seed", 42)
    split_strategy = source_cfg.get("split_strategy")
    ratios = source_cfg.get("split_ratios", DEFAULT_RATIOS)

    if source_type == "kb_jsonl":
        if input_path is None:
            raise ValueError(
                "KB source requires data.input_path (or --input on the CLI) to be set — "
                "point it at kb_full.jsonl."
            )
        records = kb_converter.convert_kb_file(Path(input_path), eos_token, missing_value_token)
    elif source_type == "web_corpus":
        records = web_converter.convert_web_file(
            Path(input_path) if input_path else None,
            eos_token,
            text_format=source_cfg.get("text_format", "cleaned_web_text"),
            min_chars_per_doc=source_cfg.get("min_chars_per_doc", 0),
            deduplicate_exact_text=source_cfg.get("deduplicate_exact_text", True),
        )
    else:
        raise ValueError(f"Unknown data.source_type: {source_type!r}")

    if split_strategy == "group_by_company":
        return splitters.group_by_company(
            records, seed=seed, ratios=ratios, stratify_fields=source_cfg.get("stratify_columns"),
        )
    if split_strategy == "group_by_source":
        return splitters.group_by_source(records, seed=seed, ratios=ratios)
    if split_strategy == "train_all":
        return splitters.train_all(records)
    raise ValueError(f"Unknown data.split_strategy: {split_strategy!r}")


def tokenize_split(
    records: list[TextRecord],
    tokenizer: Any,
    max_seq_length: int,
    packing: bool,
    chunk_stride: int = 0,
) -> list[dict]:
    if not records:
        return []

    examples: list[dict] = []
    if not packing:
        for r in records:
            enc = tokenizer(r.text, truncation=True, max_length=max_seq_length, padding=False)
            example = {
                "input_ids": list(enc["input_ids"]),
                "attention_mask": list(enc["attention_mask"]),
                "labels": list(enc["input_ids"]),
            }
            schemas.assert_no_qa_fields(example)
            examples.append(example)
        return examples

    # packing=True: concatenate token streams across all records in this split (each
    # record already ends with an eos_token marking the document boundary), then chunk
    # into fixed-size blocks. A trailing partial block is dropped rather than padded.
    all_ids: list[int] = []
    for r in records:
        enc = tokenizer(r.text, truncation=False, padding=False)
        all_ids.extend(enc["input_ids"])

    step = max_seq_length - chunk_stride if chunk_stride else max_seq_length
    if step <= 0:
        raise ValueError(f"chunk_stride ({chunk_stride}) must be smaller than max_seq_length ({max_seq_length})")

    for start in range(0, len(all_ids) - max_seq_length + 1, step):
        block = all_ids[start:start + max_seq_length]
        example = {
            "input_ids": block,
            "attention_mask": [1] * len(block),
            "labels": list(block),
        }
        schemas.assert_no_qa_fields(example)
        examples.append(example)
    return examples


def build_datasets(
    data_cfg: dict,
    tokenizer: Any,
    processed_dir: Optional[Path] = None,
    input_path_override: Optional[Path] = None,
) -> tuple[dict[str, list[dict]], dict]:
    """Returns (mixed_examples, manifest_info). mixed_examples has keys
    "train"/"validation"/"test", each a list of {input_ids, attention_mask, labels}."""
    eos_token = tokenizer.eos_token
    sources = _normalize_sources(data_cfg)

    per_source_text: dict[str, dict[str, list[TextRecord]]] = {}
    leakage_reports: dict[str, LeakageReport] = {}
    for name, src_cfg in sources.items():
        override = input_path_override if (input_path_override and name == "kb") else None
        split_records, report = build_examples_for_source(src_cfg, eos_token, input_path_override=override)
        per_source_text[name] = split_records
        leakage_reports[name] = report

    per_source_tokenized: dict[str, dict[str, list[dict]]] = {}
    for name, src_cfg in sources.items():
        max_seq_length = src_cfg.get("max_seq_length", data_cfg.get("max_seq_length", 1024))
        packing = src_cfg.get("packing", False)
        chunk_stride = src_cfg.get("chunk_stride", 0)
        per_source_tokenized[name] = {
            split_name: tokenize_split(recs, tokenizer, max_seq_length, packing, chunk_stride)
            for split_name, recs in per_source_text[name].items()
        }

    if len(sources) > 1:
        sampling_weights = data_cfg.get("sampling_weights", {name: 1.0 for name in sources})
        seed = data_cfg.get("seed", 42)
        # Exposure-control mode: if train_presentations is set, present every example of
        # each source that many times (all web once + each KB record 50x) instead of the
        # fixed-epoch weighted sampling. Decouples per-source exposure from token ratio.
        train_presentations = data_cfg.get("train_presentations")
        mixed = mix_datasets(per_source_tokenized, sampling_weights, seed, train_presentations=train_presentations)
    else:
        mixed = next(iter(per_source_tokenized.values()))

    manifest_info = {
        "sources": list(sources.keys()),
        "per_source_split_sizes": {
            name: {split: len(recs) for split, recs in splits.items()}
            for name, splits in per_source_text.items()
        },
        "mixed_split_sizes": {split: len(examples) for split, examples in mixed.items()},
    }

    if processed_dir is not None:
        processed_dir = Path(processed_dir)
        manifests.write_data_manifest(
            processed_dir / "data_manifest.json",
            input_files={name: src.get("input_path") for name, src in sources.items()},
            processed_records={name: sum(splits.values(), []) for name, splits in per_source_text.items()},
        )
        manifests.write_split_manifest(processed_dir / "split_manifest.json", per_source_text)
        manifests.write_leakage_report(processed_dir / "leakage_report.json", leakage_reports)

    manifest_info["leakage_reports"] = {name: r.as_dict() for name, r in leakage_reports.items()}
    return mixed, manifest_info


def prepare_text_splits(
    data_cfg: dict,
    eos_token: str,
    processed_dir: Path,
    input_path_override: Optional[Path] = None,
) -> dict:
    """Text-only variant (no tokenizer call beyond eos_token) used by `ssft prepare-kb`:
    converts + splits, writes readable per-split JSONL + manifests, no model needed."""
    processed_dir = Path(processed_dir)
    sources = _normalize_sources(data_cfg)

    per_source_text: dict[str, dict[str, list[TextRecord]]] = {}
    leakage_reports: dict[str, LeakageReport] = {}
    for name, src_cfg in sources.items():
        override = input_path_override if (input_path_override and name == "kb") else None
        split_records, report = build_examples_for_source(src_cfg, eos_token, input_path_override=override)
        per_source_text[name] = split_records
        leakage_reports[name] = report

        for split_name, records in split_records.items():
            out_path = processed_dir / name / f"{split_name}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for r in records:
                    row = {"record_id": r.record_id, "group_key": r.group_key, "text": r.text}
                    schemas.assert_no_qa_fields(row)
                    f.write(json.dumps(row) + "\n")

    manifests.write_data_manifest(
        processed_dir / "data_manifest.json",
        input_files={name: src.get("input_path") for name, src in sources.items()},
        processed_records={name: sum(splits.values(), []) for name, splits in per_source_text.items()},
    )
    manifests.write_split_manifest(processed_dir / "split_manifest.json", per_source_text)
    manifests.write_leakage_report(processed_dir / "leakage_report.json", leakage_reports)

    return {
        "processed_dir": str(processed_dir),
        "sizes": {
            name: {split: len(recs) for split, recs in splits.items()}
            for name, splits in per_source_text.items()
        },
        "leakage_reports": {name: r.as_dict() for name, r in leakage_reports.items()},
    }
