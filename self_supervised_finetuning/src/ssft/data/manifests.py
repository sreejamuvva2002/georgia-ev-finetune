"""Write/read data_manifest.json, split_manifest.json, leakage_report.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from ssft.data.splitters import LeakageReport
from ssft.data.text_formatters import TextRecord
from ssft.utils.hashing import sha256_file, sha256_text


def _write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str, sort_keys=False)


def read_manifest(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_data_manifest(
    path: Path,
    *,
    input_files: dict[str, Union[Path, str, None]],
    processed_records: dict[str, list[TextRecord]],
) -> dict:
    manifest = {
        "input_files": {
            name: {"path": str(p), "sha256": sha256_file(Path(p))}
            for name, p in input_files.items() if p and Path(p).exists()
        },
        "sources": {
            name: {
                "n_records": len(recs),
                "processed_text_sha256": sha256_text("".join(r.text for r in recs)),
            }
            for name, recs in processed_records.items()
        },
    }
    _write_json(path, manifest)
    return manifest


def write_split_manifest(path: Path, split_records: dict[str, dict[str, list[TextRecord]]]) -> dict:
    """split_records: source_name -> split_name -> list[TextRecord]."""
    manifest = {
        "splits": {
            source: {
                split_name: sorted({r.group_key for r in recs})
                for split_name, recs in per_split.items()
            }
            for source, per_split in split_records.items()
        },
        "sizes": {
            source: {split_name: len(recs) for split_name, recs in per_split.items()}
            for source, per_split in split_records.items()
        },
    }
    _write_json(path, manifest)
    return manifest


def write_leakage_report(path: Path, report: Union[LeakageReport, dict[str, LeakageReport]]) -> dict:
    if isinstance(report, dict):
        payload = {name: r.as_dict() for name, r in report.items()}
    else:
        payload = report.as_dict()
    _write_json(path, payload)
    return payload
