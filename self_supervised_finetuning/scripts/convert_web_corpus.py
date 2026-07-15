"""Convert the LLM-wiki markdown pages into the {source_url, text} JSONL the
web-corpus pipeline expects (src/ssft/data/web_converter.py).

Each page looks like:

    ---
    title: "SK Battery America"
    entity_type: company
    supply_chain_category: battery_cell_manufacturing
    page_id: page_000001
    publication_date: 2023-01-30
    source_url: https://gov.georgia.gov/press-releases/...
    generated_from: v15 single-page extraction (pre Stage-6 merge)
    ---

    # SK Battery America
    ... markdown body (## Facts, ## Links, etc.) ...

We take `source_url` from the frontmatter (falling back to `page_id` as
`document_id` if a page ever lacks a URL) and use the markdown BODY (everything
after the closing `---`) as the text. Filenames are ignored entirely, per the
plan — the frontmatter + body are the source of truth. Empty-body pages are
dropped; the downstream converter additionally dedups exact-identical text.

Usage (from repo root):
    self_supervised_finetuning/.venv/bin/python \
        self_supervised_finetuning/scripts/convert_web_corpus.py \
        --input-dir "self_supervised_finetuning/data/raw/llm_wiki_pages" \
        --output    "self_supervised_finetuning/data/raw/web_corpus.jsonl"
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body). Minimal line parser (key: value) — values may
    contain ':' (URLs), so we split on the FIRST ': ' only. No YAML dep needed since
    every field here is a simple scalar."""
    m = FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw.strip()
    fm_block, body = m.group(1), m.group(2)
    fm: dict = {}
    for line in fm_block.splitlines():
        line = line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        fm[key.strip()] = val
    return fm, body.strip()


def convert(input_dir: Path, output: Path) -> dict:
    pages = sorted(input_dir.glob("*.md"))
    if not pages:
        raise FileNotFoundError(f"No .md pages under {input_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_empty = n_no_url = 0
    seen_urls: set[str] = set()
    total_chars = 0

    with open(output, "w") as out:
        for page in pages:
            raw = page.read_text(encoding="utf-8", errors="replace")
            fm, body = _parse_frontmatter(raw)
            if not body:
                n_empty += 1
                continue
            source_url = fm.get("source_url", "").strip()
            record: dict[str, str] = {}
            if source_url:
                record["source_url"] = source_url
                seen_urls.add(source_url)
            else:
                # Fall back to page_id as a stable document_id (grouping key downstream).
                n_no_url += 1
                doc_id = fm.get("page_id") or page.stem
                record["document_id"] = doc_id
            record["text"] = body
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
            total_chars += len(body)

    stats = {
        "pages_seen": len(pages),
        "rows_written": n_written,
        "empty_body_skipped": n_empty,
        "pages_without_source_url": n_no_url,
        "unique_source_urls": len(seen_urls),
        "total_body_chars": total_chars,
        "approx_tokens_chars_over_4": total_chars // 4,
        "output": str(output),
    }
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert LLM-wiki .md pages -> web_corpus.jsonl")
    ap.add_argument("--input-dir", required=True, help="Directory of .md wiki pages.")
    ap.add_argument("--output", required=True, help="Output JSONL path.")
    args = ap.parse_args()

    stats = convert(Path(args.input_dir), Path(args.output))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
