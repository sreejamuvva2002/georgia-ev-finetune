"""Ingest the human-validated KB gold Excel into structured records — stdlib only
(openpyxl absent). An .xlsx is a zip of XML; we read sharedStrings + sheet1.

Prints an auditable row census and `assert parsed == EXPECT` (default 42) — NEVER
silently drops a row. With --length-report, tokenizes gold answers to choose
max_new_tokens from data (median / p95 / max).

Usage:
  ingest_kb_gold.py --xlsx "Human validated questions.xlsx" --out kb_gold_raw.json
  ingest_kb_gold.py --xlsx "Human validated questions.xlsx" --length-report \
     --model-config self_supervised_finetuning/configs/models/qwen2p5_14b_base.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _colnum(ref: str) -> int:
    m = re.match(r"([A-Z]+)", ref)
    c = 0
    for ch in m.group(1):
        c = c * 26 + (ord(ch) - 64)
    return c


def read_xlsx_rows(path: Path) -> list[list[str]]:
    z = zipfile.ZipFile(path)
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        r = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in r.findall(f"{NS}si"):
            ss.append("".join(t.text or "" for t in si.iter(f"{NS}t")))
    sheet = sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml", n))[0]
    root = ET.fromstring(z.read(sheet))
    rows = []
    for row in root.iter(f"{NS}row"):
        cells = {}
        for c in row.findall(f"{NS}c"):
            ref, t = c.get("r"), c.get("t")
            v, istr = c.find(f"{NS}v"), c.find(f"{NS}is")
            val = ""
            if t == "s" and v is not None:
                val = ss[int(v.text)]
            elif istr is not None:
                val = "".join(x.text or "" for x in istr.iter(f"{NS}t"))
            elif v is not None:
                val = v.text
            cells[_colnum(ref)] = val
        if cells:
            maxc = max(cells)
            rows.append([cells.get(i + 1, "") for i in range(maxc)])
    return rows


def parse_gold(path: Path, expect: int = 42) -> list[dict]:
    rows = read_xlsx_rows(path)
    worksheet_rows = len(rows)
    header = [str(c).strip().lower() for c in rows[0]]
    def col(name_substr):
        for i, h in enumerate(header):
            if name_substr in h:
                return i
        return None
    c_num, c_cat = col("num"), col("use case")
    c_q, c_ans = col("question"), col("answer")
    assert None not in (c_q, c_ans), f"missing Question/Answer columns in header {header}"

    records, skipped_blank, seen_q = [], 0, {}
    skipped_dup = 0
    for r in rows[1:]:
        def g(i):
            return str(r[i]).strip() if (i is not None and i < len(r)) else ""
        q, ans = g(c_q), g(c_ans)
        if not q and not ans:
            skipped_blank += 1
            continue
        if q in seen_q:
            skipped_dup += 1
            continue
        seen_q[q] = True
        records.append({"num": g(c_num), "use_case_category": g(c_cat),
                        "question": q, "gold_answer": ans})
    print(f"Worksheet rows including header: {worksheet_rows}")
    print(f"Parsed question records:        {len(records)}")
    print(f"Skipped blank rows:             {skipped_blank}")
    print(f"Skipped duplicate rows:         {skipped_dup}")
    assert len(records) == expect, (
        f"expected {expect} questions, parsed {len(records)} — refusing to proceed "
        f"(never silently drop/keep an unexpected count). Inspect the xlsx.")
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--expect", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--length-report", action="store_true")
    ap.add_argument("--model-config", default=None)
    args = ap.parse_args()

    records = parse_gold(Path(args.xlsx), args.expect)

    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=2, ensure_ascii=False))
        print(f"wrote {args.out}")

    if args.length_report:
        from ssft.utils.yaml_utils import load_yaml
        from transformers import AutoTokenizer
        mc = load_yaml(Path(args.model_config))["model"]
        tok = AutoTokenizer.from_pretrained(mc["name_or_path"], trust_remote_code=True)
        lens = sorted(len(tok(r["gold_answer"])["input_ids"]) for r in records)
        n = len(lens)
        def pct(p):
            return lens[min(n - 1, int(round(p * (n - 1))))]
        print("\n=== gold-answer token lengths ===")
        print(f"  min {lens[0]}  median {pct(0.5)}  p90 {pct(0.9)}  p95 {pct(0.95)}  max {lens[-1]}")
        rec = min(round(lens[-1] * 1.3), 512)
        print(f"  suggested max_new_tokens = min(round(max*1.3), 512) = {rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
