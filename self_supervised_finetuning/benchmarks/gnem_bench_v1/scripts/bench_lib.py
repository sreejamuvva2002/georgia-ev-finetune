"""Shared helpers for GNEM-Bench-v1 scoring: entity extraction against the 205 KB
company names (symmetric for gold and predictions), plus value normalizers. Reuses
eval_cloze normalization where possible.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ssft.eval.eval_cloze import _normalize_for_match  # reuse

KB_PATH = Path("/home/sreeja/georgia-ev-finetune/kb_full.jsonl")

_CORP_SUFFIX = re.compile(r"\b(inc|llc|corp|corporation|co|ltd|ltda|lp|llp|gmbh|ag|sa|america|"
                          r"north america|usa|us|company|group|manufacturing|industries)\b\.?", re.I)


def _canon_company(name: str) -> str:
    s = _CORP_SUFFIX.sub(" ", name or "")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def load_kb_companies() -> list[dict]:
    """Return [{company, canon, employment, row}] for the 205 KB rows."""
    out = []
    for line in open(KB_PATH):
        r = json.loads(line)
        c = r.get("Company", "")
        canon = _canon_company(c)
        if len(canon) >= 3:
            out.append({"company": c, "canon": canon, "employment": r.get("Employment"), "row": r})
    return out


_KB = None
def kb_companies() -> list[dict]:
    global _KB
    if _KB is None:
        _KB = load_kb_companies()
    return _KB


def companies_in_text(text: str) -> set[str]:
    """Set of canonical KB company keys whose (canonicalized) name appears in `text`.
    Symmetric extractor for gold and prediction — both scored in the KB name space."""
    t = " " + re.sub(r"[^a-z0-9]+", " ", (text or "").lower()) + " "
    found = set()
    for kc in kb_companies():
        canon = kc["canon"]
        if canon and f" {canon} " in t:
            found.add(canon)
    return found


# ---- value normalizers -------------------------------------------------------------
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_SCALE = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}


def parse_count(text: str):
    """First integer (digits or number-word) in text, e.g. 'There are 18 ...' -> 18."""
    if not text:
        return None
    m = re.search(r"\b(\d{1,4})\b", text.replace(",", ""))
    if m:
        return int(m.group(1))
    for w, v in _NUM_WORDS.items():
        if re.search(rf"\b{w}\b", text, re.I):
            return v
    return None


def parse_currency(text: str):
    """Return dollar amount in USD as int, handling '$5.54 billion', '500 million', etc."""
    if not text:
        return None
    m = re.search(r"\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(billion|million|thousand)?", text, re.I)
    if not m:
        return None
    digits = m.group(1).replace(",", "").strip(".")
    if not digits:
        return None
    val = float(digits)
    if m.group(2):
        val *= _SCALE[m.group(2).lower()]
    return int(round(val))


def parse_gwh(text: str):
    m = re.search(r"([\d.]+)\s*GWh", text or "", re.I)
    return float(m.group(1)) if m else None


def parse_year(text: str):
    m = re.search(r"\b(20\d{2})\b", text or "")
    return int(m.group(1)) if m else None


def norm_county(text: str):
    m = re.search(r"([A-Z][a-zA-Z]+)\s+County", text or "")
    return m.group(1).lower() if m else None


def set_prf(pred: set, gold: set) -> dict:
    if not gold:
        return {"precision": None, "recall": None, "f1": None, "completeness": None}
    inter = len(pred & gold)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(gold)
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "completeness": round(r, 4), "n_pred": len(pred), "n_gold": len(gold),
            "n_correct": inter}


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson interval for a binomial proportion (n small -> use this, not normal)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))
