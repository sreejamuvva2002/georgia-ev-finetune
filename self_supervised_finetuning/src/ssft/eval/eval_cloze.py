"""Deterministic cloze-style factual probes generated from KB fields.

Evaluation only — probes are never used for training. Short fields (Location,
Category, EV/Battery Relevant, Employment, Primary OEMs) are scored with
exact/normalized-exact match; the long free-text field (Product/Service) is scored
with token-level F1 + normalized containment. Results are split into seen-company
(memorization/absorption signal) vs held-out-company (generalization signal) using
the run's split_manifest.json.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ssft.data.text_formatters import coalesce_missing

CLOZE_TEMPLATES = [
    {"template_name": "location", "prompt_template": "Company: {Company}\nLocation:", "gold_column": "Location", "field_type": "short"},
    {"template_name": "category", "prompt_template": "Company: {Company}\nCategory:", "gold_column": "Category", "field_type": "short"},
    {"template_name": "ev_battery_relevant", "prompt_template": "Company: {Company}\nEV or Battery Relevant:", "gold_column": "EV / Battery Relevant", "field_type": "short"},
    {"template_name": "employment", "prompt_template": "Company: {Company}\nEmployment:", "gold_column": "Employment", "field_type": "short"},
    {"template_name": "primary_oems", "prompt_template": "Company: {Company}\nPrimary OEMs:", "gold_column": "Primary OEMs", "field_type": "short"},
    {"template_name": "product_or_service", "prompt_template": "Company: {Company}\nProduct or Service:", "gold_column": "Product / Service", "field_type": "long"},
]


@dataclass
class ClozeProbe:
    probe_id: str
    template_name: str
    company: str
    prompt: str
    gold_answer: str
    field_type: str


def generate_cloze_probes(kb_records: list[dict], seed: int = 42) -> list[ClozeProbe]:
    probes = []
    for row in sorted(kb_records, key=lambda r: r.get("row_id", 0)):
        company = coalesce_missing(row.get("Company"))
        for t in CLOZE_TEMPLATES:
            gold = coalesce_missing(row.get(t["gold_column"]))
            prompt = t["prompt_template"].format(Company=company)
            probes.append(ClozeProbe(
                probe_id=f"row{row.get('row_id')}-{t['template_name']}",
                template_name=t["template_name"], company=company, prompt=prompt,
                gold_answer=gold, field_type=t["field_type"],
            ))
    return probes


def _normalize_for_match(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def score_short_field(pred: str, gold: str) -> dict:
    pred, gold = (pred or "").strip(), (gold or "").strip()
    return {
        "exact_match": pred == gold,
        "normalized_exact_match": _normalize_for_match(pred) == _normalize_for_match(gold),
    }


def score_long_field(pred: str, gold: str) -> dict:
    pred_tokens = _normalize_for_match(pred or "").split()
    gold_tokens = _normalize_for_match(gold or "").split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0 or not pred_tokens or not gold_tokens:
        f1 = 0.0
    else:
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
    norm_pred, norm_gold = _normalize_for_match(pred or ""), _normalize_for_match(gold or "")
    containment = bool(norm_gold) and (norm_gold in norm_pred or norm_pred in norm_gold)
    return {"token_f1": f1, "normalized_containment": containment}


def _aggregate(rows: list[dict], key: str) -> dict:
    if not rows:
        return {}
    # Short-field and long-field probes produce different score keys (exact_match vs token_f1),
    # so aggregate each metric only over the rows that actually have it instead of assuming a
    # uniform schema from rows[0].
    all_keys: list = []
    for r in rows:
        for sk in r[key].keys():
            if sk not in all_keys:
                all_keys.append(sk)
    agg = {}
    for score_key in all_keys:
        vals = [r[key][score_key] for r in rows if score_key in r[key]]
        if not vals:
            continue
        agg[score_key] = (sum(1 for v in vals if v) / len(vals)) if isinstance(vals[0], bool) else (sum(vals) / len(vals))
    return agg


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 16) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.split("\n")[0].strip()


def run_cloze_eval(run_dir: Path, probes_path: Optional[Path] = None, max_new_tokens: int = 16) -> dict:
    from ssft.data.kb_converter import load_kb_jsonl
    from ssft.eval.eval_perplexity import load_adapter_for_eval
    from ssft.utils.paths import PROBES_ROOT

    run_dir = Path(run_dir)
    resolved, base_model, adapter_model, tokenizer = load_adapter_for_eval(run_dir)

    kb_input_path = resolved.data_cfg.get("input_path") or (
        (resolved.data_cfg.get("sources", {}) or {}).get("kb", {}) or {}
    ).get("input_path")
    if not kb_input_path:
        raise ValueError("resolved data config has no KB input_path — cannot regenerate cloze probes for this run")
    kb_rows = load_kb_jsonl(Path(kb_input_path))
    probes = generate_cloze_probes(kb_rows, seed=resolved.seed)

    probes_path = Path(probes_path) if probes_path else (PROBES_ROOT / "kb_cloze_probes.jsonl")
    probes_path.parent.mkdir(parents=True, exist_ok=True)
    with open(probes_path, "w") as f:
        for p in probes:
            f.write(json.dumps(asdict(p)) + "\n")

    held_out_companies: set = set()
    split_manifest_path = run_dir / "split_manifest.json"
    if split_manifest_path.exists():
        with open(split_manifest_path) as f:
            split_manifest = json.load(f)
        kb_splits = (split_manifest.get("splits", {}) or {}).get("kb", {})
        held_out_companies = set(kb_splits.get("validation", [])) | set(kb_splits.get("test", []))

    rows = []
    for p in probes:
        # base_model is the same object as adapter_model (adapter attached in place), so get the
        # true base prediction with the adapter disabled rather than from base_model directly.
        with adapter_model.disable_adapter():
            base_pred = _generate(adapter_model, tokenizer, p.prompt, max_new_tokens)
        adapter_pred = _generate(adapter_model, tokenizer, p.prompt, max_new_tokens)
        scorer = score_long_field if p.field_type == "long" else score_short_field
        rows.append({
            "probe_id": p.probe_id, "template_name": p.template_name, "company": p.company,
            "gold_answer": p.gold_answer, "field_type": p.field_type,
            "is_seen_company": p.company not in held_out_companies,
            "base_prediction": base_pred, "adapter_prediction": adapter_pred,
            "base_score": scorer(base_pred, p.gold_answer),
            "adapter_score": scorer(adapter_pred, p.gold_answer),
        })

    seen_rows = [r for r in rows if r["is_seen_company"]]
    held_out_rows = [r for r in rows if not r["is_seen_company"]]
    summary = {
        "n_probes": len(rows),
        "seen_company": {
            "n": len(seen_rows),
            "base": _aggregate(seen_rows, "base_score"),
            "adapter": _aggregate(seen_rows, "adapter_score"),
            "note": "Seen-company recall reflects memorization/absorption, not generalization.",
        },
        "held_out_company": {
            "n": len(held_out_rows),
            "base": _aggregate(held_out_rows, "base_score"),
            "adapter": _aggregate(held_out_rows, "adapter_score"),
        },
    }
    result = {"summary": summary, "rows": rows}

    out_path = run_dir / "eval" / "cloze_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result
