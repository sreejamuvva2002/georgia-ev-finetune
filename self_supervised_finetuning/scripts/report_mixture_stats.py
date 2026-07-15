"""Pre-training mixture verification for the KB+web mixed data config.

Reuses the real pipeline (`dataset_builder.build_examples_for_source` + `tokenize_split`
+ `dataset_mixer.mix_split`) but loads ONLY the tokenizer (no 14B model / no GPU), so it
runs in ~1-2 min. Reports, per the plan:
  - per-source example AND token counts (train/val/test),
  - avg tokens/example per source,
  - effective POST-MIX train token ratio (mixer weights per-example, not per-token),
  - KB-record exposure over N epochs,
  - exact-duplicate drop count for the web corpus,
and emits the web TRAIN page list (source_urls) → the pool to author GNEM-Web-18 from.

Usage (from repo root):
  self_supervised_finetuning/.venv/bin/python \
    self_supervised_finetuning/scripts/report_mixture_stats.py \
    --data-config  self_supervised_finetuning/configs/data/kb_web_mixed_kbdominant.yaml \
    --model-config self_supervised_finetuning/configs/models/qwen2p5_14b_base.yaml \
    --epochs 5 \
    --web-train-out self_supervised_finetuning/outputs/question_eval/web_train_pages.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import re

from ssft.data.dataset_builder import _normalize_sources, build_examples_for_source, tokenize_split
from ssft.data.dataset_mixer import mix_datasets, mix_split
from ssft.utils.yaml_utils import load_yaml


def _tok_counts(tokenized: list[dict]) -> int:
    return sum(len(ex["input_ids"]) for ex in tokenized)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--web-train-out", default=None, help="Write web TRAIN page list here (JSON).")
    args = ap.parse_args()

    data_cfg = load_yaml(Path(args.data_config))["data"]
    model_cfg = load_yaml(Path(args.model_config))["model"]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"], trust_remote_code=model_cfg.get("trust_remote_code", True)
    )
    eos = tok.eos_token
    sources = _normalize_sources(data_cfg)

    # Per-source: build text splits, tokenize with each source's own packing/max_seq.
    per_source_text = {}
    per_source_tok = {}
    raw_record_counts = {}
    for name, scfg in sources.items():
        split_records, _report = build_examples_for_source(scfg, eos)
        per_source_text[name] = split_records
        raw_record_counts[name] = sum(len(v) for v in split_records.values())
        max_seq = scfg.get("max_seq_length", data_cfg.get("max_seq_length", 1024))
        packing = scfg.get("packing", False)
        stride = scfg.get("chunk_stride", 0)
        per_source_tok[name] = {
            sp: tokenize_split(recs, tok, max_seq, packing, stride) for sp, recs in split_records.items()
        }

    print("=" * 78)
    print("PER-SOURCE EXAMPLE + TOKEN COUNTS")
    print("=" * 78)
    src_stats = {}
    for name in sources:
        src_stats[name] = {}
        for sp in ("train", "validation", "test"):
            toks = per_source_tok[name].get(sp, [])
            n_ex = len(toks)
            n_tok = _tok_counts(toks)
            avg = (n_tok / n_ex) if n_ex else 0.0
            src_stats[name][sp] = {"examples": n_ex, "tokens": n_tok, "avg_tokens_per_example": round(avg, 1)}
            print(f"  {name:4s} {sp:11s}: {n_ex:6d} ex  {n_tok:9d} tok  avg {avg:7.1f} tok/ex")

    # Materialize the EXACT mixed TRAIN set (source-tagged) honoring the config's mode:
    # weighted sampling OR train_presentations (exposure control). All effective stats
    # below are computed from what ACTUALLY enters training, valid for either mode.
    weights = data_cfg.get("sampling_weights", {n: 1.0 for n in sources})
    train_presentations = data_cfg.get("train_presentations")
    seed = data_cfg.get("seed", 42)
    tagged = {n: [dict(ex, _source=n) for ex in splits.get("train", [])]
              for n, splits in per_source_tok.items()}
    tagged_full = {n: {"train": tagged[n], "validation": [], "test": []} for n in tagged}
    mixed = mix_datasets(tagged_full, weights, seed, train_presentations=train_presentations)
    train_ex = mixed["train"]

    print("\n" + "=" * 78)
    mode = f"train_presentations={train_presentations}" if train_presentations else f"weights={weights}"
    print(f"EFFECTIVE MATERIALIZED TRAIN ({mode})")
    print("=" * 78)
    eff = {}
    for name in sources:
        exs = [e for e in train_ex if e.get("_source") == name]
        ntok = _tok_counts(exs)
        eff[name] = {"examples": len(exs), "tokens": ntok}
        print(f"  {name:4s}: {len(exs):7d} example-presentations  {ntok:10d} tokens")
    tot_tok = sum(v["tokens"] for v in eff.values()) or 1
    print(f"  total materialized train examples: {len(train_ex)}  "
          f"(optimizer steps/epoch @ ebs16 = {len(train_ex)//16})")
    print("  --- effective TRAIN TOKEN ratio ---")
    for name, v in eff.items():
        print(f"      {name:4s}: {100*v['tokens']/tot_tok:5.1f}% of train tokens")

    # KB exposure: presentations of each KB record over the whole run.
    if "kb" in sources and per_source_tok["kb"].get("train"):
        kb_pool = len(per_source_tok["kb"]["train"])
        kb_pres = eff["kb"]["examples"]
        per_rec = kb_pres / kb_pool if kb_pool else 0
        print("\n" + "=" * 78)
        print("KB-RECORD EXPOSURE")
        print("=" * 78)
        print(f"  KB pool: {kb_pool} records; presentations in materialized train: {kb_pres}")
        print(f"  -> each KB record seen ~{per_rec:.1f}x per epoch, "
              f"~{per_rec*args.epochs:.1f}x over {args.epochs} epoch(s)")
    if "web" in sources and per_source_tok["web"].get("train"):
        web_pool = len(per_source_tok["web"]["train"])
        web_pres = eff["web"]["examples"]
        print(f"  WEB pool: {web_pool} chunks; presentations: {web_pres} "
              f"({'ALL chunks' if web_pres>=web_pool else f'{100*web_pres/web_pool:.0f}% of chunks'})"
              f" -> ~{web_pres/web_pool:.1f}x per epoch")

    print("\n" + "=" * 78)
    print(f"WEB RECORDS kept after exact-text dedup: {raw_record_counts.get('web', 0)}")
    print("=" * 78)

    # ---- EFFECTIVE web-page provenance (the valid pool to author GNEM-Web from) -----
    if args.web_train_out:
        web_examples = [ex for ex in train_ex if ex.get("_source") == "web"]
        # distinct chunks (dedupe by input_ids identity)
        seen_ids, distinct = set(), []
        for ex in web_examples:
            key = tuple(ex["input_ids"])
            if key not in seen_ids:
                seen_ids.add(key); distinct.append(ex)
        effective_web_text = "\n".join(
            tok.decode(ex["input_ids"], skip_special_tokens=True) for ex in distinct
        )
        norm_eff = _norm(effective_web_text)
        print("\n" + "=" * 78)
        print("EFFECTIVE WEB TRAINING SET (what actually enters training)")
        print("=" * 78)
        print(f"  distinct web chunks sampled into train: {len(distinct)}  "
              f"(~{sum(len(e['input_ids']) for e in distinct)} tokens)")

        # Which web TRAIN pages are provably inside the sampled chunks? Use a distinctive
        # mid-body snippet per page to survive tokenizer round-trip + chunk boundaries.
        web_train = per_source_text.get("web", {}).get("train", [])
        proven = []
        for r in web_train:
            body = _norm(r.text)
            if len(body) < 80:
                continue
            snip = body[40:200]  # mid-body slice, avoids the shared "# Title / **Type**" header
            occ = norm_eff.count(snip)
            if occ > 0:
                proven.append({"record_id": r.record_id, "source_url": r.group_key,
                               "document_id": (r.metadata or {}).get("document_id"),
                               "n_chars": len(r.text), "occurrences_in_effective_train": occ,
                               "exposure_over_epochs": occ * args.epochs})
        print(f"  web pages provably in effective training: {len(proven)} / {len(web_train)} train pages")

        outp = Path(args.web_train_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({
            "weights": weights, "seed": data_cfg.get("seed", 42), "epochs": args.epochs,
            "distinct_web_chunks_in_train": len(distinct),
            "n_web_pages_proven_in_effective_training": len(proven),
            "pages": proven}, indent=2))
        (outp.parent / "effective_web_text.txt").write_text(effective_web_text)
        print(f"  wrote {len(proven)} PROVEN web pages -> {outp}")
        print(f"  wrote decoded effective web text -> {outp.parent / 'effective_web_text.txt'}")

    print("\n" + "=" * 78)
    print("SUMMARY (is KB drowned?): compare KB vs web effective TRAIN TOKEN % above.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
