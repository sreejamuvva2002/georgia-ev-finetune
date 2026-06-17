#!/usr/bin/env python3
"""Validate train/valid/test JSONL before training (v4).

Structural checks PLUS no-compromise integrity gates:
  - 0 leakage of the 50 test questions and the probe-benchmark questions into train/valid
  - count == number of listed companies on every enumerated list answer
  - refusal fraction within a sane band (not drowned, not over-refusing)
  - KB-consistency: every role bucket and OEM tag's listed count matches the KB unique count
  - sequence-length safety vs the trainer's max_seq_length
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd            # load_kb, _qnorm, by_role, by_oem, uniq, ROLE_BUCKETS, OEM_CANON

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOGS = ROOT / "logs"
MAX_SEQ = 1536                        # must match lora_14b.yaml
random.seed(7)

REQUIRED_ROLES = ["system", "user", "assistant"]


def load(path):
    return [json.loads(l) for l in open(path)]


def structural(rows, name, problems):
    for i, obj in enumerate(rows, 1):
        msgs = obj.get("messages", [])
        if [m.get("role") for m in msgs] != REQUIRED_ROLES:
            problems.append(f"{name} line {i}: bad roles")
        for m in msgs:
            if not isinstance(m.get("content"), str) or not m["content"].strip():
                problems.append(f"{name} line {i}: empty content for {m.get('role')}")


def main():
    problems = []
    train = load(DATA / "train.jsonl")
    valid = load(DATA / "valid.jsonl")
    test = load(DATA / "test.jsonl")
    structural(train, "train", problems)
    structural(valid, "valid", problems)
    structural(test, "test", problems)

    def q(r): return r["messages"][1]["content"]
    def a(r): return r["messages"][2]["content"]

    # ---- 1. leakage: 0 overlap with the 50 test + the probe benchmark ----
    test_qs = {bd._qnorm(q(r)) for r in test}
    probe_qs = set()
    if bd.PROBE_FILE.exists():
        for l in open(bd.PROBE_FILE):
            probe_qs.add(bd._qnorm(json.loads(l)["question"]))
    trainvalid_qs = {bd._qnorm(q(r)) for r in train + valid}
    leak_test = trainvalid_qs & test_qs
    leak_probe = trainvalid_qs & probe_qs
    gold_ans = {a(r).strip() for r in test}
    leak_ans = {a(r).strip() for r in train} & gold_ans
    if leak_test: problems.append(f"LEAKAGE: {len(leak_test)} test questions in train/valid")
    if leak_probe: problems.append(f"LEAKAGE: {len(leak_probe)} probe questions in train/valid")
    if leak_ans: problems.append(f"LEAKAGE: {len(leak_ans)} human gold answers reproduced in train")

    # ---- 2. count == number of listed companies (enumerated list answers) ----
    count_mismatch = 0
    for r in train + valid:
        ans = a(r)
        m = re.search(r"there (?:are|is) (\d+) ", ans)
        if not m or "\n" not in ans or "listing the first" in ans:
            continue
        comp_lines = [l for l in ans.split("\n")[1:] if " | " in l]
        if comp_lines and int(m.group(1)) != len(comp_lines):
            count_mismatch += 1
            if count_mismatch <= 5:
                problems.append(f"COUNT!=LISTED ({m.group(1)} vs {len(comp_lines)}): {ans.splitlines()[0][:80]}")
    if count_mismatch:
        problems.append(f"COUNT!=LISTED total: {count_mismatch}")

    # ---- 3. refusal fraction band ----
    # (recount from the regenerated train tags via the dataset report is simpler, but recompute here)
    refusal_like = sum(1 for r in train if "knowledge base does not provide" in a(r))
    frac = refusal_like / max(1, len(train))
    if not (0.02 <= frac <= 0.15):
        problems.append(f"refusal fraction out of band: {frac:.3f}")

    # ---- 4. KB-consistency: role buckets + OEM tags listed counts == KB unique counts ----
    df = bd.load_kb()
    def headline(pattern):
        """Return the count from the FIRST train answer matching a precise regex (the main
        list answer, not a cross answer that merely mentions the same role/OEM)."""
        rx = re.compile(pattern, re.I)
        for r in train:
            mm = rx.search(a(r))
            if mm:
                return int(mm.group(1))
        return None
    kb_problems = []
    for tag in bd.ROLE_BUCKETS:
        kb_n = len(bd.uniq(bd.by_role(df, tag)))
        # main list_role answer: "there are N companies in the {tag} role:" (N directly before 'compan')
        got = headline(rf"there (?:are|is) (\d+) compan(?:ies|y) in the {re.escape(tag)} role:")
        if kb_n <= 50 and got is not None and got != kb_n:
            kb_problems.append(f"role {tag}: train says {got}, KB unique {kb_n}")
    for tag in bd.OEM_CANON:
        kb_n = len(bd.uniq(bd.by_oem(df, tag)))
        if kb_n == 0:
            continue
        got = headline(rf"there (?:are|is) (\d+) Georgia compan(?:ies|y) linked to {re.escape(tag)} \(via")
        if kb_n <= 50 and got is not None and got != kb_n:
            kb_problems.append(f"OEM {tag}: train says {got}, KB unique {kb_n}")
    problems.extend(kb_problems)

    # ---- 5. sequence-length safety (word-count proxy; ~1.3 tok/word) ----
    too_long = 0
    for r in train:
        words = sum(len(m["content"].split()) for m in r["messages"])
        if words * 1.4 > MAX_SEQ:
            too_long += 1
    if too_long:
        problems.append(f"~{too_long} train examples may exceed max_seq_length={MAX_SEQ} (clip risk)")

    # ---- report ----
    lens = [sum(len(m["content"].split()) for m in r["messages"]) for r in train]
    lens.sort()
    print(f"train={len(train)} valid={len(valid)} test={len(test)}")
    print(f"words/example: median={lens[len(lens)//2]} p99={lens[int(.99*len(lens))]} max={lens[-1]}")
    print(f"leakage: test={len(leak_test)} probe={len(leak_probe)} gold_answers={len(leak_ans)}")
    print(f"count!=listed: {count_mismatch} | refusal_frac={frac:.3f} | KB-consistency issues={len(kb_problems)} | seq-risk={too_long}")
    print("\n=== 4 random train examples ===")
    for s in random.sample(train, 4):
        print("-" * 60); print("Q:", q(s)); print("A:", a(s)[:300] + ("..." if len(a(s)) > 300 else ""))

    if problems:
        print("\nPROBLEMS:")
        for p in problems[:25]:
            print("  !", p)
        print("\nVALIDATION: FAIL")
        raise SystemExit(1)
    print("\nVALIDATION: PASS")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
