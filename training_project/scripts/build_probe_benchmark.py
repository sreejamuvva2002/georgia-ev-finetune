#!/usr/bin/env python3
"""Held-out PROBE BENCHMARK — the primary 'no-compromise' coverage metric for v4.

Generates a few hundred questions across every shape, with gold computed from the KB
under the SAME canonical convention as build_dataset (unique companies + RoleNorm + OEM
split). The phrasings here are RESERVED (held out) — different from the training paraphrase
bank — and build_dataset decontaminates against this file, so a passing model has genuinely
recalled the fact and generalized to an unseen phrasing (not string-matched a trained question).

Headline metrics:
- pass rate (overall + per kind)
- company RECALL on list questions  ->  missing-company rate = 1 - recall  (the coverage gate)
- precision / hallucination rate

Usage:
  python build_probe_benchmark.py            # build probes -> data/probe_benchmark.jsonl
  python build_probe_benchmark.py score ANSWERS.json   # score model answers (list aligned to probes)
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd          # load_kb, by_role, by_oem, uniq, ROLE_BUCKETS, OEM_CANON, emp_str
import evaluate as ev               # company_mentions, numbers_in, lead_count

random.seed(7)                      # distinct from build_dataset's seed
PROBE_FILE = bd.PROBE_FILE
NAMES_CAP = 25                      # sets larger than this are tested as COUNT probes, not name lists

# ---- reserved phrasings (held out from training) ----
RT_ROLE = ["Provide the complete list of Georgia suppliers categorized under {x}.",
           "Enumerate every Georgia firm whose EV supply chain role is {x}; list them all.",
           "I need the full roster of {x} suppliers in Georgia — who are they?"]
RT_CAT = ["Give the complete roster of Georgia companies in the {x} classification.",
          "Enumerate all Georgia suppliers assigned to {x}; list each one.",
          "Who are all the {x} companies in the Georgia EV knowledge base?"]
RT_OEM = ["Provide the complete supplier list tied to {x} in Georgia (per Primary OEMs).",
          "Enumerate every Georgia company whose Primary OEMs include {x}.",
          "I want the full {x} supplier roster in Georgia — list them all."]
RT_COUNTY = ["Provide the complete list of Georgia EV companies situated in {x}.",
             "Enumerate every supplier the KB records in {x}.",
             "Who are all the companies the Georgia EV KB places in {x}?"]
RT_CROSS = ["Provide the complete list of {a} Georgia companies that are also {b}.",
            "Enumerate every Georgia supplier matching both {a} and {b}.",
            "Which Georgia firms satisfy {a} and {b}? List them all."]
RT_COUNT = ["State the exact number of {x} in the Georgia EV KB.",
            "How many entries does the KB record for {x}? Give the precise count.",
            "Report the precise count of {x}."]
RT_SUP = ["Identify the single {x}.",
          "Which one is the {x}? Name it.",
          "Report the {x} from the Georgia EV KB."]
RT_AGG = ["Compute the {x}.",
          "What is the {x}? Give the figure.",
          "Report the {x} from the Georgia EV KB."]


def pick(ts):
    return random.choice(ts)


PROBES = []


def emit(question, kind, **gold):
    PROBES.append({"question": question, "kind": kind, **gold})


def build():
    df = bd.load_kb()
    ALL = sorted(df["Company"].dropna().unique())

    # 1) role buckets
    for tag in bd.ROLE_BUCKETS:
        comps = sorted(bd.uniq(bd.by_role(df, tag))["Company"].tolist())
        if not comps:
            continue
        if len(comps) <= NAMES_CAP:
            emit(pick(RT_ROLE).format(x=f"the {tag} role"), "names",
                 expected_companies=comps, expected_count=len(comps))
        emit(pick(RT_COUNT).format(x=f"companies in the {tag} role"), "count", expected_count=len(comps))

    # 2) categories
    for cat in sorted(df["CategoryNorm"].dropna().unique()):
        comps = sorted(bd.uniq(df[df["CategoryNorm"] == cat])["Company"].tolist())
        if len(comps) <= NAMES_CAP:
            emit(pick(RT_CAT).format(x=cat), "names", expected_companies=comps, expected_count=len(comps))
        emit(pick(RT_COUNT).format(x=f"{cat} companies"), "count", expected_count=len(comps))

    # 3) OEM linkage (Rivian is the critical one)
    for tag in bd.OEM_CANON:
        comps = sorted(bd.uniq(bd.by_oem(df, tag))["Company"].tolist())
        if not comps:
            continue
        if len(comps) <= NAMES_CAP:
            emit(pick(RT_OEM).format(x=tag), "names", expected_companies=comps, expected_count=len(comps))
        else:
            emit(pick(RT_COUNT).format(x=f"suppliers linked to {tag}"), "count", expected_count=len(comps))

    # 4) counties (those with 2..NAMES_CAP get names; sample to ~30)
    counties = [c for c in sorted(df["County"].dropna().unique())
                if 2 <= len(bd.uniq(df[df["County"] == c])) <= NAMES_CAP]
    for c in random.sample(counties, min(30, len(counties))):
        comps = sorted(bd.uniq(df[df["County"] == c])["Company"].tolist())
        emit(pick(RT_COUNTY).format(x=c), "names", expected_companies=comps, expected_count=len(comps))

    # 5) Category x Role crosses (non-empty, small)
    for cat in sorted(df["CategoryNorm"].dropna().unique()):
        for tag in bd.ROLE_BUCKETS:
            comps = sorted(bd.uniq(bd.by_role(df[df["CategoryNorm"] == cat], tag))["Company"].tolist())
            if 1 <= len(comps) <= NAMES_CAP:
                emit(pick(RT_CROSS).format(a=f"{cat}", b=f"in the {tag} role"),
                     "names", expected_companies=comps, expected_count=len(comps))

    # 6) Category x Relevance
    for cat in sorted(df["CategoryNorm"].dropna().unique()):
        for rl in ["Yes", "Indirect", "No"]:
            comps = sorted(bd.uniq(df[(df["CategoryNorm"] == cat) & (df["EV / Battery Relevant"] == rl)])["Company"].tolist())
            if 1 <= len(comps) <= NAMES_CAP:
                emit(pick(RT_CROSS).format(a=f"{cat}", b=f"EV/Battery relevant = {rl}"),
                     "names", expected_companies=comps, expected_count=len(comps))

    # 7) Role x Relevance
    for tag in bd.ROLE_BUCKETS:
        for rl in ["Yes", "Indirect"]:
            comps = sorted(bd.uniq(bd.by_role(df[df["EV / Battery Relevant"] == rl], tag))["Company"].tolist())
            if 1 <= len(comps) <= NAMES_CAP:
                emit(pick(RT_CROSS).format(a=f"the {tag} role", b=f"EV/Battery relevant = {rl}"),
                     "names", expected_companies=comps, expected_count=len(comps))

    # 8) category + role counts
    for cat in sorted(df["CategoryNorm"].dropna().unique()):
        emit(pick(RT_COUNT).format(x=f"{cat} companies"), "count",
             expected_count=len(bd.uniq(df[df["CategoryNorm"] == cat])))

    # 9) relevance distribution (aggregate: the three counts must appear)
    rel = bd.uniq(df)["EV / Battery Relevant"].value_counts()
    emit("Report the EV/Battery relevance distribution (the Yes, No, and Indirect counts).", "aggregate",
         expected_numbers=[str(int(rel.get(k, 0))) for k in ["Yes", "No", "Indirect"]])

    # 10) argmax: global highest employment
    top = df.dropna(subset=["Employment"]).sort_values("Employment", ascending=False).drop_duplicates("Company")
    r0 = top.iloc[0]
    emit(pick(RT_SUP).format(x="company with the highest employment in the Georgia EV KB"), "superlative",
         expected_company=r0["Company"], expected_numbers=[str(int(r0["Employment"]))])

    # 11) per-county argmax (sample) + per-county sums (sample)
    big_counties = [c for c, g in df.dropna(subset=["County", "Employment"]).groupby("County")
                    if len(bd.uniq(g)) >= 2]
    for c in random.sample(big_counties, min(15, len(big_counties))):
        g = df[(df["County"] == c)].dropna(subset=["Employment"])
        tp = g.loc[g["Employment"].idxmax()]
        emit(pick(RT_SUP).format(x=f"company with the highest employment in {c}"), "superlative",
             expected_company=tp["Company"], expected_numbers=[str(int(tp["Employment"]))])
    for c in random.sample(big_counties, min(12, len(big_counties))):
        s = int(df[df["County"] == c]["Employment"].dropna().sum())
        emit(pick(RT_AGG).format(x=f"total combined employment across companies in {c}"), "aggregate",
             expected_numbers=[str(s)])

    # 12) total employment + per-category sums
    emit(pick(RT_AGG).format(x="total combined employment across all companies in the Georgia EV KB"),
         "aggregate", expected_numbers=[str(int(df["Employment"].dropna().sum()))])
    for cat in sorted(df["CategoryNorm"].dropna().unique()):
        s = int(df[df["CategoryNorm"] == cat]["Employment"].dropna().sum())
        if s > 0:
            emit(pick(RT_AGG).format(x=f"total employment of all {cat} companies"), "aggregate",
                 expected_numbers=[str(s)])

    # 13) single-source roles count
    role_counts = df.dropna(subset=["EV Supply Chain Role"]).groupby("EV Supply Chain Role")["Company"].nunique()
    singles = [r for r, c in role_counts.items() if c == 1]
    emit(pick(RT_COUNT).format(x="EV supply chain roles served by exactly one company"), "count",
         expected_count=len(singles))

    # 14) gap counties count
    bat_counties = set(bd.by_role(df, "Battery Cell")["County"].dropna()) | set(bd.by_role(df, "Battery Pack")["County"].dropna())
    t1_counties = set(df[df["CategoryNorm"] == "Tier 1"]["County"].dropna())
    gap = sorted(t1_counties - bat_counties)
    emit("State the exact number of Georgia counties that have Tier 1 suppliers but no Battery Cell or Battery Pack suppliers.",
         "count", expected_count=len(gap))

    # 15) keyword searches (small sets only) -> names
    hay = (df["Product / Service"].fillna("") + " || " + df["EV Supply Chain Role"].fillna(""))
    import re
    for kw, desc in list(bd.KEYWORDS.items()):
        sub = df[hay.str.contains(re.escape(kw), case=False)]
        comps = sorted(bd.uniq(sub)["Company"].tolist())
        if 1 <= len(comps) <= NAMES_CAP:
            emit(f"Provide the complete list of Georgia companies associated with {desc}.",
                 "names", expected_companies=comps, expected_count=len(comps))

    # 16) refusals (reserved, distinct from training)
    refusals = [
        "What is the gross profit margin of Kia Georgia Inc.?",
        "Who sits on the board of directors at Hyundai Motor Group?",
        "What is the share price of Novelis today?",
        "Where are the public EV fast-charging stations in Macon?",
        "How many electric cars were registered in Georgia in 2025?",
        "What is the humidity in Savannah right now?",
        "Which Georgia supplier will go bankrupt next year?",
        "In your opinion, which is the best-managed supplier in the KB?",
        "Give me the fax number for SK Battery America.",
        "What is the LinkedIn URL of Yazaki North America?",
        "List the EV battery factories in North Carolina.",
        "What is the dividend yield of Adient?",
        "How many vehicles per day are assembled at the Hyundai Metaplant?",
        "What is the bond rating of ZF Gainesville LLC?",
        "Who founded a company called Peachtree EV Motors LLC?",
    ]
    for q in refusals:
        emit(q, "refusal")

    # 17) none-match (reserved): empty cross-filters + empty COUNTIES (held out from training
    # via bd.RESERVED_ABSTAIN_COUNTIES) — measures the v5 abstention fix (v4 hallucinated on Glynn).
    nones = [
        "List the Georgia Charging Infrastructure companies located in DeKalb County.",
        "Which Tier 1 Georgia companies are in the Battery Cell role?",
        "Identify Georgia Vehicle Assembly companies located in Chatham County.",
    ]
    for c in sorted(bd.RESERVED_ABSTAIN_COUNTIES):
        nones.append(f"List all companies located in {c} County.")
    for q in nones:
        emit(q, "none", expected_companies=[], expected_count=0)

    with open(PROBE_FILE, "w") as f:
        for p in PROBES:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    by_kind = {}
    for p in PROBES:
        by_kind[p["kind"]] = by_kind.get(p["kind"], 0) + 1
    print(f"wrote {len(PROBES)} probes -> {PROBE_FILE}")
    print("by kind:", by_kind)
    return PROBES


# ----------------------------------------------------------------------- scoring
REFUSAL_MARKERS = ["does not", "doesn't", "not provide", "no information", "cannot",
                   "only covers", "does not contain", "not include", "no companies",
                   "there are no", "there is no", "none "]

def _ncore(name):
    """Full normalized name (NO suffix stripping). KB spelling-duplicates are already
    canonicalized in build_dataset.load_kb, so full-name matching with longest-first masking
    is both precise (distinguishes 'X' from 'X Inc.') and complete (handles mid-name 'Co.')."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", name.lower())).strip()


def detect(text, companies):
    """Precise company detector for the coverage gate. A company is present iff the core of
    its name (legal suffix stripped) appears as a contiguous substring of the answer. Cores
    are matched LONGEST-FIRST and each match consumes its text span, so a shorter contained
    name ("Hitachi Astemo") cannot also match inside a longer listed one ("Hitachi Astemo
    Americas Inc."). Models trained on canonical KB names reproduce them verbatim, so this is
    ~100% accurate on correct answers — unlike the fuzzy prefix matcher used by the legacy
    50-question scorer."""
    work = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())) + " "
    found = set()
    for c, core in sorted(((c, _ncore(c)) for c in companies), key=lambda x: -len(x[1])):
        core = core or re.sub(r"[^a-z0-9 ]", " ", c.lower()).strip()
        toks = core.split()
        if len(toks) == 1:
            span = f" {toks[0]} "
            if span in work:
                found.add(c)
                work = work.replace(span, " _ ", 1)
        elif core and core in work:
            found.add(c)
            work = work.replace(core, " _ ", 1)
    return found


def _strip_oem_clauses(text):
    """Remove 'Primary OEMs: ...' / 'OEMs: ...' field segments before company detection — the
    OEM values are themselves KB company names and must not be counted as the model's answer
    set (the listed entity is at the start of each line, before the first ' | ')."""
    out = []
    for line in text.splitlines():
        segs = re.split(r"\s*\|\s*", line)
        segs = [s for s in segs if not re.match(r"\s*(primary\s+)?oems?\s*:", s, re.I)]
        out.append(" | ".join(segs))
    return "\n".join(out)


def score_one(p, pred, ALL):
    kind = p["kind"]
    pred_comps = detect(_strip_oem_clauses(pred), ALL)
    pred_nums = set(ev.numbers_in(pred))
    model_count = ev.lead_count(pred)

    if kind == "names":
        exp = set(p["expected_companies"])
        inter = exp & pred_comps
        recall = len(inter) / len(exp) if exp else 1.0
        precision = len(inter) / len(pred_comps) if pred_comps else (1.0 if not exp else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        count_ok = (model_count == str(p["expected_count"])) if model_count is not None else None
        passed = recall >= 1.0 and precision >= 0.90
        return passed, dict(recall=round(recall, 3), precision=round(precision, 3), f1=round(f1, 3),
                            count_ok=count_ok, missing=sorted(exp - pred_comps)[:8],
                            hallucinated=sorted(pred_comps - exp)[:8])

    if kind == "count":
        ok = model_count == str(p["expected_count"])
        return bool(ok), dict(expected=p["expected_count"], model_count=model_count)

    if kind == "superlative":
        comp = p["expected_company"]
        comp_ok = comp in pred_comps
        if not comp_ok:
            comp_ok = ev.norm(comp).split()[0] in ev.norm(pred)
        num_ok = all(n in pred_nums for n in p.get("expected_numbers", []))
        return bool(comp_ok and num_ok), dict(expected_company=comp, comp_ok=comp_ok, num_ok=num_ok)

    if kind == "aggregate":
        nums = p.get("expected_numbers", [])
        present = [n for n in nums if n in pred_nums]
        ok = len(present) == len(nums)
        return bool(ok), dict(expected_numbers=nums, present=present)

    if kind == "refusal":
        declined = any(m in pred.lower() for m in REFUSAL_MARKERS)
        return bool(declined and len(pred_comps) <= 1), dict(declined=declined, named=len(pred_comps))

    if kind == "none":
        declined = any(m in pred.lower() for m in REFUSAL_MARKERS) or model_count == "0"
        return bool(declined and len(pred_comps) <= 1), dict(declined=declined, named=len(pred_comps))

    return False, {}


def score(answers_path):
    df = bd.load_kb()
    ALL = sorted(df["Company"].dropna().unique())
    probes = [json.loads(l) for l in open(PROBE_FILE)]
    raw = json.load(open(answers_path))
    # answers may be a list aligned to probes, or a dict {question: answer}
    if isinstance(raw, dict):
        answers = [raw.get(p["question"], "") for p in probes]
    else:
        answers = raw
    assert len(answers) == len(probes), f"answers ({len(answers)}) != probes ({len(probes)})"

    by_kind = {}
    recalls, precisions = [], []
    rows = []
    for p, pred in zip(probes, answers):
        ok, detail = score_one(p, pred or "", ALL)
        k = p["kind"]
        by_kind.setdefault(k, [0, 0])
        by_kind[k][0] += int(ok)
        by_kind[k][1] += 1
        if k == "names":
            recalls.append(detail["recall"])
            precisions.append(detail["precision"])
        rows.append({"question": p["question"], "kind": k, "pass": ok, **detail})

    total_pass = sum(v[0] for v in by_kind.values())
    total = sum(v[1] for v in by_kind.values())
    print(f"\n=== PROBE BENCHMARK ===  overall {total_pass}/{total} = {100*total_pass/total:.1f}%")
    for k in sorted(by_kind):
        c, n = by_kind[k]
        print(f"  {k:12s} {c}/{n} = {100*c/n:.0f}%")
    if recalls:
        mean_recall = sum(recalls) / len(recalls)
        print(f"\nCOVERAGE GATE — list questions: mean company recall = {mean_recall:.3f} "
              f"(missing-company rate = {100*(1-mean_recall):.1f}%); mean precision = {sum(precisions)/len(precisions):.3f}")
    out = bd.ROOT / "eval_results" / "probe_benchmark_scores.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"detail -> {out}")
    return rows


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "score":
        score(sys.argv[2])
    else:
        build()
