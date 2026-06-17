#!/usr/bin/env python3
"""Build instruction-tuning dataset (chat JSONL) from the Georgia EV KB — v4.

No-compromise coverage rewrite. Key invariants:
- train/valid: generated ONLY from the KB Excel (facts + precomputed aggregates).
- test: the 50 human-validated Q&A, NEVER used for training (held out, decontaminated).
- ONE counting convention everywhere: UNIQUE COMPANIES (drop_duplicates("Company")).
- Canonical RoleNorm (substring buckets, multi-tag) reproduces the gold's semantic role
  groupings (thermal->5, power-electronics/charging->4, harness->2) and stops the under-listing
  that came from filtering on exact role strings (the 26 free-text one-off roles were dropped).
- OEM token splitter: "Hyundai Kia Rivian" feeds Hyundai AND Kia AND Rivian supplier lists
  (Rivian = 6 companies, the set v3 missed).
- Exhaustive enumeration: every 1-way filter + key 2-way/3-way crosses emit FULL unique
  membership + a count == len(list) (list-then-count discipline).
- Offline teacher paraphrase bank: many natural phrasings per fact so the model binds the fact,
  not one string. Probe-benchmark phrasings are reserved (held out) for honest measurement.
- Refusals + explicit "none-match" answers restored and balanced (fixes v3's refusal regression).
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

random.seed(42)

KB_PATH = "/Users/surya/Downloads/GNEM - Auto Landscape Lat Long Updated (1).xlsx"
QA_PATH = "/Users/surya/Downloads/Human validated 50 questions (2).xlsx"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOGS = ROOT / "logs"
PROBE_FILE = DATA / "probe_benchmark.jsonl"   # held-out probe questions to decontaminate against

SYSTEM = ("You are a Georgia EV supply chain assistant. Answer only using the "
          "Georgia EV knowledge base. If the KB does not contain enough "
          "information, say so clearly.")

NO_INFO = "The Georgia EV KB does not provide that information."


# ---------------------------------------------------------------- canonical layer
ROLE_BUCKETS = ["General Automotive", "Materials", "Vehicle Assembly", "Battery Cell",
                "Battery Pack", "Thermal Management", "Power Electronics",
                "Charging Infrastructure", "Wiring Harness", "OEM Corporate Footprint"]


def role_tags(role):
    """Map a raw EV-supply-chain-role string -> set of canonical buckets (multi-tag).
    Substring rules chosen to reproduce the human gold's semantic groupings. The 26
    free-text one-off roles that match no keyword get NO bucket here; they are still
    covered by per-company facts and by the exact-role single-company generator."""
    if not isinstance(role, str):
        return frozenset()
    s = role.lower()
    t = set()
    if "battery cell" in s: t.add("Battery Cell")
    if "battery pack" in s: t.add("Battery Pack")
    if "thermal" in s: t.add("Thermal Management")
    if "power electronics" in s: t.add("Power Electronics")
    if "charging infrastructure" in s: t.add("Charging Infrastructure")
    if "harness" in s: t.add("Wiring Harness")
    if "vehicle assembly" in s: t.add("Vehicle Assembly")
    if "footprint" in s: t.add("OEM Corporate Footprint")   # role bucket; distinct from the 'OEM Footprint' Category
    if s == "materials": t.add("Materials")
    if s == "general automotive": t.add("General Automotive")
    return frozenset(t)


# Canonical OEM tokens present in the Primary OEMs column.
OEM_CANON = ["Hyundai", "Kia", "Rivian", "Blue Bird", "Club Car", "Porsche",
             "Mercedes-Benz", "Yamaha", "Textron", "SK Battery America", "Archer Aviation"]


def oem_tags(primary_oems):
    """Split a Primary OEMs cell into canonical OEM tags. 'Multiple OEMs' and missing
    values are NOT specific OEMs -> empty (handled separately)."""
    if not isinstance(primary_oems, str):
        return frozenset()
    s = primary_oems.lower()
    if s == "multiple oems":
        return frozenset()
    return frozenset(tok for tok in OEM_CANON if tok.lower() in s)


def category_norm(cat):
    """Merge the duplicate OEM-footprint category labels."""
    if not isinstance(cat, str):
        return None
    return "OEM Footprint" if cat in ("OEM (Footprint)", "OEM Footprint") else cat


def uniq(sub):
    return sub.drop_duplicates("Company")


# ---------------------------------------------------------------- load & clean
NULL_TOKENS = {"nan", "none", "n/a", "na", "null", ""}


def _clean(v):
    """Normalize any cell to a clean string or Python None (no float-nan / '<NA>' / 'nan')."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return None if s.lower() in NULL_TOKENS else s


def _name_core(n):
    s = re.sub(r"[^a-z0-9 ]", " ", n.lower())
    s = re.sub(r"\b(inc|llc|corp|corporation|ltd|lp|co|company)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_kb():
    df = pd.read_excel(KB_PATH)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(_clean)            # robust: every object cell is str or None
    df["Company"] = df["Company"].map(lambda v: str(v).strip() if v is not None else None)
    # Canonicalize KB spelling-duplicate company names (same core, different spelling) to ONE
    # spelling (the longest). KB has 2 such pairs — e.g. 'Trenton Pressing'/'Trenton Pressing
    # Inc.' (same site) — which otherwise inflate unique-company counts and confuse matching.
    canon = {}
    for nm in df["Company"].dropna().unique():
        c = _name_core(nm)
        if c not in canon or len(nm) > len(canon[c]):
            canon[c] = nm
    df["Company"] = df["Company"].map(lambda n: canon.get(_name_core(n), n) if isinstance(n, str) else n)
    df["Employment"] = pd.to_numeric(df["Employment"], errors="coerce")

    def parse_loc(v):
        if not v or not isinstance(v, str):
            return None, None
        parts = [p.strip() for p in v.split(",")]
        city = parts[0] if parts else None
        county = next((p for p in parts[1:] if "county" in p.lower()), None)
        return city, county
    df["City"], df["County"] = zip(*df["Updated Location"].map(parse_loc))
    df["FacilityNorm"] = df["Primary Facility Type"].map(
        lambda v: re.sub(r"\s+", " ", v).strip().lower() if isinstance(v, str) else None)
    df["IndustryNorm"] = df["Industry Group"].map(
        lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else None)
    df["CategoryNorm"] = df["Category"].map(category_norm)
    df["RoleTags"] = df["EV Supply Chain Role"].map(role_tags)
    df["OemTags"] = df["Primary OEMs"].map(oem_tags)
    return df


def emp_str(v):
    if pd.isna(v):
        return None
    return f"{int(v):,}" if v >= 10000 else str(int(v))


def fact(row, col):
    v = row.get(col)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v)


# helpers to filter by a canonical tag
def by_role(df, tag):
    return df[df["RoleTags"].map(lambda s: tag in s)]


def by_oem(df, tag):
    return df[df["OemTags"].map(lambda s: tag in s)]


EX = []                  # generated examples
SKIPPED = []             # questions dropped by the leakage guard

# Populated in main() before any add(): the held-out questions (50 test + probe benchmark),
# so we never train on a question that appears in either eval set (verbatim or near-dup).
TEST_Q_NORM = set()
TEST_Q_TOKENS = []


def _qnorm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def _is_test_question(q):
    """True if q matches a held-out question exactly (normalized) or by >=0.85
    token-Jaccard (near-duplicate)."""
    n = _qnorm(q)
    if n in TEST_Q_NORM:
        return True
    toks = set(n.split())
    if not toks:
        return False
    for tt in TEST_Q_TOKENS:
        inter = len(toks & tt)
        union = len(toks | tt)
        if union and inter / union >= 0.85:
            return True
    return False


def _add_one(q, a, tag):
    if _is_test_question(q):
        SKIPPED.append(q)
        return
    EX.append({"tag": tag, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]})


def add(q, a, tag):
    """q may be a single question or a list of paraphrase variants (one example each)."""
    if isinstance(q, (list, tuple)):
        for qq in q:
            _add_one(qq, a, tag)
    else:
        _add_one(q, a, tag)


def pick(*options):
    return random.choice(options)


def vary(templates, k, **slots):
    """Render k sampled paraphrases from a template bank with slot substitution."""
    rendered = [t.format(**slots) for t in templates]
    k = min(k, len(rendered))
    return random.sample(rendered, k)


# ---------------------------------------------------------------- paraphrase bank
# Offline "teacher": many natural phrasings per question shape. {x} = the filter label.
P_ROLE = [
    "Which Georgia companies have the {x} role?",
    "List the Georgia {x} suppliers.",
    "Identify the companies classified under {x} in the Georgia EV KB.",
    "Show all {x} companies in Georgia.",
    "Name the Georgia suppliers whose EV supply chain role is {x}.",
    "Map the {x} suppliers in Georgia.",
    "Find the Georgia companies in the {x} role.",
    "Which suppliers in the Georgia EV KB are {x} companies?",
    "Give me every Georgia company with a {x} EV supply chain role.",
    "Enumerate the {x} suppliers in the Georgia EV knowledge base.",
    "Who are the {x} companies in Georgia?",
    "List all Georgia firms classified as {x}.",
]
P_CATEGORY = [
    "Which Georgia companies are classified as {x}?",
    "List the {x} suppliers in the Georgia EV KB.",
    "Show all {x} companies in Georgia with their roles and products.",
    "Identify every {x} company in the Georgia EV knowledge base.",
    "Name the Georgia suppliers in the {x} category.",
    "Which suppliers fall under the {x} classification?",
    "Give me the {x} companies in Georgia.",
    "Enumerate the Georgia {x} suppliers.",
    "List all Georgia firms in the {x} tier.",
]
P_COUNTY = [
    "Which companies are located in {x}, Georgia?",
    "List the companies in {x}.",
    "Which suppliers does the Georgia EV KB place in {x}?",
    "Show the Georgia EV companies in {x}.",
    "Identify the companies based in {x}.",
    "Who operates in {x} according to the Georgia EV KB?",
    "Name the suppliers located in {x}.",
]
P_CITY = [
    "Which companies are located in {x}, Georgia?",
    "List the Georgia EV KB companies in {x}.",
    "Which suppliers are based in {x}?",
    "Show the companies in {x}.",
    "Identify the firms located in {x}.",
]
P_INDUSTRY = [
    "Which Georgia companies are in the {x} industry group?",
    "List the companies in the {x} industry group.",
    "Identify Georgia suppliers under the {x} industry group.",
    "Show the {x} industry-group companies in the Georgia EV KB.",
    "Name the Georgia firms classified in {x}.",
]
P_RELEVANCE = [
    "Which Georgia companies are marked '{x}' for EV/Battery relevance?",
    "List the companies with EV/Battery relevance = {x}.",
    "Identify the Georgia suppliers classified '{x}' for EV/battery relevance.",
    "Show every company marked '{x}' for EV relevance.",
    "Which suppliers are '{x}' EV/battery relevant?",
]
P_OEM = [
    "Which Georgia suppliers are linked to {x} through their Primary OEMs?",
    "Show the supplier network linked to {x} in Georgia, with tier and EV supply chain role.",
    "List the Georgia companies whose Primary OEMs include {x}.",
    "Identify the suppliers connected to {x} in the Georgia EV KB.",
    "Which companies serve {x} according to the Primary OEMs column?",
    "Map the {x} supplier base in Georgia.",
    "Name the Georgia suppliers tied to {x}.",
]
P_FACILITY = [
    "Which Georgia companies have a {x} facility type?",
    "List the {x} facilities in the Georgia EV KB.",
    "Identify Georgia suppliers whose primary facility type is {x}.",
    "Show the {x} sites in Georgia.",
]
P_COUNT_CAT = [
    "How many companies are classified as {x} in the Georgia EV KB?",
    "What is the exact number of {x} companies in the KB?",
    "Count the {x} companies in the Georgia EV knowledge base.",
    "How many Georgia suppliers fall under {x}?",
]
P_COUNT_ROLE = [
    "How many Georgia companies have the {x} role?",
    "What is the number of {x} suppliers in the Georgia EV KB?",
    "Count the {x} companies in Georgia.",
]


# ---------------------------------------------------------------- formatting
def fmt_list(rows, fields, max_full=50):
    """rows: list of dicts; fields: list of (label, key). Lists ALL members up to
    max_full; only genuine mega-lists (e.g. General Automotive ~135) get capped, and
    the answer says so explicitly so truncation is never taught as completeness. The
    coverage-critical filtered lists are all well under the cap (fully enumerated)."""
    lines = []
    for rr in rows:
        segs = [rr["Company"]]
        for label, key in fields:
            v = rr.get(key)
            if key == "Employment":
                v = emp_str(v)
            elif isinstance(v, float) and pd.isna(v):
                v = None
            segs.append(f"{label}: {v if v else 'not provided'}")
        lines.append(" | ".join(segs))
    if len(lines) > max_full:
        shown = "\n".join(lines[:max_full])
        return f"{shown}\n(listing the first {max_full} of {len(lines)} companies)"
    return "\n".join(lines)


def listed_answer(sub, desc, fields, k_cap=50):
    """Build a list-then-count answer from unique companies of `sub`. Count == len(list)."""
    u = uniq(sub)
    n = len(u)
    body = fmt_list(u.to_dict("records"), fields, max_full=k_cap)
    head = (f"According to the Georgia EV KB, there {'are' if n != 1 else 'is'} {n} "
            f"{desc}:" if n != 1 else
            f"According to the Georgia EV KB, there is 1 {desc}:")
    return f"{head}\n{body}", n


# ---------------------------------------------------------- per-company items
def company_examples(df):
    for name, g in df.groupby("Company", sort=False):
        rows = g.to_dict("records")
        r = rows[0]
        multi = len(rows) > 1

        # -- summary
        parts = []
        if multi:
            parts.append(f"According to the Georgia EV KB, {name} has {len(rows)} entries in Georgia.")
            for rr in rows:
                seg = []
                if fact(rr, "Updated Location"): seg.append(f"located in {rr['Updated Location']}")
                if fact(rr, "Primary Facility Type"): seg.append(f"facility type: {rr['Primary Facility Type']}")
                if fact(rr, "Product / Service"): seg.append(f"product/service: {rr['Product / Service']}")
                parts.append(f"- {name} | " + " | ".join(seg))
            parts.append(f"Category: {r['Category']}. EV Supply Chain Role: {r['EV Supply Chain Role']}.")
        else:
            s = f"According to the Georgia EV KB, {name} is a {r['Category']} company"
            if fact(r, "Updated Location"): s += f" located in {r['Updated Location']}, Georgia"
            s += "."
            if fact(r, "EV Supply Chain Role"): s += f" Its EV supply chain role is {r['EV Supply Chain Role']}."
            if fact(r, "Product / Service"): s += f" Product/service: {r['Product / Service']}."
            if fact(r, "Primary Facility Type"): s += f" Primary facility type: {r['Primary Facility Type']}."
            if emp_str(r["Employment"]): s += f" Employment: {emp_str(r['Employment'])}."
            if fact(r, "Primary OEMs"): s += f" Primary OEMs: {r['Primary OEMs']}."
            if fact(r, "EV / Battery Relevant"): s += f" EV/battery relevance: {r['EV / Battery Relevant']}."
            parts.append(s)
        add(pick(f"Summarize {name} from the Georgia EV KB.",
                 f"Give me an overview of {name}.",
                 f"Tell me about {name} in the Georgia EV knowledge base.",
                 f"What does the Georgia EV KB say about {name}?"),
            "\n".join(parts), "company_summary")

        # -- role
        roles = sorted({fact(rr, "EV Supply Chain Role") for rr in rows if fact(rr, "EV Supply Chain Role")})
        a = (f"According to the Georgia EV KB, {name}'s EV supply chain role is "
             f"{' and '.join(roles)}.") if roles else NO_INFO
        add(pick(f"What is {name}'s EV supply chain role?",
                 f"What EV supply chain role does {name} have?",
                 f"Which EV supply chain role is {name} classified under?"), a, "company_role")

        # -- products
        prods = [fact(rr, "Product / Service") for rr in rows if fact(rr, "Product / Service")]
        if prods:
            if len(prods) == 1:
                a = f"According to the Georgia EV KB, {name} provides: {prods[0]}."
            else:
                a = f"According to the Georgia EV KB, {name} provides the following across its entries:\n" + \
                    "\n".join(f"- {p}" for p in prods)
        else:
            a = NO_INFO
        add(pick(f"What products or services does {name} provide?",
                 f"What does {name} make or supply?",
                 f"Which products or services are associated with {name}?"), a, "company_products")

        # -- location
        locs = sorted({fact(rr, "Updated Location") for rr in rows if fact(rr, "Updated Location")})
        addrs = [fact(rr, "Address") for rr in rows if fact(rr, "Address")]
        if locs:
            a = f"According to the Georgia EV KB, {name} is located in {'; '.join(locs)}, Georgia."
            if addrs and not multi:
                a += f" Address: {addrs[0]}."
        else:
            a = NO_INFO
        add(pick(f"Where is {name} located?",
                 f"In which Georgia city and county is {name} located?",
                 f"What is the location of {name}?"), a, "company_location")

        # -- category
        a = f"According to the Georgia EV KB, {name} is classified as {r['Category']}."
        add(pick(f"What supplier category is {name}?",
                 f"What tier classification does {name} have in the Georgia EV KB?",
                 f"Which category is {name} assigned to?"), a, "company_category")

        # -- OEMs
        oems = sorted({fact(rr, "Primary OEMs") for rr in rows if fact(rr, "Primary OEMs")})
        a = (f"According to the Georgia EV KB, {name}'s primary OEMs are: {'; '.join(oems)}."
             if oems else f"The Georgia EV KB does not list primary OEMs for {name}.")
        add(pick(f"Which OEMs is {name} connected to?",
                 f"What are the primary OEMs of {name}?",
                 f"Which OEMs does {name} supply according to the KB?"), a, "company_oems")

        # -- employment
        emps = [emp_str(rr["Employment"]) for rr in rows if emp_str(rr["Employment"])]
        if emps:
            if multi and len(set(emps)) > 1:
                a = f"According to the Georgia EV KB, {name} employment by entry: {', '.join(emps)}."
            else:
                a = f"According to the Georgia EV KB, {name} has an employment of {emps[0]}."
        else:
            a = f"The Georgia EV KB does not provide an employment figure for {name}."
        add(pick(f"What is the employment number at {name}?",
                 f"How many people does {name} employ according to the KB?",
                 f"What is {name}'s headcount in the Georgia EV KB?"), a, "company_employment")

        # -- facility type / EV relevance
        ft = sorted({fact(rr, "Primary Facility Type") for rr in rows if fact(rr, "Primary Facility Type")})
        rel = fact(r, "EV / Battery Relevant")
        if ft:
            a = f"According to the Georgia EV KB, {name}'s primary facility type is {'; '.join(ft)}."
            if rel:
                a += f" Its EV/battery relevance is marked as '{rel}'."
            add(pick(f"What is the primary facility type of {name}?",
                     f"What kind of facility does {name} operate, and is it EV relevant?"), a, "company_facility")

        # -- multi-entry questions (mirrors eval Q6/Q11 style)
        if multi:
            lines = [f"{name} has {len(rows)} operating entries in the Georgia EV KB."]
            for rr in rows:
                lines.append(f"- Location: {fact(rr,'Updated Location') or 'not provided'} | "
                             f"Facility type: {fact(rr,'Primary Facility Type') or 'not provided'} | "
                             f"Product: {fact(rr,'Product / Service') or 'not provided'}")
            add(f"What locations does {name} operate in, and what facility types and products are associated with each?",
                "\n".join(lines), "company_multientry")


# ---------------------------------------------------------- exhaustive lists
def list_examples(df):
    # ---- 1-way: by Category (normalized) ----
    for cat in sorted(df["CategoryNorm"].dropna().unique()):
        sub = df[df["CategoryNorm"] == cat]
        a, _ = listed_answer(sub, f"{cat} compan{'ies' if len(uniq(sub))!=1 else 'y'} in Georgia",
                             [("Role", "EV Supply Chain Role"), ("Product", "Product / Service")])
        add(vary(P_CATEGORY, 7, x=cat), a, "list_category")

    # ---- 1-way: by canonical role bucket ----
    for tag in ROLE_BUCKETS:
        sub = by_role(df, tag)
        if len(uniq(sub)) == 0:
            continue
        a, _ = listed_answer(sub, f"compan{'ies' if len(uniq(sub))!=1 else 'y'} in the {tag} role",
                             [("Category", "Category"), ("Primary OEMs", "Primary OEMs"),
                              ("Employment", "Employment")])
        add(vary(P_ROLE, 8, x=tag), a, "list_role")

    # ---- exact-role single/small lists for the free-text one-off roles ----
    for role in sorted(df["EV Supply Chain Role"].dropna().unique()):
        if role_tags(role):           # already covered by a canonical bucket
            continue
        sub = df[df["EV Supply Chain Role"] == role]
        a, _ = listed_answer(sub, f"compan{'ies' if len(uniq(sub))!=1 else 'y'} with the exact "
                                  f"EV supply chain role '{role}'",
                             [("Category", "Category"), ("Product", "Product / Service")])
        add([f"Which Georgia company has the EV supply chain role '{role}'?",
             f"Identify the Georgia supplier classified as '{role}'."], a, "list_exact_role")

    # ---- 1-way: by County ----
    for county in sorted(df["County"].dropna().unique()):
        sub = df[df["County"] == county]
        a, _ = listed_answer(sub, f"compan{'ies' if len(uniq(sub))!=1 else 'y'} in {county}",
                             [("Location", "Updated Location"), ("Category", "Category"),
                              ("Role", "EV Supply Chain Role")])
        add(vary(P_COUNTY, 5, x=county), a, "list_county")

    # ---- 1-way: by City (>=2) ----
    for city in sorted(df["City"].dropna().unique()):
        sub = df[df["City"] == city]
        if len(uniq(sub)) < 2:
            continue
        a, _ = listed_answer(sub, f"companies in {city}",
                             [("Category", "Category"), ("Role", "EV Supply Chain Role")])
        add(vary(P_CITY, 4, x=city), a, "list_city")

    # ---- 1-way: by Industry group (normalized) ----
    for ind in sorted(df["IndustryNorm"].dropna().unique()):
        sub = df[df["IndustryNorm"] == ind]
        if len(uniq(sub)) < 2:
            continue
        a, _ = listed_answer(sub, f"compan{'ies' if len(uniq(sub))!=1 else 'y'} in the {ind} industry group",
                             [("Category", "Category"), ("Product", "Product / Service")])
        add(vary(P_INDUSTRY, 5, x=ind), a, "list_industry")

    # ---- 1-way: by EV/Battery relevance ----
    for rel in ["Yes", "No", "Indirect"]:
        sub = df[df["EV / Battery Relevant"] == rel]
        a, _ = listed_answer(sub, f"compan{'ies' if len(uniq(sub))!=1 else 'y'} marked '{rel}' for EV/Battery relevance",
                             [("Category", "Category"), ("Role", "EV Supply Chain Role")])
        add(vary(P_RELEVANCE, 5, x=rel), a, "list_relevance")

    # ---- 1-way: by OEM (the Rivian fix — full membership per OEM) ----
    for tag in OEM_CANON:
        sub = by_oem(df, tag)
        if len(uniq(sub)) == 0:
            continue
        a, _ = listed_answer(sub, f"Georgia compan{'ies' if len(uniq(sub))!=1 else 'y'} linked to {tag} "
                                  f"(via the Primary OEMs column)",
                             [("Category", "Category"), ("Role", "EV Supply Chain Role"),
                              ("Primary OEMs", "Primary OEMs")])
        add(vary(P_OEM, 7, x=tag), a, "list_oem")

    # companies serving "Multiple OEMs" (diversified base), by tier
    mo = df[df["Primary OEMs"] == "Multiple OEMs"]
    for cat in ["Tier 1", "Tier 1/2", "Tier 2/3"]:
        sub = mo[mo["CategoryNorm"] == cat]
        if len(uniq(sub)) == 0:
            continue
        a, _ = listed_answer(sub, f"{cat} Georgia suppliers serving Multiple OEMs (diversified customer base)",
                             [("Role", "EV Supply Chain Role")])
        add([f"Which Georgia {cat} suppliers serve Multiple OEMs (a diversified customer base)?",
             f"List the {cat} companies whose Primary OEMs are 'Multiple OEMs'.",
             f"Identify diversified {cat} Georgia suppliers (Multiple OEMs)."], a, "list_multi_oem")

    # ---- 1-way: by facility type (R&D, Headquarters) ----
    for keys, label in [(["r&d"], "R&D"),
                        (["headquarters", "north american headquarters"], "Headquarters")]:
        sub = df[df["FacilityNorm"].isin(keys)]
        if len(uniq(sub)) == 0:
            continue
        a, _ = listed_answer(sub, f"{label} facilit{'ies' if len(uniq(sub))!=1 else 'y'} in the Georgia EV KB",
                             [("Location", "Updated Location"), ("Product", "Product / Service")])
        add(vary(P_FACILITY, 4, x=label), a, "list_facility")


# ---------------------------------------------------------- cross-product lists
def cross_examples(df):
    cats = sorted(df["CategoryNorm"].dropna().unique())

    # ---- 2-way: Category x Role bucket ----
    for cat in cats:
        for tag in ROLE_BUCKETS:
            sub = by_role(df[df["CategoryNorm"] == cat], tag)
            n = len(uniq(sub))
            if n == 0:
                continue
            a, _ = listed_answer(sub, f"{cat} compan{'ies' if n!=1 else 'y'} in the {tag} role",
                                 [("Product", "Product / Service"), ("Primary OEMs", "Primary OEMs")])
            add([f"Which {cat} Georgia companies are in the {tag} role?",
                 f"List the {cat} suppliers with the {tag} EV supply chain role.",
                 f"Identify {cat} companies classified under {tag}.",
                 f"Show the {cat} {tag} suppliers in Georgia."], a, "cross_cat_role")

    # ---- 2-way: Category x Relevance ----
    relword = {"Yes": "directly EV/battery relevant", "No": "not EV/battery relevant",
               "Indirect": "indirectly EV/battery relevant"}
    for cat in cats:
        for rl, word in relword.items():
            sub = df[(df["CategoryNorm"] == cat) & (df["EV / Battery Relevant"] == rl)]
            n = len(uniq(sub))
            if n == 0:
                continue
            a, _ = listed_answer(sub, f"{cat} compan{'ies' if n!=1 else 'y'} that {'are' if n!=1 else 'is'} {word}",
                                 [("Role", "EV Supply Chain Role")])
            add([f"Which {cat} Georgia companies are {word} (EV/Battery Relevant = {rl})?",
                 f"List the {cat} suppliers marked '{rl}' for EV/battery relevance, with their roles.",
                 f"Identify {cat} companies whose EV/battery relevance is '{rl}'."], a, "cross_cat_relevance")

    # ---- 2-way: Category x OEM ----
    for cat in ["Tier 1", "Tier 1/2", "Tier 2/3", "OEM"]:
        for tag in ["Hyundai", "Kia", "Rivian"]:
            sub = by_oem(df[df["CategoryNorm"] == cat], tag)
            n = len(uniq(sub))
            if n == 0:
                continue
            a, _ = listed_answer(sub, f"{cat} Georgia compan{'ies' if n!=1 else 'y'} linked to {tag}",
                                 [("Role", "EV Supply Chain Role"), ("Primary OEMs", "Primary OEMs")])
            add([f"Which {cat} Georgia suppliers are linked to {tag}?",
                 f"List the {cat} companies whose Primary OEMs include {tag}."], a, "cross_cat_oem")

    # ---- 2-way: Role x Relevance ----
    for tag in ROLE_BUCKETS:
        for rl, word in relword.items():
            sub = df[(df["EV / Battery Relevant"] == rl)]
            sub = by_role(sub, tag)
            n = len(uniq(sub))
            if n == 0:
                continue
            a, _ = listed_answer(sub, f"{tag} compan{'ies' if n!=1 else 'y'} marked '{rl}' for EV/battery relevance",
                                 [("Category", "Category")])
            add([f"Which Georgia {tag} companies are marked '{rl}' for EV/battery relevance?",
                 f"List {tag} suppliers whose EV/battery relevance is '{rl}'."], a, "cross_role_relevance")

    # ---- 2-way: employment threshold x constraint (Q25/Q29/Q32/Q38 shapes) ----
    def t23_ga_300(d):
        s = d[(d["CategoryNorm"] == "Tier 2/3") & (d["Employment"] > 300)]
        return by_role(s, "General Automotive")
    sub = t23_ga_300(df)
    a, _ = listed_answer(sub, "Tier 2/3 General Automotive suppliers with employment over 300",
                         [("Employment", "Employment")])
    add(["Find Tier 2/3 Georgia suppliers with more than 300 employees classified as General Automotive.",
         "Which Tier 2/3 General Automotive Georgia companies employ over 300 people?",
         "List Georgia Tier 2/3 General Automotive suppliers whose employment exceeds 300."],
        a, "cross_emp_t23ga300")

    sub = df[(df["RoleTags"].map(lambda s: ("Thermal Management" in s) or ("Power Electronics" in s)))
             & (df["Employment"] < 200)]
    a, _ = listed_answer(sub, "Thermal Management or Power Electronics companies with fewer than 200 employees",
                         [("Employment", "Employment"), ("Role", "EV Supply Chain Role")])
    add(["Which Georgia Thermal Management or Power Electronics companies have fewer than 200 employees?",
         "List small (under 200 employees) Thermal Management or Power Electronics suppliers in Georgia.",
         "Identify Georgia Thermal Management/Power Electronics firms with headcount below 200."],
        a, "cross_emp_tmpe200")

    sub = df[(df["Employment"] > 1000) & (df["EV / Battery Relevant"] == "Indirect")]
    a, _ = listed_answer(sub, "companies with over 1,000 employees marked only Indirectly EV-relevant",
                         [("Employment", "Employment"), ("Role", "EV Supply Chain Role")])
    add(['Which Georgia companies employ over 1,000 workers but are only Indirectly EV-relevant?',
         "List large Georgia employers (1,000+) categorized as Indirectly relevant to EVs.",
         "Identify Georgia firms with 1,000+ employees that are only indirectly EV relevant."],
        a, "cross_emp_big_indirect")

    # ---- 3-way: Category x Role x Relevance (key combos that are non-empty) ----
    for cat in cats:
        for tag in ["Battery Cell", "Battery Pack", "Thermal Management", "Power Electronics",
                    "Materials", "Vehicle Assembly"]:
            for rl in ["Yes", "Indirect"]:
                sub = by_role(df[(df["CategoryNorm"] == cat) & (df["EV / Battery Relevant"] == rl)], tag)
                n = len(uniq(sub))
                if n == 0:
                    continue
                a, _ = listed_answer(sub, f"{cat} {tag} compan{'ies' if n!=1 else 'y'} marked '{rl}' for EV/battery relevance",
                                     [("Product", "Product / Service")])
                add([f"Which {cat} Georgia {tag} companies are '{rl}' EV/battery relevant?",
                     f"List {cat} {tag} suppliers with EV/battery relevance '{rl}'."], a, "cross_3way")


# ---------------------------------------------------------- counts / aggregates
def agg_examples(df):
    u = uniq(df)

    # ---- category distribution + total ----
    vc = u["CategoryNorm"].value_counts()
    tbl = "\n".join(f"{k}: {v}" for k, v in vc.items())
    add(["How many companies are in each supplier Category in the Georgia EV KB?",
         "Give the company count for each supplier Category.",
         "Break down the Georgia EV KB by Category with the number of companies in each.",
         "What is the distribution of companies across supplier categories?"],
        f"According to the Georgia EV KB, the number of companies in each Category is:\n{tbl}\n"
        f"Total: {u['Company'].nunique()} companies.", "agg_category_dist")
    for cat in vc.index:
        sub = uniq(df[df["CategoryNorm"] == cat])   # per-category unique (a company with an OEM row
        n = len(sub)                                # counts toward OEM even if its first row differs)
        if n <= 50:   # ground the count in an enumerated list (count == listed)
            body = fmt_list(sub.to_dict("records"), [("Role", "EV Supply Chain Role")])
            ans = f"According to the Georgia EV KB, there are {n} companies classified as {cat}:\n{body}"
        else:
            ans = f"According to the Georgia EV KB, there are {n} companies classified as {cat}."
        add(vary(P_COUNT_CAT, 3, x=cat), ans, "agg_category_count")

    # ---- relevance distribution ----
    rel = u["EV / Battery Relevant"].value_counts()
    line = ", ".join(f"{k}: {int(rel.get(k, 0))}" for k in ["Yes", "No", "Indirect"])
    add(["How many companies are marked Yes, No, and Indirect for EV/Battery relevance?",
         "Break down the Georgia EV KB by EV/Battery relevance with counts.",
         "What is the EV/Battery relevance distribution across all companies?"],
        f"According to the Georgia EV KB, companies by EV/Battery relevance are — {line}.",
        "agg_relevance_dist")

    # ---- role distribution (canonical buckets) ----
    rb = {tag: len(uniq(by_role(df, tag))) for tag in ROLE_BUCKETS}
    rb = {k: v for k, v in rb.items() if v}
    tbl = "\n".join(f"{k}: {v}" for k, v in sorted(rb.items(), key=lambda x: -x[1]))
    add(["How many Georgia companies have each EV supply chain role?",
         "Break down the companies by EV supply chain role with counts."],
        f"According to the Georgia EV KB, the number of companies in each EV supply chain role is:\n{tbl}",
        "agg_role_dist")
    for tag, n in rb.items():
        if n <= 50:   # ground the count in an enumerated list
            body = fmt_list(uniq(by_role(df, tag)).to_dict("records"), [("Category", "Category")])
            ans = (f"According to the Georgia EV KB, there {'are' if n != 1 else 'is'} {n} "
                   f"compan{'ies' if n != 1 else 'y'} in the {tag} role:\n{body}")
        else:
            ans = f"According to the Georgia EV KB, there are {n} companies in the {tag} role."
        add(vary(P_COUNT_ROLE, 3, x=tag), ans, "agg_role_count")

    # ---- single-source roles (SPOF): exact roles with exactly one company ----
    role_counts = df.dropna(subset=["EV Supply Chain Role"]).groupby("EV Supply Chain Role")["Company"].nunique()
    singles = sorted([role for role, c in role_counts.items() if c == 1])
    body = "\n".join(f"{role} | Only company: {df[df['EV Supply Chain Role']==role]['Company'].iloc[0]}"
                     for role in singles)
    add(["Which EV supply chain roles in Georgia are served by only a single company (single-point-of-failure risk)?",
         "Identify the EV supply chain roles covered by exactly one Georgia company.",
         "List the single-source EV roles in the Georgia EV KB, flagging concentration risk."],
        f"According to the Georgia EV KB, there are {len(singles)} EV supply chain roles served by only a single company:\n{body}",
        "agg_single_source")

    # ---- employment sums + argmax ----
    add(["What is the total combined employment across all companies in the Georgia EV KB?",
         "Sum the employment across the entire Georgia EV KB.",
         "What is the aggregate headcount of all companies in the KB?"],
        f"According to the Georgia EV KB, the total combined employment across all companies is "
        f"{emp_str(df['Employment'].dropna().sum())}.", "agg_total_emp")

    top = df.dropna(subset=["Employment"]).sort_values("Employment", ascending=False).drop_duplicates("Company")
    r0 = top.iloc[0]
    add(["Which company in the Georgia EV KB has the highest employment, and how many employees?",
         "What is the single largest employer in the Georgia EV KB?",
         "Name the company with the most employees in the KB and its headcount."],
        f"According to the Georgia EV KB, {r0['Company']} has the highest employment with "
        f"{emp_str(r0['Employment'])} employees.", "agg_argmax_emp")
    for k in (5, 10):
        body = fmt_list(top.head(k).to_dict("records"), [("Employment", "Employment"), ("Role", "EV Supply Chain Role")])
        add([f"List the top {k} Georgia companies by employment size.",
             f"Which {k} companies have the most employees in the Georgia EV KB?"],
            f"According to the Georgia EV KB, the top {k} companies by employment are:\n{body}", "agg_topk_emp")

    # ---- per-county + per-category employment sums and county argmax ----
    cat_sum = df.dropna(subset=["Employment"]).groupby("CategoryNorm")["Employment"].sum()
    for cat, s in cat_sum.items():
        if s > 0:
            add([f"What is the total employment of all {cat} companies in the Georgia EV KB?"],
                f"According to the Georgia EV KB, the total employment of all {cat} companies is {emp_str(s)}.",
                "agg_cat_total_emp")

    county_emp = df.dropna(subset=["County", "Employment"]).groupby("County")["Employment"].sum().sort_values(ascending=False)
    add(["Which Georgia county has the highest total employment across all companies, and what is the combined total?",
         "Across all companies, which county has the greatest combined employment?",
         "Sum employment by county — which county is highest, and what is the figure?"],
        f"According to the Georgia EV KB, {county_emp.index[0]} has the highest total employment across all companies: "
        f"{emp_str(county_emp.iloc[0])}.", "agg_top_emp_county")
    t1_emp = df[df["CategoryNorm"] == "Tier 1"].dropna(subset=["County", "Employment"]).groupby("County")["Employment"].sum().sort_values(ascending=False)
    add(["Among Tier 1 suppliers only, which Georgia county has the highest total employment?",
         "Looking at Tier 1 companies, which county leads in combined employment, and what is the total?"],
        f"According to the Georgia EV KB, {t1_emp.index[0]} has the highest total employment among Tier 1 suppliers, "
        f"with a total of {emp_str(t1_emp.iloc[0])} employees.", "agg_top_emp_county_t1")

    for county, g in df.dropna(subset=["County", "Employment"]).groupby("County"):
        if len(uniq(g)) < 2:
            continue
        tp = g.loc[g["Employment"].idxmax()]
        add([f"In {county}, which company has the highest employment, and what is its EV supply chain role?",
             f"Who is the largest employer in {county} per the Georgia EV KB, and what role do they have?"],
            f"According to the Georgia EV KB, {tp['Company']} has the highest employment in {county} "
            f"(Employment: {emp_str(tp['Employment'])}; Role: {tp['EV Supply Chain Role']}).",
            "agg_county_argmax")
        s = g["Employment"].sum()
        add([f"What is the total combined employment across companies in {county}?"],
            f"According to the Georgia EV KB, the combined employment across companies in {county} is {emp_str(s)}.",
            "agg_county_total_emp")

    # ---- site-selection: gap counties (Tier 1 but no Battery Cell/Pack) ----
    bat_counties = set(by_role(df, "Battery Cell")["County"].dropna()) | set(by_role(df, "Battery Pack")["County"].dropna())
    t1_counties = set(df[df["CategoryNorm"] == "Tier 1"]["County"].dropna())
    gap = sorted(t1_counties - bat_counties)
    add(["Which Georgia counties have Tier 1 suppliers but no Battery Cell or Battery Pack suppliers?",
         "Identify Georgia areas with Tier 1 infrastructure but lacking battery cell/pack suppliers.",
         "List the counties that have Tier 1 suppliers yet no battery cell or pack presence (using Updated Location)."],
        f"According to the Georgia EV KB, there are {len(gap)} Georgia counties that have Tier 1 suppliers but no "
        f"Battery Cell/Pack suppliers (using Updated Location):\n" + ", ".join(gap), "agg_gap_counties")

    # ---- OEM Footprint / OEM Supply Chain categories (Q33) ----
    sub = df[df["CategoryNorm"].isin(["OEM Footprint", "OEM Supply Chain"])]
    a, _ = listed_answer(sub, "Georgia companies classified as OEM Footprint or OEM Supply Chain",
                         [("Category", "Category"), ("Role", "EV Supply Chain Role")])
    add(["Which Georgia companies are classified as OEM Footprint or OEM Supply Chain?",
         "Identify the OEM Footprint and OEM Supply Chain companies in the Georgia EV KB.",
         "List Georgia companies in the OEM Footprint or OEM Supply Chain categories."], a, "agg_oem_footprint")

    # ---- Original Equipment Manufacturer affiliation type (Q5) ----
    sub = df[df["Supplier or Affiliation Type"] == "Original Equipment Manufacturer"]
    a, _ = listed_answer(sub, "Georgia companies classified as Original Equipment Manufacturer",
                         [("Role", "EV Supply Chain Role")])
    add(["Which Georgia companies are classified as Original Equipment Manufacturer, and what roles do they cover?",
         "List the Original Equipment Manufacturer companies in the Georgia EV KB with their EV supply chain roles."],
        a, "agg_oem_affiliation")


# ---------------------------------------------------------- product keyword search
KEYWORDS = {
    "battery": "battery-related products",
    "lithium": "lithium-ion battery materials",
    "thermal": "thermal-related products or services",
    "wiring harness": "wiring harnesses",
    "harness": "harness-related products",
    "copper foil": "copper foil",
    "electrodeposited": "electrodeposited materials",
    "powder coating": "powder coating-related products or services",
    "recycl": "battery or materials recycling",
    "composite": "composite materials",
    "aluminum": "aluminum products",
    "plastic": "plastics or polymer products",
    "polymer": "polymer materials",
    "DC-to-DC": "DC-to-DC converters",
    "inverter": "inverters",
    "motor controller": "motor controllers",
    "high-voltage": "high-voltage components",
    "high voltage": "high-voltage components",
    "chemical": "chemical products",
    "electrolyte": "battery electrolytes",
    "anode": "anode materials",
    "cathode": "cathode materials",
    "capacitor": "capacitors",
    "R&D": "research and development activity",
    "prototyp": "prototyping activity",
    "stamping": "stamping products",
    "stamped": "stamped components",
    "seat": "seating products",
    "tire": "tire products",
    "axle": "axles",
    "transmission": "transmissions",
    "brake": "brake products",
    "steel": "steel products",
    "glass": "glass products",
    "connector": "connectors",
    "sensor": "sensors",
    "textile": "textile or interior components",
}


def keyword_examples(df):
    hay = (df["Product / Service"].fillna("") + " || " + df["EV Supply Chain Role"].fillna(""))
    for kw, desc in KEYWORDS.items():
        mask = hay.str.contains(re.escape(kw), case=False)
        sub = df[mask]
        n = len(uniq(sub))
        if n == 0:
            add([f"Which Georgia companies provide {desc}, according to the KB?",
                 f"List Georgia companies whose products relate to {desc}."],
                f"According to the Georgia EV KB, no companies are identified as providing {desc}.",
                "kw_none")
            continue
        a, _ = listed_answer(sub, f"Georgia compan{'ies' if n!=1 else 'y'} matching {desc}",
                             [("Category", "Category"), ("Product", "Product / Service")])
        add([f"Which Georgia companies provide {desc}, and what tier are they classified under?",
             f"Find Georgia companies whose products relate to {desc}.",
             f"List the Georgia suppliers associated with {desc}.",
             f"Identify Georgia companies producing {desc}."], a, "kw_search")


# ---------------------------------------------------------- refusals / none-match
def refusal_examples():
    refusals = [
        ("What is the annual revenue of Kia Georgia Inc.?", "revenue or financial-performance"),
        ("What was Hyundai Motor Group's revenue last year?", "revenue or financial-performance"),
        ("When was Novelis Inc. founded?", "company founding dates"),
        ("What year did SK Battery America open?", "company founding or opening dates"),
        ("Who is the CEO of Hyundai MOBIS (Georgia)?", "executive or leadership"),
        ("Who runs Kia Georgia Inc.?", "executive or leadership"),
        ("What is the stock price of Adient?", "stock or financial-market"),
        ("What is the market capitalization of Novelis?", "stock or financial-market"),
        ("List EV charging stations in Atlanta.", "EV charging-station location"),
        ("Where can I charge my EV in Savannah?", "EV charging-station location"),
        ("How many EVs were sold in Georgia last year?", "EV sales-volume"),
        ("What is the population of Gwinnett County?", "demographic or population"),
        ("What is the weather in Atlanta today?", "weather"),
        ("Which Georgia companies will expand in 2027?", "future plans or predictions"),
        ("Predict which supplier will win the next battery contract.", "predictions or forecasts"),
        ("What is your opinion on the best EV supplier in Georgia?", "opinions or recommendations"),
        ("Rate the Georgia EV suppliers from best to worst.", "opinions, ratings, or rankings of quality"),
        ("What are the phone numbers of the Tier 1 suppliers?", "contact phone-number"),
        ("Give me the email addresses of these companies.", "contact email"),
        ("What is the website of Hyundai Transys Georgia Powertrain?", "website or URL"),
        ("How much did Rivian invest in Georgia?", "investment-amount"),
        ("What are the EV suppliers in Alabama?", "out-of-state (Georgia only)"),
        ("List the EV battery plants in Tennessee.", "out-of-state (Georgia only)"),
        ("Which companies in South Carolina make batteries?", "out-of-state (Georgia only)"),
        ("What is the credit rating of Yazaki North America?", "credit-rating"),
        ("How many patents does ZF Gainesville hold?", "patent"),
        ("What union represents workers at Kia Georgia?", "labor-union"),
        ("What is the square footage of the Hyundai Metaplant?", "facility square-footage"),
        ("When does the next shift start at SK Battery America?", "operational scheduling"),
        ("What is the VIN of vehicles made at Kia Georgia?", "vehicle identification"),
        ("Who are the customers of a company called Tesla Georgia LLC?", "a company not in the KB"),
        ("Tell me about Panasonic's gigafactory in Kansas.", "out-of-state (Georgia only)"),
        ("What is the EBITDA of Adient?", "financial-performance"),
        ("Which supplier has the best work culture?", "opinions on work culture"),
    ]
    for q, kind in refusals:
        a = f"The Georgia EV knowledge base does not provide {kind} information."
        add(q, a, "refusal")

    # explicit none-match for plausible-but-empty KB filters (teach 'none', not hallucinate)
    nones = [
        ("Identify Georgia Tier 2/3 companies in the Electronic and Other Electrical Equipment and Components "
         "industry group that could be upgraded to supply EV power electronics.",
         "There are no Georgia Tier 2/3 companies listed under the Electronic and Other Electrical Equipment and "
         "Components industry group, so none can be identified for that upgrade based on the KB."),
        ("Which Georgia Tier 1/2 companies produce engineered plastics, polymers, or composite materials applicable "
         "to EV structural or thermal components?",
         "No Georgia Tier 1/2 companies are explicitly identified in the KB as producing engineered plastics, polymers, "
         "or composite materials applicable to EV structural or thermal components."),
        ("Which Georgia companies are classified under the Charging Infrastructure role and located in Fulton County?",
         "According to the Georgia EV KB, there are no Charging Infrastructure companies located in Fulton County."),
    ]
    for q, a in nones:
        add(q, a, "none_match")


# The 159 Georgia counties (public reference data). Used to teach ABSTENTION: the KB only
# covers ~50 of them, so a query about any other county must say "none", not hallucinate
# (v4 invented companies in Glynn County — the abstention failure this fixes).
GA_COUNTIES = [
    "Appling", "Atkinson", "Bacon", "Baker", "Baldwin", "Banks", "Barrow", "Bartow", "Ben Hill",
    "Berrien", "Bibb", "Bleckley", "Brantley", "Brooks", "Bryan", "Bulloch", "Burke", "Butts",
    "Calhoun", "Camden", "Candler", "Carroll", "Catoosa", "Charlton", "Chatham", "Chattahoochee",
    "Chattooga", "Cherokee", "Clarke", "Clay", "Clayton", "Clinch", "Cobb", "Coffee", "Colquitt",
    "Columbia", "Cook", "Coweta", "Crawford", "Crisp", "Dade", "Dawson", "Decatur", "DeKalb",
    "Dodge", "Dooly", "Dougherty", "Douglas", "Early", "Echols", "Effingham", "Elbert", "Emanuel",
    "Evans", "Fannin", "Fayette", "Floyd", "Forsyth", "Franklin", "Fulton", "Gilmer", "Glascock",
    "Glynn", "Gordon", "Grady", "Greene", "Gwinnett", "Habersham", "Hall", "Hancock", "Haralson",
    "Harris", "Hart", "Heard", "Henry", "Houston", "Irwin", "Jackson", "Jasper", "Jeff Davis",
    "Jefferson", "Jenkins", "Johnson", "Jones", "Lamar", "Lanier", "Laurens", "Lee", "Liberty",
    "Lincoln", "Long", "Lowndes", "Lumpkin", "Macon", "Madison", "Marion", "McDuffie", "McIntosh",
    "Meriwether", "Miller", "Mitchell", "Monroe", "Montgomery", "Morgan", "Murray", "Muscogee",
    "Newton", "Oconee", "Oglethorpe", "Paulding", "Peach", "Pickens", "Pierce", "Pike", "Polk",
    "Pulaski", "Putnam", "Quitman", "Rabun", "Randolph", "Richmond", "Rockdale", "Schley", "Screven",
    "Seminole", "Spalding", "Stephens", "Stewart", "Sumter", "Talbot", "Taliaferro", "Tattnall",
    "Taylor", "Telfair", "Terrell", "Thomas", "Tift", "Toombs", "Towns", "Treutlen", "Troup",
    "Turner", "Twiggs", "Union", "Upson", "Walker", "Walton", "Ware", "Warren", "Washington",
    "Wayne", "Webster", "Wheeler", "White", "Whitfield", "Wilcox", "Wilkes", "Wilkinson", "Worth"]

ABSTAIN_COUNTY_Q = [
    "List all companies located in {x} County.",
    "Which companies in the dataset are located in {x} County?",
    "Show the Georgia EV KB companies in {x} County.",
    "Identify the suppliers based in {x} County.",
    "What companies does the Georgia EV KB list in {x} County?"]


# held out for the probe benchmark (NOT trained on) so abstention generalization is measured
RESERVED_ABSTAIN_COUNTIES = {"Glynn", "Echols", "Quitman", "Taliaferro", "Webster"}


def abstain_examples(df):
    """Teach the model to say 'none' for empty queries instead of hallucinating."""
    kb_counties = {str(c).replace(" County", "").strip().lower() for c in df["County"].dropna()}
    missing = [c for c in GA_COUNTIES if c.lower() not in kb_counties and c not in RESERVED_ABSTAIN_COUNTIES]
    random.shuffle(missing)
    for c in missing[:80]:
        add(vary(ABSTAIN_COUNTY_Q, 2, x=c),
            f"According to the Georgia EV KB, there are no companies located in {c} County.",
            "abstain_county")
    # empty Category x Role-bucket combinations -> say none
    empties = []
    for cat in sorted(df["CategoryNorm"].dropna().unique()):
        for tag in ROLE_BUCKETS:
            if len(uniq(by_role(df[df["CategoryNorm"] == cat], tag))) == 0:
                empties.append((cat, tag))
    random.shuffle(empties)
    for cat, tag in empties[:45]:
        add([f"Which {cat} Georgia companies are in the {tag} role?",
             f"List the {cat} suppliers with the {tag} EV supply chain role."],
            f"According to the Georgia EV KB, there are no {cat} companies in the {tag} role.",
            "abstain_cross")


# ----------------------------------------------------------------------- main
def main():
    df = load_kb()

    # Load held-out questions BEFORE generating so the leakage guard drops any collision.
    # (1) the 50 human Q&A; (2) the probe-benchmark questions, if already generated.
    qa = pd.read_excel(QA_PATH)
    held_out = list(qa["Question"].astype(str))
    if PROBE_FILE.exists():
        for line in open(PROBE_FILE):
            try:
                rec = json.loads(line)
                held_out.append(rec["question"])
            except Exception:
                pass
    for q in held_out:
        n = _qnorm(q)
        TEST_Q_NORM.add(n)
        TEST_Q_TOKENS.append(set(n.split()))

    company_examples(df)
    list_examples(df)
    cross_examples(df)
    agg_examples(df)
    keyword_examples(df)
    refusal_examples()
    abstain_examples(df)

    random.shuffle(EX)
    n_valid = max(1, int(len(EX) * 0.08))
    valid, train = EX[:n_valid], EX[n_valid:]

    # Balanced oversampling. Lists/crosses/aggregates carry the exact memberships, so they
    # get a modest boost; refusals + none-match are lifted so they are NOT drowned (the v3
    # refusal regression). Per-company facts already number in the thousands -> no boost.
    boost = defaultdict(int)
    counts = defaultdict(int)
    for e in train:
        counts[e["tag"]] += 1
    boosted = []
    for e in train:
        boosted.append(e)
        tag = e["tag"]
        if tag in ("refusal", "none_match"):
            boosted.extend([e] * 5)               # protect refusals (v3 regression)
        elif tag in ("abstain_county", "abstain_cross"):
            boosted.extend([e] * 1)               # teach 'none' for empty queries (v4 Glynn hallucination); modest so we don't over-abstain
        elif tag.startswith(("list_", "cross_", "agg_", "kw_")):
            boosted.extend([e] * 2)
        elif tag == "company_multientry":
            boosted.extend([e])
    train = boosted
    random.shuffle(train)

    DATA.mkdir(exist_ok=True)
    with open(DATA / "train.jsonl", "w") as f:
        for e in train:
            f.write(json.dumps({"messages": e["messages"]}, ensure_ascii=False) + "\n")
    with open(DATA / "valid.jsonl", "w") as f:
        for e in valid:
            f.write(json.dumps({"messages": e["messages"]}, ensure_ascii=False) + "\n")

    # test set = the 50 human Q&A (NEVER trained on)
    with open(DATA / "test.jsonl", "w") as f:
        for _, r in qa.iterrows():
            f.write(json.dumps({
                "question_id": int(r["Num"]),
                "use_case_category": str(r["Use Case Category"]),
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": str(r["Question"]).strip()},
                    {"role": "assistant", "content": str(r["Human validated answers"]).strip()},
                ]}, ensure_ascii=False) + "\n")

    # stats
    tags = defaultdict(int)
    refusal_n = none_n = 0
    for e in train:
        tags[e["tag"]] += 1
        if e["tag"] == "refusal":
            refusal_n += 1
        if e["tag"] == "none_match":
            none_n += 1
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / "dataset_report.md", "w") as f:
        f.write("# Dataset Report (v4)\n\n")
        f.write(f"- KB source: `{KB_PATH}` ({len(df)} rows, {df['Company'].nunique()} unique companies)\n")
        f.write(f"- Convention: **unique companies** everywhere; RoleNorm buckets: {ROLE_BUCKETS}\n")
        f.write(f"- Held-out (never trained): 50 human Q&A"
                f"{' + probe benchmark' if PROBE_FILE.exists() else ''}\n")
        f.write(f"- **Leakage guard:** {len(SKIPPED)} generated questions dropped (exact or >=0.85 Jaccard).\n\n")
        f.write(f"| split | examples |\n|---|---|\n| train | {len(train)} |\n| valid | {len(valid)} |\n| test | {len(qa)} |\n\n")
        f.write(f"- refusal examples in train: {refusal_n}  ({100*refusal_n/max(1,len(train)):.1f}%)\n")
        f.write(f"- none_match examples in train: {none_n}\n\n")
        f.write("## Examples by generator tag (post-oversample)\n\n")
        for t, c in sorted(tags.items(), key=lambda x: -x[1]):
            f.write(f"- {t}: {c}\n")
    print(f"train={len(train)} valid={len(valid)} test={len(qa)} | leakage-dropped={len(SKIPPED)} "
          f"| refusals={refusal_n} none_match={none_n}")
    print("top tags:", dict(sorted(tags.items(), key=lambda x: -x[1])[:20]))


if __name__ == "__main__":
    main()
