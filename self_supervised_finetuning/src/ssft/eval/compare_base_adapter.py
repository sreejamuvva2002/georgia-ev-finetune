"""Orchestrates perplexity + cloze evaluation into one comparison.json artifact."""
from __future__ import annotations

import json
from pathlib import Path


def run_comparison(run_dir: Path) -> dict:
    from ssft.eval.eval_cloze import run_cloze_eval
    from ssft.eval.eval_perplexity import run_perplexity_eval

    run_dir = Path(run_dir)
    perplexity = run_perplexity_eval(run_dir)
    cloze = run_cloze_eval(run_dir)

    result = {"perplexity": perplexity, "cloze": cloze}
    out_path = run_dir / "eval" / "comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result
