"""Tiny fixed sanity prompt set — a damage check only, NOT the main benchmark.

Confirms continued pretraining on the tiny KB didn't badly break arithmetic,
summarization, basic instruction-following, or the ability to say "I don't know"
about invented entities. Not scored against KB facts.
"""
from __future__ import annotations

import json
from pathlib import Path

SANITY_PROMPTS = [
    {"category": "arithmetic", "prompt": "Q: What is 12 + 7?\nA:", "expected_substring": "19"},
    {"category": "arithmetic", "prompt": "Q: What is 9 times 6?\nA:", "expected_substring": "54"},
    {"category": "summarization", "prompt": "Summarize in one sentence: The cat sat on the mat and then fell asleep in the sun.\nSummary:", "expected_substring": None},
    {"category": "instruction_following", "prompt": "List three colors, separated by commas.\nAnswer:", "expected_substring": None},
    {"category": "refusal_to_invent", "prompt": "Q: What is the phone number of Zyxwabc Corp?\nA:", "expected_substring": None},
]


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 40) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_instruction_sanity(run_dir: Path) -> dict:
    from ssft.eval.eval_perplexity import load_adapter_for_eval

    run_dir = Path(run_dir)
    _, base_model, adapter_model, tokenizer = load_adapter_for_eval(run_dir)

    rows = []
    for item in SANITY_PROMPTS:
        base_out = _generate(base_model, tokenizer, item["prompt"])
        adapter_out = _generate(adapter_model, tokenizer, item["prompt"])
        degraded = None
        if item["expected_substring"]:
            base_ok = item["expected_substring"] in base_out
            adapter_ok = item["expected_substring"] in adapter_out
            degraded = base_ok and not adapter_ok
        rows.append({
            "category": item["category"], "prompt": item["prompt"],
            "base_output": base_out, "adapter_output": adapter_out, "degraded": degraded,
        })

    result = {
        "note": "Damage check only — not the main benchmark. Confirms continued pretraining "
                "did not badly break basic behavior.",
        "rows": rows,
    }
    out_path = run_dir / "eval" / "instruction_sanity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result
