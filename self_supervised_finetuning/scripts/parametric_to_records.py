"""Convert my parametric raw_outputs (base/kb_only/kb_web) into the JSONL schema the
LLM_Context repo's `eval_common.direct_records()` reads, so the repo's Deterministic-V2
and DeepEval scripts can grade them UNCHANGED. Only the 42 KB questions are emitted
(kb_q01..kb_q42 -> question_number 1..42); the GNEM-Web-18 are excluded (not in the
repo's 42-question benchmark). arm is left as direct_context by the repo reader and
relabeled to `parametric` in the merge step.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/sreeja/georgia-ev-finetune/self_supervised_finetuning")
RAW = ROOT / "outputs/question_eval/raw_outputs"
OUT = ROOT / "external/LLM_Context/results/parametric/raw_json"
SYSTEMS = ["base", "kb_only", "kb_web"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for sys_name in SYSTEMS:
        data = json.loads((RAW / f"{sys_name}.json").read_text())
        kb_rows = [r for r in data["results"] if r.get("section") == "kb"]
        assert len(kb_rows) == 42, f"{sys_name}: expected 42 KB rows, got {len(kb_rows)}"
        out_path = OUT / f"answers_{sys_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in kb_rows:
                qn = int(r["question_id"].split("_q")[1])  # kb_q07 -> 7
                rec = {
                    "model_name": sys_name,
                    "question_number": qn,
                    "question": r["question"],
                    "model_response": r["output"],
                    "status": "ok",
                    "eval_count": r.get("generated_tokens"),
                    "prompt_eval_count": None,
                    "num_ctx_requested": None,
                    "total_duration": r.get("generation_time"),  # seconds; ns_to_seconds keeps <1e6 as-is
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"wrote {out_path}  ({len(kb_rows)} answers, model_name={sys_name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
