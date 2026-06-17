# Georgia EV Supply-Chain — LLM Fine-Tuning

Fine-tuning a local LLM to answer questions about Georgia's electric-vehicle supply chain
(companies, supply-chain roles, OEM relationships, counties, employment, etc.) from a curated
knowledge base — with **maximal accuracy and full company coverage**, trained entirely on-device.

- **Base model:** `Qwen2.5-14B-Instruct` (MLX 8-bit)
- **Method:** LoRA fine-tuning via [`mlx-lm`](https://github.com/ml-explore/mlx-lm) on Apple Silicon (Metal). Pure fine-tuning, no retrieval.
- **Data:** auto-generated from the KB under one canonical convention (unique-company counting,
  normalized supply-chain roles, OEM-token splitting), plus an offline paraphrase bank, refusals,
  and "none-match" classes. The 50 human-validated Q&A are **held out** (never trained on).
- **Evaluation:** a held-out **probe benchmark** (primary metric — pass rate + company recall) plus
  the 50 human Q&A as a secondary reference.

## Repository layout

```
training_project/
  scripts/
    build_dataset.py          # generate train/valid/test from the KB (canonical conventions)
    build_probe_benchmark.py  # held-out probe benchmark + scorer (primary metric)
    validate_dataset.py       # integrity / leakage gates
    train_mlx.py              # LoRA fine-tune (--smoke / --stability / --resume)
    lora_14b.yaml             # mlx-lm LoRA config
    evaluate_mlx.py           # score the 50 Q&A + probe benchmark
    serve_gradio_mlx.py       # local Gradio demo
  data/                       # train/valid/test + probe_benchmark (jsonl)
  logs/                       # training & eval logs
docs/                         # per-version notes (v1..v5)
deployment/                   # deployment notes (vLLM)
SESSION_LOG*.md               # detailed engineering logs / handoffs
```

> Model weights, LoRA adapters, and release archives are **not** tracked in git (size) — they are
> reproduced by running the training pipeline below.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install mlx-lm pandas      # plus transformers, openpyxl as needed
PY=.venv/bin/python

$PY training_project/scripts/build_probe_benchmark.py   # build held-out probes (before dataset)
$PY training_project/scripts/build_dataset.py           # build train/valid (decontaminates vs probes)
$PY training_project/scripts/validate_dataset.py        # integrity + 0-leakage gates
$PY training_project/scripts/train_mlx.py --stability   # 90-iter stability probe
$PY training_project/scripts/train_mlx.py               # full LoRA run (config in lora_14b.yaml)
$PY training_project/scripts/evaluate_mlx.py generate && $PY training_project/scripts/evaluate_mlx.py score
```

## Versions

See `docs/` and the `SESSION_LOG*.md` files for the full per-version history (v1/v2 = 7B, v3/v4/v5 = 14B).
The latest iteration (v5) targets v4's known weak spots (count grounding and "none-match" handling).
