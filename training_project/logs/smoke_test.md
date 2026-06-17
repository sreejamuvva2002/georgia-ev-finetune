# Smoke Test — PASSED (v2, r=32)

- Date: 2026-06-13
- Command: `python training_project/scripts/train_qlora.py --max-steps 5 --output-dir training_project/outputs/smoke_v2`
- Config: LoRA **r=32 / α=64** / dropout 0.05, max_seq_length 1536, eff. batch 16 (1×16), lr 2e-4, bf16 MPS
- Base model: Qwen/Qwen2.5-Coder-7B-Instruct
- Trainable params: **80,740,352 (1.05%)** — double the v1 r=16 adapter
- Result: 5 steps OK, no errors. train_loss 3.044, eval_loss 2.084 (token acc 0.671)
- Dataset: 2,535 train / 196 valid (clean — 0 test-question or gold-answer leakage; 73 leaky generations dropped)
- Decision: passed → full v2 training started automatically.

(v1 smoke test for r=16 is superseded; v1 adapter preserved at `adapters/georgia_ev_lora_v1/`.)
