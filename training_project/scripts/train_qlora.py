#!/usr/bin/env python3
"""QLoRA/LoRA fine-tuning of Qwen2.5-Coder-7B-Instruct on the Georgia EV KB dataset.

- On CUDA (Linux GPU): 4-bit NF4 QLoRA via bitsandbytes.
- On Apple Silicon (MPS): bf16 LoRA (bitsandbytes is CUDA-only).
Adapter output is standard PEFT format, directly servable with vLLM --enable-lora.
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=-1, help="limit steps (smoke test)")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--output-dir", default=str(ROOT / "adapters" / "georgia_ev_lora"))
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-length", type=int, default=1536)
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() and not use_cuda
    device = "cuda" if use_cuda else ("mps" if use_mps else "cpu")
    bf16_ok = use_mps or (use_cuda and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    print(f"device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(dtype=dtype)
    if use_cuda:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    if use_cuda:
        model = prepare_model_for_kbit_training(model)
    else:
        model.to(device)
    model.config.use_cache = False

    lora = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={
        "train": str(ROOT / "data" / "train.jsonl"),
        "valid": str(ROOT / "data" / "valid.jsonl"),
    })

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        max_length=args.max_seq_length,
        gradient_checkpointing=True,
        bf16=bf16_ok,
        fp16=not bf16_ok and use_cuda,
        optim="adamw_torch",
        report_to=[],
        seed=42,
        dataloader_pin_memory=False,
        use_cpu=device == "cpu",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["valid"],
        processing_class=tokenizer,
    )

    has_ckpt = any(Path(args.output_dir).glob("checkpoint-*"))
    result = trainer.train(resume_from_checkpoint=has_ckpt or None)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = {"train": result.metrics}
    try:
        metrics["eval"] = trainer.evaluate()
    except Exception as e:
        metrics["eval_error"] = str(e)
    with open(Path(args.output_dir) / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(ROOT / "logs" / "training_log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2, default=str)
    print("FINAL METRICS:", json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
