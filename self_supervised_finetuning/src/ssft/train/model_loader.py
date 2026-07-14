"""Model + tokenizer loading: AutoModelForCausalLM / AutoTokenizer only.

pad_token falls back to eos_token when missing; no new special tokens are ever
added and embeddings are never resized. SDPA attention by default (flash_attention_2
only if explicitly configured and importable). use_cache=False during training.
dtype is auto-detected (bf16 if supported, else fp16) and training on a config that
isn't `debug` hard-fails without CUDA rather than silently running slow/wrong.
"""
from __future__ import annotations

from typing import Any, Optional


def resolve_dtype(model_cfg: dict, training_cfg: Optional[dict] = None):
    import torch

    debug = bool((training_cfg or {}).get("debug"))
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    if debug:
        return torch.float32
    raise RuntimeError(
        "No CUDA device available. Real (non-debug) training configs require a CUDA GPU for "
        "QLoRA/bitsandbytes. Use configs/training/debug_cpu_or_small_gpu.yaml "
        "(training.debug: true) for a CPU smoke test instead."
    )


def load_tokenizer(model_cfg: dict):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        padding_side=model_cfg.get("tokenizer_padding_side", "right"),
    )
    strategy = model_cfg.get("tokenizer_pad_token_strategy", "eos_if_missing")
    if tokenizer.pad_token is None:
        if strategy != "eos_if_missing":
            raise ValueError(
                f"tokenizer for {model_cfg['name_or_path']} has no pad_token and "
                f"tokenizer_pad_token_strategy={strategy!r} does not handle it. This "
                "framework never adds new special tokens by default."
            )
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_quantization_config(method_cfg: dict):
    quant = method_cfg.get("quantization")
    if not quant:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    # 8-bit LoRA: no NF4 quantization noise floor (recovers ~1% over 4-bit), ~15GB for a 14B
    # base — still fits 24GB with r64 LoRA + gradient checkpointing. Checked before 4-bit so a
    # config can't accidentally request both.
    if quant.get("load_in_8bit"):
        return BitsAndBytesConfig(load_in_8bit=True)
    if not quant.get("load_in_4bit"):
        return None

    compute_dtype = quant.get("bnb_4bit_compute_dtype", "auto")
    if compute_dtype == "auto":
        compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    else:
        compute_dtype = getattr(torch, compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model(model_cfg: dict, method_cfg: dict, dtype, training_cfg: Optional[dict] = None):
    from transformers import AutoModelForCausalLM

    attn_implementation = model_cfg.get("attn_implementation", "sdpa")
    if attn_implementation == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "model config requests attn_implementation=flash_attention_2 but the "
                "flash-attn package is not installed. Install it explicitly or switch back "
                "to attn_implementation: sdpa."
            ) from e

    # device_map: "auto" (default) lets accelerate place/split the model. For MoE models the
    # auto planner both (a) spuriously CPU-offloads on a single GPU that actually fits, and
    # (b) when it splits across GPUs, model-parallel loss reads labels off the wrong device
    # (spurious nll_loss out-of-range asserts). Setting model.device_map: "single" pins the
    # whole model to one GPU (index 0 of the visible set), which is correct whenever it fits.
    device_map = model_cfg.get("device_map", "auto")
    if device_map == "single":
        device_map = {"": 0}

    quantization_config = build_quantization_config(method_cfg)
    model_kwargs: dict[str, Any] = dict(
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        attn_implementation=attn_implementation,
        dtype=dtype,
        device_map=device_map,
    )
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(model_cfg["name_or_path"], **model_kwargs)
    model.config.use_cache = model_cfg.get("use_cache", False)

    gradient_checkpointing = method_cfg.get("gradient_checkpointing", False)
    if quantization_config is not None:
        # Quantized (4-bit/8-bit) base: prepare_model_for_kbit_training upcasts only the
        # non-quantized params (layernorms) to fp32 and wires up gradient checkpointing.
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing,
        )
    elif gradient_checkpointing:
        # Non-quantized (bf16/fp16) base: enable gradient checkpointing WITHOUT
        # prepare_model_for_kbit_training, which for a non-kbit model upcasts the ENTIRE base
        # to fp32 (~2x memory — e.g. 28GB bf16 -> 56GB — and pointless here).
        # enable_input_require_grads is required so gradients reach the LoRA adapters through
        # the checkpointed frozen base.
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model


def load_model_and_tokenizer(model_cfg: dict, method_cfg: dict, training_cfg: Optional[dict] = None):
    dtype = resolve_dtype(model_cfg, training_cfg)
    tokenizer = load_tokenizer(model_cfg)
    model = load_model(model_cfg, method_cfg, dtype, training_cfg)
    return model, tokenizer
