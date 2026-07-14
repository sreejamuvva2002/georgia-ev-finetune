"""Build a peft LoraConfig from a method config and wrap the model — after verifying
every target_modules entry actually names a Linear layer present in the model.
Never silently trains with missing/wrong target modules.
"""
from __future__ import annotations


def list_linear_module_suffixes(model) -> list[str]:
    import torch.nn as nn

    suffixes = set()
    for name, module in model.named_modules():
        is_linear_like = isinstance(module, nn.Linear) or type(module).__name__ in (
            "Linear4bit", "Linear8bitLt", "Linear",
        )
        if is_linear_like:
            suffixes.add(name.split(".")[-1])
    return sorted(suffixes)


def validate_target_modules(model, target_modules: list[str]) -> None:
    available = list_linear_module_suffixes(model)
    missing = [m for m in target_modules if m not in available]
    if missing:
        raise ValueError(
            f"LoRA target_modules {missing} do not match any Linear module in this model. "
            f"Available Linear module name suffixes: {available}. Refusing to silently train "
            "with missing/wrong target modules — fix method_config.method.lora.target_modules."
        )


def build_lora_config(method_cfg: dict):
    from peft import LoraConfig

    if method_cfg.get("status") == "not_implemented":
        raise NotImplementedError(
            f"method '{method_cfg.get('name')}' ({method_cfg.get('method_id_slug')}) is a "
            "placeholder and has no implementation yet — choose a different --method-config."
        )
    lora_cfg = method_cfg.get("lora")
    if not lora_cfg:
        raise ValueError(f"method config '{method_cfg.get('method_id_slug')}' has no 'lora:' block")
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=list(lora_cfg["target_modules"]),
        # rsLoRA (alpha/sqrt(r) scaling — keeps gradients from collapsing at high rank) and
        # DoRA are opt-in per method config; default False so existing configs are unchanged.
        use_rslora=lora_cfg.get("use_rslora", False),
        use_dora=lora_cfg.get("use_dora", False),
    )


def build_lora_model(model, method_cfg: dict):
    from peft import get_peft_model

    lora_config = build_lora_config(method_cfg)
    validate_target_modules(model, lora_config.target_modules)
    return get_peft_model(model, lora_config)
