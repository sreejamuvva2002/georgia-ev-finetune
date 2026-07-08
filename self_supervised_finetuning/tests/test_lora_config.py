"""LoRA config contains the expected target modules; missing target modules fail
loudly instead of training silently on the wrong layers."""
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
peft = pytest.importorskip("peft")

from ssft.train.lora_factory import build_lora_config, validate_target_modules  # noqa: E402

METHOD_CFG = {
    "method_id_slug": "qlora-lora-r16-a32-d005", "name": "qlora_lora",
    "lora": {
        "r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)
        self.gate_proj = nn.Linear(4, 4)
        self.up_proj = nn.Linear(4, 4)
        self.down_proj = nn.Linear(4, 4)


class IncompleteModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)


def test_build_lora_config_has_expected_target_modules():
    cfg = build_lora_config(METHOD_CFG)
    assert set(cfg.target_modules) == set(METHOD_CFG["lora"]["target_modules"])
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05


def test_validate_target_modules_passes_for_matching_model():
    validate_target_modules(TinyModel(), METHOD_CFG["lora"]["target_modules"])  # should not raise


def test_validate_target_modules_fails_loudly_for_missing_modules():
    with pytest.raises(ValueError, match="do not match any Linear module"):
        validate_target_modules(IncompleteModel(), METHOD_CFG["lora"]["target_modules"])


def test_placeholder_method_raises_not_implemented():
    placeholder = {
        "method_id_slug": "adalora-placeholder", "name": "adalora", "status": "not_implemented",
        "lora": {"r": 1, "lora_alpha": 1, "target_modules": []},
    }
    with pytest.raises(NotImplementedError):
        build_lora_config(placeholder)
