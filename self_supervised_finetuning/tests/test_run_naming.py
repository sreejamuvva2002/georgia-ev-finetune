"""training_slug / format_lr_slug / effective_batch_size match the spec's own examples;
the output path includes model/method/dataset/split/training/seed."""
from pathlib import Path

from ssft.train.hyperparams import ResolvedConfig
from ssft.train.run_naming import (
    build_output_dir,
    build_run_id,
    build_training_slug,
    effective_batch_size,
    format_lr_slug,
)


def _resolved(training_overrides=None, data_overrides=None):
    model_cfg = {"model_id_slug": "qwen2p5-14b", "name_or_path": "Qwen/Qwen2.5-14B"}
    method_cfg = {
        "method_id_slug": "qlora-lora-r16-a32-d005", "name": "qlora_lora",
        "lora": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj"]},
    }
    data_cfg = {
        "dataset_variant_slug": "kb-only-company-split", "split_strategy_slug": "group-by-company",
        "split_strategy": "group_by_company", "max_seq_length": 1024,
    }
    training_cfg = {
        "num_train_epochs": 8, "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16, "learning_rate": 0.00008,
    }
    training_cfg.update(training_overrides or {})
    data_cfg.update(data_overrides or {})
    return ResolvedConfig(model_cfg, method_cfg, data_cfg, training_cfg, seed=42)


def test_format_lr_slug_matches_spec_examples():
    assert format_lr_slug(0.00008) == "lr8e5"
    assert format_lr_slug(0.00005) == "lr5e5"
    assert format_lr_slug(0.00010) == "lr1e4"
    assert format_lr_slug(0.00015) == "lr15e5"


def test_effective_batch_size_matches_spec_examples():
    assert effective_batch_size(1, 8) == 8
    assert effective_batch_size(1, 16) == 16
    assert effective_batch_size(1, 32) == 32
    assert effective_batch_size(2, 8) == 16
    assert effective_batch_size(2, 16) == 32
    assert effective_batch_size(2, 32) == 64


def test_training_slug_matches_tiny_kb_conservative_example():
    assert build_training_slug(_resolved()) == "ep8-bs1-ga16-ebs16-lr8e5-seq1024"


def test_training_slug_matches_tiny_kb_memorization_example():
    resolved = _resolved(training_overrides={"num_train_epochs": 20, "learning_rate": 0.00010})
    assert build_training_slug(resolved) == "ep20-bs1-ga16-ebs16-lr1e4-seq1024"


def test_training_slug_matches_web_default_example():
    resolved = _resolved(training_overrides={"num_train_epochs": 1, "learning_rate": 0.00010}, data_overrides={"max_seq_length": 2048})
    assert build_training_slug(resolved) == "ep1-bs1-ga16-ebs16-lr1e4-seq2048"


def test_training_slug_reacts_to_batch_sweep_overrides():
    resolved = _resolved(training_overrides={"per_device_train_batch_size": 2, "gradient_accumulation_steps": 8})
    assert build_training_slug(resolved) == "ep8-bs2-ga8-ebs16-lr8e5-seq1024"


def test_output_dir_includes_all_identity_fields():
    resolved = _resolved()
    out = build_output_dir(resolved, Path("/tmp/adapters"), "20260708_153000", "a1b2c3d4")
    parts = out.parts
    assert "qwen2p5-14b" in parts
    assert "qlora-lora-r16-a32-d005" in parts
    assert "kb-only-company-split" in parts
    assert "group-by-company" in parts
    assert "ep8-bs1-ga16-ebs16-lr8e5-seq1024" in parts
    assert "seed42" in parts
    assert "20260708_153000_a1b2c3d4" in parts


def test_run_id_is_double_underscore_joined():
    resolved = _resolved()
    run_id = build_run_id(resolved, "20260708_153000", "a1b2c3d4")
    segments = run_id.split("__")
    assert segments[0] == "qwen2p5-14b"
    assert segments[1] == "qlora-lora-r16-a32-d005"
    assert segments[2] == "kb-only-company-split"
    assert segments[3] == "group-by-company"
    assert segments[4] == "ep8-bs1-ga16-ebs16-lr8e5-seq1024"
    assert segments[5] == "seed42"
    assert segments[6] == "20260708_153000_a1b2c3d4"
