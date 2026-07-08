"""Sweep expansion creates the expected number of runs (deterministic cartesian
product across models x methods x data x override-value-lists x seeds)."""
import pytest
import yaml

from ssft.train.sweep_runner import expand_sweep


def test_expand_sweep_matches_kb_epoch_sweep_shape():
    sweep_cfg = {
        "models": ["m1.yaml"], "methods": ["method1.yaml"], "data": ["data1.yaml"],
        "base_training_config": "training1.yaml",
        "overrides": {
            "training.num_train_epochs": [1, 3, 5, 8, 12],
            "training.learning_rate": [0.00008],
            "training.gradient_accumulation_steps": [16],
        },
        "seeds": [42],
    }
    points = expand_sweep(sweep_cfg)
    assert len(points) == 5  # 1*1*1 * 5 epochs*1 lr*1 ga * 1 seed


def test_expand_sweep_matches_kb_full_factorial_small_shape():
    sweep_cfg = {
        "models": ["m1.yaml"], "methods": ["method_low.yaml", "method_default.yaml"], "data": ["data1.yaml"],
        "base_training_config": "training1.yaml",
        "overrides": {
            "training.num_train_epochs": [3, 8],
            "training.learning_rate": [0.00005, 0.00008, 0.00010],
            "training.gradient_accumulation_steps": [8, 16],
        },
        "seeds": [42],
    }
    points = expand_sweep(sweep_cfg)
    # 1 model * 2 methods * 1 data * (2 epochs * 3 lr * 2 ga) * 1 seed = 24
    assert len(points) == 24


def test_expand_sweep_multi_model_multi_seed():
    sweep_cfg = {
        "models": ["m1.yaml", "m2.yaml", "m3.yaml"], "methods": ["method1.yaml"], "data": ["data1.yaml"],
        "base_training_config": "training1.yaml", "overrides": {}, "seeds": [42, 43],
    }
    points = expand_sweep(sweep_cfg)
    assert len(points) == 3 * 1 * 1 * 1 * 2


def test_expand_sweep_respects_max_runs():
    sweep_cfg = {
        "models": ["m1.yaml"], "methods": ["method1.yaml"], "data": ["data1.yaml"],
        "base_training_config": "training1.yaml",
        "overrides": {"training.num_train_epochs": [1, 2, 3, 4, 5]},
        "seeds": [42], "max_runs": 2,
    }
    points = expand_sweep(sweep_cfg)
    assert len(points) == 2


def test_expand_sweep_requires_a_training_config():
    sweep_cfg = {"models": ["m1.yaml"], "methods": ["method1.yaml"], "data": ["data1.yaml"], "overrides": {}, "seeds": [42]}
    with pytest.raises(ValueError):
        expand_sweep(sweep_cfg)


def test_expand_sweep_requires_models_methods_data():
    with pytest.raises(ValueError):
        expand_sweep({"methods": ["a"], "data": ["b"], "base_training_config": "t.yaml"})


def test_sweep_indices_are_sequential():
    sweep_cfg = {
        "models": ["m1.yaml"], "methods": ["method1.yaml"], "data": ["data1.yaml"],
        "base_training_config": "training1.yaml",
        "overrides": {"training.num_train_epochs": [1, 2, 3]}, "seeds": [42],
    }
    points = expand_sweep(sweep_cfg)
    assert [p.sweep_index for p in points] == [0, 1, 2]


def test_expand_sweep_training_config_by_dataset(tmp_path):
    data_path = tmp_path / "data1.yaml"
    data_path.write_text(yaml.safe_dump({"data": {"dataset_variant_slug": "kb-only-company-split"}}))
    sweep_cfg = {
        "models": ["m1.yaml"], "methods": ["method1.yaml"], "data": [str(data_path)],
        "training_config_by_dataset": {"kb-only-company-split": "training1.yaml"},
        "overrides": {}, "seeds": [42],
    }
    points = expand_sweep(sweep_cfg)
    assert len(points) == 1
    assert points[0].training_config == "training1.yaml"


def test_expand_sweep_training_config_by_dataset_raises_for_unknown_variant(tmp_path):
    data_path = tmp_path / "data1.yaml"
    data_path.write_text(yaml.safe_dump({"data": {"dataset_variant_slug": "some-other-variant"}}))
    sweep_cfg = {
        "models": ["m1.yaml"], "methods": ["method1.yaml"], "data": [str(data_path)],
        "training_config_by_dataset": {"kb-only-company-split": "training1.yaml"},
        "overrides": {}, "seeds": [42],
    }
    with pytest.raises(ValueError):
        expand_sweep(sweep_cfg)
