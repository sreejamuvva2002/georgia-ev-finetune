#!/usr/bin/env python3
"""Freeze the current model as an immutable, restorable version snapshot.

Run this after every training to save that version. It bundles the trained
adapter (final weights only, no bulky checkpoints), the exact dataset, the
scripts, the eval results, a manifest, and SHA-256 checksums into
`releases/<version>_<date>/`, plus a single `.tar.gz` you can copy off-machine,
then write-protects the folder so it can't be clobbered later.

Usage:
    python training_project/scripts/save_version.py v3
    python training_project/scripts/save_version.py v3 --notes "added county-sum data, rank 48"
    python training_project/scripts/save_version.py v1 \
        --adapter training_project/adapters/georgia_ev_lora_v1 \
        --notes "initial run (its exact training data was later overwritten)"
"""
import argparse
import hashlib
import json
import os
import shutil
import tarfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]            # training_project/
PROJ = ROOT.parent                                    # repo root
RELEASES = PROJ / "releases"
DEFAULT_ADAPTER = ROOT / "adapters" / "georgia_ev_lora"
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
SCRIPTS = ["build_dataset.py", "validate_dataset.py", "train_qlora.py",
           "evaluate.py", "chat.py", "serve_gradio.py", "save_version.py",
           "train_mlx.py", "lora_14b.yaml", "diagnose_v2.py", "evaluate_mlx.py"]
EVAL_FILES = ["evaluation.xlsx", "summary.md", "base_answers.json",
              "finetuned_answers.json", "finetuned_answers_v1.json",
              "answers_base_v1_v2.xlsx", "scores_base_v1_v2.xlsx",
              # 14B (MLX) artifacts
              "summary_14b.md", "answers_14b.json", "diagnostic_14b_answers.json",
              "scores_14b_vs_v2.xlsx", "diagnostic_14b.xlsx",
              "diagnostic_v2.xlsx", "diagnostic_v2_answers.json"]


def base_revision(model=BASE_MODEL):
    if Path(model).exists():            # local model dir (e.g. the MLX 8-bit build)
        return "local"
    ref = Path.home() / ".cache/huggingface/hub" / \
        f"models--{model.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else "unknown"


def copy_some(files, src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        p = src_dir / f
        if p.exists():
            shutil.copy2(p, dst_dir / f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version", help="version label, e.g. v3")
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER), help="adapter dir to snapshot")
    ap.add_argument("--base-model", default=BASE_MODEL, help="base model id/path this adapter applies to")
    ap.add_argument("--notes", default="", help="free-text description of this version")
    ap.add_argument("--date", default=date.today().isoformat())
    a = ap.parse_args()
    base_model = a.base_model

    adapter = Path(a.adapter).resolve()
    # PEFT writes adapter_model.safetensors; mlx-lm writes adapters.safetensors
    if not ((adapter / "adapter_model.safetensors").exists() or (adapter / "adapters.safetensors").exists()):
        raise SystemExit(f"No adapter weights (adapter_model.safetensors / adapters.safetensors) in {adapter} — nothing to save.")

    snap = RELEASES / f"{a.version}_{a.date}"
    if snap.exists():
        raise SystemExit(f"{snap} already exists — pick a different version/date (won't overwrite a save point).")
    print(f"Saving {a.version} -> {snap}")

    # 1. final adapter only (skip training checkpoints + optimizer state;
    #    also skip mlx-lm intermediate checkpoints like 0000400_adapters.safetensors)
    (snap / "adapter").mkdir(parents=True)
    for p in adapter.iterdir():
        if p.is_file() and not (p.name[:1].isdigit() and "_adapters" in p.name):
            shutil.copy2(p, snap / "adapter" / p.name)
    # 2. exact dataset, 3. scripts, 4. eval artifacts
    copy_some(["train.jsonl", "valid.jsonl", "test.jsonl"], ROOT / "data", snap / "data")
    copy_some(SCRIPTS, ROOT / "scripts", snap / "scripts")
    copy_some(EVAL_FILES, ROOT / "eval_results", snap / "eval")

    # metrics from the adapter (best-effort)
    metrics, cfg = {}, {}
    fm = adapter / "final_metrics.json"
    if fm.exists():
        metrics = json.loads(fm.read_text())
    ac = adapter / "adapter_config.json"
    if ac.exists():
        cfg = json.loads(ac.read_text())

    lp = cfg.get("lora_parameters", {})          # mlx-lm stores rank/scale/dropout here
    version_meta = {
        "version": a.version,
        "created": a.date,
        "notes": a.notes,
        "base_model": base_model,
        "base_model_revision": base_revision(base_model),
        "source_adapter": str(adapter.relative_to(PROJ)) if adapter.is_relative_to(PROJ) else str(adapter),
        "lora": {"r": cfg.get("r", lp.get("rank")), "alpha": cfg.get("lora_alpha", lp.get("scale")),
                 "dropout": cfg.get("lora_dropout", lp.get("dropout")),
                 "target_modules": cfg.get("target_modules", f"all layers (num_layers={cfg.get('num_layers')})" if "num_layers" in cfg else None)},
        "final_train_loss": metrics.get("train", {}).get("train_loss"),
        "final_eval_loss": metrics.get("eval", {}).get("eval_loss"),
        "eval_token_accuracy": metrics.get("eval", {}).get("eval_mean_token_accuracy"),
    }
    (snap / "VERSION.json").write_text(json.dumps(version_meta, indent=2))

    # checksums over adapter/data/eval
    lines = []
    for sub in ("adapter", "data", "eval"):
        for p in sorted((snap / sub).rglob("*")):
            if p.is_file():
                lines.append(f"{sha256(p)}  {p.relative_to(snap)}")
    (snap / "checksums.sha256").write_text("\n".join(lines) + "\n")

    # human-readable manifest
    acc_note = ""
    summ = ROOT / "eval_results" / "summary.md"
    if (snap / "eval" / "summary.md").exists():
        acc_note = "\nSee `eval/summary.md` for accuracy on the 50 held-out questions.\n"
    (snap / "MANIFEST.md").write_text(f"""# Snapshot — {a.version} ({a.date})

{a.notes or "Frozen, restorable copy of this model version."}

- **Base model:** `{base_model}` @ `{version_meta['base_model_revision']}` (not stored here)
- **Adapter:** LoRA r={version_meta['lora']['r']} / α(scale)={version_meta['lora']['alpha']} — from `{version_meta['source_adapter']}`
- **Final train / eval loss:** {version_meta['final_train_loss']} / {version_meta['final_eval_loss']}
{acc_note}
## Contents
`adapter/` (the model — final weights only) · `data/` (exact train/valid/test) · `scripts/` · `eval/` · `VERSION.json` · `checksums.sha256`

## Verify
```bash
cd {snap.relative_to(PROJ)} && shasum -a 256 -c checksums.sha256
```

## Roll back to this version
```bash
cd {PROJ}
mv training_project/adapters/georgia_ev_lora training_project/adapters/georgia_ev_lora_OLD 2>/dev/null || true
cp -R {snap.relative_to(PROJ)}/adapter training_project/adapters/georgia_ev_lora
```
""")

    # portable tarball + write-protect
    tar_path = RELEASES / f"georgia_ev_{a.version}_{a.date}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(snap, arcname=snap.name)
    for p in snap.rglob("*"):
        if p.is_file():
            p.chmod(0o444)

    size = sum(f.stat().st_size for f in snap.rglob("*") if f.is_file()) / 1e6
    print(f"  snapshot: {snap}  ({size:.0f} MB)")
    print(f"  archive:  {tar_path}  ({tar_path.stat().st_size/1e6:.0f} MB)")
    print(f"  files checksummed: {len(lines)}")
    print("Done. Folder is read-only; copy the .tar.gz off-machine for backup.")


if __name__ == "__main__":
    main()
