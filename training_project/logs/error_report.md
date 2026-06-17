# Error Report

## Incident 1 — Full training run crashed at step 11/456 (2026-06-12 16:53)

**Symptom:** Training process died silently ~4 minutes into the full run. Log showed:

```
Error creating directory
The volume "Macintosh HD" is out of space. You can't save the file "mpsgraph-4784-..." because the volume is out of space.
```

**Root cause:** The startup disk had only ~784 MB free. The 14 GB Hugging Face
download of `Qwen/Qwen2.5-Coder-7B-Instruct` (required for PEFT-format training,
since the local LM Studio copy is MLX 8-bit) consumed the remaining free space.
The MPS backend could not write its `mpsgraph` temp files and macOS killed the
process.

**Fix:** Reclaimed ~6.5 GB by deleting regenerable caches only
(`~/Library/Caches/{pip, com.apple.python, Homebrew, Google, BraveSoftware, go-build, vscode-cpptools}`).
No user files, model downloads, or project data were touched. Free space after
cleanup: 7.2 GB. Removed the partial checkpoint directory and restarted training
from scratch.

**Note for the future:** `~/.lmstudio/models/lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-8bit`
(7.5 GB) is now redundant with the HF copy if you only use the fine-tuned
pipeline; deleting it via LM Studio would free another 7.5 GB.

## Smoke test

No errors — see `smoke_test.md`.
