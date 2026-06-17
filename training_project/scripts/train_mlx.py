#!/usr/bin/env python3
"""v4 trainer: LoRA fine-tune the MLX 8-bit Qwen2.5-14B on the Georgia EV data via mlx-lm.

Always passes --data and --adapter-path (ROOT-relative) so a moved project can never train
on stale yaml paths. The full run uses the iters/lr-schedule in lora_14b.yaml (kept in sync).

  python scripts/train_mlx.py --smoke        # 8 iters, temp adapter — verify the loop runs
  python scripts/train_mlx.py --stability     # 90 iters — confirm clean loss descent before committing
  python scripts/train_mlx.py                 # full run (uses yaml iters), adapter -> georgia_ev_14b_mlx_v4
  python scripts/train_mlx.py --resume        # resume the full run from the saved adapter
  python scripts/train_mlx.py --iters N        # override iters (full run)
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YAML = ROOT / "scripts" / "lora_14b.yaml"
DATA = ROOT / "data"
ADAPTER = ROOT / "adapters" / "georgia_ev_14b_mlx_v5"
SMOKE = ROOT / "adapters" / "_smoke_14b"
STAB = ROOT / "adapters" / "_stab_14b"


def n_train():
    return sum(1 for _ in open(DATA / "train.jsonl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="8-iter smoke test to a temp adapter")
    ap.add_argument("--stability", action="store_true", help="90-iter stability probe to a temp adapter")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--iters", type=int, default=None, help="override iters for the full run")
    a = ap.parse_args()

    cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", str(YAML), "--data", str(DATA)]
    if a.smoke:
        cmd += ["--iters", "8", "--steps-per-eval", "8", "--save-every", "8",
                "--adapter-path", str(SMOKE)]
    elif a.stability:
        cmd += ["--iters", "90", "--steps-per-report", "10", "--steps-per-eval", "90",
                "--save-every", "90", "--adapter-path", str(STAB)]
    else:
        cmd += ["--adapter-path", str(ADAPTER)]
        if a.iters:
            cmd += ["--iters", str(a.iters)]
        if a.resume:
            cmd += ["--resume-adapter-file", str(ADAPTER / "adapters.safetensors")]
        print(f"[train_mlx] {n_train()} train examples; iters from yaml (3300 ~= 2.5 epochs) "
              f"unless --iters given; adapter -> {ADAPTER}")
    print("[train_mlx] running:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
