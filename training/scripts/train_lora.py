"""LoRA fine-tuning script wrapping mlx_lm.lora.

Trains a LoRA adapter on the BIM Coordinator dataset for either Gemma 4 E2B
or E4B. Saves checkpoints, logs metrics, and runs a quick smoke-test
generation at the end so you know if the adapter actually trained.

Usage:
  python train_lora.py --model E2B  # ~3-5h on M4 Max
  python train_lora.py --model E4B  # ~6-10h on M4 Max
  python train_lora.py --model E2B --iters 200 --batch-size 1  # quick smoke

Config defaults are conservative for stability; aggressive enough to learn the
domain in a single overnight run. Tune --iters / --learning-rate if you have
training run data showing under/overfitting.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# Map our shorthand model names to actual MLX model paths
MODEL_MAP = {
    "E2B": "mlx-community/gemma-4-e2b-it-bf16",
    "E4B": "mlx-community/gemma-4-e4b-it-bf16",
}


def _resolve_repo_root() -> Path:
    """Find the bim-coordinator repo root from this script's location."""
    here = Path(__file__).resolve().parent
    # training/scripts/train_lora.py -> bim-coordinator/
    return here.parent.parent


def _write_lora_config(
    output: Path,
    model: str,
    data_dir: Path,
    adapter_path: Path,
    iters: int,
    batch_size: int,
    lr: float,
    lora_layers: int,
    rank: int,
    alpha: float,
    val_batches: int,
    save_every: int,
    seed: int,
) -> None:
    """Write the YAML config consumed by mlx_lm.lora --config."""
    cfg = {
        "model": model,
        "train": True,
        "fine_tune_type": "lora",
        "data": str(data_dir),
        "seed": seed,
        "num_layers": lora_layers,           # how many transformer layers to LoRA-fy
        "batch_size": batch_size,
        "iters": iters,
        "val_batches": val_batches,
        "learning_rate": lr,
        "steps_per_report": 25,
        "steps_per_eval": max(50, iters // 20),
        "resume_adapter_file": None,
        "adapter_path": str(adapter_path),
        "save_every": save_every,
        "test": False,
        "test_batches": 0,
        "max_seq_length": 2048,
        "lr_schedule": {
            "name": "cosine_decay",
            "warmup": max(50, iters // 50),
            "warmup_init": 1e-7,
            "arguments": [lr, iters, 1e-7],
        },
        "lora_parameters": {
            "rank": rank,
            "scale": alpha / rank,           # mlx-lm uses 'scale' = alpha / rank
            "dropout": 0.05,
        },
    }
    output.write_text(_yaml_dump(cfg))


def _yaml_dump(d) -> str:
    """Tiny YAML emitter — avoids pulling pyyaml as a dep."""
    def emit(v, indent=0):
        sp = "  " * indent
        if isinstance(v, dict):
            lines = []
            for k, vv in v.items():
                if isinstance(vv, (dict, list)):
                    lines.append(f"{sp}{k}:")
                    lines.append(emit(vv, indent + 1))
                else:
                    lines.append(f"{sp}{k}: {_scalar(vv)}")
            return "\n".join(lines)
        if isinstance(v, list):
            return "\n".join(f"{sp}- {_scalar(x)}" for x in v)
        return f"{sp}{_scalar(v)}"

    def _scalar(v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, str):
            return v
        return str(v)

    return emit(d) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_MAP.keys()), required=True)
    parser.add_argument("--data-dir", default="training/data/processed",
                        help="dir containing train.jsonl, valid.jsonl")
    parser.add_argument("--checkpoint-base", default="training/checkpoints",
                        help="parent dir for adapter checkpoints")
    parser.add_argument("--iters", type=int, default=1500,
                        help="training iterations (1500 ≈ 2 epochs at 6000 examples / batch 8)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="per-step batch (M4 Max 64GB: 2 for E4B, 4 for E2B)")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-layers", type=int, default=16,
                        help="number of transformer layers to LoRA-fy from the top")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick 50-iter run for plumbing validation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the command + config that would be run, exit 0")
    args = parser.parse_args()

    repo_root = _resolve_repo_root()
    data_dir = (repo_root / args.data_dir).resolve()
    if not (data_dir / "train.jsonl").exists():
        print(f"ERROR: {data_dir}/train.jsonl not found. Run build_dataset.py first.",
              file=sys.stderr)
        return 2

    if args.smoke_test:
        args.iters = 50
        args.save_every = 25

    model_id = MODEL_MAP[args.model]
    run_name = f"gemma4-{args.model.lower()}-lora-r{args.rank}-iters{args.iters}-{int(time.time())}"
    adapter_path = (repo_root / args.checkpoint_base / run_name).resolve()
    adapter_path.mkdir(parents=True, exist_ok=True)
    config_path = adapter_path / "config.yaml"

    _write_lora_config(
        config_path,
        model=model_id,
        data_dir=data_dir,
        adapter_path=adapter_path,
        iters=args.iters,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        lora_layers=args.lora_layers,
        rank=args.rank,
        alpha=args.alpha,
        val_batches=args.val_batches,
        save_every=args.save_every,
        seed=args.seed,
    )

    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_path)]
    print(f"\n=== LoRA fine-tune: {args.model} ===")
    print(f"  model:        {model_id}")
    print(f"  data:         {data_dir}")
    print(f"  adapter:      {adapter_path}")
    print(f"  iters:        {args.iters} (save every {args.save_every})")
    print(f"  batch_size:   {args.batch_size}")
    print(f"  lr:           {args.learning_rate}")
    print(f"  rank/alpha:   {args.rank}/{args.alpha}")
    print(f"  lora_layers:  {args.lora_layers}")
    print(f"\nCommand: {' '.join(cmd)}")

    if args.dry_run:
        print("\n--- config.yaml ---")
        print(config_path.read_text())
        return 0

    print(f"\nLog file: {adapter_path / 'training.log'}\n")

    # Run training, mirroring stdout to log file
    log_path = adapter_path / "training.log"
    t0 = time.time()
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
        rc = proc.wait()
    dt = time.time() - t0
    print(f"\n=== Training finished in {dt/60:.1f} min, return code {rc} ===")

    if rc != 0:
        print("Training failed — see log for details.", file=sys.stderr)
        return rc

    # Save a manifest of what was trained
    manifest = {
        "run_name": run_name,
        "base_model": model_id,
        "data_dir": str(data_dir),
        "adapter_path": str(adapter_path),
        "iters": args.iters,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "rank": args.rank,
        "alpha": args.alpha,
        "lora_layers": args.lora_layers,
        "duration_s": int(dt),
    }
    (adapter_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {adapter_path / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
