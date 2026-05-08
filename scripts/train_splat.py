"""Train a 3D Gaussian Splat from a synthetic BIM dataset.

Reads /tmp/splat_dataset/<template_id>/  (transforms.json + images/)
Writes /tmp/splat_dataset/<template_id>/output.ply

Then optionally converts to SOG using PlayCanvas SuperSplat tools, but
we leave that as a separate step.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("template_id")
    ap.add_argument("--iterations", type=int, default=7000)
    ap.add_argument("--num-downscales", type=int, default=0)
    ap.add_argument("--dataset-root", type=Path,
                     default=Path("/tmp/splat_dataset"))
    ap.add_argument("--output", type=Path, default=None,
                     help="Override .ply output path")
    args = ap.parse_args()

    dataset_dir = args.dataset_root / args.template_id
    if not dataset_dir.exists():
        sys.exit(f"dataset not found: {dataset_dir}\n"
                  f"Run: python scripts/render_splat_dataset.py {args.template_id}")

    transforms = dataset_dir / "transforms.json"
    if not transforms.exists():
        sys.exit(f"transforms.json missing in {dataset_dir}")

    out_ply = args.output or (dataset_dir / "output.ply")
    print(f"[train] dataset = {dataset_dir}")
    print(f"[train] iterations = {args.iterations}")
    print(f"[train] output = {out_ply}")

    import msplat

    t0 = time.time()
    print(f"[train] loading dataset ...")
    dataset = msplat.load_dataset(str(dataset_dir), eval_mode=False)
    print(f"[train]   loaded in {time.time()-t0:.1f}s")

    config = msplat.TrainingConfig(
        iterations=args.iterations,
        num_downscales=args.num_downscales,
    )
    trainer = msplat.GaussianTrainer(dataset, config)

    last_print = time.time()
    last_iter = 0
    print_every = 100

    def _on_step(stats):
        nonlocal last_print, last_iter
        now = time.time()
        if now - last_print >= 5.0 or stats.iteration == args.iterations:
            it_per_s = (stats.iteration - last_iter) / max(now - last_print, 1e-3)
            print(f"[train]   step {stats.iteration:6d}/{args.iterations}  "
                  f"splats={stats.splat_count:>8,d}  "
                  f"{it_per_s:6.1f} it/s")
            last_print = now
            last_iter = stats.iteration

    print(f"[train] training ...")
    train_t0 = time.time()
    trainer.train(_on_step, callback_every=print_every)
    train_t = time.time() - train_t0
    print(f"[train] training finished in {train_t:.1f}s")

    print(f"[train] exporting PLY ...")
    trainer.export_ply(str(out_ply))
    print(f"[train] wrote {out_ply}  "
          f"({out_ply.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
