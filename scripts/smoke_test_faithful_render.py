"""End-to-end smoke test for depth-conditioned faithful rendering.

Pipeline:
  template JSON  ->  3D scene + depth map (trimesh raycast)
                 ->  SDXL-turbo + Depth ControlNet
                 ->  photorealistic interior matching layout

Outputs:
  /tmp/faithful_<id>_depth.png   — depth map (gray)
  /tmp/faithful_<id>_render.png  — final photorealistic image
  /tmp/faithful_<id>_compare.png — side-by-side
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from backend.app.depth_renderer import render_template_depth
from backend.app.image_renderer import (
    warmup,
    render_faithful_from_template,
    build_prompt_from_template,
)


def main(template_paths: list[str]) -> None:
    print("=== Warming up SDXL-turbo + Depth ControlNet ===")
    t0 = time.time()
    info = warmup(include_controlnet=True)
    print(json.dumps(info, indent=2))
    print(f"warmup total: {time.time() - t0:.1f}s\n")

    for tpath in template_paths:
        template = json.loads(Path(tpath).read_text())
        tid = template.get("id", Path(tpath).stem)

        # Step 1: depth map alone (for inspection)
        d = render_template_depth(template)
        depth_path = f"/tmp/faithful_{tid}_depth.png"
        d["depth_image"].save(depth_path)
        print(f"[{tid}] focus={d['focus_room_name']} ({d['focus_room_type']}, "
              f"{d['focus_room_area']} m²), ceiling={d['ceiling_height_m']:.1f}m")
        print(f"  depth -> {depth_path}")

        # Step 2: faithful render
        out = render_faithful_from_template(template)
        r = out["result"]
        if r.error:
            print(f"  ERROR: {r.error}")
            continue
        render_path = f"/tmp/faithful_{tid}_render.png"
        Path(render_path).write_bytes(r.image_bytes)
        print(f"  render -> {render_path} ({r.latency_s:.2f}s, {r.width}x{r.height})")
        print(f"  prompt: {r.prompt[:120]}...")

        # Side-by-side compare
        depth_img = d["depth_image"].convert("RGB")
        render_img = Image.open(io_for(r.image_bytes)).convert("RGB")
        # Match heights
        h = max(depth_img.height, render_img.height)
        depth_img = depth_img.resize(
            (int(depth_img.width * h / depth_img.height), h)
        )
        render_img = render_img.resize(
            (int(render_img.width * h / render_img.height), h)
        )
        compare = Image.new("RGB", (depth_img.width + render_img.width, h), (0, 0, 0))
        compare.paste(depth_img, (0, 0))
        compare.paste(render_img, (depth_img.width, 0))
        compare_path = f"/tmp/faithful_{tid}_compare.png"
        compare.save(compare_path)
        print(f"  compare -> {compare_path}\n")


def io_for(b: bytes):
    import io
    return io.BytesIO(b)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: a couple of contrasting layouts
        templates = [
            "data/templates/europe/eu_de_2zimmer_berlin_altbau.json",
        ]
    else:
        templates = sys.argv[1:]
    main(templates)
