"""Stylistic vs Faithful render comparison — pre-generates demo material.

For each template:
  1. Renders stylistic (prompt-only, ~1.5s)
  2. Renders faithful (depth-conditioned, ~2s)
  3. Generates the depth map (~0.4s)
  4. Stitches a 3-up comparison image: depth | stylistic | faithful

Output:
  /tmp/compare_<id>_3up.png       — side-by-side
  /tmp/compare_<id>_stylistic.png — solo stylistic
  /tmp/compare_<id>_faithful.png  — solo faithful
  /tmp/compare_<id>_depth.png     — solo depth
  /tmp/compare_index.html         — gallery to view all in a browser
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

from backend.app.depth_renderer import render_template_depth
from backend.app.image_renderer import (
    warmup,
    render_from_template,
    render_faithful_from_template,
)


LABEL_H = 28


def _label_strip(text: str, width: int) -> Image.Image:
    img = Image.new("RGB", (width, LABEL_H), (32, 32, 32))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
    d.text((8, 6), text, fill=(220, 220, 220), font=font)
    return img


def _stitch_3up(depth: Image.Image, stylistic: Image.Image,
                 faithful: Image.Image, title: str) -> Image.Image:
    # Equalise heights
    h = max(depth.height, stylistic.height, faithful.height)
    def resize(im):
        return im.resize((int(im.width * h / im.height), h))
    d = resize(depth.convert("RGB"))
    s = resize(stylistic.convert("RGB"))
    f = resize(faithful.convert("RGB"))

    pad = 6
    total_w = d.width + s.width + f.width + 4 * pad
    canvas = Image.new("RGB", (total_w, h + LABEL_H + 3 * pad), (16, 16, 16))

    canvas.paste(_label_strip(title, total_w), (0, 0))
    y = LABEL_H + pad
    x = pad
    canvas.paste(_label_strip(f"Depth (BIM scene)  {d.width}×{h}", d.width),
                 (x, LABEL_H - 0))
    canvas.paste(d, (x, y)); x += d.width + pad
    canvas.paste(_label_strip(f"Stylistic (prompt-only)  {s.width}×{h}", s.width),
                 (x, LABEL_H - 0))
    canvas.paste(s, (x, y)); x += s.width + pad
    canvas.paste(_label_strip(f"Faithful (depth ControlNet)  {f.width}×{h}", f.width),
                 (x, LABEL_H - 0))
    canvas.paste(f, (x, y))
    return canvas


def main(template_paths: list[str]) -> None:
    print("=== Warming up SDXL-turbo + Depth ControlNet ===")
    t0 = time.time()
    info = warmup(include_controlnet=True)
    print(f"  loaded={info.get('loaded')} cn_loaded={info.get('controlnet_loaded')} "
          f"in {time.time()-t0:.1f}s")

    rows = []
    for tpath in template_paths:
        template = json.loads(Path(tpath).read_text())
        tid = template.get("id", Path(tpath).stem)
        country = template.get("metadata", {}).get("country", "")
        descr = template.get("metadata", {}).get("style", "")[:80]

        print(f"\n[{tid}] {country} — {descr}")

        # 1. Depth map
        t1 = time.time()
        depth_info = render_template_depth(template)
        depth_img = depth_info["depth_image"]
        Path(f"/tmp/compare_{tid}_depth.png").write_bytes(_to_png(depth_img))
        print(f"  depth: {time.time()-t1:.2f}s  focus={depth_info['focus_room_name']} "
              f"({depth_info['focus_room_type']}, {depth_info['focus_room_area']} m²)")

        # 2. Stylistic render
        t1 = time.time()
        s = render_from_template(template, width=768, height=512, steps=2)
        if s.error:
            print(f"  stylistic ERROR: {s.error}"); continue
        Path(f"/tmp/compare_{tid}_stylistic.png").write_bytes(s.image_bytes)
        stylistic_img = Image.open(io.BytesIO(s.image_bytes))
        print(f"  stylistic: {s.latency_s:.2f}s")

        # 3. Faithful render
        t1 = time.time()
        out = render_faithful_from_template(template, width=768, height=512,
                                              steps=5, controlnet_scale=0.55)
        f = out["result"]
        if f.error:
            print(f"  faithful ERROR: {f.error}"); continue
        Path(f"/tmp/compare_{tid}_faithful.png").write_bytes(f.image_bytes)
        faithful_img = Image.open(io.BytesIO(f.image_bytes))
        print(f"  faithful: {f.latency_s:.2f}s")

        # 4. 3-up
        title = (f"{tid}  |  {country}  |  focus: {depth_info['focus_room_name']} "
                 f"({depth_info['focus_room_area']} m², "
                 f"ceiling {depth_info['ceiling_height_m']:.1f}m)")
        three_up = _stitch_3up(depth_img, stylistic_img, faithful_img, title)
        three_up.save(f"/tmp/compare_{tid}_3up.png")
        print(f"  3-up -> /tmp/compare_{tid}_3up.png")

        rows.append({
            "tid": tid, "country": country, "title": title,
            "stylistic_s": round(s.latency_s, 2),
            "faithful_s": round(f.latency_s, 2),
            "focus": f"{depth_info['focus_room_name']} ({depth_info['focus_room_type']})",
        })

    # Build a tiny HTML gallery
    html = ['<!doctype html><meta charset=utf-8>',
            '<title>Stylistic vs Faithful — BIM render compare</title>',
            '<style>body{font-family:system-ui;background:#111;color:#ddd;'
            'margin:0;padding:24px} h1{font-size:18px;margin:0 0 18px}'
            'figure{margin:0 0 32px} figcaption{font-size:12px;opacity:.7;'
            'margin-top:6px} img{max-width:100%;border-radius:4px;'
            'box-shadow:0 2px 12px rgba(0,0,0,.4)}</style>',
            f'<h1>Faithful vs Stylistic render — {len(rows)} templates '
            f'({time.strftime("%Y-%m-%d %H:%M")})</h1>']
    for r in rows:
        html += [
            f'<figure><img src="compare_{r["tid"]}_3up.png" alt="{r["tid"]}">'
            f'<figcaption>{r["title"]} &middot; '
            f'stylistic {r["stylistic_s"]}s · faithful {r["faithful_s"]}s'
            f'</figcaption></figure>'
        ]
    Path("/tmp/compare_index.html").write_text("\n".join(html))
    print(f"\nDone. Gallery: /tmp/compare_index.html ({len(rows)} entries)")


def _to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        templates = sys.argv[1:]
    else:
        # Default demo set: 6 contrasting countries/styles
        templates = [
            "data/templates/europe/eu_de_2zimmer_berlin_altbau.json",
            "data/templates/europe/eu_fr_1bed_paris_marais.json",
            "data/templates/global/gl_jp_1ldk_tokyo_mansion.json",
            "data/templates/global/gl_ae_1bed_dubai_marina.json",
            "data/templates/india/in_1bhk_bangalore_whitefield.json",
            "data/templates/global/gl_au_2bed_sydney_modern.json",
        ]
    main(templates)
