"""Pre-render dollhouse cutaway views for the demo gallery.

For each template, fetches:
  - dollhouse depth map     (BIM scene, no ceiling)
  - dollhouse photoreal     (depth ControlNet -> Matterport-style image)
  - interior photoreal      (existing single-room view)

Saves a 3-up: depth | dollhouse | interior — and an HTML index.
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

from backend.app.depth_renderer import render_template_dollhouse_depth
from backend.app.image_renderer import (
    warmup,
    render_faithful_dollhouse_from_template,
    render_faithful_from_template,
)


LABEL_H = 28


def _label(text: str, width: int) -> Image.Image:
    img = Image.new("RGB", (width, LABEL_H), (32, 32, 32))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
    d.text((8, 6), text, fill=(220, 220, 220), font=font)
    return img


def _stitch_3up(depth: Image.Image, dollhouse: Image.Image,
                 interior: Image.Image, title: str) -> Image.Image:
    h = max(depth.height, dollhouse.height, interior.height)
    def rsz(im):
        return im.resize((int(im.width * h / im.height), h))
    d = rsz(depth.convert("RGB"))
    dh = rsz(dollhouse.convert("RGB"))
    it = rsz(interior.convert("RGB"))
    pad = 6
    total_w = d.width + dh.width + it.width + 4 * pad
    canvas = Image.new("RGB", (total_w, h + LABEL_H * 2 + 3 * pad), (16, 16, 16))
    canvas.paste(_label(title, total_w), (0, 0))
    canvas.paste(_label(f"Dollhouse depth", d.width), (pad, LABEL_H + pad))
    canvas.paste(_label(f"Faithful dollhouse (full flat)", dh.width), (pad * 2 + d.width, LABEL_H + pad))
    canvas.paste(_label(f"Faithful interior (one room)", it.width),
                 (pad * 3 + d.width + dh.width, LABEL_H + pad))
    y = LABEL_H * 2 + pad * 2
    canvas.paste(d, (pad, y))
    canvas.paste(dh, (pad * 2 + d.width, y))
    canvas.paste(it, (pad * 3 + d.width + dh.width, y))
    return canvas


def main(template_paths: list[str]) -> None:
    print("=== Warming up SDXL-turbo + Depth ControlNet ===")
    t0 = time.time()
    info = warmup(include_controlnet=True)
    print(f"  loaded={info.get('loaded')} cn_loaded={info.get('controlnet_loaded')} "
          f"in {time.time()-t0:.1f}s\n")

    rows = []
    for tpath in template_paths:
        template = json.loads(Path(tpath).read_text())
        tid = template.get("id", Path(tpath).stem)
        country = template.get("metadata", {}).get("country", "")
        print(f"[{tid}] {country}")

        # 1. Dollhouse depth (for inspection)
        d_info = render_template_dollhouse_depth(template)
        depth_img = d_info["depth_image"]
        Path(f"/tmp/dh_{tid}_depth.png").write_bytes(_to_png(depth_img))
        print(f"  depth: {d_info['rooms_count']} rooms, "
              f"{d_info['boundary_w']:.1f}x{d_info['boundary_d']:.1f}m")

        # 2. Dollhouse photoreal
        out = render_faithful_dollhouse_from_template(template, width=768, height=512)
        dh = out["result"]
        if dh.error:
            print(f"  dollhouse ERROR: {dh.error}"); continue
        Path(f"/tmp/dh_{tid}_dollhouse.png").write_bytes(dh.image_bytes)
        dh_img = Image.open(io.BytesIO(dh.image_bytes))
        print(f"  dollhouse: {dh.latency_s:.2f}s")

        # 3. Interior photoreal (for compare)
        out_i = render_faithful_from_template(template, width=768, height=512)
        it = out_i["result"]
        if it.error:
            print(f"  interior ERROR: {it.error}"); continue
        Path(f"/tmp/dh_{tid}_interior.png").write_bytes(it.image_bytes)
        interior_img = Image.open(io.BytesIO(it.image_bytes))
        print(f"  interior:  {it.latency_s:.2f}s")

        title = (f"{tid}  |  {country}  |  {d_info['rooms_count']} rooms, "
                 f"{d_info['boundary_w']:.1f}×{d_info['boundary_d']:.1f}m, "
                 f"ceiling {d_info['ceiling_height_m']:.1f}m")
        three_up = _stitch_3up(depth_img, dh_img, interior_img, title)
        three_up.save(f"/tmp/dh_{tid}_3up.png")
        print(f"  3-up -> /tmp/dh_{tid}_3up.png\n")
        rows.append({
            "tid": tid, "country": country, "title": title,
            "dollhouse_s": round(dh.latency_s, 2),
            "interior_s": round(it.latency_s, 2),
        })

    html = ['<!doctype html><meta charset=utf-8>',
            '<title>Dollhouse vs Interior — BIM render compare</title>',
            '<style>body{font-family:system-ui;background:#111;color:#ddd;'
            'margin:0;padding:24px} h1{font-size:18px;margin:0 0 18px}'
            'figure{margin:0 0 32px} figcaption{font-size:12px;opacity:.7;'
            'margin-top:6px} img{max-width:100%;border-radius:4px;'
            'box-shadow:0 2px 12px rgba(0,0,0,.4)}</style>',
            f'<h1>Dollhouse (full flat) vs Interior (one room) — {len(rows)} templates '
            f'({time.strftime("%Y-%m-%d %H:%M")})</h1>']
    for r in rows:
        html += [
            f'<figure><img src="dh_{r["tid"]}_3up.png" alt="{r["tid"]}">'
            f'<figcaption>{r["title"]} &middot; '
            f'dollhouse {r["dollhouse_s"]}s · interior {r["interior_s"]}s'
            f'</figcaption></figure>'
        ]
    Path("/tmp/dh_index.html").write_text("\n".join(html))
    print(f"Done. Gallery: /tmp/dh_index.html ({len(rows)} entries)")


def _to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        templates = sys.argv[1:]
    else:
        templates = [
            "data/templates/europe/eu_de_2zimmer_berlin_altbau.json",
            "data/templates/europe/eu_fr_1bed_paris_marais.json",
            "data/templates/global/gl_jp_1ldk_tokyo_mansion.json",
            "data/templates/global/gl_ae_1bed_dubai_marina.json",
            "data/templates/india/in_1bhk_bangalore_whitefield.json",
            "data/templates/global/gl_au_2bed_sydney_modern.json",
        ]
    main(templates)
