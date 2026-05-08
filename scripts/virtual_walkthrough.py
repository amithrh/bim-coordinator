"""Virtual walkthrough — render every habitable room of an apartment and
produce both:

  1. A composed contact-sheet PNG (floor plan + every room's photoreal
     interior in a grid).
  2. A self-contained HTML viewer where a customer can click "next room"
     to walk through the apartment, with a floor-plan minimap showing
     "you are here".

Per-room render:
    POST /api/render mode=faithful view=interior focus_room_id=<room>

Total wall-clock for a 5-room apartment: ~10s warm.
"""

from __future__ import annotations

import base64
import io
import json
import math
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont
import cairosvg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.floorplan_renderer import render_template_svg

API = "http://127.0.0.1:8009"

# Skip these room types (no useful interior view)
SKIP_TYPES = {"corridor", "diele", "flur", "stairs", "balcony",
                "loggia", "terrace", "store", "wardrobe", "utility",
                "abstellraum", "passage"}
# Include even if small (these rooms are small but worth seeing)
INCLUDE_SMALL = {"bathroom", "wc", "bad", "kitchen", "kueche"}


# ---------------------------------------------------------------------------
# Drawing helpers (subset of full_cycle_demo)
# ---------------------------------------------------------------------------

PAPER = (22, 22, 22)
HEADER_BG = (16, 18, 22)
LABEL_BG = (28, 28, 28)
TXT_HI = (245, 245, 245)
TXT_MD = (180, 180, 180)
TXT_LO = (140, 140, 140)
ACCENT = (0, 200, 160)


def _font(size, bold=False):
    paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size, index=1 if bold else 0)
        except Exception:
            continue
    return ImageFont.load_default()


def label_strip(text, w, h, fontsize=14, bg=LABEL_BG, fg=TXT_HI,
                  bold=False, align="center"):
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    f = _font(fontsize, bold=bold)
    bbox = d.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if align == "center":
        x = (w - tw) // 2
    elif align == "right":
        x = w - tw - 12
    else:
        x = 12
    y = (h - th) // 2 - 2
    d.text((x, y), text, fill=fg, font=f)
    return img


def fit_height(im, target_h):
    return im.resize(
        (int(im.width * target_h / im.height), target_h),
        Image.Resampling.LANCZOS,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def fetch_template(tid: str) -> dict:
    r = requests.get(f"{API}/api/templates/{tid}/json")
    r.raise_for_status()
    return r.json()


def pick_walkthrough_rooms(template: dict) -> list[dict]:
    """Return ordered list of rooms worth rendering an interior shot for."""
    rooms = template.get("rooms") or []
    if not rooms and template.get("floors"):
        rooms = []
        for fl in template["floors"]:
            rooms.extend(fl.get("rooms", []))
    chosen = []
    for r in rooms:
        rtype = (r.get("type") or "").lower()
        area = r.get("area_sqm") or 0
        if rtype in SKIP_TYPES:
            continue
        if rtype not in INCLUDE_SMALL and area < 4:
            continue
        chosen.append(r)
    # Order: living first, then bedrooms, then kitchen/bath/etc.
    order = {"living": 0, "dining": 1, "master_bedroom": 2, "bedroom": 3,
             "kitchen": 4, "kueche": 4, "office": 5, "study": 5,
             "bathroom": 6, "bad": 6, "wc": 7}
    chosen.sort(key=lambda r: (order.get((r.get("type") or "").lower(), 9),
                                  -(r.get("area_sqm") or 0)))
    return chosen


def render_room(tid: str, room_id: str) -> tuple[Image.Image, dict, float]:
    t0 = time.time()
    r = requests.post(f"{API}/api/render", json={
        "template_id": tid, "mode": "faithful", "view": "interior",
        "focus_room_id": room_id,
    })
    r.raise_for_status()
    elapsed = time.time() - t0
    headers = {k.lower(): v for k, v in r.headers.items()
                if k.lower().startswith("x-render")}
    return Image.open(io.BytesIO(r.content)).convert("RGB"), headers, elapsed


# ---------------------------------------------------------------------------
# Floor plan minimap with highlighted room
# ---------------------------------------------------------------------------

def render_minimap_for_room(template: dict, highlight_room_id: str,
                              size: int = 600) -> Image.Image:
    """Render the floor plan with one room highlighted (overlay an
    accent-colored translucent polygon on top of the standard plan)."""
    # 1) Render the standard architectural plan
    svg_path = f"/tmp/walkthrough_minimap.svg"
    png_path = f"/tmp/walkthrough_minimap.png"
    render_template_svg(template, svg_path, size=size)
    cairosvg.svg2png(url=svg_path, write_to=png_path,
                      output_width=size, background_color="white")
    base = Image.open(png_path).convert("RGB")

    # 2) Compute the highlight room's bounding box in IMAGE pixels
    # (We need to mirror the renderer's coordinate transform.)
    boundary = template.get("boundary", {}).get("polygon", [])
    if not boundary:
        return base
    bxs = [p[0] for p in boundary]
    bys = [p[1] for p in boundary]
    bx_min, bx_max = min(bxs), max(bxs)
    by_min, by_max = min(bys), max(bys)
    bw = bx_max - bx_min
    bh = by_max - by_min

    pad = 32
    canvas_w = canvas_h = size
    scale = min((canvas_w - 2 * pad) / bw, (canvas_h - 2 * pad) / bh)
    cw = bw * scale
    ch = bh * scale
    ox = pad + (canvas_w - 2 * pad - cw) / 2 - bx_min * scale
    oy = pad + canvas_h - 2 * pad - (canvas_h - 2 * pad - ch) / 2 + by_min * scale

    def tx(p):
        return (ox + p[0] * scale, oy - p[1] * scale)

    rooms = template.get("rooms") or []
    target = next((r for r in rooms if r.get("id") == highlight_room_id), None)
    if not target:
        return base

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    pts = [tx(p) for p in target["polygon"]]
    d.polygon(pts, fill=(255, 200, 60, 110),
                outline=(255, 200, 60, 230), width=4)

    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def build_contact_sheet(template: dict, results: list[dict], out_path: str):
    """Grid: floor plan large on the left, room cards 2-up on the right.

    Layout for 5 rooms:
        +--------+------+------+
        |        | A    | B    |
        |  PLAN  +------+------+
        |        | C    | D    |
        |        +------+------+
        |        |    E (wide) |
        +--------+--------------+
    """
    n = len(results)
    PAGE_W = 1980
    PAD = 16
    HEADER_H = 110
    plan_w = 800
    cards_w = PAGE_W - plan_w - 3 * PAD
    cols = 2
    rows = math.ceil(n / cols)
    card_h = 320
    page_h = HEADER_H + 2 * PAD + max(plan_w, rows * (card_h + PAD * 2 + 30))

    page = Image.new("RGB", (PAGE_W, page_h), (10, 10, 10))

    # Header
    header = Image.new("RGB", (PAGE_W, HEADER_H), HEADER_BG)
    d = ImageDraw.Draw(header)
    md = template.get("metadata", {})
    d.text((24, 18), f"Virtual walkthrough — {template['id']}",
            fill=TXT_HI, font=_font(28, bold=True))
    sub = (f"{md.get('country','?')} · {md.get('city_inspiration','?')} · "
           f"{md.get('total_area_sqm','?')} m² · {md.get('bedrooms','?')} BR · "
           f"{n} rooms toured")
    d.text((24, 56), sub, fill=TXT_MD, font=_font(14))
    page.paste(header, (0, 0))

    # Floor plan (no highlight — overall view)
    svg_path = "/tmp/walkthrough_plan_full.svg"
    png_path = "/tmp/walkthrough_plan_full.png"
    render_template_svg(template, svg_path, size=1024)
    cairosvg.svg2png(url=svg_path, write_to=png_path,
                      output_width=plan_w, background_color="white")
    plan = Image.open(png_path).convert("RGB")
    plan = fit_height(plan, max(plan_w, rows * (card_h + PAD * 2 + 30)) - 70)
    if plan.width > plan_w:
        plan = plan.resize(
            (plan_w, int(plan.height * plan_w / plan.width)),
            Image.Resampling.LANCZOS,
        )
    plan_y = HEADER_H + PAD + 30
    page.paste(label_strip("Floor plan", plan_w, 30, fontsize=15,
                              bold=True, align="left"),
                (PAD, HEADER_H + PAD))
    page.paste(plan, (PAD, plan_y))

    # Room cards grid
    cell_w = (cards_w - PAD) // cols
    for i, r in enumerate(results):
        row = i // cols
        col = i % cols
        x = plan_w + 2 * PAD + col * (cell_w + PAD)
        y = HEADER_H + PAD + row * (card_h + PAD * 2 + 30)
        # Card title
        title = f"{i+1}. {r['name']} · {r['area']} m²"
        page.paste(label_strip(title, cell_w, 30, fontsize=14,
                                  bold=True, align="left"), (x, y))
        # Image
        img = fit_height(r["image"], card_h)
        if img.width > cell_w:
            img = img.resize(
                (cell_w, int(img.height * cell_w / img.width)),
                Image.Resampling.LANCZOS,
            )
        page.paste(img, (x + (cell_w - img.width) // 2, y + 30))
        # Caption
        cap = f"{r['type']}  ·  {r['latency_s']:.2f}s"
        page.paste(label_strip(cap, cell_w, 22, fontsize=11,
                                  fg=TXT_LO, bg=PAPER),
                    (x, y + 30 + img.height))

    page.save(out_path)
    return out_path


def build_html_viewer(template: dict, results: list[dict],
                        out_path: str) -> str:
    """A standalone HTML page with a floor-plan minimap and arrow keys to
    walk between rooms. All images embedded as base64 so the file is
    self-contained — the customer can open it offline."""
    md = template.get("metadata", {})
    pages = []
    for r in results:
        # Per-room minimap with highlight
        minimap = render_minimap_for_room(template, r["id"], size=480)
        buf_m = io.BytesIO(); minimap.save(buf_m, "PNG")
        b64_m = base64.b64encode(buf_m.getvalue()).decode("ascii")
        # Per-room photo
        buf_p = io.BytesIO(); r["image"].save(buf_p, "PNG")
        b64_p = base64.b64encode(buf_p.getvalue()).decode("ascii")
        pages.append({
            "id": r["id"], "name": r["name"], "type": r["type"],
            "area": r["area"], "latency": r["latency_s"],
            "minimap_b64": b64_m, "photo_b64": b64_p,
        })

    title = (f"Walkthrough — {template['id']} "
             f"({md.get('country','')} · {md.get('total_area_sqm','?')} m² · "
             f"{md.get('bedrooms','?')} BR)")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
:root {{ --bg:#0f1115; --card:#1a1d23; --hi:#f5f5f5; --md:#aaa;
         --lo:#666; --accent:#FFC83D; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--hi);
         font-family:-apple-system,BlinkMacSystemFont,Helvetica,sans-serif; }}
header {{ padding:18px 24px; background:#0a0c10; border-bottom:1px solid #222; }}
h1 {{ margin:0; font-size:18px; font-weight:600; }}
.sub {{ color:var(--md); font-size:13px; margin-top:4px; }}
.layout {{ display:grid; grid-template-columns:1fr 380px;
            gap:24px; padding:24px; }}
.photo {{ background:var(--card); border-radius:10px; overflow:hidden;
            position:relative; aspect-ratio:3/2; }}
.photo img {{ width:100%; height:100%; object-fit:cover; display:none; }}
.photo img.active {{ display:block; }}
.caption {{ position:absolute; bottom:0; left:0; right:0;
              background:linear-gradient(transparent,rgba(0,0,0,.85));
              padding:32px 18px 14px; }}
.caption .room-name {{ font-size:24px; font-weight:600; }}
.caption .room-meta {{ color:var(--md); font-size:13px; margin-top:4px; }}
.controls {{ position:absolute; top:50%; left:0; right:0;
                transform:translateY(-50%); pointer-events:none;
                display:flex; justify-content:space-between; padding:0 16px; }}
.controls button {{ pointer-events:auto;
            background:rgba(0,0,0,0.55); color:white; border:1px solid #444;
            border-radius:50%; width:48px; height:48px; font-size:24px;
            cursor:pointer; transition:background .15s; }}
.controls button:hover {{ background:rgba(0,0,0,0.85); }}
.controls button:disabled {{ opacity:0.3; cursor:default; }}
.minimap {{ background:var(--card); border-radius:10px; overflow:hidden;
              padding:12px; }}
.minimap-img {{ width:100%; display:none; border-radius:6px;
                  background:white; }}
.minimap-img.active {{ display:block; }}
.minimap-title {{ font-size:13px; color:var(--md);
                     margin:0 0 10px 4px; }}
.room-list {{ background:var(--card); border-radius:10px; padding:12px;
                margin-top:16px; }}
.room-list h3 {{ margin:0 0 10px 4px; font-size:13px; color:var(--md);
                   font-weight:500; }}
.room-list ol {{ list-style:none; padding:0; margin:0; }}
.room-list li {{ padding:8px 12px; border-radius:6px; cursor:pointer;
                   color:var(--md); font-size:14px;
                   display:flex; justify-content:space-between;
                   transition:background .15s; }}
.room-list li:hover {{ background:#272a32; color:var(--hi); }}
.room-list li.active {{ background:rgba(255,200,61,0.15);
                           color:var(--accent); font-weight:600; }}
.room-list .area {{ color:var(--lo); font-size:12px; }}
footer {{ text-align:center; padding:16px; color:var(--lo); font-size:12px; }}
</style>
</head><body>

<header>
  <h1>{title}</h1>
  <div class="sub">Use ← → keys, click rooms in the list, or click the chevrons.</div>
</header>

<div class="layout">
  <div class="photo" id="photo">
    {"".join(f'<img id="img-{p["id"]}" '
              f'src="data:image/png;base64,{p["photo_b64"]}" '
              f'alt="{p["name"]}">' for p in pages)}
    <div class="caption" id="caption"></div>
    <div class="controls">
      <button id="prev">‹</button>
      <button id="next">›</button>
    </div>
  </div>
  <aside>
    <div class="minimap">
      <div class="minimap-title" id="minimap-title">You are here</div>
      {"".join(f'<img id="map-{p["id"]}" class="minimap-img" '
                f'src="data:image/png;base64,{p["minimap_b64"]}" '
                f'alt="{p["name"]} location">' for p in pages)}
    </div>
    <div class="room-list">
      <h3>Rooms ({len(pages)})</h3>
      <ol id="rooms">
        {"".join(f'<li data-id="{p["id"]}" data-idx="{i}">'
                  f'<span>{i+1}. {p["name"]}</span>'
                  f'<span class="area">{p["area"]} m²</span></li>'
                  for i, p in enumerate(pages))}
      </ol>
    </div>
  </aside>
</div>

<footer>
  Generated by BIM Coordinator · SDXL-turbo + Depth ControlNet ·
  one-shot per-room photoreal render
</footer>

<script>
const rooms = {json.dumps([{"id": p["id"], "name": p["name"],
                              "type": p["type"], "area": p["area"],
                              "latency": p["latency"]} for p in pages])};
let idx = 0;

function show(i) {{
  idx = Math.max(0, Math.min(rooms.length - 1, i));
  const r = rooms[idx];
  document.querySelectorAll('#photo img').forEach(el => el.classList.remove('active'));
  document.getElementById('img-' + r.id).classList.add('active');
  document.querySelectorAll('.minimap-img').forEach(el => el.classList.remove('active'));
  document.getElementById('map-' + r.id).classList.add('active');
  document.querySelectorAll('#rooms li').forEach(el => el.classList.remove('active'));
  document.querySelector(`#rooms li[data-idx='${{idx}}']`).classList.add('active');
  document.getElementById('caption').innerHTML =
    `<div class="room-name">${{r.name}}</div>` +
    `<div class="room-meta">${{r.type}}  ·  ${{r.area}} m²  ·  rendered in ${{r.latency.toFixed(2)}}s</div>`;
  document.getElementById('prev').disabled = (idx === 0);
  document.getElementById('next').disabled = (idx === rooms.length - 1);
}}

document.getElementById('prev').onclick = () => show(idx - 1);
document.getElementById('next').onclick = () => show(idx + 1);
document.querySelectorAll('#rooms li').forEach(el => {{
  el.onclick = () => show(parseInt(el.dataset.idx));
}});
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft')  show(idx - 1);
  if (e.key === 'ArrowRight') show(idx + 1);
}});
show(0);
</script>
</body></html>
"""
    Path(out_path).write_text(html)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(tid: str) -> None:
    print(f"\n=== Virtual walkthrough — {tid} ===\n")
    template = fetch_template(tid)

    rooms = pick_walkthrough_rooms(template)
    print(f"Rooms to tour: {len(rooms)}")
    for r in rooms:
        print(f"  · {r['name']:24s}  ({r['type']:18s}  {r.get('area_sqm','?')} m²)")

    results = []
    for r in rooms:
        print(f"\n→ Rendering {r['name']} ({r['id']})...")
        try:
            img, headers, elapsed = render_room(tid, r["id"])
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        results.append({
            "id": r["id"], "name": r["name"], "type": r["type"],
            "area": r.get("area_sqm", 0),
            "image": img, "latency_s": elapsed,
            "render_s": float(headers.get("x-render-latency-s", elapsed)),
        })
        print(f"  rendered in {elapsed:.2f}s "
              f"(SDXL: {headers.get('x-render-latency-s','?')}s)")

    # Outputs
    cs_path = f"/tmp/walkthrough_{tid}_contact.png"
    build_contact_sheet(template, results, cs_path)
    print(f"\ncontact-sheet -> {cs_path}")

    html_path = f"/tmp/walkthrough_{tid}.html"
    build_html_viewer(template, results, html_path)
    print(f"html viewer  -> {html_path}")
    print(f"             open it in any browser; ← → walks between rooms")


if __name__ == "__main__":
    tid = sys.argv[1] if len(sys.argv) > 1 else "eu_de_1bed_munich_schwabing"
    main(tid)
