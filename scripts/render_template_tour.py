#!/usr/bin/env python3
"""End-to-end: take a BIM template_id, emit a HyperFrames composition that
renders a 47s broker-mode tour. One script per demo template:

    python scripts/render_template_tour.py eu_fr_1bed_paris_marais

Steps:
  1. Load template JSON from data/templates/<region>/<id>.json
  2. Compute room layout (positions in %-of-canvas) from polygons
  3. Select tour stops (max 5 in canonical order: entry → living → kitchen → bath → bedroom)
  4. For each stop, build BIM-conditioned panoramic prompt + render with SDXL-turbo
  5. Emit experiments/hyperframes-bim-tour-<id>/{index.html, hyperframes.json, meta.json, assets/}
  6. Call `npx hyperframes render` → produces out/<id>_tour.mp4

All copy is in the template's local language (or a sensible English fallback)
based on the country.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent  # repo root (scripts/ → ..)
TEMPLATES_DIR = ROOT / "data/templates"
EXPERIMENTS_DIR = ROOT / "experiments"
PUBLIC_DIR = ROOT / "frontend/public"

sys.path.insert(0, str(ROOT))
from backend.app.image_renderer import render_from_prompt  # noqa: E402

# ── Style anchors per country (truncated to fit CLIP's 77-token budget) ────
# Each anchor is mid-prompt — keep them ≤25 words so the room-specific
# detail at the START of the prompt always survives tokenisation.
STYLE_ANCHORS = {
    "Germany":            "Munich Schwabing Jugendstil Altbau, oak parquet, period mouldings, soft daylight",
    "France":             "Paris Marais Haussmannian, herringbone parquet, tall French windows, period mouldings, refined neutrals",
    "India":              "Bangalore vastu-compliant apartment, polished granite floor, neutral walls, contemporary Indian design",
    "Japan":              "Tokyo modern minimalist, light wood floor, white walls, low furniture, paper-shaded lighting, Nordic-Japanese",
    "United Arab Emirates":"Dubai Marina luxe modern, marble floor, neutral palette, floor-to-ceiling windows, skyline view",
    "United States":      "NYC tenement-style apartment, hardwood floor, white walls, exposed brick accents, contemporary",
    "Australia":          "Sydney modern coastal, polished concrete or oak floor, abundant daylight, indoor-outdoor flow",
}

# Per-room-type prompt fragments. The script inserts area_sqm + window count.
ROOM_TYPE_PROMPT = {
    "entry":          "an entry foyer with paneled walls, oak floor, console table, period sconce and coat hooks",
    "corridor":       "a small genkan / entry corridor with shoe storage and a step up to the main floor",
    "living":         "a spacious living room with sofa, coffee table, framed art and refined furnishings",
    "kitchen":        "a clean modern kitchen with custom cabinetry, marble or stone countertop, integrated appliances",
    "bathroom":       "a beautifully appointed bathroom with tile floor, freestanding tub or shower, period fittings",
    "wc":             "a compact WC with tile floor, basin, mirror, soft diffused light",
    "utility":        "a small utility / laundry room with stacked washer-dryer and shelving",
    "store":          "a small storage / laundry room",
    "master_bedroom": "a master bedroom with queen bed, walnut headboard, matching nightstands, linen palette",
    "bedroom":        "a tranquil bedroom with bed, side table, soft lighting and warm wood floor",
    "balcony":        "a balcony or terrace with potted plants, outdoor seating and city or garden view",
    "dining":         "a dining area with table and chairs, pendant lighting, framed art on the wall",
}

# Caption labels per stop (English fallback used everywhere — concise and
# consistent across templates).
STOP_LABELS = {
    "entry":          ("Entry",       "entry foyer"),
    "corridor":       ("Entry",       "entry corridor"),
    "living":         ("Living room", "living room"),
    "kitchen":        ("Kitchen",     "kitchen"),
    "bathroom":       ("Bath",        "bathroom"),
    "wc":             ("WC",          "powder room"),
    "utility":        ("Utility",     "utility / laundry"),
    "store":          ("Storage",     "storage / laundry"),
    "master_bedroom": ("Master bedroom", "master bedroom"),
    "bedroom":        ("Bedroom",     "bedroom"),
    "balcony":        ("Balcony",     "balcony / terrace"),
    "dining":         ("Dining",      "dining area"),
}

# Tour-stop selection priority: pick at most 5 in this canonical order.
STOP_PRIORITY = [
    "entry", "corridor",
    "living",
    "kitchen",
    "bathroom",
    "master_bedroom", "bedroom",
    "balcony",
    "dining",
]

# Room background colours (map BIM type → CSS class on the .room div).
ROOM_TYPE_TO_CLASS = {
    "entry":          "r-foyer",
    "corridor":       "r-foyer",
    "living":         "r-living",
    "kitchen":        "r-kitchen",
    "bathroom":       "r-bath",
    "wc":             "r-bath",
    "utility":        "r-bath",
    "store":          "r-bath",
    "master_bedroom": "r-bedroom",
    "bedroom":        "r-bedroom",
    "balcony":        "r-balcony",
    "dining":         "r-living",
}

# Panoramic dims for room photos (2:1 aspect).
PHOTO_W, PHOTO_H, SDXL_STEPS = 1536, 768, 4


# ─────────────────────────────────────────────────────────────────────────────
def find_template(template_id: str) -> Path:
    for region_dir in TEMPLATES_DIR.iterdir():
        if not region_dir.is_dir():
            continue
        candidate = region_dir / f"{template_id}.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"template {template_id!r} not found under {TEMPLATES_DIR}")


def bbox_of_polygon(poly: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def select_stops(rooms: list[dict]) -> list[dict]:
    """Pick up to 5 rooms in canonical tour order."""
    by_type: dict[str, list[dict]] = {}
    for r in rooms:
        by_type.setdefault(r["type"], []).append(r)

    stops: list[dict] = []
    for ty in STOP_PRIORITY:
        if ty in by_type:
            # If multiple of the same type, pick the largest.
            r = max(by_type[ty], key=lambda r: r.get("area_sqm", 0))
            stops.append(r)
            if len(stops) >= 5:
                break
    return stops


def windows_for_room(template: dict, room_id: str) -> list[dict]:
    return [w for w in template.get("windows", []) if w.get("room") == room_id]


def build_room_prompt(template: dict, room: dict, style_anchor: str) -> str:
    rtype = room["type"]
    area = room.get("area_sqm", 0)
    poly = room["polygon"]
    minx, miny, maxx, maxy = bbox_of_polygon(poly)
    w_m = maxx - minx
    h_m = maxy - miny
    n_windows = len(windows_for_room(template, room["id"]))
    ceiling_m = template.get("boundary", {}).get("ceiling_height_mm", 2700) / 1000.0

    base = ROOM_TYPE_PROMPT.get(rtype, "an interior room")
    win_text = ""
    if n_windows == 1:
        win_text = ", one large window letting in soft daylight"
    elif n_windows >= 2:
        win_text = f", {n_windows} large windows letting in soft daylight"

    return (
        f"Photorealistic panoramic interior photograph, wide horizontal view, of "
        f"{base}, approximately {area:g} square meters ({w_m:g}m × {h_m:g}m), "
        f"{ceiling_m:.1f} meter ceiling{win_text}. "
        f"{style_anchor}. Wide-angle, sharp focus, no people."
    )


def render_room_photos(template: dict, stops: list[dict], assets_dir: Path) -> dict[str, str]:
    """Render one panoramic per stop. Returns {stop_id: filename}."""
    style_anchor = STYLE_ANCHORS.get(
        template.get("metadata", {}).get("country", ""),
        "modern apartment, neutral palette, abundant natural light",
    )
    assets_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for stop in stops:
        rid = stop["id"].replace("r_", "")
        filename = f"room_{rid}.png"
        out_path = assets_dir / filename
        prompt = build_room_prompt(template, stop, style_anchor)
        print(f"  [{stop['name']:<16}] {PHOTO_W}×{PHOTO_H}  {SDXL_STEPS} steps")
        t0 = time.time()
        result = render_from_prompt(prompt, width=PHOTO_W, height=PHOTO_H, steps=SDXL_STEPS)
        if result.error:
            print(f"     ERROR: {result.error}")
            continue
        out_path.write_bytes(result.image_bytes)
        print(f"     wrote {filename} ({len(result.image_bytes)//1024} KB) in {time.time()-t0:.1f}s")
        out[stop["id"]] = filename
    return out


# ─────────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE_PATH = ROOT / "experiments/hyperframes-bim-floorplan-tour/index.html"


def emit_index_html(
    template: dict,
    stops: list[dict],
    photo_files: dict[str, str],
    out_path: Path,
) -> None:
    """Generate index.html by patching the Munich-template HTML.

    We use the Munich HTML as a starting point and rewrite the chunks that
    are template-specific:
      • header (city, name, area + ceiling)
      • room divs (positions from polygons)
      • ENTRY marker x-position
      • photo frames + captions
      • STOPS array in JS
      • outro stats
    """
    md = template.get("metadata", {})
    city  = md.get("city_inspiration", "")
    name  = md.get("size_label", "")
    area  = md.get("total_area_sqm", 0)
    ceiling_m = template.get("boundary", {}).get("ceiling_height_mm", 2700) / 1000.0

    # Layout: compute boundary bbox → each room's CSS rect.
    bminx, bminy, bmaxx, bmaxy = bbox_of_polygon(template["boundary"]["polygon"])
    bw = bmaxx - bminx
    bh = bmaxy - bminy

    def room_rect(room: dict) -> tuple[float, float, float, float]:
        rminx, rminy, rmaxx, rmaxy = bbox_of_polygon(room["polygon"])
        # CSS y is inverted (BIM north grows up; CSS top grows down)
        left   = (rminx - bminx) / bw * 100
        top    = (bmaxy - rmaxy) / bh * 100
        width  = (rmaxx - rminx) / bw * 100
        height = (rmaxy - rminy) / bh * 100
        return left, top, width, height

    def room_centroid_pct(room: dict) -> tuple[float, float]:
        rminx, rminy, rmaxx, rmaxy = bbox_of_polygon(room["polygon"])
        cx = (rminx + rmaxx) / 2
        cy = (rminy + rmaxy) / 2
        return ((cx - bminx) / bw * 100, (bmaxy - cy) / bh * 100)

    # Build all room divs (NOT just stops — show every room in the floor plan).
    room_divs = []
    for r in template["rooms"]:
        left, top, w, h = room_rect(r)
        cls = ROOM_TYPE_TO_CLASS.get(r["type"], "r-living")
        room_divs.append(
            f'          <div class="room {cls}" id="rm-{r["id"]}"\n'
            f'               style="left: {left:.1f}%; top: {top:.1f}%; '
            f'width: {w:.1f}%; height: {h:.1f}%;">\n'
            f'            <div class="room-name">{r["name"]}</div>'
            f'<div class="room-area">{r.get("area_sqm",0):g} m²</div>\n'
            f'          </div>'
        )

    # Find the main entry door + which boundary edge it's on, so the ENTRY
    # marker can point at the actual front door (south/north/west/east).
    main_entry = None
    for d in template.get("doors", []):
        if d.get("is_main_entry") or d.get("from") == "outside":
            main_entry = d
            break
    if main_entry is not None:
        ex, ey = main_entry["position"]
        ex_pct      = (ex - bminx) / bw * 100               # 0 = west, 100 = east
        ey_pct_bim  = (ey - bminy) / bh * 100               # 0 = south, 100 = north
        ey_pct_css  = 100 - ey_pct_bim                       # 0 = top, 100 = bottom
        # Pick edge by which dimension is "stuck" to the boundary.
        if ey_pct_bim < 5:
            entry_edge = "south"
            entry_inline = f"left: {ex_pct:.1f}%;"
        elif ey_pct_bim > 95:
            entry_edge = "north"
            entry_inline = f"left: {ex_pct:.1f}%;"
        elif ex_pct < 5:
            entry_edge = "west"
            entry_inline = f"top: {ey_pct_css:.1f}%;"
        else:
            entry_edge = "east"
            entry_inline = f"top: {ey_pct_css:.1f}%;"
    else:
        # Fall back to centroid of first stop on south edge
        cx, _ = room_centroid_pct(stops[0]) if stops else (50, 100)
        entry_edge = "south"
        entry_inline = f"left: {cx:.1f}%;"

    # Build photo-frame HTML.
    photo_frames = []
    for i, stop in enumerate(stops, start=1):
        eyebrow_label, body_label = STOP_LABELS.get(stop["type"], (stop["type"], stop["type"]))
        rid = stop["id"].replace("r_", "")
        photo_file = photo_files.get(stop["id"], f"room_{rid}.png")
        n_windows = len(windows_for_room(template, stop["id"]))
        meta_bits = [f'{stop.get("area_sqm",0):g} m²', body_label]
        if n_windows >= 1:
            meta_bits.append(f"{n_windows} window{'s' if n_windows != 1 else ''}")
        photo_meta = " · ".join(meta_bits)
        photo_frames.append(
            f'        <div class="photo-frame" id="photo-{rid}">\n'
            f'          <img src="assets/{photo_file}" alt="{stop["name"]}">\n'
            f'          <div class="look360-pill">360° look</div>\n'
            f'          <div class="photo-overlay">\n'
            f'            <div class="photo-stop">Stop {i} / {len(stops)} · {eyebrow_label}</div>\n'
            f'            <div class="photo-name">{stop["name"]}</div>\n'
            f'            <div class="photo-meta">{photo_meta}</div>\n'
            f'          </div>\n'
            f'        </div>'
        )

    # Build the JS STOPS array (just the centroids needed at runtime).
    stops_js_entries = []
    for stop in stops:
        cx, cy = room_centroid_pct(stop)
        rid = stop["id"].replace("r_", "")
        stops_js_entries.append(
            f'        {{ id: "{rid}", x: {cx:.1f}, y: {cy:.1f}, '
            f'photoId: "photo-{rid}", roomId: "rm-{stop["id"]}" }},'
        )
    stops_js = "\n".join(stops_js_entries)

    # Header text. Pick a 1-2-word "accent" from the style label that gets
    # gold-coloured in the title.
    style = md.get("style", name)
    name_main = name
    name_accent = ""
    # E.g. "1-Zimmer-Wohnung" + "Jugendstil Universitätsviertel" → main="1 BR", accent="Jugendstil"
    accent_words = []
    for keyword in ["Jugendstil", "Haussmann", "Vastu", "Modern", "Marina",
                    "Tenement", "Mansion", "Coastal", "Altbau"]:
        if keyword.lower() in style.lower():
            accent_words.append(keyword)
            break
    name_accent = accent_words[0] if accent_words else "Modern"
    bedrooms = md.get("bedrooms", 0)
    name_main = f"{bedrooms} BR" if bedrooms else "Studio"

    n_rooms = len(template.get("rooms", []))
    hdr_meta = f"{int(area)} m² · {n_rooms} rooms · {ceiling_m:.1f} m ceiling · IFC valid 35/35"

    # Outro text.
    outro_eyebrow = f"{city} {name_main}".strip()
    outro_title_main = f"{n_rooms} rooms · "
    outro_title_accent = f"{int(area)} m²"
    outro_subtitle = "tour in 30 seconds"

    # Assemble. We start from the Munich HTML's structure but override the
    # data-bearing chunks. Using string replacement on landmark fragments.
    html = HTML_TEMPLATE_PATH.read_text()

    # 1. Header
    html = html.replace(
        '<div class="hdr-eyebrow">Munich Schwabing</div>',
        f'<div class="hdr-eyebrow">{city}</div>',
    )
    html = html.replace(
        '<div class="hdr-title">1 BR <span>Jugendstil</span></div>',
        f'<div class="hdr-title">{name_main} <span>{name_accent}</span></div>',
    )
    html = html.replace(
        '<div class="hdr-meta">50 m² · 5 rooms · 3.1 m ceiling · IFC valid 35/35</div>',
        f'<div class="hdr-meta">{hdr_meta}</div>',
    )

    # 2. Room divs — replace the full Munich room block with ours.
    munich_rooms_start = '          <!-- Plan rooms positioned by BIM coords (10m × 5m → percentages)'
    munich_rooms_end = '          </div>\n\n          <!-- North arrow'
    start_idx = html.index(munich_rooms_start)
    end_idx = html.index(munich_rooms_end)
    new_rooms_block = (
        f'          <!-- Plan rooms — auto-laid-out from {template["id"]} polygons. -->\n'
        + "\n".join(room_divs) + "\n\n"
    )
    # Re-insert the comment block + room divs, then continue with the existing
    # compass/entry markup.
    html = html[:start_idx] + new_rooms_block + html[end_idx:]

    # 3. ENTRY marker — add .entry-<edge> class + inline position override.
    # West/east pills get partially clipped by the 1080px canvas edge, so for
    # those edges we shorten the label to "ENT." which fits in ~50px.
    pill_label = "Entry" if entry_edge in ("south", "north") else "Door"
    html = html.replace(
        '<div class="entry-marker" id="entry-marker">',
        f'<div class="entry-marker entry-{entry_edge}" id="entry-marker" '
        f'style="{entry_inline}">',
    )
    html = html.replace(
        '<div class="entry-pill">Entry</div>',
        f'<div class="entry-pill">{pill_label}</div>',
    )

    # 4. Photo frames — replace the entire <div class="photo-stage"> body.
    photo_stage_start = '<div class="photo-stage">\n        <div class="photo-frame" id="photo-flur">'
    photo_stage_end = '          </div>\n        </div>\n      </div>\n\n      <!-- Outro -->'
    start_idx = html.index(photo_stage_start)
    end_idx = html.index(photo_stage_end) + len('          </div>\n        </div>\n      </div>')
    new_photos = (
        '<div class="photo-stage">\n'
        + "\n".join(photo_frames)
        + "\n      </div>"
    )
    html = html[:start_idx] + new_photos + html[end_idx:]

    # 5. Outro
    html = html.replace(
        '<div class="outro-eyebrow">Munich Schwabing 1 BR</div>',
        f'<div class="outro-eyebrow">{outro_eyebrow}</div>',
    )
    html = html.replace(
        '<div class="outro-title">5 rooms · <span>50 m²</span><br>tour in 30 seconds</div>',
        f'<div class="outro-title">{n_rooms} rooms · <span>{outro_title_accent}</span><br>{outro_subtitle}</div>',
    )

    # 6. STOPS JS array — replace the full hardcoded Munich array.
    munich_stops_start = '      const STOPS = ['
    munich_stops_end = '      ];'
    start_idx = html.index(munich_stops_start)
    end_idx = html.index(munich_stops_end, start_idx) + len(munich_stops_end)
    new_stops = (
        '      const STOPS = [\n'
        + stops_js + '\n'
        + '      ];'
    )
    html = html[:start_idx] + new_stops + html[end_idx:]

    out_path.write_text(html)


# ─────────────────────────────────────────────────────────────────────────────
def emit_meta(experiment_dir: Path, template_id: str, name: str) -> None:
    (experiment_dir / "meta.json").write_text(json.dumps({
        "id": f"hyperframes-bim-tour-{template_id}",
        "name": name,
        "createdAt": "2026-05-09T00:00:00.000Z",
    }, indent=2))


def emit_hyperframes_json(experiment_dir: Path) -> None:
    (experiment_dir / "hyperframes.json").write_text(json.dumps({
        "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
        "registry": "https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry",
        "paths": {
            "blocks": "compositions",
            "components": "compositions/components",
            "assets": "assets"
        }
    }, indent=2))


def render_mp4(experiment_dir: Path, template_id: str) -> Path | None:
    out_dir = experiment_dir / "out"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{template_id}_tour.mp4"
    print(f"  npx hyperframes render → {out_path.name} ...")
    t0 = time.time()
    proc = subprocess.run(
        ["npx", "hyperframes", "render", ".", "--output", f"out/{template_id}_tour.mp4",
         "--quality", "draft"],
        cwd=experiment_dir, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"     RENDER FAILED ({time.time()-t0:.1f}s)")
        print(proc.stderr[-1000:])
        return None
    print(f"     done in {time.time()-t0:.1f}s")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("template_id", help="e.g. eu_fr_1bed_paris_marais")
    ap.add_argument("--skip-photos", action="store_true",
                    help="skip SDXL re-render (assumes assets/ is already populated)")
    ap.add_argument("--skip-render", action="store_true",
                    help="skip the hyperframes mp4 step")
    args = ap.parse_args()

    template_path = find_template(args.template_id)
    template = json.loads(template_path.read_text())
    print(f"\n=== {args.template_id} ===")

    stops = select_stops(template["rooms"])
    print(f"  selected {len(stops)} stops: " +
          " → ".join(s["name"] for s in stops))

    experiment_dir = EXPERIMENTS_DIR / f"hyperframes-bim-tour-{args.template_id}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_photos:
        photo_files = render_room_photos(template, stops, experiment_dir / "assets")
    else:
        photo_files = {s["id"]: f"room_{s['id'].replace('r_','')}.png" for s in stops}

    emit_index_html(template, stops, photo_files, experiment_dir / "index.html")
    emit_meta(experiment_dir, args.template_id,
              template.get("metadata", {}).get("style", args.template_id))
    emit_hyperframes_json(experiment_dir)

    if not args.skip_render:
        mp4 = render_mp4(experiment_dir, args.template_id)
        if mp4 and mp4.exists():
            public_path = PUBLIC_DIR / mp4.name
            shutil.copy(mp4, public_path)
            print(f"  → http://localhost:3000/{mp4.name}  ({mp4.stat().st_size//1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
