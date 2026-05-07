"""Render 2D floor plan SVG for every template.

Now delegates to the architectural-grade renderer in
`backend.app.floorplan_renderer`, which adds wall thickness,
door swings, window glyphs, and room fixtures (sinks, beds, sofas, etc).

Usage:
    python render_svg.py                    # render every template
    python render_svg.py <template.json> <out.svg>   # render one
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import svgwrite

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "data" / "templates"
SVG_OUT_DIR = REPO_ROOT / "data" / "svg_plans"

# Make backend importable when this script is invoked directly
sys.path.insert(0, str(REPO_ROOT))
from backend.app.floorplan_renderer import (  # noqa: E402
    render_template_svg as _render_template_svg_arch,
)

COLOR_BY_TYPE = {
    "kitchen": "#FFF3CD", "kochnische": "#FFF3CD",
    "living": "#D1ECF1",
    "bedroom": "#E2D5F0", "master_bedroom": "#D5C8E0",
    "bathroom": "#D4EDDA", "wc": "#D4EDDA",
    "balcony": "#F8F9FA",
    "pooja": "#FFE5B4",
    "utility": "#E0E0E0", "abstellraum": "#E0E0E0",
    "corridor": "#F5F5F5", "diele": "#F5F5F5", "entry": "#F5F5F5",
    "dining": "#FFE5E5",
    "study": "#E5F0FF",
    "store": "#EEE", "wardrobe": "#EEE",
}


def bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}


def _render_floor_panel(dwg, boundary, rooms, doors, windows,
                         panel_x, panel_y, panel_w, panel_h, label=None):
    """Render one floor into a sub-panel of the SVG drawing."""
    bb = bbox(boundary)
    pad = 16
    scale = min((panel_w - 2 * pad) / bb["w"], (panel_h - 2 * pad) / bb["h"])
    cw, ch = bb["w"] * scale, bb["h"] * scale
    ox = panel_x + (panel_w - cw) / 2 - bb["x"] * scale
    oy = panel_y + panel_h - (panel_h - ch) / 2 + bb["y"] * scale

    def tx(p):
        return (ox + p[0] * scale, oy - p[1] * scale)

    pts = [tx(p) for p in boundary]
    dwg.add(dwg.polygon(pts, fill="white", stroke="#222", stroke_width=3))
    for room in rooms:
        rpts = [tx(p) for p in room["polygon"]]
        color = COLOR_BY_TYPE.get(room["type"], "#EFEFEF")
        dwg.add(dwg.polygon(rpts, fill=color, stroke="#888", stroke_width=1))
        cx = sum(p[0] for p in rpts) / len(rpts)
        cy = sum(p[1] for p in rpts) / len(rpts)
        dwg.add(dwg.text(room["name"], insert=(cx, cy),
                          font_size=12, text_anchor="middle",
                          font_family="Arial, sans-serif", fill="#333"))
        if "area_sqm" in room:
            dwg.add(dwg.text(f"{room['area_sqm']:g} m²",
                              insert=(cx, cy + 14),
                              font_size=10, text_anchor="middle",
                              font_family="Arial, sans-serif", fill="#666"))
    for d in doors:
        x, y = tx(d["position"])
        dwg.add(dwg.rect((x - 4, y - 4), (8, 8),
                          fill="#E66", stroke="#A00", stroke_width=1))
    for w in windows:
        x, y = tx(w["position"])
        dwg.add(dwg.rect((x - 6, y - 3), (12, 6),
                          fill="#69E", stroke="#04A", stroke_width=1))
    if label:
        dwg.add(dwg.text(label, insert=(panel_x + 8, panel_y + 18),
                          font_size=11, font_family="Arial, sans-serif",
                          fill="#1f4ed8", font_weight="bold"))


def render_one(template_path: Path, svg_path: Path, size: int = 1024, pad: int = 40):
    """Render a single template using the architectural-grade renderer.

    Falls back to the legacy basic renderer (kept below as
    `_render_one_basic`) if the architectural one fails — multi-floor
    templates take the legacy path until the new renderer adds floor
    stacking.
    """
    template = json.loads(template_path.read_text())
    if template.get("floors"):
        # Multi-floor — use the legacy basic renderer (architectural one
        # is single-floor for now; multi-floor stack is a follow-up).
        return _render_one_basic(template, svg_path, size=size, pad=pad)
    try:
        _render_template_svg_arch(template, str(svg_path), size=size)
    except Exception as e:
        print(f"[render_svg] arch renderer failed for {template_path.name}: "
              f"{e!r} — falling back to basic", file=sys.stderr)
        _render_one_basic(template, svg_path, size=size, pad=pad)


def _render_one_basic(template: dict, svg_path: Path, size: int = 1024, pad: int = 40):
    """Legacy basic renderer (rectangles + dots). Kept as fallback."""
    floors = template.get("floors")
    dwg = svgwrite.Drawing(str(svg_path), size=(size, size),
                            viewBox=f"0 0 {size} {size}")
    dwg.add(dwg.rect((0, 0), (size, size), fill="#FAFAFA"))

    if floors:
        n = len(floors)
        panel_h = (size - 2 * pad) / n
        boundary_default = template["boundary"]["polygon"]
        for i, fl in enumerate(floors):
            poly = fl.get("boundary_polygon", boundary_default)
            _render_floor_panel(
                dwg, poly, fl["rooms"], fl.get("doors", []),
                fl.get("windows", []),
                panel_x=pad,
                panel_y=pad + i * panel_h,
                panel_w=size - 2 * pad,
                panel_h=panel_h - 6,
                label=fl["name"],
            )
    else:
        _render_floor_panel(
            dwg, template["boundary"]["polygon"], template["rooms"],
            template["doors"], template["windows"],
            panel_x=pad, panel_y=pad,
            panel_w=size - 2 * pad, panel_h=size - 2 * pad,
        )

    dwg.add(dwg.text(template["id"], insert=(pad, 24),
                      font_size=12, font_family="Arial, sans-serif",
                      fill="#444"))
    dwg.save()


def render_all() -> int:
    SVG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for t in sorted(TEMPLATES_DIR.glob("*/*.json")):
        out = SVG_OUT_DIR / (t.stem + ".svg")
        render_one(t, out)
        print(f"  {t.stem}.svg")
        n += 1
    print(f"OK: rendered {n} SVGs to {SVG_OUT_DIR.relative_to(REPO_ROOT)}")
    return 0


def main(argv):
    if len(argv) == 1:
        return render_all()
    if len(argv) == 3:
        render_one(Path(argv[1]), Path(argv[2]))
        print(f"OK: wrote {argv[2]}")
        return 0
    print("usage: render_svg.py [<template.json> <out.svg>]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
