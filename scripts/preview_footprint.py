"""Quick 2D footprint preview from template JSON. CHECKPOINT-only tool —
the production renderer (render_svg.py) follows the same JSON-direct path
per the build plan §6.3 fallback."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import svgwrite

COLOR_BY_TYPE = {
    "kitchen": "#FFF3CD", "living": "#D1ECF1", "bedroom": "#E2D5F0",
    "master_bedroom": "#D5C8E0", "bathroom": "#D4EDDA", "wc": "#D4EDDA",
    "balcony": "#F8F9FA", "pooja": "#FFE5B4", "utility": "#E0E0E0",
    "corridor": "#F5F5F5", "dining": "#FFE5E5", "study": "#E5F0FF",
    "kochnische": "#FFF3CD", "diele": "#F5F5F5",
}


def bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}


def render(template_path: Path, svg_path: Path, size: int = 1024, pad: int = 40):
    template = json.loads(template_path.read_text())
    boundary = template["boundary"]["polygon"]
    bb = bbox(boundary)
    scale = (size - 2 * pad) / max(bb["w"], bb["h"])

    def tx(p):
        # Flip Y so south is bottom
        return (pad + (p[0] - bb["x"]) * scale,
                size - pad - (p[1] - bb["y"]) * scale)

    dwg = svgwrite.Drawing(str(svg_path), size=(size, size))
    dwg.add(dwg.rect((0, 0), (size, size), fill="#FAFAFA"))

    # Boundary background
    points = [tx(p) for p in boundary]
    dwg.add(dwg.polygon(points, fill="white", stroke="#222", stroke_width=4))

    # Rooms
    for room in template["rooms"]:
        rpts = [tx(p) for p in room["polygon"]]
        color = COLOR_BY_TYPE.get(room["type"], "#EFEFEF")
        dwg.add(dwg.polygon(rpts, fill=color, stroke="#888", stroke_width=1))
        # Centroid label
        cx = sum(p[0] for p in rpts) / len(rpts)
        cy = sum(p[1] for p in rpts) / len(rpts)
        dwg.add(dwg.text(room["name"], insert=(cx, cy),
                          font_size=14, text_anchor="middle",
                          font_family="Arial", fill="#333"))
        if "area_sqm" in room:
            dwg.add(dwg.text(f"{room['area_sqm']:g} m²",
                              insert=(cx, cy + 18),
                              font_size=11, text_anchor="middle",
                              font_family="Arial", fill="#666"))

    # Doors as red squares with label
    for d in template["doors"]:
        x, y = tx(d["position"])
        dwg.add(dwg.rect((x - 5, y - 5), (10, 10), fill="#E66",
                          stroke="#A00", stroke_width=1))

    # Windows as blue lines
    for w in template["windows"]:
        x, y = tx(w["position"])
        dwg.add(dwg.rect((x - 6, y - 3), (12, 6), fill="#69E",
                          stroke="#04A", stroke_width=1))

    # Legend
    dwg.add(dwg.text(template["id"], insert=(pad, pad - 10),
                      font_size=12, font_family="Arial", fill="#444"))

    dwg.save()


def main(argv):
    if len(argv) != 3:
        print("usage: preview_footprint.py <template.json> <output.svg>", file=sys.stderr)
        return 2
    render(Path(argv[1]), Path(argv[2]))
    print(f"OK: wrote {argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
