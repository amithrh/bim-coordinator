"""Architectural-grade 2D floor plan renderer.

Produces SVGs that look like real architectural plans, not colored
rectangles. Adds:

  * **Wall thickness** — solid black fill, computed from the boundary
    minus the room interiors (shapely-based negative-space).
  * **Door swings** — quarter-circle arc + leaf line, oriented based on
    which wall the door sits on.
  * **Window glyphs** — wall opening + 2 parallel lines representing glass.
  * **Room fixtures** — toilet, tub, shower, vanity (bathroom); sink,
    island, stove, fridge (kitchen); bed + nightstands (bedroom);
    sofa + coffee table (living); table + chairs (dining); desk (office).
  * **Dimension labels** — room name + dimensions in metres + area.
  * **Pastel palette** — soft warm tones similar to architectural prints.

Public API:
    render_template_svg(template, out_path, size=1024) -> None
    render_template_svg_string(template, size=1024) -> str
"""

from __future__ import annotations

import math
from typing import Any

import svgwrite
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.ops import unary_union


# ---------------------------------------------------------------------------
# Style tokens
# ---------------------------------------------------------------------------

COLOR_BY_TYPE = {
    "kitchen": "#FFE9A8", "kochnische": "#FFE9A8", "kueche": "#FFE9A8",
    "living": "#FFEFC2",
    "bedroom": "#E0D2F2", "master_bedroom": "#D2C0EB",
    "bathroom": "#CCE6F0", "wc": "#CCE6F0", "bad": "#CCE6F0",
    "balcony": "#CFE9C2", "loggia": "#CFE9C2", "terrace": "#CFE9C2",
    "pooja": "#FFD79E",
    "utility": "#E0E0E0", "abstellraum": "#E0E0E0",
    "corridor": "#F0EFE6", "diele": "#F0EFE6", "entry": "#F0EFE6",
    "flur": "#F0EFE6",
    "dining": "#FFD7BB",
    "study": "#C8DAF7", "office": "#C8DAF7",
    "store": "#DCDCDC", "wardrobe": "#DCDCDC", "walk_in_closet": "#E0D2F2",
}

WALL_FILL = "#1c1c1c"
DIM_TEXT  = "#3a3a3a"
NAME_TEXT = "#1c1c1c"
PAPER_BG  = "#FAFAF6"

# Visual wall thickness — independent of BIM. Real arch plans render
# walls thinner than the actual structural thickness for legibility.
VISUAL_WALL_M = 0.10


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _shapely(polygon):
    return Polygon(polygon)


def _wall_band(boundary_pts: list, room_polys: list[Polygon],
               wall_thickness_m: float) -> Any:
    """Return a shapely geometry of the wall MASS (the band between rooms
    and between the rooms and the exterior), as if you'd cut floor plan
    drawings from a Swiss-cheese sheet."""
    boundary = _shapely(boundary_pts)
    if not room_polys:
        return boundary.difference(boundary.buffer(-wall_thickness_m))
    # Each room's interior = polygon shrunk by wall_thickness/2. The wall
    # is exterior boundary minus the union of those interiors.
    inset = wall_thickness_m / 2
    interiors = unary_union([
        p.buffer(-inset) for p in room_polys if p.is_valid and not p.is_empty
    ])
    walls = boundary.difference(interiors)
    return walls


def _polygon_to_svg_pts(geom, tx) -> list[tuple[float, float]]:
    """Flatten a shapely Polygon/MultiPolygon to a list of ring-paths."""
    rings = []
    if geom.is_empty:
        return rings
    if isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            rings.extend(_polygon_to_svg_pts(g, tx))
        return rings
    rings.append([tx(c) for c in geom.exterior.coords])
    for hole in geom.interiors:
        rings.append([tx(c) for c in hole.coords])
    return rings


def _polygon_to_svg_path_d(geom, tx) -> str:
    """Build an SVG path 'd' string with M ... Z for each ring of a
    Polygon/MultiPolygon. When rendered with fill-rule='evenodd', holes
    are correctly subtracted (so a donut shape draws as a donut, not as
    a filled disc + filled hole)."""
    parts = []

    def add_poly(poly):
        ext = list(poly.exterior.coords)
        if len(ext) >= 3:
            seg = "M " + " L ".join(
                f"{tx(c)[0]:.2f},{tx(c)[1]:.2f}" for c in ext) + " Z"
            parts.append(seg)
        for hole in poly.interiors:
            h = list(hole.coords)
            if len(h) >= 3:
                seg = "M " + " L ".join(
                    f"{tx(c)[0]:.2f},{tx(c)[1]:.2f}" for c in h) + " Z"
                parts.append(seg)

    if isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            add_poly(p)
    elif isinstance(geom, Polygon):
        add_poly(geom)
    return " ".join(parts)


def _rotate(point, origin, angle_rad):
    ox, oy = origin
    px, py = point
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return (ox + (px - ox) * c - (py - oy) * s,
            oy + (px - ox) * s + (py - oy) * c)


def _wall_for_position(room_poly: list, pt: tuple[float, float],
                        tol: float = 0.25) -> tuple[tuple, tuple] | None:
    """Find the room edge a door/window position sits on. Returns the
    edge endpoints in BIM coordinates (or None if not found)."""
    n = len(room_poly)
    best = None
    best_d = tol
    for i in range(n):
        a = room_poly[i]
        b = room_poly[(i + 1) % n]
        # distance from pt to segment AB
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-6:
            continue
        t = max(0.0, min(1.0, ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / L2))
        cx, cy = ax + t * dx, ay + t * dy
        d = math.hypot(pt[0] - cx, pt[1] - cy)
        if d < best_d:
            best_d = d
            best = (a, b, t)
    if best is None:
        return None
    return best  # (a, b, t)


def _normal_pointing_into_room(a, b, room_poly):
    """Return a unit normal of edge AB that points INTO the room polygon."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return (0.0, 1.0)
    # Two candidate normals
    nA = (-dy / L,  dx / L)
    nB = ( dy / L, -dx / L)
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    eps = 0.05
    pA = (mid[0] + nA[0] * eps, mid[1] + nA[1] * eps)
    pB = (mid[0] + nB[0] * eps, mid[1] + nB[1] * eps)
    poly = Polygon(room_poly)
    if poly.contains(Point(pA)):
        return nA
    if poly.contains(Point(pB)):
        return nB
    return nA  # fallback


# ---------------------------------------------------------------------------
# Door / window glyphs
# ---------------------------------------------------------------------------

def _draw_door(dwg, door, rooms_by_id: dict, tx, scale,
                wall_thickness_m: float):
    """Draw a door glyph: gap in the wall + leaf line + 90° swing arc."""
    pos = door.get("position")
    if not pos:
        return
    width_m = door.get("width_mm", 900) / 1000.0
    # Find which room polygon edge this door sits on. Prefer the 'to' room
    # so the swing arc opens INTO the destination room (typical convention).
    target_room_id = door.get("to") or door.get("from")
    target_room = rooms_by_id.get(target_room_id)
    edge = None
    if target_room and target_room.get("polygon"):
        edge = _wall_for_position(target_room["polygon"], tuple(pos))
    if edge is None:
        # Try every room — door may be on the boundary
        for r in rooms_by_id.values():
            if r.get("polygon"):
                edge = _wall_for_position(r["polygon"], tuple(pos))
                if edge:
                    target_room = r
                    break
    if edge is None:
        return
    a, b, _t = edge

    # Direction vector along wall + inward normal
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    nx, ny = _normal_pointing_into_room(a, b, target_room["polygon"])

    # Door frame endpoints — width_m centred on `pos`
    half = width_m / 2
    p1 = (pos[0] - ux * half, pos[1] - uy * half)
    p2 = (pos[0] + ux * half, pos[1] + uy * half)

    # 1) Erase wall: white rectangle covering wall thickness * door width
    wt = wall_thickness_m
    rect_pts = [
        (p1[0] - nx * wt / 2, p1[1] - ny * wt / 2),
        (p2[0] - nx * wt / 2, p2[1] - ny * wt / 2),
        (p2[0] + nx * wt / 2, p2[1] + ny * wt / 2),
        (p1[0] + nx * wt / 2, p1[1] + ny * wt / 2),
    ]
    dwg.add(dwg.polygon([tx(p) for p in rect_pts], fill="white",
                         stroke="none"))

    # 2) Leaf line (the door panel itself, opened ~90°)
    hinge = p1
    # Leaf endpoint = hinge + width along the inward normal
    leaf_end = (hinge[0] + nx * width_m, hinge[1] + ny * width_m)
    dwg.add(dwg.line(tx(hinge), tx(leaf_end),
                      stroke=WALL_FILL, stroke_width=1.6))

    # 3) Swing arc: 90° from p2 (closed) to leaf_end (open)
    cx_s, cy_s = tx(hinge)
    r_s = width_m * scale
    closed = tx(p2)
    open_end = tx(leaf_end)
    # Determine sweep direction so the arc is on the inward side of the wall
    # (arc bulges toward the room interior). With SVG path A command,
    # large-arc-flag=0, sweep-flag depends on cross-product sign.
    cross = ux * ny - uy * nx
    sweep = 1 if cross > 0 else 0
    path = (f"M {closed[0]:.2f},{closed[1]:.2f} "
            f"A {r_s:.2f},{r_s:.2f} 0 0 {sweep} {open_end[0]:.2f},{open_end[1]:.2f}")
    dwg.add(dwg.path(d=path, fill="none", stroke="#666",
                      stroke_width=0.8, stroke_dasharray="3,2"))


def _draw_window(dwg, window, rooms_by_id: dict, tx, scale,
                  wall_thickness_m: float):
    """Wall opening + 2 parallel lines representing glass."""
    pos = window.get("position")
    if not pos:
        return
    width_m = window.get("width_mm", 1000) / 1000.0
    room = rooms_by_id.get(window.get("room"))
    if not room or not room.get("polygon"):
        return
    edge = _wall_for_position(room["polygon"], tuple(pos))
    if edge is None:
        return
    a, b, _ = edge
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    nx, ny = _normal_pointing_into_room(a, b, room["polygon"])
    half = width_m / 2
    p1 = (pos[0] - ux * half, pos[1] - uy * half)
    p2 = (pos[0] + ux * half, pos[1] + uy * half)
    wt = wall_thickness_m

    # Erase wall
    rect_pts = [
        (p1[0] - nx * wt / 2, p1[1] - ny * wt / 2),
        (p2[0] - nx * wt / 2, p2[1] - ny * wt / 2),
        (p2[0] + nx * wt / 2, p2[1] + ny * wt / 2),
        (p1[0] + nx * wt / 2, p1[1] + ny * wt / 2),
    ]
    dwg.add(dwg.polygon([tx(p) for p in rect_pts], fill="white",
                         stroke=WALL_FILL, stroke_width=0.6))

    # Two parallel glass lines, offset 1/3 and 2/3 of wall thickness
    for frac in (-1/6, 1/6):
        s = (p1[0] + nx * wt * frac, p1[1] + ny * wt * frac)
        e = (p2[0] + nx * wt * frac, p2[1] + ny * wt * frac)
        dwg.add(dwg.line(tx(s), tx(e), stroke="#3a6ea8",
                          stroke_width=0.8))


# ---------------------------------------------------------------------------
# Furniture / fixture symbols
# ---------------------------------------------------------------------------

def _add_rect(dwg, tx, x, y, w, h, scale, fill="white",
                stroke="#555", sw=0.7, rx=0):
    p1 = tx((x, y))
    p2 = tx((x + w, y + h))
    rx_px = (p2[0] - p1[0]) * (rx / max(w, 1e-3)) if rx else 0
    dwg.add(dwg.rect((min(p1[0], p2[0]), min(p1[1], p2[1])),
                      (abs(p2[0] - p1[0]), abs(p2[1] - p1[1])),
                      fill=fill, stroke=stroke, stroke_width=sw,
                      rx=rx_px, ry=rx_px))


def _add_circle(dwg, tx, cx, cy, r_m, scale, fill="white", stroke="#555", sw=0.7):
    c = tx((cx, cy))
    dwg.add(dwg.circle(c, r_m * scale, fill=fill, stroke=stroke, stroke_width=sw))


def _add_text(dwg, tx, x, y, text, font_size=10, fill="#666",
                anchor="middle", weight="normal", family="Arial"):
    p = tx((x, y))
    dwg.add(dwg.text(text, insert=p, font_size=font_size, fill=fill,
                      text_anchor=anchor, font_family=f"{family}, sans-serif",
                      font_weight=weight))


def _draw_bathroom(dwg, room, tx, scale):
    """Toilet, bathtub or shower, vanity. Place along the longest wall."""
    poly = room["polygon"]
    minx = min(p[0] for p in poly); maxx = max(p[0] for p in poly)
    miny = min(p[1] for p in poly); maxy = max(p[1] for p in poly)
    w = maxx - minx; h = maxy - miny
    # Bath/shower along long wall
    bath_w, bath_h = (1.7, 0.75)
    if h > w:
        bath_w, bath_h = bath_h, bath_w
    bath_x = minx + 0.1
    bath_y = miny + 0.1
    _add_rect(dwg, tx, bath_x, bath_y, bath_w, bath_h, scale,
                fill="#F4F8FA", stroke="#5b6f86", sw=1.0, rx=0.15)
    # Inner basin
    _add_rect(dwg, tx, bath_x + 0.1, bath_y + 0.1, bath_w - 0.2, bath_h - 0.2,
                scale, fill="white", stroke="#9aabbf", sw=0.6, rx=0.12)
    # Toilet — opposite corner
    if h > w:
        toilet_cx = minx + w / 2
        toilet_cy = maxy - 0.4
    else:
        toilet_cx = maxx - 0.4
        toilet_cy = miny + h / 2
    # Toilet tank (rect) + bowl (oval)
    _add_rect(dwg, tx, toilet_cx - 0.2, toilet_cy - 0.15, 0.4, 0.18,
                scale, fill="white", stroke="#5b6f86", sw=0.8)
    p1 = tx((toilet_cx, toilet_cy + 0.1))
    rx_px = 0.18 * scale; ry_px = 0.22 * scale
    dwg.add(dwg.ellipse(center=p1, r=(rx_px, ry_px),
                          fill="white", stroke="#5b6f86", stroke_width=0.8))
    # Sink/vanity — third corner
    sink_x = maxx - 0.7 if h > w else minx + 0.1
    sink_y = miny + 0.1 if h > w else maxy - 0.7
    _add_rect(dwg, tx, sink_x, sink_y, 0.6, 0.45, scale,
                fill="#F4F8FA", stroke="#5b6f86", sw=0.8, rx=0.05)
    sx_c = sink_x + 0.3
    sy_c = sink_y + 0.22
    _add_circle(dwg, tx, sx_c, sy_c, 0.15, scale,
                  fill="white", stroke="#9aabbf", sw=0.5)


def _draw_kitchen(dwg, room, tx, scale):
    """Counter run with sink + stove along longest wall, optional island."""
    poly = room["polygon"]
    minx = min(p[0] for p in poly); maxx = max(p[0] for p in poly)
    miny = min(p[1] for p in poly); maxy = max(p[1] for p in poly)
    w = maxx - minx; h = maxy - miny
    counter_d = 0.6
    if w >= h:
        # Counter along bottom wall
        cx0, cy0, cw, ch = minx + 0.1, miny + 0.05, w - 0.2, counter_d
    else:
        cx0, cy0, cw, ch = minx + 0.05, miny + 0.1, counter_d, h - 0.2
    _add_rect(dwg, tx, cx0, cy0, cw, ch, scale,
                fill="#F2EFE5", stroke="#7a6b53", sw=0.9)
    # Sink
    if w >= h:
        sx, sy, sw_, sh = cx0 + cw * 0.35, cy0 + 0.1, 0.7, 0.4
    else:
        sx, sy, sw_, sh = cx0 + 0.1, cy0 + ch * 0.35, 0.4, 0.7
    _add_rect(dwg, tx, sx, sy, sw_, sh, scale,
                fill="white", stroke="#7a6b53", sw=0.7, rx=0.05)
    # Faucet line
    if w >= h:
        _add_rect(dwg, tx, sx + sw_ / 2 - 0.04, sy - 0.06, 0.08, 0.06,
                    scale, fill="#9c8b6e", stroke="none", sw=0)
    # Stove (4 burners) — at one end
    if w >= h:
        stx, sty = cx0 + cw * 0.05, cy0 + 0.05
        sw_s, sh_s = 0.6, 0.5
    else:
        stx, sty = cx0 + 0.05, cy0 + ch * 0.05
        sw_s, sh_s = 0.5, 0.6
    _add_rect(dwg, tx, stx, sty, sw_s, sh_s, scale,
                fill="white", stroke="#7a6b53", sw=0.7)
    # 4 burners
    burner_r = 0.07
    bx = stx + sw_s / 4
    by = sty + sh_s / 4
    for ix in (0, 1):
        for iy in (0, 1):
            _add_circle(dwg, tx,
                          bx + ix * sw_s / 2,
                          by + iy * sh_s / 2,
                          burner_r, scale,
                          fill="#F4EFE2", stroke="#7a6b53", sw=0.5)
    # Refrigerator — opposite end
    if w >= h:
        fx, fy = cx0 + cw - 0.7, cy0 + 0.05
        fw, fh = 0.7, counter_d - 0.1
    else:
        fx, fy = cx0 + 0.05, cy0 + ch - 0.7
        fw, fh = counter_d - 0.1, 0.7
    _add_rect(dwg, tx, fx, fy, fw, fh, scale,
                fill="white", stroke="#7a6b53", sw=0.9)
    # Vertical line on fridge for door
    if w >= h:
        _add_rect(dwg, tx, fx + fw - 0.04, fy + 0.02, 0.04, fh - 0.04,
                    scale, fill="#9c8b6e", stroke="none")
    else:
        _add_rect(dwg, tx, fx + 0.02, fy + fh - 0.04, fw - 0.04, 0.04,
                    scale, fill="#9c8b6e", stroke="none")


def _draw_bedroom(dwg, room, tx, scale, master: bool = False):
    """Bed + 2 nightstands centred on long wall."""
    poly = room["polygon"]
    minx = min(p[0] for p in poly); maxx = max(p[0] for p in poly)
    miny = min(p[1] for p in poly); maxy = max(p[1] for p in poly)
    w = maxx - minx; h = maxy - miny
    bed_w = 1.6 if master else 1.4
    bed_d = 2.0
    if w >= h:
        # Bed along left wall
        bx, by = minx + 0.2, miny + (h - bed_d) / 2
        bw, bh = bed_d, bed_w
        head_x, head_y = bx, by
    else:
        # Bed along bottom wall
        bx, by = minx + (w - bed_w) / 2, miny + 0.2
        bw, bh = bed_w, bed_d
        head_x, head_y = bx, by + bh - 0.3
    # Mattress
    _add_rect(dwg, tx, bx, by, bw, bh, scale,
                fill="#F8F4EE", stroke="#7a6f5e", sw=0.9, rx=0.1)
    # Headboard strip
    if w >= h:
        _add_rect(dwg, tx, bx, by, 0.2, bh, scale,
                    fill="#7a6f5e", stroke="none")
    else:
        _add_rect(dwg, tx, bx, by + bh - 0.2, bw, 0.2, scale,
                    fill="#7a6f5e", stroke="none")
    # Pillows (2)
    if w >= h:
        _add_rect(dwg, tx, bx + 0.25, by + 0.15, 0.45, bh / 2 - 0.2, scale,
                    fill="white", stroke="#bbb", sw=0.4, rx=0.05)
        _add_rect(dwg, tx, bx + 0.25, by + bh / 2 + 0.05, 0.45, bh / 2 - 0.2, scale,
                    fill="white", stroke="#bbb", sw=0.4, rx=0.05)
    else:
        _add_rect(dwg, tx, bx + 0.15, by + bh - 0.7, bw / 2 - 0.2, 0.45, scale,
                    fill="white", stroke="#bbb", sw=0.4, rx=0.05)
        _add_rect(dwg, tx, bx + bw / 2 + 0.05, by + bh - 0.7, bw / 2 - 0.2, 0.45, scale,
                    fill="white", stroke="#bbb", sw=0.4, rx=0.05)
    # Nightstands flanking the head of the bed
    ns_size = 0.5
    if w >= h:
        for offset in (-ns_size - 0.05, bh + 0.05):
            _add_rect(dwg, tx, head_x, by + offset, ns_size, ns_size, scale,
                        fill="white", stroke="#7a6f5e", sw=0.5, rx=0.05)
    else:
        for offset in (-ns_size - 0.05, bw + 0.05):
            _add_rect(dwg, tx, bx + offset, head_y - ns_size + 0.3, ns_size, ns_size,
                        scale, fill="white", stroke="#7a6f5e", sw=0.5, rx=0.05)


def _draw_living(dwg, room, tx, scale):
    """L-shaped sofa + coffee table + side chair."""
    poly = room["polygon"]
    minx = min(p[0] for p in poly); maxx = max(p[0] for p in poly)
    miny = min(p[1] for p in poly); maxy = max(p[1] for p in poly)
    w = maxx - minx; h = maxy - miny
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    sofa_d = 0.85
    sofa_l = min(2.4, w * 0.55, h * 0.55)
    if w >= h:
        # Sofa centred along bottom wall facing up
        sx, sy = cx - sofa_l / 2, miny + 0.25
        sw, sh = sofa_l, sofa_d
        ct_x = cx - 0.55
        ct_y = sy + sh + 0.25
    else:
        sx, sy = minx + 0.25, cy - sofa_l / 2
        sw, sh = sofa_d, sofa_l
        ct_x = sx + sw + 0.25
        ct_y = cy - 0.35
    # Sofa frame
    _add_rect(dwg, tx, sx, sy, sw, sh, scale,
                fill="#E9E2D6", stroke="#7a6f5e", sw=0.8, rx=0.12)
    # Cushions split
    if w >= h:
        seg = sw / 3
        for i in range(3):
            _add_rect(dwg, tx, sx + i * seg + 0.05, sy + 0.2,
                        seg - 0.1, sh - 0.3, scale,
                        fill="#F4ECDC", stroke="#a89881", sw=0.4, rx=0.06)
    else:
        seg = sh / 3
        for i in range(3):
            _add_rect(dwg, tx, sx + 0.2, sy + i * seg + 0.05,
                        sw - 0.3, seg - 0.1, scale,
                        fill="#F4ECDC", stroke="#a89881", sw=0.4, rx=0.06)
    # Coffee table
    _add_rect(dwg, tx, ct_x, ct_y, 1.1, 0.7, scale,
                fill="white", stroke="#7a6f5e", sw=0.7, rx=0.06)


def _draw_dining(dwg, room, tx, scale):
    """Round dining table with 4 chairs."""
    poly = room["polygon"]
    minx = min(p[0] for p in poly); maxx = max(p[0] for p in poly)
    miny = min(p[1] for p in poly); maxy = max(p[1] for p in poly)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    _add_circle(dwg, tx, cx, cy, 0.6, scale,
                  fill="#F4ECDC", stroke="#7a6f5e", sw=0.8)
    for ang_deg in (0, 90, 180, 270):
        ang = math.radians(ang_deg)
        chx = cx + math.cos(ang) * 1.0
        chy = cy + math.sin(ang) * 1.0
        _add_rect(dwg, tx, chx - 0.18, chy - 0.18, 0.36, 0.36, scale,
                    fill="white", stroke="#7a6f5e", sw=0.5, rx=0.05)


def _draw_office(dwg, room, tx, scale):
    """Desk + chair against a wall."""
    poly = room["polygon"]
    minx = min(p[0] for p in poly); maxx = max(p[0] for p in poly)
    miny = min(p[1] for p in poly); maxy = max(p[1] for p in poly)
    w = maxx - minx; h = maxy - miny
    if w >= h:
        dx, dy, dw, dh = minx + 0.3, miny + 0.2, 1.6, 0.6
    else:
        dx, dy, dw, dh = minx + 0.2, miny + 0.3, 0.6, 1.6
    _add_rect(dwg, tx, dx, dy, dw, dh, scale,
                fill="#F4ECDC", stroke="#7a6f5e", sw=0.7, rx=0.04)
    # Chair
    if w >= h:
        _add_circle(dwg, tx, dx + dw / 2, dy + dh + 0.4, 0.25, scale,
                      fill="white", stroke="#7a6f5e", sw=0.5)
    else:
        _add_circle(dwg, tx, dx + dw + 0.4, dy + dh / 2, 0.25, scale,
                      fill="white", stroke="#7a6f5e", sw=0.5)


FIXTURE_DRAWERS = {
    "bathroom":       _draw_bathroom,
    "wc":             _draw_bathroom,
    "bad":            _draw_bathroom,
    "kitchen":        _draw_kitchen,
    "kueche":         _draw_kitchen,
    "kochnische":     _draw_kitchen,
    "bedroom":        lambda d, r, t, s: _draw_bedroom(d, r, t, s, master=False),
    "master_bedroom": lambda d, r, t, s: _draw_bedroom(d, r, t, s, master=True),
    "living":         _draw_living,
    "dining":         _draw_dining,
    "study":          _draw_office,
    "office":         _draw_office,
}


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def _render_floor(dwg, boundary_pts, rooms, doors, windows,
                   panel_x, panel_y, panel_w, panel_h,
                   wall_thickness_m: float, label: str | None = None):
    """Render one floor (or single-floor template) into the SVG drawing."""
    _bx, _by, bw, bh = _bbox(boundary_pts)
    pad = 24
    scale = min((panel_w - 2 * pad) / bw, (panel_h - 2 * pad) / bh)
    cw = bw * scale
    ch = bh * scale
    ox = panel_x + (panel_w - cw) / 2 - _bx * scale
    oy = panel_y + panel_h - (panel_h - ch) / 2 + _by * scale

    def tx(p):
        return (ox + p[0] * scale, oy - p[1] * scale)

    rooms_by_id = {r.get("id"): r for r in rooms if r.get("id")}
    room_polys = [_shapely(r["polygon"]) for r in rooms if r.get("polygon")]

    # 1) Boundary wall band (negative-space mass)
    walls = _wall_band(boundary_pts, room_polys, wall_thickness_m)

    # 2) Room fills (inset slightly so they don't clash with walls)
    inset = wall_thickness_m / 2
    for r in rooms:
        if not r.get("polygon"):
            continue
        rp = _shapely(r["polygon"]).buffer(-inset)
        if rp.is_empty:
            continue
        color = COLOR_BY_TYPE.get(r["type"], "#F0F0F0")
        d = _polygon_to_svg_path_d(rp, tx)
        if d:
            p = dwg.path(d=d, fill=color, stroke="none")
            p["fill-rule"] = "evenodd"  # svgwrite drops fill_rule kwarg silently
            dwg.add(p)

    # 3) Wall mass (drawn AFTER fills so walls cover edges). Use one
    # path with fill-rule='evenodd' so holes (rooms) are correctly
    # subtracted from the boundary instead of being re-filled black.
    walls_d = _polygon_to_svg_path_d(walls, tx)
    if walls_d:
        p = dwg.path(d=walls_d, fill=WALL_FILL, stroke="none")
        p["fill-rule"] = "evenodd"
        dwg.add(p)

    # 4) Doors and windows — these "cut" the walls (white over them)
    for d in doors:
        _draw_door(dwg, d, rooms_by_id, tx, scale, wall_thickness_m)
    for w in windows:
        _draw_window(dwg, w, rooms_by_id, tx, scale, wall_thickness_m)

    # 5) Furniture / fixtures
    for r in rooms:
        if not r.get("polygon"):
            continue
        drawer = FIXTURE_DRAWERS.get(r["type"])
        if drawer:
            try:
                drawer(dwg, r, tx, scale)
            except Exception:
                pass  # never let furniture break the plan

    # 6) Labels — name (large), dimensions (small). Position toward the
    # upper third of the room (above furniture which sits centre/lower).
    for r in rooms:
        if not r.get("polygon"):
            continue
        rpts = r["polygon"]
        cx_b = sum(p[0] for p in rpts) / len(rpts)
        cy_b = sum(p[1] for p in rpts) / len(rpts)
        minx = min(p[0] for p in rpts); maxx = max(p[0] for p in rpts)
        miny = min(p[1] for p in rpts); maxy = max(p[1] for p in rpts)
        w_m = maxx - minx; h_m = maxy - miny
        # Place label in the upper third of the room (BIM coords: higher y)
        label_y = miny + h_m * 0.78
        # Font size scales with room area but bounded for legibility
        name_sz = max(13, min(22, int(math.sqrt(w_m * h_m) * 5.2)))
        dim_sz  = max(9, min(13, int(math.sqrt(w_m * h_m) * 3.0)))
        _add_text(dwg, tx, cx_b, label_y, r["name"],
                    font_size=name_sz,
                    fill=NAME_TEXT, anchor="middle", weight="bold",
                    family="Georgia")
        dim_text = f"{w_m:g} × {h_m:g} m"
        if r.get("area_sqm"):
            dim_text += f"  ·  {r['area_sqm']:g} m²"
        # Offset dim text below the name (BIM y decreases for "below")
        _add_text(dwg, tx, cx_b, label_y - 0.32, dim_text,
                    font_size=dim_sz, fill=DIM_TEXT, anchor="middle",
                    weight="normal", family="Arial")

    # Floor label
    if label:
        dwg.add(dwg.text(label, insert=(panel_x + 12, panel_y + 22),
                          font_size=13, font_family="Georgia, serif",
                          fill="#1f4ed8", font_weight="bold"))


def render_template_svg(template: dict, out_path, size: int = 1024) -> None:
    """Render a template to an SVG file at out_path."""
    out_path = str(out_path)
    pad = 32

    floors = template.get("floors")
    boundary = template.get("boundary", {})
    # Visual wall thickness fixed for legibility (architectural plans
    # render walls thinner than actual structural thickness).
    wall_thickness_m = VISUAL_WALL_M

    dwg = svgwrite.Drawing(out_path, size=(size, size),
                            viewBox=f"0 0 {size} {size}")
    dwg.add(dwg.rect((0, 0), (size, size), fill=PAPER_BG))

    if floors:
        n = len(floors)
        panel_h = (size - 2 * pad) / n
        boundary_default = boundary.get("polygon", [])
        for i, fl in enumerate(floors):
            poly = fl.get("boundary_polygon", boundary_default)
            _render_floor(
                dwg, poly, fl.get("rooms", []), fl.get("doors", []),
                fl.get("windows", []),
                panel_x=pad,
                panel_y=pad + i * panel_h,
                panel_w=size - 2 * pad,
                panel_h=panel_h - 6,
                wall_thickness_m=wall_thickness_m,
                label=fl.get("name"),
            )
    else:
        _render_floor(
            dwg, boundary.get("polygon", []),
            template.get("rooms", []),
            template.get("doors", []),
            template.get("windows", []),
            panel_x=pad, panel_y=pad,
            panel_w=size - 2 * pad, panel_h=size - 2 * pad,
            wall_thickness_m=wall_thickness_m,
        )

    # Title block (template id + total area)
    md = template.get("metadata", {})
    title = template.get("id", "")
    subtitle_parts = []
    if md.get("country"):    subtitle_parts.append(md["country"])
    if md.get("city_inspiration"): subtitle_parts.append(md["city_inspiration"])
    if md.get("total_area_sqm"): subtitle_parts.append(f"{md['total_area_sqm']:g} m²")
    if md.get("bedrooms") is not None:
        subtitle_parts.append(f"{md['bedrooms']} BR")
    subtitle = "  ·  ".join(subtitle_parts)

    dwg.add(dwg.text(title, insert=(pad, 22),
                      font_size=14, font_family="Georgia, serif",
                      fill="#222", font_weight="bold"))
    if subtitle:
        dwg.add(dwg.text(subtitle, insert=(pad, 38),
                          font_size=11, font_family="Arial, sans-serif",
                          fill="#666"))
    dwg.save()


def render_template_svg_string(template: dict, size: int = 1024) -> str:
    """Render to an SVG string (for in-memory pipelines)."""
    import io
    buf = io.StringIO()
    # svgwrite always wants a path; dump to a temp Drawing then read it.
    import tempfile, os
    with tempfile.NamedTemporaryFile("r+", suffix=".svg", delete=False) as f:
        path = f.name
    try:
        render_template_svg(template, path, size=size)
        with open(path) as fh:
            return fh.read()
    finally:
        try: os.unlink(path)
        except OSError: pass
