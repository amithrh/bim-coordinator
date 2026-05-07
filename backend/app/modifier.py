"""Apply Tier 1 modifications to a template, with a hard validation gate.

Returns (modified_template_dict, errors). If errors is non-empty, the
caller MUST reject the modification — the original template is unchanged.
"""
from __future__ import annotations

import math
import sys
from copy import deepcopy
from pathlib import Path

# scripts is a sibling of backend; add it to sys.path so we can reuse
# the validator without packaging.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from validate_template import validate_dict  # noqa: E402


def _scale_polygon(polygon: list[list[float]], s: float) -> list[list[float]]:
    return [[round(x * s, 4), round(y * s, 4)] for x, y in polygon]


def _rotate_point(p: list[float], angle_deg: float) -> list[float]:
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [round(p[0] * ca - p[1] * sa, 4),
            round(p[0] * sa + p[1] * ca, 4)]


def _rotate_polygon(polygon: list[list[float]], angle_deg: float) -> list[list[float]]:
    return [_rotate_point(p, angle_deg) for p in polygon]


def _normalize_to_positive(template: dict) -> None:
    """After a rotation, shift the entire layout so all coordinates are >= 0."""
    points: list[list[float]] = []
    points.extend(template["boundary"]["polygon"])
    for r in template["rooms"]:
        points.extend(r["polygon"])
    points.extend(d["position"] for d in template["doors"])
    points.extend(w["position"] for w in template["windows"])
    if not points:
        return
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    if min_x >= 0 and min_y >= 0:
        return
    dx, dy = -min(0.0, min_x), -min(0.0, min_y)
    template["boundary"]["polygon"] = [
        [round(x + dx, 4), round(y + dy, 4)] for x, y in template["boundary"]["polygon"]
    ]
    for r in template["rooms"]:
        r["polygon"] = [[round(x + dx, 4), round(y + dy, 4)] for x, y in r["polygon"]]
    for d in template["doors"]:
        d["position"] = [round(d["position"][0] + dx, 4),
                          round(d["position"][1] + dy, 4)]
    for w in template["windows"]:
        w["position"] = [round(w["position"][0] + dx, 4),
                          round(w["position"][1] + dy, 4)]


def apply_modifications(template: dict, mods: dict) -> tuple[dict | None, list[str]]:
    """Returns (modified_template, errors). If errors, request rejected and
    None is returned for the modified_template."""
    out = deepcopy(template)
    out["id"] = template["id"] + "_modified"

    # 1. Area scale (uniform XY scaling)
    if mods.get("area_scale"):
        s = math.sqrt(float(mods["area_scale"]))
        out["boundary"]["polygon"] = _scale_polygon(out["boundary"]["polygon"], s)
        for r in out["rooms"]:
            r["polygon"] = _scale_polygon(r["polygon"], s)
            if "area_sqm" in r:
                r["area_sqm"] = round(r["area_sqm"] * float(mods["area_scale"]), 2)
        for d in out["doors"]:
            d["position"] = [round(c * s, 4) for c in d["position"]]
        for w in out["windows"]:
            w["position"] = [round(c * s, 4) for c in w["position"]]
        out["metadata"]["total_area_sqm"] = round(
            out["metadata"]["total_area_sqm"] * float(mods["area_scale"]), 1
        )

    # 2. Ceiling height
    if mods.get("ceiling_height_mm"):
        out["boundary"]["ceiling_height_mm"] = int(mods["ceiling_height_mm"])

    # 3. Rotation in 0/90/180/270 degrees
    if mods.get("rotation_deg"):
        deg = float(mods["rotation_deg"]) % 360
        if deg:
            out["boundary"]["polygon"] = _rotate_polygon(
                out["boundary"]["polygon"], deg
            )
            for r in out["rooms"]:
                r["polygon"] = _rotate_polygon(r["polygon"], deg)
            for d in out["doors"]:
                d["position"] = _rotate_point(d["position"], deg)
            for w in out["windows"]:
                w["position"] = _rotate_point(w["position"], deg)
            _normalize_to_positive(out)

    # 4. Swap rooms — swap names + types of two rooms (polygons stay).
    # mods["swap_rooms"]: {"a": "<room_id>", "b": "<room_id>"}
    swap = mods.get("swap_rooms")
    if swap:
        a_id, b_id = swap.get("a"), swap.get("b")
        room_a = next((r for r in out["rooms"] if r["id"] == a_id), None)
        room_b = next((r for r in out["rooms"] if r["id"] == b_id), None)
        if not room_a or not room_b:
            return None, [f"swap_rooms: unknown room id (a={a_id} b={b_id})"]
        # Swap name + type, keep polygons + ids
        room_a["name"], room_b["name"] = room_b["name"], room_a["name"]
        room_a["type"], room_b["type"] = room_b["type"], room_a["type"]

    # 5. Add balcony — carve a strip from an exterior wall of a chosen room.
    # mods["add_balcony"]: {"from_room": "<room_id>", "depth_m": 1.2}
    add_bal = mods.get("add_balcony")
    if add_bal:
        out, err = _add_balcony(out, add_bal)
        if err:
            return None, [f"add_balcony: {err}"]

    # HARD VALIDATION GATE — the build plan §7.2 contract.
    errors = validate_dict(out)
    if errors:
        return None, errors
    return out, []


# --------------------------------------------------------------------------- #
# Add-balcony helper
# --------------------------------------------------------------------------- #

def _add_balcony(template: dict, opts: dict) -> tuple[dict, str]:
    """Carve a balcony strip off a room's exterior edge.

    Strategy: find the chosen room's longest edge that lies on the boundary,
    take a `depth_m`-deep strip along it, shrink the original room by that
    strip, and add a new Balcony room over the strip. The boundary stays the
    same, so the balcony is INSIDE the original boundary (more like a recessed
    loggia than an external balcony — it always validates).

    opts:
      - from_room: room id to take a slice from (default = the largest room)
      - depth_m:   thickness of the balcony strip (default 1.2)
    """
    boundary = template["boundary"]["polygon"]
    bx_max = max(p[0] for p in boundary)
    by_max = max(p[1] for p in boundary)
    bx_min = min(p[0] for p in boundary)
    by_min = min(p[1] for p in boundary)

    depth = float(opts.get("depth_m", 1.2))
    if depth < 0.8 or depth > 2.0:
        return template, f"depth_m must be 0.8-2.0 m, got {depth}"

    target_id = opts.get("from_room")
    rooms = template["rooms"]
    if target_id:
        room = next((r for r in rooms if r["id"] == target_id), None)
        if not room:
            return template, f"unknown room id {target_id}"
    else:
        # Default: pick the largest non-bath/wc room
        candidates = [r for r in rooms if r.get("type") not in
                      ("bathroom", "wc", "entry", "balcony")]
        if not candidates:
            return template, "no eligible room to take a slice from"
        room = max(candidates, key=lambda r: r.get("area_sqm", 0))

    poly = room["polygon"]
    rx_min = min(p[0] for p in poly)
    rx_max = max(p[0] for p in poly)
    ry_min = min(p[1] for p in poly)
    ry_max = max(p[1] for p in poly)

    # Find an exterior edge of this room (one that lies on the boundary)
    # Try each side; pick the longest one
    sides = [
        ("bottom", ry_min == by_min, rx_max - rx_min),
        ("top",    ry_max == by_max, rx_max - rx_min),
        ("left",   rx_min == bx_min, ry_max - ry_min),
        ("right",  rx_max == bx_max, ry_max - ry_min),
    ]
    exterior = [(name, length) for name, is_ext, length in sides if is_ext]
    if not exterior:
        return template, f"room {room['id']} has no exterior wall"
    exterior.sort(key=lambda x: -x[1])
    edge_name = exterior[0][0]

    # Shrink the room by depth on the chosen edge, balcony takes the strip
    new_poly = [list(p) for p in poly]
    if edge_name == "bottom":
        # Shrink from bottom: ry_min += depth
        if (ry_max - ry_min) - depth < 1.5:
            return template, "room would be too narrow after balcony cut"
        new_poly = [[rx_min, ry_min + depth], [rx_max, ry_min + depth],
                    [rx_max, ry_max], [rx_min, ry_max]]
        bal_poly = [[rx_min, ry_min], [rx_max, ry_min],
                    [rx_max, ry_min + depth], [rx_min, ry_min + depth]]
        bal_window_pos = [(rx_min + rx_max) / 2, ry_min]
    elif edge_name == "top":
        if (ry_max - ry_min) - depth < 1.5:
            return template, "room would be too narrow after balcony cut"
        new_poly = [[rx_min, ry_min], [rx_max, ry_min],
                    [rx_max, ry_max - depth], [rx_min, ry_max - depth]]
        bal_poly = [[rx_min, ry_max - depth], [rx_max, ry_max - depth],
                    [rx_max, ry_max], [rx_min, ry_max]]
        bal_window_pos = [(rx_min + rx_max) / 2, ry_max]
    elif edge_name == "left":
        if (rx_max - rx_min) - depth < 1.5:
            return template, "room would be too narrow after balcony cut"
        new_poly = [[rx_min + depth, ry_min], [rx_max, ry_min],
                    [rx_max, ry_max], [rx_min + depth, ry_max]]
        bal_poly = [[rx_min, ry_min], [rx_min + depth, ry_min],
                    [rx_min + depth, ry_max], [rx_min, ry_max]]
        bal_window_pos = [rx_min, (ry_min + ry_max) / 2]
    else:  # right
        if (rx_max - rx_min) - depth < 1.5:
            return template, "room would be too narrow after balcony cut"
        new_poly = [[rx_min, ry_min], [rx_max - depth, ry_min],
                    [rx_max - depth, ry_max], [rx_min, ry_max]]
        bal_poly = [[rx_max - depth, ry_min], [rx_max, ry_min],
                    [rx_max, ry_max], [rx_max - depth, ry_max]]
        bal_window_pos = [rx_max, (ry_min + ry_max) / 2]

    # Update the source room's polygon + area
    room["polygon"] = new_poly
    new_w = abs(new_poly[1][0] - new_poly[0][0])
    new_h = abs(new_poly[3][1] - new_poly[0][1])
    room["area_sqm"] = round(new_w * new_h, 2)

    # Add the new Balcony room
    used_ids = {r["id"] for r in rooms}
    bal_id = "r_balcony_added"
    n = 2
    while bal_id in used_ids:
        bal_id = f"r_balcony_added_{n}"
        n += 1
    bal_w = abs(bal_poly[1][0] - bal_poly[0][0])
    bal_h = abs(bal_poly[3][1] - bal_poly[0][1])
    rooms.append({
        "id": bal_id,
        "name": "Balcony (added)",
        "type": "balcony",
        "polygon": bal_poly,
        "area_sqm": round(bal_w * bal_h, 2),
    })

    # Move any windows that were on the room's old exterior edge to the
    # balcony (they're now on the balcony's exterior edge, not the room's).
    src_id = room["id"]
    new_windows = []
    moved_window_count = 0
    for w in template.get("windows", []):
        if w.get("room") != src_id:
            new_windows.append(w)
            continue
        wpos = w["position"]
        # Was this window on the cut edge?
        on_cut = False
        if edge_name == "bottom" and abs(wpos[1] - ry_min) < 0.05:
            on_cut = True
        elif edge_name == "top" and abs(wpos[1] - ry_max) < 0.05:
            on_cut = True
        elif edge_name == "left" and abs(wpos[0] - rx_min) < 0.05:
            on_cut = True
        elif edge_name == "right" and abs(wpos[0] - rx_max) < 0.05:
            on_cut = True
        if on_cut:
            # Reassign the window to the balcony (it's still on the boundary,
            # just now belongs to the balcony, not the original room).
            w["room"] = bal_id
            new_windows.append(w)
            moved_window_count += 1
        else:
            new_windows.append(w)
    template["windows"] = new_windows

    # Door from source room → new balcony, on the shared edge between them.
    if edge_name in ("bottom", "top"):
        door_y = new_poly[0][1] if edge_name == "bottom" else new_poly[3][1]
        door_pos = [round((rx_min + rx_max) / 2, 2), door_y]
    else:
        door_x = new_poly[0][0] if edge_name == "left" else new_poly[1][0]
        door_pos = [door_x, round((ry_min + ry_max) / 2, 2)]
    template["doors"].append({
        "from": room["id"], "to": bal_id,
        "position": door_pos, "width_mm": 800,
    })

    return template, ""
