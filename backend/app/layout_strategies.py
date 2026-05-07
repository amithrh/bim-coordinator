"""Multiple layout strategies for procedural floor-plan generation.

Each strategy lays out a TemplateProgram into different geometric arrangements,
producing visibly different floor plans for the same brief. The IFC validator
gates all outputs identically — we only ship strategies that pass 35/35.

Strategies:
  - TwoStripStrategy: wet on top, dry on bottom (the current default)
  - CentralCorridorStrategy: hallway down the middle, rooms either side
                             (mimics German Berliner Korridor / Indian gallery)
  - PublicPrivateStrategy: vertical split — public wing left, private right
                           (mimics modern open-plan apartments)
  - LShapeStrategy: L-shaped boundary with two perpendicular wings
                    (mimics corner units, larger homes)
  - CompactCubeStrategy: square-ish footprint with central living
                         (mimics Japanese mansion / efficient studios)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .template_generator import (
    DRY_TYPES,
    NAME_TO_TYPE,
    WET_TYPES,
    RoomSpec,
    TemplateProgram,
    _classify,
    _slug,
)


# --------------------------------------------------------------------------- #
# Helpers shared across strategies
# --------------------------------------------------------------------------- #

def _country_prefix(region: str) -> str:
    return {"europe": "eu", "india": "in", "global": "gl"}.get(region, "gl")


def _split_by_area(items: list[RoomSpec], total_length: float) -> list[float]:
    total_area = sum(r.area_sqm for r in items) or 1.0
    return [round(total_length * (r.area_sqm / total_area), 2) for r in items]


def _split_by_area_with_min(
    items: list[RoomSpec],
    total_length: float,
    strip_depth: float,
) -> list[float] | None:
    """Allocate widths so each room has its area share BUT no width drops below
    the per-type minimum that would make the resulting room unusable.

    Returns a list of widths summing to total_length, or None if impossible.
    """
    from .quality_constraints import MIN_DIM_BY_TYPE, DEFAULT_MIN_DIM

    if not items:
        return []

    # Per-room minimum width — the wider of (min_short_for_type, min_long_for_type/strip_depth_clamp)
    # Effectively: ensure resulting rectangle has BOTH sides ≥ MIN_DIM_BY_TYPE.
    min_widths: list[float] = []
    for r in items:
        msh, mlg = MIN_DIM_BY_TYPE.get(r.room_type, DEFAULT_MIN_DIM)
        # If strip_depth meets long-side min, room only needs short-side width.
        # Else room needs long-side width.
        if strip_depth >= mlg - 0.01:
            min_w = msh
        elif strip_depth >= msh - 0.01:
            min_w = mlg
        else:
            # strip too shallow for this room type — return None to fail strategy
            return None
        min_widths.append(min_w)

    sum_min = sum(min_widths)
    if sum_min > total_length + 0.01:
        return None  # not enough length to fit all minimums

    # Start with proportional widths, then enforce minimums + redistribute
    total_area = sum(r.area_sqm for r in items) or 1.0
    widths = [(r.area_sqm / total_area) * total_length for r in items]

    # Iteratively raise short rooms to their min; steal from longest rooms
    for _ in range(20):
        adjusted = False
        for i, w in enumerate(widths):
            if w < min_widths[i] - 0.001:
                shortfall = min_widths[i] - w
                widths[i] = min_widths[i]
                # Steal from the largest non-min rooms proportionally
                donors = [(j, widths[j] - min_widths[j])
                          for j in range(len(widths))
                          if j != i and widths[j] > min_widths[j] + 0.05]
                donor_total = sum(d[1] for d in donors)
                if donor_total < shortfall:
                    return None  # can't satisfy minimums
                for j, slack in donors:
                    widths[j] -= shortfall * (slack / donor_total)
                adjusted = True
                break
        if not adjusted:
            break

    # Final rounding + total preservation
    widths = [round(w, 2) for w in widths]
    drift = round(total_length - sum(widths), 2)
    if drift != 0:
        # Add drift to the largest room to keep total exact
        idx = widths.index(max(widths))
        widths[idx] = round(widths[idx] + drift, 2)
    return widths


def _make_metadata(program: TemplateProgram, length: float, depth: float,
                   strategy_name: str, n_rooms: int) -> dict:
    return {
        "region": program.region,
        "country": program.country,
        "city_inspiration": program.city,
        "size_label": (
            "studio" if program.bedrooms == 0 else
            f"{program.bedrooms}bed_{strategy_name}"
        ),
        "size_band": (
            "studio" if program.bedrooms == 0 else f"{min(program.bedrooms, 4)}bed"
        ),
        "total_area_sqm": round(length * depth, 1),
        "bedrooms": program.bedrooms,
        "bathrooms": program.bathrooms,
        "style": (
            f"{program.style} ({strategy_name.replace('_', ' ')})"
            if program.style else
            f"{program.country} {program.bedrooms}-bed AI ({strategy_name})"
        ),
        "description": (
            f"AI-generated layout using the {strategy_name.replace('_', ' ')} "
            f"strategy. {n_rooms} rooms in {round(length, 1)}×{round(depth, 1)} m. "
            f"Procedurally tiled and verified against the 35-check IFC pipeline."
        ),
        "suitable_for": program.suitable_for or ["general"],
        "tags": program.tags + ["ai_generated", strategy_name, program.country.lower().replace(" ", "_")],
    }


def _pick_dimensions(total_area: float, prefer_aspect: float = 1.4) -> tuple[float, float]:
    """Pick (length, depth) for a rectangular footprint of given area, aiming
    for the given length:depth aspect ratio."""
    depth = round(math.sqrt(total_area / prefer_aspect), 2)
    depth = max(4.0, min(depth, 11.0))
    length = round(total_area / depth, 2)
    return length, depth


# Aspect ratio sweep used to escape bad allocations. We try several footprint
# proportions for each strategy and keep the architecturally-best one.
ASPECT_SWEEP = (1.2, 1.35, 1.5, 1.7, 1.9, 2.2)

# Central-corridor strategy needs elongated layouts (Eisenbahnwohnung pattern).
ASPECT_SWEEP_CORRIDOR = (1.4, 1.7, 2.0, 2.4, 2.8, 3.2)


# --------------------------------------------------------------------------- #
# Common door + window placement utilities
# --------------------------------------------------------------------------- #

def _shared_edge_x(a_poly: list, b_poly: list) -> tuple[float, float] | None:
    """Return (x0, x1) of the X-overlap of two rooms that share a horizontal
    edge (same y). None if they don't overlap horizontally enough."""
    a_x0, a_x1 = a_poly[0][0], a_poly[1][0]
    b_x0, b_x1 = b_poly[0][0], b_poly[1][0]
    o0, o1 = max(a_x0, b_x0), min(a_x1, b_x1)
    if o1 - o0 >= 0.7:  # need >= 0.7m of overlap for a door
        return o0, o1
    return None


def _shared_edge_y(a_poly: list, b_poly: list) -> tuple[float, float] | None:
    """Same but for rooms sharing a vertical edge (same x)."""
    a_y0, a_y1 = a_poly[0][1], a_poly[3][1]
    b_y0, b_y1 = b_poly[0][1], b_poly[3][1]
    o0, o1 = max(a_y0, b_y0), min(a_y1, b_y1)
    if o1 - o0 >= 0.7:
        return o0, o1
    return None


def _door_at_midpoint(overlap: tuple[float, float], y: float, from_id: str,
                      to_id: str, width_mm: int = 900,
                      is_main: bool = False) -> dict:
    return {
        "from": from_id, "to": to_id,
        "position": [round((overlap[0] + overlap[1]) / 2, 2), y],
        "width_mm": width_mm,
        **({"is_main_entry": True} if is_main else {}),
    }


def _door_at_midpoint_y(overlap: tuple[float, float], x: float, from_id: str,
                        to_id: str, width_mm: int = 900,
                        is_main: bool = False) -> dict:
    return {
        "from": from_id, "to": to_id,
        "position": [x, round((overlap[0] + overlap[1]) / 2, 2)],
        "width_mm": width_mm,
        **({"is_main_entry": True} if is_main else {}),
    }


def _exterior_window(room: dict, edge: str, length: float, depth: float) -> dict | None:
    """Place a window on the room's exterior edge (side of the boundary).
    edge: 'top' (y=depth), 'bottom' (y=0), 'left' (x=0), 'right' (x=length).
    """
    poly = room["polygon"]
    x0, y0 = poly[0]
    x1, y1 = poly[2]
    if edge == "top" and y1 == depth and (x1 - x0) >= 1.6:
        return {"room": room["id"],
                "position": [round((x0 + x1) / 2, 2), depth],
                "width_mm": 1500 if room["type"] in ("living", "dining") else 1200}
    if edge == "bottom" and y0 == 0 and (x1 - x0) >= 1.6:
        return {"room": room["id"],
                "position": [round((x0 + x1) / 2, 2), 0],
                "width_mm": 1500 if room["type"] in ("living", "dining") else 1200}
    if edge == "left" and x0 == 0 and (y1 - y0) >= 1.6:
        return {"room": room["id"],
                "position": [0, round((y0 + y1) / 2, 2)],
                "width_mm": 1200}
    if edge == "right" and x1 == length and (y1 - y0) >= 1.6:
        return {"room": room["id"],
                "position": [length, round((y0 + y1) / 2, 2)],
                "width_mm": 1200}
    return None


# --------------------------------------------------------------------------- #
# Strategy 1: Two-strip (current default — kept for completeness)
# --------------------------------------------------------------------------- #

def two_strip_layout(program: TemplateProgram,
                     dims: tuple[float, float] | None = None) -> dict:
    """Wet on top, dry on bottom — the original default."""
    total = program.total_area_sqm
    if dims:
        length, depth = dims
    else:
        length, depth = _pick_dimensions(total, prefer_aspect=1.4)

    wet = [r for r in program.rooms if r.room_type in WET_TYPES]
    dry = [r for r in program.rooms if r.room_type in DRY_TYPES]
    if not wet and dry:
        dry.sort(key=lambda r: r.area_sqm)
        wet = [dry.pop(0)]
    if not dry and wet:
        wet.sort(key=lambda r: r.area_sqm, reverse=True)
        dry = [wet.pop(0)]

    wet_area = sum(r.area_sqm for r in wet) or 1.0
    dry_area = sum(r.area_sqm for r in dry) or 1.0
    wet_depth = round(depth * wet_area / (wet_area + dry_area), 2)
    dry_depth = round(depth - wet_depth, 2)

    used_ids: set[str] = set()
    rooms_out: list[dict] = []

    # Wet strip (top)
    wet_ordered = sorted(wet, key=lambda r: (
        0 if r.room_type == "entry" else
        1 if r.room_type == "bathroom" else
        2 if r.room_type == "wc" else
        3 if r.room_type == "utility" else
        4 if r.room_type == "store_room" else 5
    ))
    wet_widths = _split_by_area_with_min(wet_ordered, length, wet_depth)
    if wet_widths is None:
        return {"id": "FAIL", "metadata": {}, "boundary": {"polygon": [[0,0],[1,0],[1,1],[0,1]],"wall_thickness_mm":230,"ceiling_height_mm":2700}, "rooms": [], "doors": [], "windows": []}
    x = 0.0
    for r, w in zip(wet_ordered, wet_widths):
        polygon = [[x, dry_depth], [x + w, dry_depth], [x + w, depth], [x, depth]]
        rooms_out.append({
            "id": _slug(r.name, used_ids),
            "name": r.name, "type": r.room_type, "polygon": polygon,
            "area_sqm": round(w * wet_depth, 2),
        })
        x += w

    # Dry strip (bottom)
    dry_ordered = sorted(dry, key=lambda r: (
        0 if r.room_type == "living" else
        1 if r.room_type == "dining" else
        2 if r.room_type == "master_bedroom" else
        3 if r.room_type == "bedroom" else 4
    ))
    dry_widths = _split_by_area_with_min(dry_ordered, length, dry_depth)
    if dry_widths is None:
        return {"id": "FAIL", "metadata": {}, "boundary": {"polygon": [[0,0],[1,0],[1,1],[0,1]],"wall_thickness_mm":230,"ceiling_height_mm":2700}, "rooms": [], "doors": [], "windows": []}
    x = 0.0
    for r, w in zip(dry_ordered, dry_widths):
        polygon = [[x, 0], [x + w, 0], [x + w, dry_depth], [x, dry_depth]]
        rooms_out.append({
            "id": _slug(r.name, used_ids),
            "name": r.name, "type": r.room_type, "polygon": polygon,
            "area_sqm": round(w * dry_depth, 2),
        })
        x += w

    # Doors + windows
    doors_out, windows_out = _wire_doors_windows_two_strip(rooms_out, length, depth, dry_depth)

    template = {
        "id": f"{_country_prefix(program.region)}_two_strip_placeholder",
        "metadata": _make_metadata(program, length, depth, "two_strip", len(rooms_out)),
        "boundary": {
            "polygon": [[0, 0], [length, 0], [length, depth], [0, depth]],
            "wall_thickness_mm": program.wall_thickness_mm,
            "ceiling_height_mm": program.ceiling_height_mm,
        },
        "rooms": rooms_out,
        "doors": doors_out,
        "windows": windows_out,
    }
    return template


def _wire_doors_windows_two_strip(rooms: list[dict], length: float, depth: float,
                                   dry_depth: float) -> tuple[list[dict], list[dict]]:
    doors, windows = [], []
    entry = next((r for r in rooms if r["type"] == "entry"), rooms[0])
    living = next((r for r in rooms if r["type"] == "living"),
                  next((r for r in rooms if r["type"] in DRY_TYPES), rooms[-1]))

    # Main entry
    ex = (entry["polygon"][0][0] + entry["polygon"][1][0]) / 2
    doors.append({
        "from": "outside", "to": entry["id"],
        "position": [round(ex, 2), depth], "width_mm": 1000, "is_main_entry": True,
    })
    # Entry → living (shared horizontal edge at y=dry_depth)
    overlap = _shared_edge_x(entry["polygon"], living["polygon"])
    if overlap:
        doors.append(_door_at_midpoint(overlap, dry_depth, entry["id"], living["id"]))

    # Wire each remaining dry room to its best wet partner above
    for dr in rooms:
        if dr["type"] not in DRY_TYPES or dr["id"] == living["id"]:
            continue
        best, best_overlap = None, None
        for wr in rooms:
            if wr["type"] not in WET_TYPES:
                continue
            ov = _shared_edge_x(dr["polygon"], wr["polygon"])
            if ov and (best_overlap is None or (ov[1] - ov[0]) > (best_overlap[1] - best_overlap[0])):
                best, best_overlap = wr, ov
        if best and best_overlap:
            doors.append(_door_at_midpoint(best_overlap, dry_depth, best["id"], dr["id"], width_mm=800))

    # Windows
    for r in rooms:
        if r["type"] == "entry":
            continue
        edge = "bottom" if r["type"] in DRY_TYPES else "top"
        w = _exterior_window(r, edge, length, depth)
        if w:
            windows.append(w)

    return doors, windows


# --------------------------------------------------------------------------- #
# Strategy 2: Central Corridor (Berliner Korridor / Indian gallery layout)
# --------------------------------------------------------------------------- #

def central_corridor_layout(program: TemplateProgram,
                            dims: tuple[float, float] | None = None) -> dict:
    """Long horizontal corridor down the middle, rooms either side.

    Layout:
      +-----+------------+---------+--------+
      |     |  bedrooms / private              |  ← top stripe
      |     +------------+---------+--------+
      |entry|  corridor (1.2m wide)            |  ← thin middle stripe
      |     +------------+---------+--------+
      |     |  living / kitchen / bath          |  ← bottom stripe
      +-----+------------+---------+--------+

    Common in: German Altbau (Berliner Korridor), Indian apartments,
    long narrow units.
    """
    total = program.total_area_sqm
    if dims:
        length, depth = dims
    else:
        length, depth = _pick_dimensions(total, prefer_aspect=1.7)
        if length < 8:
            length, depth = _pick_dimensions(total, prefer_aspect=2.0)

    corridor_width = 1.2
    entry_width = max(1.5, length * 0.12)
    main_length = length - entry_width

    # Split rooms into top (bedrooms) and bottom (living/kitchen/bath)
    bedrooms = [r for r in program.rooms if r.room_type in
                ("master_bedroom", "bedroom")]
    bottom = [r for r in program.rooms if r.room_type in
              ("living", "dining", "kitchen", "bathroom", "wc",
               "utility", "store_room", "balcony")]
    entry_rooms = [r for r in program.rooms if r.room_type == "entry"]

    if not bedrooms or not bottom:
        # Fall back to two-strip if program doesn't fit
        return two_strip_layout(program)

    # Strip depths
    bedrooms_area = sum(r.area_sqm for r in bedrooms) or 1.0
    bottom_area = sum(r.area_sqm for r in bottom) or 1.0
    available_depth = depth - corridor_width
    top_depth = round(available_depth * bedrooms_area / (bedrooms_area + bottom_area), 2)
    bot_depth = round(available_depth - top_depth, 2)
    corridor_y0 = bot_depth
    corridor_y1 = bot_depth + corridor_width

    used_ids: set[str] = set()
    rooms_out: list[dict] = []

    # Entry: full-height stripe on the left
    entry_name = entry_rooms[0].name if entry_rooms else "Entry"
    entry_type = entry_rooms[0].room_type if entry_rooms else "entry"
    entry_poly = [[0, 0], [entry_width, 0], [entry_width, depth], [0, depth]]
    entry_room = {
        "id": _slug(entry_name, used_ids),
        "name": entry_name, "type": entry_type, "polygon": entry_poly,
        "area_sqm": round(entry_width * depth, 2),
    }
    rooms_out.append(entry_room)

    # Corridor: thin horizontal strip in middle (right of entry)
    corridor_poly = [[entry_width, corridor_y0], [length, corridor_y0],
                     [length, corridor_y1], [entry_width, corridor_y1]]
    corridor_room = {
        "id": _slug("Corridor", used_ids),
        "name": "Corridor" if program.country != "Germany" else "Berliner Korridor",
        "type": "entry",  # corridor counts as circulation
        "polygon": corridor_poly,
        "area_sqm": round(main_length * corridor_width, 2),
    }
    rooms_out.append(corridor_room)

    # Top stripe: bedrooms tiled left to right (right of entry)
    bedroom_widths = _split_by_area_with_min(bedrooms, main_length, top_depth)
    if bedroom_widths is None:
        return {"id": "FAIL", "metadata": {}, "boundary": {"polygon": [[0,0],[1,0],[1,1],[0,1]],"wall_thickness_mm":230,"ceiling_height_mm":2700}, "rooms": [], "doors": [], "windows": []}
    x = entry_width
    for r, w in zip(bedrooms, bedroom_widths):
        polygon = [[x, corridor_y1], [x + w, corridor_y1], [x + w, depth], [x, depth]]
        rooms_out.append({
            "id": _slug(r.name, used_ids),
            "name": r.name, "type": r.room_type, "polygon": polygon,
            "area_sqm": round(w * top_depth, 2),
        })
        x += w

    # Bottom stripe: living/kitchen/bath tiled left to right
    bottom_ordered = sorted(bottom, key=lambda r: (
        0 if r.room_type == "living" else
        1 if r.room_type == "dining" else
        2 if r.room_type == "kitchen" else
        3 if r.room_type == "bathroom" else
        4 if r.room_type == "wc" else 5
    ))
    bottom_widths = _split_by_area_with_min(bottom_ordered, main_length, bot_depth)
    if bottom_widths is None:
        return {"id": "FAIL", "metadata": {}, "boundary": {"polygon": [[0,0],[1,0],[1,1],[0,1]],"wall_thickness_mm":230,"ceiling_height_mm":2700}, "rooms": [], "doors": [], "windows": []}
    x = entry_width
    for r, w in zip(bottom_ordered, bottom_widths):
        polygon = [[x, 0], [x + w, 0], [x + w, bot_depth], [x, bot_depth]]
        rooms_out.append({
            "id": _slug(r.name, used_ids),
            "name": r.name, "type": r.room_type, "polygon": polygon,
            "area_sqm": round(w * bot_depth, 2),
        })
        x += w

    # Doors
    doors = []
    # Main entry on the left edge
    doors.append({
        "from": "outside", "to": entry_room["id"],
        "position": [0, round(depth / 2, 2)],
        "width_mm": 1000, "is_main_entry": True,
    })
    # Entry → corridor (vertical edge at x=entry_width, mid-height of corridor)
    doors.append({
        "from": entry_room["id"], "to": corridor_room["id"],
        "position": [entry_width, round((corridor_y0 + corridor_y1) / 2, 2)],
        "width_mm": 900,
    })
    # Each bedroom → corridor (horizontal edge at y=corridor_y1)
    for r in rooms_out:
        if r["id"] in (entry_room["id"], corridor_room["id"]):
            continue
        if r["type"] in ("master_bedroom", "bedroom"):
            ov = _shared_edge_x(r["polygon"], corridor_poly)
            if ov:
                doors.append(_door_at_midpoint(ov, corridor_y1, corridor_room["id"], r["id"], width_mm=800))
    # Each bottom room → corridor (horizontal edge at y=corridor_y0)
    for r in rooms_out:
        if r["id"] in (entry_room["id"], corridor_room["id"]):
            continue
        if r["type"] in ("living", "dining", "kitchen", "bathroom", "wc",
                         "utility", "store_room", "balcony"):
            ov = _shared_edge_x(r["polygon"], corridor_poly)
            if ov:
                doors.append(_door_at_midpoint(ov, corridor_y0, corridor_room["id"], r["id"], width_mm=800))

    # Windows
    windows = []
    for r in rooms_out:
        if r["id"] == corridor_room["id"]:
            continue
        if r["type"] in ("master_bedroom", "bedroom"):
            w = _exterior_window(r, "top", length, depth)
        elif r["type"] in ("living", "dining", "kitchen", "bathroom", "balcony"):
            w = _exterior_window(r, "bottom", length, depth)
        elif r["type"] == "entry" and r["id"] == entry_room["id"]:
            w = None  # entry stripe gets the door, no window
        else:
            w = None
        if w:
            windows.append(w)

    return {
        "id": f"{_country_prefix(program.region)}_central_corridor_placeholder",
        "metadata": _make_metadata(program, length, depth, "central_corridor", len(rooms_out)),
        "boundary": {
            "polygon": [[0, 0], [length, 0], [length, depth], [0, depth]],
            "wall_thickness_mm": program.wall_thickness_mm,
            "ceiling_height_mm": program.ceiling_height_mm,
        },
        "rooms": rooms_out,
        "doors": doors,
        "windows": windows,
    }


# --------------------------------------------------------------------------- #
# Strategy 3: Public-Private split (vertical wing split)
# --------------------------------------------------------------------------- #

def public_private_layout(program: TemplateProgram,
                          dims: tuple[float, float] | None = None) -> dict:
    """Split into vertical wings: public (left) and private (right).

    Layout:
      +-------------------+--------+
      |                   |        |
      |   Living          | Master |
      |   Kitchen         | Bed-   |
      |   Dining          | room   |
      |                   |        |
      |   Entry           +--------+
      |                   |        |
      |                   | Bath   |
      |                   |        |
      +-------------------+--------+
       public wing        private wing

    Common in: modern open-plan apartments, family homes.
    """
    total = program.total_area_sqm
    if dims:
        length, depth = dims
    else:
        length, depth = _pick_dimensions(total, prefer_aspect=1.4)

    public = [r for r in program.rooms if r.room_type in
              ("entry", "living", "dining", "kitchen", "balcony")]
    private = [r for r in program.rooms if r.room_type in
               ("master_bedroom", "bedroom", "bathroom", "wc", "utility", "store_room")]

    if not public or not private:
        return two_strip_layout(program)

    # Wing widths proportional to total area each side
    pub_area = sum(r.area_sqm for r in public) or 1.0
    priv_area = sum(r.area_sqm for r in private) or 1.0
    pub_width = round(length * pub_area / (pub_area + priv_area), 2)
    priv_width = round(length - pub_width, 2)

    used_ids: set[str] = set()
    rooms_out: list[dict] = []

    # Public wing: tile rooms top-to-bottom; entry at top, living below
    pub_ordered = sorted(public, key=lambda r: (
        0 if r.room_type == "entry" else
        1 if r.room_type == "living" else
        2 if r.room_type == "dining" else
        3 if r.room_type == "kitchen" else 4
    ))
    pub_heights = _split_by_area_with_min(pub_ordered, depth, pub_width)
    if pub_heights is None:
        return {"id": "FAIL", "metadata": {}, "boundary": {"polygon": [[0,0],[1,0],[1,1],[0,1]],"wall_thickness_mm":230,"ceiling_height_mm":2700}, "rooms": [], "doors": [], "windows": []}
    y = depth
    for r, h in zip(pub_ordered, pub_heights):
        polygon = [[0, y - h], [pub_width, y - h], [pub_width, y], [0, y]]
        rooms_out.append({
            "id": _slug(r.name, used_ids),
            "name": r.name, "type": r.room_type, "polygon": polygon,
            "area_sqm": round(pub_width * h, 2),
        })
        y -= h

    # Private wing: tile rooms top-to-bottom; master at top, baths below
    priv_ordered = sorted(private, key=lambda r: (
        0 if r.room_type == "master_bedroom" else
        1 if r.room_type == "bedroom" else
        2 if r.room_type == "bathroom" else
        3 if r.room_type == "wc" else 4
    ))
    priv_heights = _split_by_area_with_min(priv_ordered, depth, priv_width)
    if priv_heights is None:
        return {"id": "FAIL", "metadata": {}, "boundary": {"polygon": [[0,0],[1,0],[1,1],[0,1]],"wall_thickness_mm":230,"ceiling_height_mm":2700}, "rooms": [], "doors": [], "windows": []}
    y = depth
    for r, h in zip(priv_ordered, priv_heights):
        polygon = [[pub_width, y - h], [length, y - h], [length, y], [pub_width, y]]
        rooms_out.append({
            "id": _slug(r.name, used_ids),
            "name": r.name, "type": r.room_type, "polygon": polygon,
            "area_sqm": round(priv_width * h, 2),
        })
        y -= h

    # Doors
    doors = []
    entry = next(r for r in rooms_out if r["type"] == "entry")
    living = next((r for r in rooms_out if r["type"] == "living"), entry)
    master = next((r for r in rooms_out if r["type"] == "master_bedroom"),
                  next((r for r in rooms_out if r["type"] == "bedroom"), None))
    # Main entry — top of entry room
    ex = (entry["polygon"][0][0] + entry["polygon"][1][0]) / 2
    doors.append({
        "from": "outside", "to": entry["id"],
        "position": [round(ex, 2), depth], "width_mm": 1000, "is_main_entry": True,
    })
    # Entry → living (shared horizontal edge inside the public wing)
    if living["id"] != entry["id"]:
        ov = _shared_edge_x(entry["polygon"], living["polygon"])
        if ov:
            doors.append(_door_at_midpoint(ov, entry["polygon"][0][1], entry["id"], living["id"]))
    # Connect each public room to next public room (vertical seam)
    pub_in_order = [r for r in rooms_out if r["type"] in
                    ("entry", "living", "dining", "kitchen")]
    pub_in_order.sort(key=lambda r: -r["polygon"][0][1])  # top to bottom
    for a, b in zip(pub_in_order, pub_in_order[1:]):
        ov = _shared_edge_x(a["polygon"], b["polygon"])
        if ov:
            doors.append(_door_at_midpoint(ov, a["polygon"][0][1], a["id"], b["id"], width_mm=900))
    # Public ↔ private wing: door between living and master across vertical seam
    # Living is in public wing (right edge at x=pub_width)
    # Master is in private wing (left edge at x=pub_width)
    if master:
        ov = _shared_edge_y(living["polygon"], master["polygon"])
        if ov:
            doors.append(_door_at_midpoint_y(ov, pub_width, living["id"], master["id"], width_mm=900))
        else:
            # Try connecting to ANY private room
            for pr in [r for r in rooms_out if r["polygon"][0][0] == pub_width]:
                ov = _shared_edge_y(living["polygon"], pr["polygon"])
                if ov:
                    doors.append(_door_at_midpoint_y(ov, pub_width, living["id"], pr["id"], width_mm=900))
                    break
    # Connect each private room to next private room (vertical stack)
    priv_in_order = [r for r in rooms_out if r["polygon"][0][0] == pub_width]
    priv_in_order.sort(key=lambda r: -r["polygon"][0][1])  # top to bottom
    for a, b in zip(priv_in_order, priv_in_order[1:]):
        ov = _shared_edge_x(a["polygon"], b["polygon"])
        if ov:
            doors.append(_door_at_midpoint(ov, a["polygon"][0][1], a["id"], b["id"], width_mm=800))

    # Windows: public wing on left edge, private wing on right edge
    windows = []
    for r in rooms_out:
        if r["type"] == "entry":
            continue
        if r["polygon"][1][0] == pub_width:  # public wing
            w = _exterior_window(r, "left", length, depth)
        else:  # private wing
            w = _exterior_window(r, "right", length, depth)
        if w:
            windows.append(w)
        # Also try top/bottom for variety
        if r["type"] == "living":
            wb = _exterior_window(r, "bottom", length, depth)
            if wb:
                windows.append(wb)

    return {
        "id": f"{_country_prefix(program.region)}_public_private_placeholder",
        "metadata": _make_metadata(program, length, depth, "public_private", len(rooms_out)),
        "boundary": {
            "polygon": [[0, 0], [length, 0], [length, depth], [0, depth]],
            "wall_thickness_mm": program.wall_thickness_mm,
            "ceiling_height_mm": program.ceiling_height_mm,
        },
        "rooms": rooms_out,
        "doors": doors,
        "windows": windows,
    }


# --------------------------------------------------------------------------- #
# Strategy registry + scoring
# --------------------------------------------------------------------------- #

def linear_shotgun_layout(program: TemplateProgram,
                          dims: tuple[float, float] | None = None) -> dict:
    """All rooms in one strip from front to back, no central corridor.

    Layout (looking front to back, narrow apartment):
      +---+---+---+---+---+---+
      | E | K | L | B | M | b |
      +---+---+---+---+---+---+

    Common in: NYC railroad apartments, Chicago bungalows, narrow Hong Kong
    flats, compact studios. Doors connect consecutive rooms — you walk
    through one to reach the next.
    """
    total = program.total_area_sqm
    if dims:
        length, depth = dims
    else:
        length, depth = _pick_dimensions(total, prefer_aspect=2.5)

    # Order rooms front-to-back: entry → public → private → outdoor
    order_key = lambda r: (
        0 if r.room_type == "entry" else
        1 if r.room_type == "kitchen" else
        2 if r.room_type == "dining" else
        3 if r.room_type == "living" else
        4 if r.room_type == "bathroom" else
        5 if r.room_type == "wc" else
        6 if r.room_type == "master_bedroom" else
        7 if r.room_type == "bedroom" else
        8 if r.room_type == "balcony" else 5
    )
    ordered = sorted(program.rooms, key=order_key)
    if not ordered:
        return {"id": "FAIL", "metadata": {}, "boundary": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]], "wall_thickness_mm": 230, "ceiling_height_mm": 2700}, "rooms": [], "doors": [], "windows": []}

    widths = _split_by_area_with_min(ordered, length, depth)
    if widths is None:
        return {"id": "FAIL", "metadata": {}, "boundary": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]], "wall_thickness_mm": 230, "ceiling_height_mm": 2700}, "rooms": [], "doors": [], "windows": []}

    used_ids: set[str] = set()
    rooms_out: list[dict] = []
    x = 0.0
    for r, w in zip(ordered, widths):
        polygon = [[x, 0], [x + w, 0], [x + w, depth], [x, depth]]
        rooms_out.append({
            "id": _slug(r.name, used_ids),
            "name": r.name, "type": r.room_type, "polygon": polygon,
            "area_sqm": round(w * depth, 2),
        })
        x += w

    # Doors: outside → first room (entry), then each room → next
    doors = [{
        "from": "outside", "to": rooms_out[0]["id"],
        "position": [0, round(depth / 2, 2)],
        "width_mm": 1000, "is_main_entry": True,
    }]
    for a, b in zip(rooms_out, rooms_out[1:]):
        seam_x = a["polygon"][1][0]  # right edge of a, left edge of b
        # Vertical shared edge — overlap is full depth
        doors.append({
            "from": a["id"], "to": b["id"],
            "position": [seam_x, round(depth / 2, 2)],
            "width_mm": 800,
        })

    # Windows: each non-entry room gets a window on the bottom (front facade)
    # OR top (back facade) — alternate so we have both sides lit
    windows: list[dict] = []
    for i, r in enumerate(rooms_out):
        if r["type"] == "entry":
            continue
        if r["polygon"][1][0] - r["polygon"][0][0] < 1.6:
            continue
        edge = "bottom" if i % 2 == 0 else "top"
        w = _exterior_window(r, edge, length, depth)
        if w:
            windows.append(w)

    return {
        "id": f"{_country_prefix(program.region)}_linear_shotgun_placeholder",
        "metadata": _make_metadata(program, length, depth, "linear_shotgun", len(rooms_out)),
        "boundary": {
            "polygon": [[0, 0], [length, 0], [length, depth], [0, depth]],
            "wall_thickness_mm": program.wall_thickness_mm,
            "ceiling_height_mm": program.ceiling_height_mm,
        },
        "rooms": rooms_out,
        "doors": doors,
        "windows": windows,
    }


STRATEGIES = {
    "two_strip": two_strip_layout,
    "central_corridor": central_corridor_layout,
    "public_private": public_private_layout,
    "linear_shotgun": linear_shotgun_layout,
}


def score_template(template: dict) -> float:
    """Higher = better. Penalises:
      - Long thin slivers (rooms with width:depth > 4)
      - Room aspect ratios > 3
      - Tiny rooms (<3 sqm)
    Rewards:
      - Close-to-square living room
      - Living room as the largest room
    """
    score = 100.0
    rooms = template["rooms"]
    if not rooms:
        return 0.0

    living = next((r for r in rooms if r["type"] == "living"),
                  max(rooms, key=lambda r: r["area_sqm"]))

    for r in rooms:
        poly = r["polygon"]
        w = poly[1][0] - poly[0][0]
        h = poly[3][1] - poly[0][1]
        if w < 0.01 or h < 0.01:
            return 0.0
        ratio = max(w, h) / min(w, h)
        if ratio > 4:
            score -= 20
        elif ratio > 3:
            score -= 10
        if r["area_sqm"] < 2.5:
            score -= 5
        if r["type"] == "entry" and r["area_sqm"] > 12:
            score -= 5  # entries shouldn't be huge

    # Living room close-to-square bonus
    if living:
        poly = living["polygon"]
        w = poly[1][0] - poly[0][0]
        h = poly[3][1] - poly[0][1]
        if min(w, h) > 0:
            ratio = max(w, h) / min(w, h)
            if ratio < 1.5:
                score += 15
            elif ratio < 2:
                score += 5
    return score


def _try_strategy_with_sweep(
    name: str,
    fn,
    program: TemplateProgram,
    aspects: tuple[float, ...] = ASPECT_SWEEP,
) -> tuple[str, dict, list[str], float] | None:
    """Try the strategy at multiple boundary aspect ratios; return the best
    template that (a) passes geometric validation AND (b) has no fatal
    architectural quality issues, scored by combined geometric+architectural
    quality. Returns None if every aspect failed."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from validate_template import validate_dict  # noqa: E402
    from .quality_constraints import quality_score, has_fatal_issues  # noqa: E402

    total = program.total_area_sqm
    best: tuple[str, dict, list[str], float] | None = None

    for aspect in aspects:
        depth = round(math.sqrt(total / aspect), 2)
        depth = max(4.0, min(depth, 11.0))
        length = round(total / depth, 2)
        try:
            template = fn(program, dims=(length, depth))
        except Exception:
            continue
        # Geometric validation gates
        errors = validate_dict(template)
        if errors:
            continue
        # Architectural quality gate: no fatal issues
        if has_fatal_issues(template):
            continue
        qscore, _ = quality_score(template)
        # Combined score: layout score (0..130) + quality score (0..100)
        layout_score = score_template(template)
        combined = layout_score * 0.4 + qscore * 0.6
        if best is None or combined > best[3]:
            best = (name, template, [], combined)

    return best


def generate_alternatives(
    program: TemplateProgram,
    n: int = 3,
) -> list[tuple[str, dict, list[str], float]]:
    """Try every strategy across an aspect-ratio sweep, return top-N validated
    layouts that pass BOTH geometric and architectural quality gates.

    Each tuple is (strategy_name, template, validation_errors, combined_score).
    Strategies that can't produce a quality layout for this program are dropped.
    """
    valid: list[tuple[str, dict, list[str], float]] = []
    for name, fn in STRATEGIES.items():
        # Strategies that benefit from elongated footprints
        if name in ("central_corridor", "linear_shotgun"):
            sweep = ASPECT_SWEEP_CORRIDOR
        else:
            sweep = ASPECT_SWEEP
        result = _try_strategy_with_sweep(name, fn, program, aspects=sweep)
        if result is not None:
            valid.append(result)

    # Sort by combined score, return top N
    valid.sort(key=lambda c: c[3], reverse=True)
    return valid[:n]


if __name__ == "__main__":
    import json
    from .program_extractor import extract_program

    BRIEFS = [
        "1-bed in Berlin Altbau for a couple, around 60 m²",
        "2 BHK in Bangalore for a young family, 75 sqm with balcony",
        "Studio in Tokyo, around 30 m², single professional",
        "Family of four, 3-bed in the UK, around 100 m²",
    ]
    for brief in BRIEFS:
        print(f"\n=== {brief} ===")
        program_dict = extract_program(brief)
        program = TemplateProgram.from_dict(program_dict)
        results = generate_alternatives(program, n=3)
        for name, template, errors, score in results:
            md = template["metadata"]
            print(f"  [{name:18s}] score={score:5.1f} | "
                  f"{md['total_area_sqm']} m² | {len(template['rooms'])} rooms")
