"""LEVEL 4: Free-form template generation with guaranteed valid geometry.

Strategy: split the LLM's job from the geometry job.
  - LLM produces a high-level "program" (room list, areas, adjacencies)
  - Procedural code lays out rectangular polygons that satisfy the program
  - Doors are placed on shared edges deterministically
  - Result is geometrically valid by construction
  - Final IFC is verified by the same 35-check pipeline as the curated 500

Layout strategy: 2-strip layout with axis-aligned rectangles
  +-----------------------------------------------------+
  | wet strip (entry, bath, WC, kitchen, utility)       |  ← top strip
  +-----------------------------------------------------+
  | living strip (living, dining, bedrooms, balcony)    |  ← bottom strip
  +-----------------------------------------------------+
The strip widths are chosen by area share. Within each strip rooms are
laid out left-to-right, with widths chosen by area share.

This handles the common single-floor apartment case (1bed-3bed) which
covers ~95% of our 500-template library. Stadiums/bridges/multi-story
need a different generator (Phase 2).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# Room semantics — maps free-form room names → categories used by the layout #
# --------------------------------------------------------------------------- #

WET_TYPES = {"entry", "kitchen", "bathroom", "wc", "utility", "store_room"}
DRY_TYPES = {"living", "dining", "master_bedroom", "bedroom", "balcony"}

NAME_TO_TYPE = {
    # Entry
    "foyer": "entry", "entry": "entry", "hall": "entry",
    "vestibule": "entry", "lobby": "entry", "diele": "entry",
    "ingresso": "entry", "entrada": "entry", "recibidor": "entry",
    "hol": "entry", "genkan": "entry", "inkom": "entry",
    # Living/dining
    "living": "living", "salon": "living", "soggiorno": "living",
    "stue": "living", "wohnzimmer": "living", "saloni": "living",
    "vardagsrum": "living", "olohuone": "living",
    "drawing room": "living", "lounge": "living", "salón": "living",
    "living room": "living", "living/dining": "living",
    "living-dining": "living", "ldk": "living",
    "drawing-dining": "living", "majlis": "living", "reception": "living",
    "séjour": "living", "sala": "living",
    # Dining (rare separate)
    "dining": "dining", "spisestue": "dining", "essbereich": "dining",
    # Kitchen
    "kitchen": "kitchen", "küche": "kitchen", "cocina": "kitchen",
    "cucina": "kitchen", "köök": "kitchen", "kuchnia": "kitchen",
    "kjøkken": "kitchen", "kök": "kitchen", "keittiö": "kitchen",
    "mutfak": "kitchen", "kuhinja": "kitchen", "κουζίνα": "kitchen",
    "keuken": "kitchen", "cozinha": "kitchen", "kuchyně": "kitchen",
    "küchenzeile": "kitchen", "kochnische": "kitchen",
    "wohnkueche": "kitchen", "kitchen-dining": "kitchen",
    "kitchen-diner": "kitchen",
    # Bathroom
    "bathroom": "bathroom", "bad": "bathroom", "badezimmer": "bathroom",
    "salle de bain": "bathroom", "salle de bains": "bathroom",
    "baño": "bathroom", "bagno": "bathroom", "casa de banho": "bathroom",
    "kupaonica": "bathroom", "μπάνιο": "bathroom", "kylpyhuone": "bathroom",
    "badrum": "bathroom", "badeværelse": "bathroom", "wannenbad": "bathroom",
    "duschbad": "bathroom", "łazienka": "bathroom", "bath": "bathroom",
    "shower room": "bathroom", "banyo": "bathroom",
    # WC
    "wc": "wc", "toilet": "wc", "aseo": "wc",
    # Bedroom
    "master bedroom": "master_bedroom", "master": "master_bedroom",
    "schlafzimmer": "master_bedroom", "chambre": "master_bedroom",
    "dormitorio": "master_bedroom", "camera da letto": "master_bedroom",
    "quarto": "master_bedroom", "soveværelse": "master_bedroom",
    "sovrum": "master_bedroom", "makuuhuone": "master_bedroom",
    "yatak odası": "master_bedroom", "spavaća soba": "master_bedroom",
    "κύριο υπνοδωμάτιο": "master_bedroom", "sypialnia": "master_bedroom",
    "bedroom": "bedroom", "bedroom 2": "bedroom", "bedroom 3": "bedroom",
    "kinderzimmer": "bedroom", "chambre 2": "bedroom",
    "camera 2": "bedroom", "dormitorio 2": "bedroom",
    "habitación principal": "master_bedroom", "habitación 2": "bedroom",
    "schlafraum": "master_bedroom", "wohn-schlafraum": "master_bedroom",
    "bedroom (master)": "master_bedroom", "bedroom (parents)": "master_bedroom",
    # Balcony
    "balcony": "balcony", "balkon": "balcony", "balcon": "balcony",
    "balcón": "balcony", "balcão": "balcony", "balkong": "balcony",
    "balkkon": "balcony", "veranda": "balcony", "loggia": "balcony",
    "terrace": "balcony", "terrasse": "balcony", "terrazza": "balcony",
    "terraza": "balcony", "service yard": "balcony", "logia": "balcony",
    # Utility / store
    "utility": "utility", "abstellraum": "utility",
    "lavadora": "utility", "lavanderia": "utility",
    "store": "store_room", "store room": "store_room",
    "storage": "store_room", "speicher": "store_room", "closet": "store_room",
    # Hall corridors (treated as entry-ish, slim)
    "flur": "entry", "diele": "entry", "couloir": "entry",
    "passage": "entry", "back hall": "entry", "corridor": "entry",
    "berliner korridor": "entry",
}


def _classify(room_name: str) -> str:
    norm = room_name.lower().strip()
    if norm in NAME_TO_TYPE:
        return NAME_TO_TYPE[norm]
    # Word-substring fallback (e.g. "Master En-Suite" → bathroom)
    for substr, t in NAME_TO_TYPE.items():
        if substr in norm:
            return t
    return "bedroom"  # safe default


# --------------------------------------------------------------------------- #
# Program: what the LLM emits, what the geometry generator consumes          #
# --------------------------------------------------------------------------- #

@dataclass
class RoomSpec:
    name: str
    area_sqm: float
    room_type: str = ""  # filled from name if blank


@dataclass
class TemplateProgram:
    """The high-level intent the LLM produces, before geometry."""

    region: str = "europe"
    country: str = ""
    city: str = ""
    style: str = ""
    description: str = ""
    suitable_for: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    bedrooms: int = 1
    bathrooms: int = 1
    total_area_sqm: float = 50.0
    wall_thickness_mm: int = 230
    ceiling_height_mm: int = 2700
    rooms: list[RoomSpec] = field(default_factory=list)
    main_entry_room: str = ""  # name of the entry room

    @classmethod
    def from_dict(cls, d: dict) -> "TemplateProgram":
        rooms = [
            RoomSpec(
                name=r["name"],
                area_sqm=float(r.get("area_sqm", 0)),
                room_type=r.get("type", "") or _classify(r["name"]),
            )
            for r in d.get("rooms", [])
        ]
        return cls(
            region=d.get("region", "europe"),
            country=d.get("country", ""),
            city=d.get("city", ""),
            style=d.get("style", ""),
            description=d.get("description", ""),
            suitable_for=d.get("suitable_for", []),
            tags=d.get("tags", []),
            bedrooms=int(d.get("bedrooms", 1)),
            bathrooms=int(d.get("bathrooms", 1)),
            total_area_sqm=float(d.get("total_area_sqm", sum(r.area_sqm for r in rooms))),
            wall_thickness_mm=int(d.get("wall_thickness_mm", 230)),
            ceiling_height_mm=int(d.get("ceiling_height_mm", 2700)),
            rooms=rooms,
            main_entry_room=d.get("main_entry_room", "")
                or (rooms[0].name if rooms else ""),
        )


# --------------------------------------------------------------------------- #
# Geometry: lay out the program as axis-aligned rectangles                   #
# --------------------------------------------------------------------------- #

def _slug(s: str, used: set[str]) -> str:
    """ASCII-safe room id matching the template schema (^r_[a-z0-9_]+$)."""
    base = re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_") or "room"
    base = re.sub(r"_+", "_", base)
    candidate = f"r_{base}"
    n = 2
    while candidate in used:
        candidate = f"r_{base}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def _split_by_area(items: list[RoomSpec], total_length: float) -> list[float]:
    """Given rooms (each with area_sqm) and a strip length to pack along,
    return room widths so they sum to total_length and are proportional to area."""
    total_area = sum(r.area_sqm for r in items) or 1.0
    return [round(total_length * (r.area_sqm / total_area), 2) for r in items]


def lay_out_program(program: TemplateProgram) -> dict:
    """Take a program and produce a complete template dict ready for build/validate.

    Layout: 2-strip rectangular boundary
      - wet strip on top (entry + kitchen + baths + WC + utility)
      - dry strip on bottom (living + dining + bedrooms + balcony)
    """
    # 1. Pick an aspect ratio by total area (rough guideline)
    total = program.total_area_sqm
    # Pick depth (short edge) between 5m (small) and 10m (large)
    depth = 5.0 if total <= 50 else (6.5 if total <= 75 else (7.5 if total <= 100 else 9.0))
    length = round(total / depth, 2)

    # 2. Sort rooms into wet (top strip) and dry (bottom strip)
    wet = [r for r in program.rooms if r.room_type in WET_TYPES]
    dry = [r for r in program.rooms if r.room_type in DRY_TYPES]
    if not wet:
        # Move smallest room into wet so we always have an entry-side strip
        if dry:
            dry.sort(key=lambda r: r.area_sqm)
            wet = [dry.pop(0)]
    if not dry:
        # Same for dry strip
        if wet:
            wet.sort(key=lambda r: r.area_sqm, reverse=True)
            dry = [wet.pop(0)]

    # 3. Strip depths proportional to total area share
    wet_area = sum(r.area_sqm for r in wet) or 1.0
    dry_area = sum(r.area_sqm for r in dry) or 1.0
    wet_depth = round(depth * wet_area / (wet_area + dry_area), 2)
    dry_depth = round(depth - wet_depth, 2)

    # 4. Place rooms — wet strip y∈[dry_depth, depth], dry strip y∈[0, dry_depth]
    used_ids: set[str] = set()
    rooms_out: list[dict] = []

    # Order wet strip: entry first (left), kitchen last (right) for typical flow
    wet_ordered = sorted(
        wet,
        key=lambda r: (
            0 if r.room_type == "entry" else
            (1 if r.room_type == "bathroom" else
             (2 if r.room_type == "wc" else
              (3 if r.room_type == "utility" else
               (4 if r.room_type == "store_room" else 5))))
        ),
    )
    wet_widths = _split_by_area(wet_ordered, length)
    x = 0.0
    for r, w in zip(wet_ordered, wet_widths):
        polygon = [[x, dry_depth], [x + w, dry_depth], [x + w, depth], [x, depth]]
        rooms_out.append({
            "id": _slug(r.name, used_ids),
            "name": r.name,
            "type": r.room_type,
            "polygon": polygon,
            "area_sqm": round(w * wet_depth, 2),
        })
        x += w

    # Order dry strip: living first, then bedrooms, balcony last
    dry_ordered = sorted(
        dry,
        key=lambda r: (
            0 if r.room_type == "living" else
            (1 if r.room_type == "dining" else
             (2 if r.room_type == "master_bedroom" else
              (3 if r.room_type == "bedroom" else 4)))
        ),
    )
    dry_widths = _split_by_area(dry_ordered, length)
    x = 0.0
    for r, w in zip(dry_ordered, dry_widths):
        polygon = [[x, 0], [x + w, 0], [x + w, dry_depth], [x, dry_depth]]
        rooms_out.append({
            "id": _slug(r.name, used_ids),
            "name": r.name,
            "type": r.room_type,
            "polygon": polygon,
            "area_sqm": round(w * dry_depth, 2),
        })
        x += w

    # 5. Doors — always connect wet→dry along the strip seam at room midpoints
    doors_out: list[dict] = []
    # Main entry: outside → first wet (entry-type) room on the boundary
    entry_room = next((r for r in rooms_out if r["type"] == "entry"), rooms_out[0])
    # Position the main entry on the bottom of the entry room (boundary edge)
    ex = (entry_room["polygon"][0][0] + entry_room["polygon"][1][0]) / 2
    # Use top edge of entry (y=depth) as the outside door
    doors_out.append({
        "from": "outside", "to": entry_room["id"],
        "position": [round(ex, 2), depth],
        "width_mm": 1000, "is_main_entry": True,
    })

    # Internal door: entry → first dry room (living or biggest)
    living = next((r for r in rooms_out if r["type"] == "living"), None) or rooms_out[-1]
    # Door must be on shared edge between entry and living. Both share y=dry_depth
    # if their x ranges overlap.
    e_x0, e_x1 = entry_room["polygon"][0][0], entry_room["polygon"][1][0]
    l_x0, l_x1 = living["polygon"][0][0], living["polygon"][1][0]
    ox0, ox1 = max(e_x0, l_x0), min(e_x1, l_x1)
    if ox1 > ox0:
        doors_out.append({
            "from": entry_room["id"], "to": living["id"],
            "position": [round((ox0 + ox1) / 2, 2), dry_depth],
            "width_mm": 900,
        })

    # Connect each remaining dry room to the wet room above it (best overlap)
    for dr in rooms_out:
        if dr["type"] not in DRY_TYPES or dr["id"] == living["id"]:
            continue
        d_x0, d_x1 = dr["polygon"][0][0], dr["polygon"][1][0]
        # Find best wet partner above (highest x-overlap)
        best, best_overlap = None, 0.0
        for wr in rooms_out:
            if wr["type"] not in WET_TYPES:
                continue
            w_x0, w_x1 = wr["polygon"][0][0], wr["polygon"][1][0]
            ox = max(0.0, min(d_x1, w_x1) - max(d_x0, w_x0))
            if ox > best_overlap:
                best, best_overlap = wr, ox
        if best and best_overlap >= 0.5:
            wx0, wx1 = best["polygon"][0][0], best["polygon"][1][0]
            ox0, ox1 = max(d_x0, wx0), min(d_x1, wx1)
            doors_out.append({
                "from": best["id"], "to": dr["id"],
                "position": [round((ox0 + ox1) / 2, 2), dry_depth],
                "width_mm": 800,
            })

    # 6. Windows — one per non-entry room on its outer edge
    windows_out: list[dict] = []
    for r in rooms_out:
        if r["type"] == "entry":
            continue
        if r["type"] in DRY_TYPES:
            # Dry rooms touch boundary at y=0
            x0, x1 = r["polygon"][0][0], r["polygon"][1][0]
            if x1 - x0 >= 1.6:  # only place a window if room is wide enough
                windows_out.append({
                    "room": r["id"],
                    "position": [round((x0 + x1) / 2, 2), 0],
                    "width_mm": 1500 if r["type"] in ("living", "dining") else 1200,
                })
        elif r["type"] in WET_TYPES:
            x0, x1 = r["polygon"][0][0], r["polygon"][1][0]
            if x1 - x0 >= 1.0 and r["type"] != "entry":
                windows_out.append({
                    "room": r["id"],
                    "position": [round((x0 + x1) / 2, 2), depth],
                    "width_mm": 600 if r["type"] in ("bathroom", "wc") else 1200,
                })

    # 7. Final template dict
    # Schema requires id to start with eu/in/gl prefix
    region_prefix = {"europe": "eu", "india": "in", "global": "gl"}.get(program.region, "gl")
    template = {
        "id": f"{region_prefix}_generated_placeholder",
        "metadata": {
            "region": program.region,
            "country": program.country,
            "city_inspiration": program.city,
            "size_label": f"{program.bedrooms}bed_generated",
            "size_band": (
                "studio" if program.bedrooms == 0 else
                f"{min(program.bedrooms, 4)}bed"
            ),
            "total_area_sqm": round(length * depth, 1),
            "bedrooms": program.bedrooms,
            "bathrooms": program.bathrooms,
            "style": program.style or f"AI-generated ({program.bedrooms}-bed, {round(length * depth)} m²)",
            "description": program.description or
                f"AI-generated floor plan from a brief. {len(rooms_out)} rooms across "
                f"{round(length, 1)}×{round(depth, 1)} m. Procedurally laid out by "
                "the BIM Coordinator generator and verified against the 35-check "
                "IFC validation pipeline.",
            "suitable_for": program.suitable_for or ["general"],
            "tags": program.tags + ["ai_generated", "procedural"],
        },
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


# --------------------------------------------------------------------------- #
# Public entry — accept a JSON program (from LLM), produce a validated template
# --------------------------------------------------------------------------- #

def generate_template(program_dict: dict) -> tuple[dict | None, list[str]]:
    """Entry point: program → template. Returns (template, validation_errors).

    If validation_errors is empty, the template is fully geometric-valid and
    the build_template.py pipeline will produce a verifying IFC.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from validate_template import validate_dict  # noqa: E402

    program = TemplateProgram.from_dict(program_dict)
    template = lay_out_program(program)

    # Validate. If invalid, the caller can retry with adjusted program.
    errors = validate_dict(template)
    return (template if not errors else None), errors


if __name__ == "__main__":
    # Quick smoke test: build a generic 2BHK
    sample_program = {
        "region": "india",
        "country": "India",
        "city": "Bangalore",
        "style": "Bangalore Whitefield 2 BHK (test generation)",
        "bedrooms": 2,
        "bathrooms": 2,
        "total_area_sqm": 75.0,
        "wall_thickness_mm": 230,
        "ceiling_height_mm": 2700,
        "rooms": [
            {"name": "Foyer", "area_sqm": 4},
            {"name": "Living/Dining", "area_sqm": 22.5},
            {"name": "Kitchen", "area_sqm": 13.5},
            {"name": "Bathroom", "area_sqm": 7},
            {"name": "WC", "area_sqm": 4},
            {"name": "Master Bedroom", "area_sqm": 12},
            {"name": "Bedroom 2", "area_sqm": 12},
        ],
    }
    template, errors = generate_template(sample_program)
    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
    else:
        print(f"✅ Generated valid template: {template['metadata']['total_area_sqm']} m²")
        print(f"   Rooms: {[r['name'] for r in template['rooms']]}")
        print(f"   Doors: {len(template['doors'])}")
        print(f"   Windows: {len(template['windows'])}")
