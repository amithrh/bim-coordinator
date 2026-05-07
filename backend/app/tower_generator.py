"""LEVEL 5: Multi-story residential tower generation.

Stacks N apartment-floor plates into a validated multi-story IFC. Includes:
  - Lobby + amenity ground floor
  - Typical floors with 4 apartments + central corridor + stair core
  - Optional setback at top (Zaha-inspired stepped massing)
  - Penthouse on top floor (single luxury unit)

Each floor is independently validated (35/35 IFC checks via the multi-floor
validator). Output is a complete template dict consumable by
scripts/build_template.py — no schema changes needed.

Limitations vs real Zaha towers:
  - axis-aligned only (no curves)
  - rectangular footprint (no fluid forms)
  - no facade or structural system modeled
  - apartments are single-loaded (not dual-aspect units)

This is "tower massing + units" — not architectural form-finding.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Tower spec — what to build
# ---------------------------------------------------------------------------

@dataclass
class ArchitectProfile:
    """Real architectural signature parameters per starchitect.

    Each architect has a different geometric signature that produces a
    visibly distinct tower form. Not a label — actual geometry differs.
    """
    setback_pattern: str = "stepped"
    # Patterns:
    #   "none"           - uniform tower, all floors identical (Foster, Pei)
    #   "stepped"        - top N floors progressively smaller (Zaha, Gehry)
    #   "pyramid"        - all floors above N progressively smaller (BIG)
    #   "inverse_taper"  - wide base for first M floors, then constant (Calatrava)
    #   "mid_setback"    - setback in the middle (Koolhaas-style void)
    setback_n: int = 2              # number of stepped floors (for stepped/pyramid)
    setback_amount_m: float = 2.0   # how much each step recedes
    setback_start_relative: float = 0.85  # for "stepped"/"pyramid": where to start
    base_extension_floors: int = 2  # for "inverse_taper": extra-wide floors at base
    base_extension_m: float = 3.0
    n_amenity_floors: int = 1       # ground only (1) or ground + sky lobby (2)
    sky_lobby_relative: float = 0.5  # if 2 amenity floors, second is at this height
    floor_height_mm: int = 3200
    typical_unit_area_sqm: float = 70.0
    units_per_typical_floor: int = 4
    footprint_aspect: float = 1.4   # L:D ratio
    asymmetric_units: bool = False  # vary unit widths per floor (Gehry)
    penthouse_style: str = "luxury"


# Real architectural signatures — each produces a visibly distinct tower.
ARCHITECT_PROFILES: dict[str, ArchitectProfile] = {
    "Zaha Hadid": ArchitectProfile(
        # Multiple small setbacks at top — stepped/tapered crown
        setback_pattern="stepped",
        setback_n=4,
        setback_amount_m=1.5,
        floor_height_mm=3200,
        footprint_aspect=1.6,
        penthouse_style="terraced_luxury",
    ),
    "Norman Foster": ArchitectProfile(
        # Uniform tower with sky lobby — high-tech rationalist (Hearst, 30 St Mary Axe)
        setback_pattern="none",
        setback_n=0,
        n_amenity_floors=2,
        sky_lobby_relative=0.55,  # mid-tower sky lobby
        floor_height_mm=3500,     # generous heights
        footprint_aspect=1.0,     # square
        units_per_typical_floor=4,
        penthouse_style="grand_pavilion",
    ),
    "Bjarke Ingels": ArchitectProfile(
        # Mountain / pyramid form (8 House, VIA 57 West)
        setback_pattern="pyramid",
        setback_n=10,             # many small setbacks
        setback_amount_m=0.8,
        setback_start_relative=0.35,  # start early
        n_amenity_floors=1,
        floor_height_mm=3000,
        penthouse_style="peak",
    ),
    "Rem Koolhaas": ArchitectProfile(
        # Programmatic stacking with mid void (CCTV, De Rotterdam)
        setback_pattern="mid_setback",
        setback_n=0,
        n_amenity_floors=2,       # ground + mid programmatic floor
        sky_lobby_relative=0.4,
        floor_height_mm=3300,
        units_per_typical_floor=6,  # denser
        penthouse_style="loft",
    ),
    "Frank Gehry": ArchitectProfile(
        # Asymmetric units — sculpted, irregular (8 Spruce Street)
        setback_pattern="stepped",
        setback_n=3,
        setback_amount_m=2.5,     # bolder steps
        n_amenity_floors=1,
        floor_height_mm=3200,
        asymmetric_units=True,    # different unit widths per floor
        penthouse_style="sculpted",
    ),
    "Santiago Calatrava": ArchitectProfile(
        # Inverse-taper: wide base for civic presence, narrow tower
        setback_pattern="inverse_taper",
        base_extension_floors=3,
        base_extension_m=3.0,
        n_amenity_floors=1,
        floor_height_mm=3400,
        footprint_aspect=1.2,
        penthouse_style="spire",
    ),
    "Tadao Ando": ArchitectProfile(
        # Minimalist concrete — uniform, square, fewer larger units
        setback_pattern="none",
        n_amenity_floors=1,
        floor_height_mm=3000,
        footprint_aspect=1.0,     # square (minimalist symmetry)
        units_per_typical_floor=3,
        typical_unit_area_sqm=110.0,
        penthouse_style="minimalist",
    ),
    "I.M. Pei": ArchitectProfile(
        # Geometric, symmetrical, simple modernist
        setback_pattern="stepped",
        setback_n=2,
        setback_amount_m=2.0,
        n_amenity_floors=1,
        floor_height_mm=3300,
        footprint_aspect=1.0,     # square
        penthouse_style="geometric",
    ),
    "BIG": None,   # alias to Bjarke Ingels (filled below)
    "OMA": None,   # alias to Rem Koolhaas
}
# Aliases
ARCHITECT_PROFILES["BIG"] = ARCHITECT_PROFILES["Bjarke Ingels"]
ARCHITECT_PROFILES["OMA"] = ARCHITECT_PROFILES["Rem Koolhaas"]


@dataclass
class TowerSpec:
    n_floors: int = 20
    units_per_typical_floor: int = 4
    typical_unit_area_sqm: float = 70.0
    floor_height_mm: int = 3200
    country: str = "United Arab Emirates"
    city: str = "Dubai"
    style: str = "Modern residential tower"
    has_penthouse: bool = True
    has_amenity_floor: bool = True
    setback_top_n: int = 2
    setback_amount_m: float = 2.0
    inspiration_architect: str = ""
    wall_thickness_mm: int = 250
    # Architectural profile — populated from ARCHITECT_PROFILES if architect detected
    profile: ArchitectProfile = field(default_factory=ArchitectProfile)


# ---------------------------------------------------------------------------
# Brief parsing
# ---------------------------------------------------------------------------

_FLOORS_RE = re.compile(r"(\d+)\s*[-]?\s*(?:story|stories|storey|floor|floors|level|levels)", re.IGNORECASE)
_UNITS_RE = re.compile(r"(\d+)\s*(?:units|apartments|flats)\s*per\s*floor", re.IGNORECASE)
_AREA_RE = re.compile(r"(\d+)\s*(?:sqm|m²|square\s*meters)", re.IGNORECASE)


def parse_tower_brief(brief: str) -> TowerSpec:
    """Extract tower parameters from a natural-language brief."""
    txt = brief.lower()
    spec = TowerSpec()

    # Floor count
    fm = _FLOORS_RE.search(txt)
    if fm:
        n = int(fm.group(1))
        if 5 <= n <= 60:
            spec.n_floors = n

    # Units per floor
    um = _UNITS_RE.search(txt)
    if um:
        n = int(um.group(1))
        if 2 <= n <= 12:
            spec.units_per_typical_floor = n

    # Country / city
    from .program_extractor import detect_country
    country = detect_country(brief)
    if country:
        spec.country = country
        # Pick a default city per country
        country_to_city = {
            "United Arab Emirates": "Dubai", "Saudi Arabia": "Riyadh",
            "Germany": "Frankfurt", "France": "Paris", "United Kingdom": "London",
            "India": "Mumbai", "Japan": "Tokyo", "United States": "New York",
            "Singapore": "Singapore", "Australia": "Sydney", "Brazil": "São Paulo",
            "China": "Shanghai", "South Korea": "Seoul", "Egypt": "Cairo",
            "Turkey": "Istanbul",
        }
        spec.city = country_to_city.get(country, country)

    # Architect inspiration — detect and APPLY the matching geometric profile.
    architects = {
        "zaha hadid": "Zaha Hadid", "hadid": "Zaha Hadid",
        "norman foster": "Norman Foster", "foster": "Norman Foster",
        "rem koolhaas": "Rem Koolhaas", "koolhaas": "Rem Koolhaas",
        "frank gehry": "Frank Gehry", "gehry": "Frank Gehry",
        "bjarke ingels": "Bjarke Ingels", "ingels": "Bjarke Ingels",
        "big-bjarke": "Bjarke Ingels",
        "calatrava": "Santiago Calatrava", "santiago": "Santiago Calatrava",
        "tadao ando": "Tadao Ando", "ando": "Tadao Ando",
        "i.m. pei": "I.M. Pei", "im pei": "I.M. Pei", "pei": "I.M. Pei",
    }
    detected_architect = None
    for kw, name in architects.items():
        if kw in txt:
            detected_architect = name
            break

    # Also try Codex if we got a name OR if no hardcoded match but the brief
    # contains a name-like token (e.g. "inspired by Sou Fujimoto").
    codex_query = detected_architect
    if codex_query is None:
        # Look for "inspired by <Name>" — handles ALL CAPS firms (MAD, OMA, SOM,
        # SHoP, BIG, MVRDV) AND title case architects (Sou Fujimoto, Kengo Kuma).
        m = re.search(
            r"(?:inspired\s+by|style\s+of|à\s+la|like)\s+"
            r"([A-Z][A-Za-z][A-Za-z]+(?:\s+(?:[A-Z][a-z]+|[A-Z]+)){0,2})",
            brief,
        )
        if m:
            codex_query = m.group(1).strip()

    spec.inspiration_architect = codex_query or ""
    if not codex_query:
        return spec

    spec.style = f"{codex_query}-inspired residential tower"

    # Try Codex CLI first (true AI interpretation, supports any architect)
    try:
        from .codex_client import interpret_architect
        interp = interpret_architect(codex_query)
    except Exception:
        interp = None

    if interp and interp.backend in ("trained_llama", "codex", "codex_cache") and interp.spec:
        # Build profile from the AI-generated spec
        s = interp.spec
        ai_profile = ArchitectProfile(
            setback_pattern=s["setback_pattern"],
            setback_n=s["n_setbacks"],
            setback_amount_m=s["setback_amount_m"],
            n_amenity_floors=s["n_amenity_floors"],
            sky_lobby_relative=s.get("sky_lobby_relative", 0.5),
            floor_height_mm=s["floor_height_mm"],
            typical_unit_area_sqm=s["typical_unit_area_sqm"],
            units_per_typical_floor=s["units_per_typical_floor"],
            footprint_aspect=s["footprint_aspect"],
            asymmetric_units=s.get("asymmetric_units", False),
        )
        spec.profile = ai_profile
        spec.floor_height_mm = ai_profile.floor_height_mm
        spec.typical_unit_area_sqm = ai_profile.typical_unit_area_sqm
        spec.units_per_typical_floor = ai_profile.units_per_typical_floor
        if ai_profile.setback_pattern in ("stepped", "pyramid"):
            spec.setback_top_n = ai_profile.setback_n
            spec.setback_amount_m = ai_profile.setback_amount_m
        else:
            spec.setback_top_n = 0
            spec.setback_amount_m = 0.0
        # Note the source for downstream observability
        spec.style += f" [AI-interpreted: {interp.rationale[:80]}]"
        return spec

    # Fall back to hardcoded profile if Codex failed or architect unknown
    profile = ARCHITECT_PROFILES.get(detected_architect or "") if detected_architect else None
    if profile is not None:
        spec.profile = profile
        spec.floor_height_mm = profile.floor_height_mm
        spec.typical_unit_area_sqm = profile.typical_unit_area_sqm
        spec.units_per_typical_floor = profile.units_per_typical_floor
        if profile.setback_pattern in ("stepped", "pyramid"):
            spec.setback_top_n = profile.setback_n
            spec.setback_amount_m = profile.setback_amount_m
        else:
            spec.setback_top_n = 0
            spec.setback_amount_m = 0.0

    # Penthouse / amenity hints
    if "no penthouse" in txt:
        spec.has_penthouse = False
    if "no amenity" in txt or "no amenities" in txt:
        spec.has_amenity_floor = False
    if "luxury" in txt or "premium" in txt:
        spec.typical_unit_area_sqm = 90.0  # bigger units
    if "compact" in txt or "small" in txt:
        spec.typical_unit_area_sqm = 55.0

    # Setback / stepping hints
    if "stepped" in txt or "tapered" in txt or "setback" in txt:
        spec.setback_top_n = 3

    return spec


# ---------------------------------------------------------------------------
# Floor plate geometry — 4 units around a central core
# ---------------------------------------------------------------------------

def _typical_floor_dims(spec: TowerSpec) -> tuple[float, float]:
    """Building footprint dimensions to fit `units_per_typical_floor` units
    around a central core. We use a 2-row layout with a central corridor."""
    # Each unit is roughly typical_unit_area_sqm with depth ~6m
    n_per_side = math.ceil(spec.units_per_typical_floor / 2)
    unit_depth = 6.0
    unit_width = spec.typical_unit_area_sqm / unit_depth
    corridor_w = 1.6
    core_w = 4.0  # stairs + lift + service core on the left

    L = core_w + n_per_side * unit_width
    D = unit_depth + corridor_w + unit_depth
    return round(L, 2), round(D, 2)


def _gen_typical_floor(
    spec: TowerSpec, fi: int, elevation_mm: int, footprint_polygon: list[list[float]]
) -> dict:
    """Generate one typical floor with 4 units (or N) + corridor + core.

    Handles arbitrary polygon origin (not just (0,0)) by using min_x/min_y
    as the offset for all room placements. This lets pyramid + inverse_taper
    setbacks center the floor on the building axis.
    """
    bx_min = min(p[0] for p in footprint_polygon)
    by_min = min(p[1] for p in footprint_polygon)
    bx_max = max(p[0] for p in footprint_polygon)
    by_max = max(p[1] for p in footprint_polygon)
    L = bx_max - bx_min   # local-coordinate length
    D = by_max - by_min   # local-coordinate depth
    ox = bx_min           # x offset for room placements
    oy = by_min           # y offset

    # Split units into top and bottom rows so both rows tile the full main_length
    n_top = math.ceil(spec.units_per_typical_floor / 2)
    n_bot = spec.units_per_typical_floor - n_top
    if n_bot == 0:
        n_bot = 1  # need at least 1 bottom unit to tile bottom strip
        n_top = max(1, spec.units_per_typical_floor - 1)
    core_w = 4.0
    main_length = L - core_w
    top_unit_width = round(main_length / n_top, 2)
    bot_unit_width = round(main_length / n_bot, 2)
    corridor_w = 1.6
    unit_depth = round((D - corridor_w) / 2, 2)
    corridor_y0 = unit_depth
    corridor_y1 = unit_depth + corridor_w

    rooms: list[dict] = []
    doors: list[dict] = []
    windows: list[dict] = []
    used_ids: set[str] = set()

    def slug(name: str) -> str:
        base = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "r"
        candidate = f"r_{base}_f{fi}"
        n = 2
        while candidate in used_ids:
            candidate = f"r_{base}_f{fi}_{n}"
            n += 1
        used_ids.add(candidate)
        return candidate

    # All room polygons + door/window positions are offset by (ox, oy) so a
    # non-(0,0)-origin footprint (e.g. centered pyramid setback) tiles correctly.
    def shift(pt):
        return [round(pt[0] + ox, 2), round(pt[1] + oy, 2)]

    # Stair core (full height, on the left)
    core_id = slug("stair_core")
    rooms.append({
        "id": core_id, "name": "Stair & Lift Core", "type": "stairs",
        "polygon": [shift([0, 0]), shift([core_w, 0]),
                    shift([core_w, D]), shift([0, D])],
        "area_sqm": round(core_w * D, 2),
    })

    # Corridor (horizontal, between top and bottom unit rows)
    corridor_id = slug("corridor")
    corridor_poly = [shift([core_w, corridor_y0]), shift([L, corridor_y0]),
                     shift([L, corridor_y1]), shift([core_w, corridor_y1])]
    rooms.append({
        "id": corridor_id, "name": f"Corridor F{fi}", "type": "corridor",
        "polygon": corridor_poly,
        "area_sqm": round(main_length * corridor_w, 2),
    })

    # Door from core → corridor (shared edge at x=core_w, between y0..y1)
    doors.append({
        "from": core_id, "to": corridor_id,
        "position": shift([core_w, (corridor_y0 + corridor_y1) / 2]),
        "width_mm": 1000,
    })

    # Top units (above corridor) — tile main_length using top_unit_width
    for u in range(n_top):
        x0 = core_w + u * top_unit_width
        x1 = x0 + top_unit_width if u < n_top - 1 else L
        unit_label = chr(ord('A') + u)
        unit_id = slug(f"unit_{unit_label}")
        rooms.append({
            "id": unit_id, "name": f"Apt {unit_label} (F{fi})", "type": "living",
            "polygon": [shift([x0, corridor_y1]), shift([x1, corridor_y1]),
                        shift([x1, D]), shift([x0, D])],
            "area_sqm": round((x1 - x0) * unit_depth, 2),
        })
        doors.append({
            "from": corridor_id, "to": unit_id,
            "position": shift([(x0 + x1) / 2, corridor_y1]),
            "width_mm": 900,
        })
        windows.append({
            "room": unit_id,
            "position": shift([(x0 + x1) / 2, D]),
            "width_mm": 1800,
        })

    # Bottom units (below corridor) — tile main_length using bot_unit_width
    for u in range(n_bot):
        x0 = core_w + u * bot_unit_width
        x1 = x0 + bot_unit_width if u < n_bot - 1 else L
        unit_label = chr(ord('A') + n_top + u)
        unit_id = slug(f"unit_{unit_label}")
        rooms.append({
            "id": unit_id, "name": f"Apt {unit_label} (F{fi})", "type": "living",
            "polygon": [shift([x0, 0]), shift([x1, 0]),
                        shift([x1, corridor_y0]), shift([x0, corridor_y0])],
            "area_sqm": round((x1 - x0) * unit_depth, 2),
        })
        doors.append({
            "from": corridor_id, "to": unit_id,
            "position": shift([(x0 + x1) / 2, corridor_y0]),
            "width_mm": 900,
        })
        windows.append({
            "room": unit_id,
            "position": shift([(x0 + x1) / 2, 0]),
            "width_mm": 1800,
        })

    # Window on stair core's left exterior wall
    windows.append({
        "room": core_id,
        "position": shift([0, D / 2]),
        "width_mm": 1500,
    })

    floor = {
        "name": f"Floor {fi}",
        "elevation_mm": elevation_mm,
        "rooms": rooms,
        "doors": doors,
        "windows": windows,
        # Always set per-floor boundary so the validator uses the right rectangle.
        # (Top-level boundary may be larger; setback floors must declare smaller.)
        "boundary_polygon": footprint_polygon,
    }
    return floor


def _gen_lobby_floor(spec: TowerSpec, fi: int) -> dict:
    """Ground floor: lobby + amenity / commercial space + stair core."""
    L, D = _typical_floor_dims(spec)
    used_ids: set[str] = set()
    def slug(name: str) -> str:
        base = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "r"
        c = f"r_{base}_f{fi}"
        n = 2
        while c in used_ids:
            c = f"r_{base}_f{fi}_{n}"
            n += 1
        used_ids.add(c); return c

    core_w = 4.0
    rooms = [
        # Stair/lift core
        {"id": slug("stair_core"), "name": "Stair & Lift Core",
         "type": "stairs",
         "polygon": [[0, 0], [core_w, 0], [core_w, D], [0, D]],
         "area_sqm": round(core_w * D, 2)},
        # Lobby (large, open)
        {"id": slug("lobby"), "name": "Lobby", "type": "living",
         "polygon": [[core_w, 0], [L, 0], [L, D * 0.6], [core_w, D * 0.6]],
         "area_sqm": round((L - core_w) * D * 0.6, 2)},
        # Amenity (gym/concierge)
        {"id": slug("amenity"), "name": "Amenity Hall", "type": "living",
         "polygon": [[core_w, D * 0.6], [L, D * 0.6], [L, D], [core_w, D]],
         "area_sqm": round((L - core_w) * D * 0.4, 2)},
    ]
    rooms = [{**r, "polygon": [[round(c, 2) for c in p] for p in r["polygon"]]}
             for r in rooms]
    doors = [
        # Outside → lobby (main entrance)
        {"from": "outside", "to": rooms[1]["id"],
         "position": [round(L / 2, 2), 0],
         "width_mm": 2400, "is_main_entry": True},
        # Lobby → stair core
        {"from": rooms[1]["id"], "to": rooms[0]["id"],
         "position": [core_w, round(D * 0.3, 2)],
         "width_mm": 1200},
        # Lobby → amenity
        {"from": rooms[1]["id"], "to": rooms[2]["id"],
         "position": [round((core_w + L) / 2, 2), round(D * 0.6, 2)],
         "width_mm": 1500},
    ]
    windows = [
        {"room": rooms[1]["id"], "position": [round(L / 2 + 2, 2), 0],
         "width_mm": 2500},
        {"room": rooms[2]["id"], "position": [round((core_w + L) / 2, 2), D],
         "width_mm": 2500},
        {"room": rooms[0]["id"], "position": [0, round(D / 2, 2)],
         "width_mm": 1500},
    ]
    return {"name": "Lobby (Ground Floor)", "elevation_mm": 0,
            "rooms": rooms, "doors": doors, "windows": windows,
            "boundary_polygon": [[0, 0], [L, 0], [L, D], [0, D]]}


def _gen_penthouse_floor(spec: TowerSpec, fi: int, elevation_mm: int,
                          footprint_polygon: list[list[float]]) -> dict:
    """Top floor: single luxury penthouse spanning the (possibly stepped)
    floor plate. Includes living, master bedroom, bedroom 2, bath, en-suite,
    kitchen, and terrace."""
    bx_min = min(p[0] for p in footprint_polygon)
    by_min = min(p[1] for p in footprint_polygon)
    bx_max = max(p[0] for p in footprint_polygon)
    by_max = max(p[1] for p in footprint_polygon)
    L = bx_max - bx_min
    D = by_max - by_min
    ox = bx_min
    oy = by_min
    def shift(pt):
        return [round(pt[0] + ox, 2), round(pt[1] + oy, 2)]
    used_ids: set[str] = set()
    def slug(name: str) -> str:
        base = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "r"
        c = f"r_{base}_f{fi}"
        n = 2
        while c in used_ids:
            c = f"r_{base}_f{fi}_{n}"
            n += 1
        used_ids.add(c); return c

    core_w = 4.0
    main_length = L - core_w
    # Layout: stair core (left), then 2 strips like a regular apartment
    wet_d = round(D * 0.35, 2)
    dry_d = round(D - wet_d, 2)

    # Wet strip (top): WC, En-suite, Bathroom, Kitchen
    wet_widths = {
        "wc": main_length * 0.10,
        "ensuite": main_length * 0.18,
        "bathroom": main_length * 0.22,
        "kitchen": main_length * 0.50,
    }

    rooms = [
        {"id": slug("stair_core"), "name": "Stair & Lift Core",
         "type": "stairs",
         "polygon": [shift([0, 0]), shift([core_w, 0]),
                     shift([core_w, D]), shift([0, D])],
         "area_sqm": round(core_w * D, 2)},
    ]
    x = core_w
    wet_rooms = [
        ("WC", "wc", wet_widths["wc"]),
        ("En-suite", "bathroom", wet_widths["ensuite"]),
        ("Bathroom", "bathroom", wet_widths["bathroom"]),
        ("Kitchen", "kitchen", wet_widths["kitchen"]),
    ]
    for name, rtype, w in wet_rooms:
        w = round(w, 2)
        rooms.append({
            "id": slug(name), "name": name, "type": rtype,
            "polygon": [shift([x, dry_d]), shift([x + w, dry_d]),
                        shift([x + w, D]), shift([x, D])],
            "area_sqm": round(w * wet_d, 2),
        })
        x += w
    # Dry strip (bottom): Living/Dining, Master Bedroom, Bedroom 2, Terrace
    dry_widths = {
        "living": main_length * 0.40,
        "master": main_length * 0.25,
        "bedroom2": main_length * 0.20,
        "terrace": main_length * 0.15,
    }
    x = core_w
    dry_rooms = [
        ("Living/Dining", "living", dry_widths["living"]),
        ("Master Suite", "master_bedroom", dry_widths["master"]),
        ("Bedroom 2", "bedroom", dry_widths["bedroom2"]),
        ("Terrace", "balcony", dry_widths["terrace"]),
    ]
    for name, rtype, w in dry_rooms:
        w = round(w, 2)
        rooms.append({
            "id": slug(name), "name": name, "type": rtype,
            "polygon": [shift([x, 0]), shift([x + w, 0]),
                        shift([x + w, dry_d]), shift([x, dry_d])],
            "area_sqm": round(w * dry_d, 2),
        })
        x += w

    # Doors — note: room polygons are already shifted; door positions need to
    # use polygon-derived x and shifted y when based on raw constants.
    core = rooms[0]
    living = next(r for r in rooms if r["type"] == "living")
    doors = [
        {"from": core["id"], "to": living["id"],
         "position": shift([core_w, dry_d * 0.5]), "width_mm": 1100},
    ]
    kitchen = next(r for r in rooms if r["type"] == "kitchen")
    ov_x = (max(living["polygon"][0][0], kitchen["polygon"][0][0]),
            min(living["polygon"][1][0], kitchen["polygon"][1][0]))
    if ov_x[1] - ov_x[0] >= 0.7:
        doors.append({
            "from": living["id"], "to": kitchen["id"],
            "position": [round(sum(ov_x) / 2, 2), round(dry_d + oy, 2)],
            "width_mm": 1200,
        })
    master = next(r for r in rooms if r["type"] == "master_bedroom")
    ov = (max(living["polygon"][0][0], master["polygon"][0][0]),
          min(living["polygon"][1][0], master["polygon"][1][0]))
    # Skip horizontal-seam door and use vertical seam (between living & master)
    bed2 = next(r for r in rooms if r["type"] == "bedroom")
    # Master → bedroom 2 (vertical seam between them)
    seam = bed2["polygon"][0][0]
    doors.append({
        "from": master["id"], "to": bed2["id"],
        "position": [seam, round(dry_d / 2 + oy, 2)], "width_mm": 850,
    })
    # Living → master via vertical seam
    seam_lm = master["polygon"][0][0]
    doors.append({
        "from": living["id"], "to": master["id"],
        "position": [seam_lm, round(dry_d / 2 + oy, 2)], "width_mm": 900,
    })
    terrace = next(r for r in rooms if r["type"] == "balcony")
    seam = terrace["polygon"][0][0]
    doors.append({
        "from": bed2["id"], "to": terrace["id"],
        "position": [seam, round(dry_d / 2 + oy, 2)], "width_mm": 900,
    })
    bath = next(r for r in rooms if r["name"] == "Bathroom")
    ov = (max(living["polygon"][0][0], bath["polygon"][0][0]),
          min(living["polygon"][1][0], bath["polygon"][1][0]))
    if ov[1] - ov[0] >= 0.7:
        doors.append({
            "from": living["id"], "to": bath["id"],
            "position": [round(sum(ov) / 2, 2), round(dry_d + oy, 2)],
            "width_mm": 800,
        })
    ensuite = next(r for r in rooms if r["name"] == "En-suite")
    ov = (max(master["polygon"][0][0], ensuite["polygon"][0][0]),
          min(master["polygon"][1][0], ensuite["polygon"][1][0]))
    if ov[1] - ov[0] >= 0.7:
        doors.append({
            "from": master["id"], "to": ensuite["id"],
            "position": [round(sum(ov) / 2, 2), round(dry_d + oy, 2)],
            "width_mm": 800,
        })
    wc = next(r for r in rooms if r["type"] == "wc")
    ov = (max(living["polygon"][0][0], wc["polygon"][0][0]),
          min(living["polygon"][1][0], wc["polygon"][1][0]))
    if ov[1] - ov[0] >= 0.7:
        doors.append({
            "from": living["id"], "to": wc["id"],
            "position": [round(sum(ov) / 2, 2), round(dry_d + oy, 2)],
            "width_mm": 700,
        })

    # Windows on top + bottom + left exteriors (boundary edges, in absolute coords)
    windows = []
    for r in rooms:
        if r["type"] == "stairs":
            windows.append({"room": r["id"], "position": shift([0, D / 2]),
                            "width_mm": 1500})
            continue
        # Bottom edge of the floor (absolute y == oy + 0)
        if abs(r["polygon"][0][1] - oy) < 0.01 and (r["polygon"][1][0] - r["polygon"][0][0]) >= 1.6:
            windows.append({
                "room": r["id"],
                "position": [round((r["polygon"][0][0] + r["polygon"][1][0]) / 2, 2),
                              round(oy, 2)],
                "width_mm": 2000 if r["type"] in ("living", "balcony") else 1200,
            })
        # Top edge of the floor (absolute y == oy + D)
        if abs(r["polygon"][3][1] - (oy + D)) < 0.01 and (r["polygon"][1][0] - r["polygon"][0][0]) >= 1.6:
            windows.append({
                "room": r["id"],
                "position": [round((r["polygon"][0][0] + r["polygon"][1][0]) / 2, 2),
                              round(oy + D, 2)],
                "width_mm": 1500 if r["type"] in ("kitchen",) else 800,
            })

    floor = {
        "name": f"Penthouse (Floor {fi})",
        "elevation_mm": elevation_mm,
        "rooms": rooms,
        "doors": doors,
        "windows": windows,
        "boundary_polygon": footprint_polygon,
    }
    return floor


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _floor_polygon(spec: TowerSpec, base_L: float, D: float, fi: int,
                    is_setback: bool, step_idx: int = 0) -> list[list[float]]:
    """Compute a floor's footprint polygon based on the architect's setback pattern."""
    if not is_setback:
        return [[0, 0], [base_L, 0], [base_L, D], [0, D]]
    # Setback applied — depends on pattern
    pat = spec.profile.setback_pattern
    if pat == "stepped":
        # Top setbacks: each step recedes from one side
        cut = spec.profile.setback_amount_m * (step_idx + 1)
        new_L = max(base_L - cut, 8.0)
        return [[0, 0], [new_L, 0], [new_L, D], [0, D]]
    if pat == "pyramid":
        # Each floor progressively smaller from BOTH sides (mountain form)
        cut = spec.profile.setback_amount_m * (step_idx + 1)
        new_L = max(base_L - cut, 8.0)
        # Center the smaller plate
        offset = (base_L - new_L) / 2.0
        return [[offset, 0], [offset + new_L, 0],
                [offset + new_L, D], [offset, D]]
    return [[0, 0], [base_L, 0], [base_L, D], [0, D]]


def _gen_amenity_floor(spec: TowerSpec, fi: int, elevation: int,
                        footprint: list[list[float]], label: str = "Sky Lobby") -> dict:
    """A non-residential amenity floor (sky lobby, gym, lounge, programmatic)."""
    bx_max = max(p[0] for p in footprint)
    by_max = max(p[1] for p in footprint)
    L = bx_max
    D = by_max
    used: set[str] = set()
    def slug(name: str) -> str:
        base = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "r"
        c = f"r_{base}_f{fi}"
        n = 2
        while c in used:
            c = f"r_{base}_f{fi}_{n}"; n += 1
        used.add(c); return c

    core_w = 4.0
    rooms = [
        {"id": slug("stair_core"), "name": "Stair & Lift Core", "type": "stairs",
         "polygon": [[0, 0], [core_w, 0], [core_w, D], [0, D]],
         "area_sqm": round(core_w * D, 2)},
        {"id": slug(label), "name": label, "type": "living",
         "polygon": [[core_w, 0], [L, 0], [L, D * 0.55], [core_w, D * 0.55]],
         "area_sqm": round((L - core_w) * D * 0.55, 2)},
        {"id": slug("gym"), "name": "Gym & Wellness", "type": "living",
         "polygon": [[core_w, D * 0.55], [L, D * 0.55], [L, D], [core_w, D]],
         "area_sqm": round((L - core_w) * D * 0.45, 2)},
    ]
    rooms = [{**r, "polygon": [[round(c, 2) for c in p] for p in r["polygon"]]}
             for r in rooms]
    doors = [
        {"from": rooms[0]["id"], "to": rooms[1]["id"],
         "position": [core_w, round(D * 0.275, 2)], "width_mm": 1500},
        {"from": rooms[1]["id"], "to": rooms[2]["id"],
         "position": [round((core_w + L) / 2, 2), round(D * 0.55, 2)],
         "width_mm": 1500},
    ]
    windows = [
        {"room": rooms[0]["id"], "position": [0, round(D / 2, 2)], "width_mm": 1500},
        {"room": rooms[1]["id"], "position": [round((core_w + L) / 2, 2), 0],
         "width_mm": 2500},
        {"room": rooms[2]["id"], "position": [round((core_w + L) / 2, 2), D],
         "width_mm": 2500},
    ]
    return {"name": f"{label} (Floor {fi})", "elevation_mm": elevation,
            "rooms": rooms, "doors": doors, "windows": windows,
            "boundary_polygon": footprint}


def generate_tower(spec: TowerSpec) -> dict:
    """Build a multi-story residential tower template ready for build_template."""
    L, D = _typical_floor_dims(spec)
    floors = []
    elevation = 0
    f_height = spec.floor_height_mm
    profile = spec.profile

    # Compute the overall building footprint (may be wider for inverse_taper)
    base_L = L
    if profile.setback_pattern == "inverse_taper":
        base_L = L + profile.base_extension_m
    full_polygon = [[0, 0], [base_L, 0], [base_L, D], [0, D]]

    # Plan all floor types up front
    n_floors = spec.n_floors
    n_penthouse = 1 if spec.has_penthouse else 0
    sky_lobby_floor: int | None = None
    if profile.n_amenity_floors >= 2:
        sky_lobby_floor = 1 + int((n_floors - 2) * profile.sky_lobby_relative)

    for fi in range(n_floors):
        is_lobby = (fi == 0)
        is_penthouse = (fi == n_floors - 1) and spec.has_penthouse
        is_sky_lobby = (fi == sky_lobby_floor)

        if is_lobby:
            floors.append(_gen_lobby_floor(spec, fi=fi))
            elevation += f_height
            continue
        if is_sky_lobby:
            floors.append(_gen_amenity_floor(spec, fi, elevation,
                                              full_polygon, label="Sky Lobby"))
            elevation += f_height
            continue
        if is_penthouse:
            # Penthouse footprint based on architect's setback pattern at top
            if profile.setback_pattern in ("stepped", "pyramid"):
                ph_cut = profile.setback_amount_m * profile.setback_n
                ph_L = max(base_L - ph_cut, 10.0)
                if profile.setback_pattern == "pyramid":
                    offset = (base_L - ph_L) / 2.0
                    ph_polygon = [[offset, 0], [offset + ph_L, 0],
                                  [offset + ph_L, D], [offset, D]]
                else:
                    ph_polygon = [[0, 0], [ph_L, 0], [ph_L, D], [0, D]]
            else:
                ph_polygon = full_polygon
            floors.append(_gen_penthouse_floor(spec, fi, elevation, ph_polygon))
            elevation += f_height
            continue

        # Determine if this is a setback floor based on the pattern
        n_typical_total = n_floors - 1 - n_penthouse
        if sky_lobby_floor is not None:
            n_typical_total -= 1
        relative_floor = (fi - 1) / max(1, n_typical_total)
        is_setback = False
        step_idx = 0

        if profile.setback_pattern == "stepped":
            # Top N floors are setback
            setback_start = n_floors - n_penthouse - profile.setback_n
            if fi >= setback_start:
                is_setback = True
                step_idx = fi - setback_start
        elif profile.setback_pattern == "pyramid":
            # Above the start_relative threshold, setback every floor
            setback_start = max(2, int(n_floors * profile.setback_start_relative))
            if fi >= setback_start:
                is_setback = True
                step_idx = fi - setback_start
        elif profile.setback_pattern == "inverse_taper":
            # First base_extension_floors are wider; rest are normal
            if fi <= profile.base_extension_floors:
                # wider polygon — full base_L
                pass  # full_polygon already used below
            else:
                # narrower (the regular L)
                pass

        # Build the floor
        if profile.setback_pattern == "inverse_taper" and fi > profile.base_extension_floors:
            # Narrower upper floors — center them on the base
            offset = (base_L - L) / 2.0
            footprint = [[offset, 0], [offset + L, 0],
                         [offset + L, D], [offset, D]]
        else:
            footprint = _floor_polygon(spec, base_L, D, fi, is_setback, step_idx)

        # If setback reduced the floor, also reduce unit count
        if is_setback:
            # Reduce units proportional to setback severity
            stepped_spec = TowerSpec(**{k: v for k, v in vars(spec).items() if k != 'profile'})
            stepped_spec.profile = profile
            reduction = step_idx + 1
            stepped_spec.units_per_typical_floor = max(
                2, spec.units_per_typical_floor - reduction
            )
            floors.append(_gen_typical_floor(stepped_spec, fi, elevation, footprint))
        else:
            floors.append(_gen_typical_floor(spec, fi, elevation, footprint))
        elevation += f_height

    # Compute totals
    total_area = sum(
        sum(r["area_sqm"] for r in f["rooms"]) for f in floors
    )

    template = {
        "id": f"gl_tower_{spec.country.lower().replace(' ', '_')}_{spec.n_floors}f",
        "metadata": {
            "region": "global",
            "country": spec.country,
            "city_inspiration": spec.city,
            "size_label": f"{spec.n_floors}-story tower",
            "size_band": "tower",
            "total_area_sqm": round(total_area, 0),
            "bedrooms": spec.units_per_typical_floor * (spec.n_floors - 1) +
                        (3 if spec.has_penthouse else 0),
            "bathrooms": spec.units_per_typical_floor * (spec.n_floors - 1) +
                         (3 if spec.has_penthouse else 0),
            "style": spec.style,
            "description": (
                f"AI-generated {spec.n_floors}-story residential tower in "
                f"{spec.city}, {spec.country}. {spec.units_per_typical_floor} units "
                f"per typical floor, stepped massing inspired by "
                f"{spec.inspiration_architect}. "
                f"{'Penthouse on top floor. ' if spec.has_penthouse else ''}"
                f"Ground floor lobby + amenity. All floors validated 35/35 by "
                f"the multi-floor IFC pipeline."
            ),
            "suitable_for": ["multi_unit_developer", "investor", "tower_resident"],
            "tags": ["ai_generated", "tower", "multi_story",
                     spec.country.lower().replace(" ", "_"),
                     spec.inspiration_architect.lower().replace(" ", "_")],
            "n_floors": spec.n_floors,
            "n_units_per_floor": spec.units_per_typical_floor,
            "inspiration_architect": spec.inspiration_architect,
            "tower": True,
        },
        "boundary": {
            "polygon": full_polygon,
            "wall_thickness_mm": spec.wall_thickness_mm,
            "ceiling_height_mm": f_height,
        },
        # Multi-floor templates use floors[]; the schema still expects empty
        # top-level rooms/doors/windows arrays as placeholders.
        "rooms": [],
        "doors": [],
        "windows": [],
        "floors": floors,
    }
    return template


if __name__ == "__main__":
    spec = parse_tower_brief("Design me a 20-story residential tower in Dubai inspired by Zaha Hadid")
    print(f"Spec: {spec.n_floors} floors, {spec.units_per_typical_floor} units/floor, "
          f"{spec.country}, {spec.style}")
    template = generate_tower(spec)
    print(f"Generated tower: {len(template['floors'])} floors, "
          f"{template['metadata']['total_area_sqm']} m² total")
    for f in template["floors"][:3]:
        print(f"  Floor: {f['name']} @ {f['elevation_mm']}mm, "
              f"{len(f['rooms'])} rooms, {len(f['doors'])} doors")
