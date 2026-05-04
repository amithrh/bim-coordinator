"""Validate a floor-plan template JSON: schema check + geometry sanity.

Usage:
    python validate_template.py <path-to-template.json>
    # or programmatically:
    from validate_template import validate_dict
    errors = validate_dict(template)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jsonschema
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "data" / "_schema.json"

# 10mm grid + 1mm slack — matches build_template.ON_EDGE_TOL
ON_EDGE_TOL = 0.011
HOLE_AREA_TOL = 1.0  # sqm — flag unaccounted boundary area larger than this
AREA_SUM_RATIO = 0.05  # tighter than the original 0.20


def _load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text())


def _shape(polygon: list[list[float]]) -> ShapelyPolygon:
    return ShapelyPolygon(polygon)


def _polygon_edges(polygon: list[list[float]]):
    """Yield (p1, p2) tuples for each edge of a closed polygon."""
    pts = [tuple(p) for p in polygon]
    for i in range(len(pts)):
        yield (pts[i], pts[(i + 1) % len(pts)])


def _point_on_edge(p1, p2, point, tol: float = ON_EDGE_TOL) -> bool:
    """True if `point` lies on the edge p1->p2 within `tol`."""
    import math
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return False
    ux, uy = dx / L, dy / L
    qx, qy = point[0] - p1[0], point[1] - p1[1]
    perp = abs(qx * uy - qy * ux)
    if perp > tol:
        return False
    t = qx * ux + qy * uy
    return -tol <= t <= L + tol


def _point_on_any_edge(point, polygon, tol: float = ON_EDGE_TOL) -> bool:
    return any(_point_on_edge(p1, p2, point, tol) for p1, p2 in _polygon_edges(polygon))


def _validate_floor(floor_label: str,
                     boundary_polygon: list[list[float]],
                     rooms: list[dict],
                     doors: list[dict],
                     windows: list[dict],
                     declared_total_for_check: float | None = None) -> list[str]:
    """Per-floor geometry checks. `floor_label` prefixes errors so callers
    can tell which floor failed in a multi-floor template."""
    errors: list[str] = []
    boundary = _shape(boundary_polygon)
    if not boundary.is_valid:
        errors.append(f"{floor_label}geometry: boundary polygon is invalid (self-intersecting?)")
        return errors

    if not rooms:
        errors.append(f"{floor_label}rooms: at least one room required")
        return errors

    summed = sum(_shape(r["polygon"]).area for r in rooms)
    boundary_area = boundary.area
    if boundary_area > 0:
        ratio = abs(summed - boundary_area) / boundary_area
        if ratio > AREA_SUM_RATIO:
            errors.append(
                f"{floor_label}geometry: room area sum {summed:.2f} sqm differs from "
                f"boundary area {boundary_area:.2f} sqm by {ratio*100:.1f}% "
                f"(>{AREA_SUM_RATIO*100:.0f}%) — rooms do not tile the boundary"
            )
    rooms_union = unary_union([_shape(r["polygon"]) for r in rooms])
    leftover = boundary.difference(rooms_union)
    if leftover.area > HOLE_AREA_TOL:
        errors.append(
            f"{floor_label}geometry: {leftover.area:.2f} sqm of boundary is not covered "
            f"by any room (>{HOLE_AREA_TOL:g} sqm threshold)"
        )

    # Room overlap check
    shapes = {r["id"]: _shape(r["polygon"]) for r in rooms}
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            inter = shapes[a["id"]].intersection(shapes[b["id"]]).area
            if inter > 0.05:
                errors.append(
                    f"{floor_label}geometry: rooms {a['id']} and {b['id']} overlap "
                    f"by {inter:.3f} sqm"
                )

    # Rooms within boundary
    boundary_buffered = boundary.buffer(0.01)
    for r in rooms:
        rs = shapes[r["id"]]
        if not boundary_buffered.contains(rs):
            outside = rs.difference(boundary_buffered).area
            if outside > 0.05:
                errors.append(
                    f"{floor_label}geometry: room {r['id']} extends {outside:.3f} sqm "
                    f"outside boundary"
                )

    # Door + window edge checks
    room_polys = {r["id"]: r["polygon"] for r in rooms}
    room_ids = set(room_polys) | {"outside"}
    for i, d in enumerate(doors):
        if d["from"] not in room_ids:
            errors.append(f"{floor_label}door[{i}]: 'from' references unknown room/outside '{d['from']}'")
            continue
        if d["to"] not in room_ids:
            errors.append(f"{floor_label}door[{i}]: 'to' references unknown room/outside '{d['to']}'")
            continue
        pos = tuple(d["position"])
        for endpoint in (d["from"], d["to"]):
            if endpoint == "outside":
                if not _point_on_any_edge(pos, boundary_polygon):
                    errors.append(
                        f"{floor_label}door[{i}] ({d['from']}→{d['to']}) at {pos}: "
                        "'outside' endpoint requires position on the boundary"
                    )
            else:
                if not _point_on_any_edge(pos, room_polys[endpoint]):
                    errors.append(
                        f"{floor_label}door[{i}] ({d['from']}→{d['to']}) at {pos}: "
                        f"position is not on any edge of room '{endpoint}'"
                    )
    for i, w in enumerate(windows):
        if w["room"] not in room_polys:
            errors.append(f"{floor_label}window[{i}]: references unknown room '{w['room']}'")
            continue
        pos = tuple(w["position"])
        if not _point_on_any_edge(pos, room_polys[w["room"]]):
            errors.append(
                f"{floor_label}window[{i}] ({w['room']}) at {pos}: "
                f"position is not on any edge of room '{w['room']}'"
            )
    return errors


def validate_dict(template: dict[str, Any]) -> list[str]:
    """Return a list of error strings. Empty list means the template passes."""
    errors: list[str] = []

    # 1. Schema validation
    schema = _load_schema()
    validator = jsonschema.Draft7Validator(schema)
    for err in sorted(validator.iter_errors(template), key=lambda e: list(e.path)):
        path = "/".join(str(p) for p in err.path) or "<root>"
        errors.append(f"schema: {path}: {err.message}")
    if errors:
        return errors

    # 2. Multi-floor branch
    floors = template.get("floors")
    if floors:
        if template.get("rooms"):
            errors.append("structure: multi-floor template has both top-level rooms and floors[] — pick one")
        if len({f["name"] for f in floors}) != len(floors):
            errors.append("floors: floor names must be unique")
        # Validate at least one main entry across all floors
        all_main = sum(1 for f in floors for d in f.get("doors", []) if d.get("is_main_entry"))
        if all_main == 0:
            errors.append("door: no door has is_main_entry=true (across all floors)")
        # Per-floor checks
        boundary_default = template["boundary"]["polygon"]
        for fi, fl in enumerate(floors):
            label = f"floor[{fi}] '{fl['name']}': "
            poly = fl.get("boundary_polygon", boundary_default)
            errors.extend(_validate_floor(label, poly, fl["rooms"], fl.get("doors", []),
                                            fl.get("windows", [])))
        # Sanity: at least one bathroom or wc somewhere if total > 25 sqm
        total_room_count = sum(len(f["rooms"]) for f in floors)
        if total_room_count == 0:
            errors.append("rooms: no rooms defined across any floor")
        return errors

    boundary = _shape(template["boundary"]["polygon"])
    if not boundary.is_valid:
        errors.append("geometry: boundary polygon is invalid (self-intersecting?)")
        return errors

    # 2. Sum of room areas vs. boundary area (tighter than declared total).
    #    Also flag unaccounted holes inside the boundary.
    declared_total = float(template["metadata"]["total_area_sqm"])
    summed = sum(_shape(r["polygon"]).area for r in template["rooms"])
    boundary_area = boundary.area
    if boundary_area > 0:
        ratio = abs(summed - boundary_area) / boundary_area
        if ratio > AREA_SUM_RATIO:
            errors.append(
                f"geometry: room area sum {summed:.2f} sqm differs from boundary area "
                f"{boundary_area:.2f} sqm by {ratio*100:.1f}% (>{AREA_SUM_RATIO*100:.0f}%) "
                "— rooms do not tile the boundary"
            )
    # Hole detection: boundary minus union of all rooms
    rooms_union = unary_union([_shape(r["polygon"]) for r in template["rooms"]])
    leftover = boundary.difference(rooms_union)
    if leftover.area > HOLE_AREA_TOL:
        errors.append(
            f"geometry: {leftover.area:.2f} sqm of boundary is not covered by any "
            f"room (>{HOLE_AREA_TOL:g} sqm threshold) — unaccounted area"
        )
    # Sanity: declared total roughly matches summed
    if declared_total > 0:
        decl_ratio = abs(summed - declared_total) / declared_total
        if decl_ratio > 0.10:
            errors.append(
                f"metadata: declared total_area_sqm {declared_total:.1f} differs from "
                f"computed room sum {summed:.2f} by {decl_ratio*100:.1f}% (>10%)"
            )

    # 3. No two rooms overlap by more than 0.05 sqm
    rooms = list(template["rooms"])
    shapes = {r["id"]: _shape(r["polygon"]) for r in rooms}
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            inter = shapes[a["id"]].intersection(shapes[b["id"]]).area
            if inter > 0.05:
                errors.append(
                    f"geometry: rooms {a['id']} and {b['id']} overlap by {inter:.3f} sqm"
                )

    # 4. Every room polygon contained within boundary (allow tiny tolerance)
    boundary_buffered = boundary.buffer(0.01)
    for r in rooms:
        rs = shapes[r["id"]]
        if not boundary_buffered.contains(rs):
            outside = rs.difference(boundary_buffered).area
            if outside > 0.05:
                errors.append(
                    f"geometry: room {r['id']} extends {outside:.3f} sqm outside boundary"
                )

    # 5. Every door connects two valid rooms or one room + "outside",
    #    AND its position lies on an edge of every connected room (or boundary).
    room_polys = {r["id"]: r["polygon"] for r in rooms}
    room_ids = set(room_polys) | {"outside"}
    main_entry_count = 0
    boundary_polygon = template["boundary"]["polygon"]
    for i, d in enumerate(template["doors"]):
        if d["from"] not in room_ids:
            errors.append(f"door[{i}]: 'from' references unknown room/outside '{d['from']}'")
            continue
        if d["to"] not in room_ids:
            errors.append(f"door[{i}]: 'to' references unknown room/outside '{d['to']}'")
            continue
        if d.get("is_main_entry"):
            main_entry_count += 1
        pos = tuple(d["position"])
        # Position must lie on an edge of each non-outside endpoint.
        for endpoint in (d["from"], d["to"]):
            if endpoint == "outside":
                if not _point_on_any_edge(pos, boundary_polygon):
                    errors.append(
                        f"door[{i}] ({d['from']}→{d['to']}) at {pos}: "
                        "'outside' endpoint requires position on the boundary"
                    )
            else:
                if not _point_on_any_edge(pos, room_polys[endpoint]):
                    errors.append(
                        f"door[{i}] ({d['from']}→{d['to']}) at {pos}: "
                        f"position is not on any edge of room '{endpoint}'"
                    )

    # 6. At least one main entry
    if main_entry_count == 0:
        errors.append("door: no door has is_main_entry=true")

    # 7. Every window references a valid room AND its position lies on an
    #    edge of that room (typically a boundary edge for daylight).
    for i, w in enumerate(template["windows"]):
        if w["room"] not in room_polys:
            errors.append(f"window[{i}]: references unknown room '{w['room']}'")
            continue
        pos = tuple(w["position"])
        if not _point_on_any_edge(pos, room_polys[w["room"]]):
            errors.append(
                f"window[{i}] ({w['room']}) at {pos}: "
                f"position is not on any edge of room '{w['room']}'"
            )

    # 8. At least one bathroom or WC if total_area_sqm > 25
    if declared_total > 25:
        bath_types = {"bathroom", "wc"}
        if not any(r["type"] in bath_types for r in rooms):
            errors.append("rooms: no bathroom or wc despite total_area > 25 sqm")

    return errors


def validate_file(path: Path) -> list[str]:
    template = json.loads(path.read_text())
    return validate_dict(template)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: validate_template.py <path-to-template.json>", file=sys.stderr)
        return 2
    target = Path(argv[1])
    errors = validate_file(target)
    if errors:
        print(f"FAIL: {target.name}")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"OK: {target.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
