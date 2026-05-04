"""Build an IFC4 file from a floor-plan template JSON.

Approach: SEGMENTED WALLS. Walls are broken into segments where doors and
windows go. The gap IS the opening. No IfcOpeningElement, no
IfcRelVoidsElement. Visually identical in 3D, dramatically simpler code.

See Day-Zero-Build-Plan-v3.2.docx §6.1 for rationale.

Usage:
    python build_template.py <template.json> <output.ifc>
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import ifcopenshell
from ifcopenshell.api import (
    aggregate, context, geometry, project, root, spatial, unit,
)

# ---------- tolerances ----------
COLLINEAR_TOL = 1e-3       # tight tolerance for collinearity (perpendicular distance)
ON_EDGE_TOL = 0.011        # 10mm grid + 1mm slack for "is this point on this edge"
EDGE_TOL = COLLINEAR_TOL   # back-compat alias for collinearity checks
MIN_SEGMENT_LEN = 0.05     # ignore segments shorter than 5cm
COORD_QUANTIZE = 1000      # mm grid for endpoint deduplication

# ---------- 2D geometry primitives ----------

Vec = tuple[float, float]
Edge = tuple[Vec, Vec]


def length(seg: Edge) -> float:
    (x1, y1), (x2, y2) = seg
    return math.hypot(x2 - x1, y2 - y1)


def unit_vec(seg: Edge) -> Vec:
    (x1, y1), (x2, y2) = seg
    L = length(seg)
    if L < EDGE_TOL:
        return (1.0, 0.0)
    return ((x2 - x1) / L, (y2 - y1) / L)


def perp(v: Vec) -> Vec:
    return (-v[1], v[0])


def project_onto_edge(p1: Vec, p2: Vec, point: Vec) -> float:
    """Parametric position of `point` projected onto edge p1->p2.
    0 = at p1, length(p1,p2) = at p2. May lie outside [0, L]."""
    ux, uy = unit_vec((p1, p2))
    return (point[0] - p1[0]) * ux + (point[1] - p1[1]) * uy


def perp_distance(p1: Vec, p2: Vec, point: Vec) -> float:
    ux, uy = unit_vec((p1, p2))
    dx, dy = point[0] - p1[0], point[1] - p1[1]
    return abs(dx * uy - dy * ux)


def point_on_edge(p1: Vec, p2: Vec, point: Vec, tol: float = ON_EDGE_TOL) -> bool:
    """True if `point` lies on edge p1->p2 (within `tol`, including endpoints).
    Uses the larger ON_EDGE_TOL by default to match §6.1's 10mm grid snapping."""
    if perp_distance(p1, p2, point) > tol:
        return False
    t = project_onto_edge(p1, p2, point)
    return -tol <= t <= length((p1, p2)) + tol


def collinear_overlap(seg_a: Edge, seg_b: Edge,
                       tol: float = EDGE_TOL) -> Edge | None:
    """If two segments are collinear and overlap by more than MIN_SEGMENT_LEN,
    return the overlap segment as (p_lo, p_hi). Otherwise return None."""
    a1, a2 = seg_a
    L_a = length(seg_a)
    if L_a < tol:
        return None
    ux, uy = unit_vec(seg_a)
    # Check b1 and b2 are on the line through a1 with direction (ux, uy)
    if perp_distance(a1, a2, seg_b[0]) > tol:
        return None
    if perp_distance(a1, a2, seg_b[1]) > tol:
        return None
    # Project b1 and b2 onto a-line, compute overlap interval
    t_b1 = project_onto_edge(a1, a2, seg_b[0])
    t_b2 = project_onto_edge(a1, a2, seg_b[1])
    b_lo, b_hi = (t_b1, t_b2) if t_b1 <= t_b2 else (t_b2, t_b1)
    overlap_lo = max(0.0, b_lo)
    overlap_hi = min(L_a, b_hi)
    if overlap_hi - overlap_lo < MIN_SEGMENT_LEN:
        return None
    p_lo = (a1[0] + ux * overlap_lo, a1[1] + uy * overlap_lo)
    p_hi = (a1[0] + ux * overlap_hi, a1[1] + uy * overlap_hi)
    return (p_lo, p_hi)


def quantize(p: Vec) -> tuple[int, int]:
    return (round(p[0] * COORD_QUANTIZE), round(p[1] * COORD_QUANTIZE))


def edge_key(seg: Edge) -> tuple[tuple[int, int], tuple[int, int]]:
    a, b = quantize(seg[0]), quantize(seg[1])
    return (a, b) if a <= b else (b, a)


# ---------- segment computation ----------

def find_openings_on_edge(p1: Vec, p2: Vec,
                           openings: Iterable[dict]) -> list[dict]:
    """Return the openings whose `position` is on edge p1->p2."""
    return [o for o in openings if point_on_edge(p1, p2, tuple(o["position"]))]


def opening_endpoints_on_edge(p1: Vec, p2: Vec, opening: dict) -> Edge:
    """Compute (start_point, end_point) of the opening's gap on the edge.
    The opening straddles its centerpoint by half its width along the edge.
    Endpoints are clamped to the edge's [0, L] parametric range so wide
    openings near a corner produce a flush edge, not a reverse-direction
    "wall" extending past the corner."""
    ux, uy = unit_vec((p1, p2))
    cx, cy = opening["position"]
    half_w = (opening["width_mm"] / 1000.0) / 2.0
    L = length((p1, p2))
    t_center = project_onto_edge(p1, p2, (cx, cy))
    t_start = max(0.0, t_center - half_w)
    t_end = min(L, t_center + half_w)
    p_start = (p1[0] + ux * t_start, p1[1] + uy * t_start)
    p_end = (p1[0] + ux * t_end, p1[1] + uy * t_end)
    return (p_start, p_end)


def segment_edge_by_openings(p1: Vec, p2: Vec,
                              openings_on_edge: list[dict]) -> list[Edge]:
    """Return wall segments (between openings) along edge p1->p2."""
    if not openings_on_edge:
        return [(p1, p2)] if length((p1, p2)) > MIN_SEGMENT_LEN else []
    sorted_ops = sorted(
        openings_on_edge,
        key=lambda o: project_onto_edge(p1, p2, tuple(o["position"])),
    )
    segments: list[Edge] = []
    cursor = p1
    for op in sorted_ops:
        op_start, op_end = opening_endpoints_on_edge(p1, p2, op)
        if length((cursor, op_start)) > MIN_SEGMENT_LEN:
            segments.append((cursor, op_start))
        cursor = op_end
    if length((cursor, p2)) > MIN_SEGMENT_LEN:
        segments.append((cursor, p2))
    return segments


def compute_exterior_segments(boundary_polygon: list[list[float]],
                                doors: list[dict],
                                windows: list[dict]) -> list[Edge]:
    """Walk each boundary edge, slice out doors and windows, return wall segments."""
    openings = doors + windows
    segments: list[Edge] = []
    n = len(boundary_polygon)
    for i in range(n):
        p1 = tuple(boundary_polygon[i])
        p2 = tuple(boundary_polygon[(i + 1) % n])
        ops_here = find_openings_on_edge(p1, p2, openings)
        segments.extend(segment_edge_by_openings(p1, p2, ops_here))
    return segments


def compute_interior_segments(rooms: list[dict],
                                doors: list[dict]) -> list[Edge]:
    """Find edges shared by two rooms. Segment each by interior doors on it."""
    # Collect (room_id, edge) pairs
    edges: list[tuple[str, Edge]] = []
    for room in rooms:
        poly = room["polygon"]
        for i in range(len(poly)):
            e = (tuple(poly[i]), tuple(poly[(i + 1) % len(poly)]))
            edges.append((room["id"], e))

    # Find collinear overlaps between edges of different rooms
    shared_overlaps: dict[tuple, Edge] = {}
    for i, (a_id, a_edge) in enumerate(edges):
        for b_id, b_edge in edges[i + 1:]:
            if a_id == b_id:
                continue
            overlap = collinear_overlap(a_edge, b_edge)
            if overlap is None:
                continue
            key = edge_key(overlap)
            shared_overlaps.setdefault(key, overlap)

    # Segment each shared edge by doors that lie on it
    interior_doors = [d for d in doors if d["from"] != "outside" and d["to"] != "outside"]
    final: list[Edge] = []
    for seg in shared_overlaps.values():
        ops_here = find_openings_on_edge(seg[0], seg[1], interior_doors)
        final.extend(segment_edge_by_openings(seg[0], seg[1], ops_here))
    return final


# ---------- IFC creation helpers ----------

def cart_point(model: ifcopenshell.file, *coords: float):
    return model.create_entity("IfcCartesianPoint",
                                Coordinates=[float(c) for c in coords])


def direction(model: ifcopenshell.file, *ratios: float):
    return model.create_entity("IfcDirection",
                                DirectionRatios=[float(r) for r in ratios])


def axis2_placement_3d(model: ifcopenshell.file,
                        location: tuple[float, float, float] = (0.0, 0.0, 0.0),
                        axis_z: tuple[float, float, float] = (0.0, 0.0, 1.0),
                        ref_x: tuple[float, float, float] = (1.0, 0.0, 0.0)):
    return model.create_entity(
        "IfcAxis2Placement3D",
        Location=cart_point(model, *location),
        Axis=direction(model, *axis_z),
        RefDirection=direction(model, *ref_x),
    )


def axis2_placement_2d(model: ifcopenshell.file,
                        location: tuple[float, float] = (0.0, 0.0),
                        ref_x: tuple[float, float] = (1.0, 0.0)):
    return model.create_entity(
        "IfcAxis2Placement2D",
        Location=cart_point(model, *location),
        RefDirection=direction(model, *ref_x),
    )


def local_placement(model: ifcopenshell.file,
                     location: tuple[float, float, float],
                     x_axis_2d: Vec,
                     parent_placement=None):
    """Create an IfcLocalPlacement with origin at `location`, X-axis along
    `x_axis_2d` (normalized 2D vector), Z-axis straight up."""
    ux, uy = x_axis_2d
    rel = axis2_placement_3d(
        model,
        location=location,
        axis_z=(0.0, 0.0, 1.0),
        ref_x=(ux, uy, 0.0),
    )
    return model.create_entity(
        "IfcLocalPlacement",
        PlacementRelTo=parent_placement,
        RelativePlacement=rel,
    )


def polyline_2d(model: ifcopenshell.file, points: list[Vec], close: bool = True):
    pts = [cart_point(model, *p) for p in points]
    if close:
        # Use a fresh cart_point for closing rather than appending the same instance
        first = points[0]
        last = points[-1]
        if abs(first[0] - last[0]) > EDGE_TOL or abs(first[1] - last[1]) > EDGE_TOL:
            pts.append(cart_point(model, *first))
    return model.create_entity("IfcPolyline", Points=pts)


# ---------- entity builders ----------

def create_wall(model, body_ctx, seg: Edge, thickness: float, height: float,
                 storey_placement, external: bool):
    p1, p2 = seg
    L = length(seg)
    if L < MIN_SEGMENT_LEN:
        return None
    ux, uy = unit_vec(seg)

    wall = root.create_entity(model, ifc_class="IfcWall",
                               name=("Exterior Wall" if external else "Interior Wall"))
    # add_wall_representation builds the wall body extending from local origin
    # along +X by `length`. Place at p1 (segment start), with local +X along
    # the segment direction, so the body lands exactly on the segment.
    wall.ObjectPlacement = local_placement(model,
                                            location=(p1[0], p1[1], 0.0),
                                            x_axis_2d=(ux, uy),
                                            parent_placement=storey_placement)

    # Wall extends from -L/2 to +L/2 along its local X axis.
    # add_wall_representation builds the body relative to that axis;
    # offset = -thickness/2 centers the wall thickness on the axis.
    rep = geometry.add_wall_representation(
        model,
        context=body_ctx,
        length=L,
        height=height,
        thickness=thickness,
        offset=-thickness / 2.0,
    )
    geometry.assign_representation(model, product=wall, representation=rep)
    return wall


def find_host_direction(position: Vec, host_edges: list[Edge]) -> Vec:
    """Return the unit direction vector of the edge `position` lies on.
    Falls back to (1, 0) if no host edge is found."""
    for edge in host_edges:
        if point_on_edge(edge[0], edge[1], position):
            return unit_vec(edge)
    return (1.0, 0.0)


def create_door_entity(model, body_ctx, door: dict, host_edges: list[Edge],
                        storey_placement):
    width = door["width_mm"] / 1000.0
    height = door.get("height_mm", 2100) / 1000.0
    cx, cy = door["position"]
    ux, uy = find_host_direction((cx, cy), host_edges)
    # Place at (cx, cy) shifted back by width/2 along the wall direction so the
    # body — which extends from local (0, ...) to (width, ...) along +X — ends
    # up centered on the JSON position.
    px = cx - ux * (width / 2)
    py = cy - uy * (width / 2)
    door_entity = root.create_entity(model, ifc_class="IfcDoor", name="Door")
    door_entity.OverallWidth = width
    door_entity.OverallHeight = height
    door_entity.ObjectPlacement = local_placement(
        model, location=(px, py, 0.0), x_axis_2d=(ux, uy),
        parent_placement=storey_placement,
    )
    rep = geometry.add_door_representation(
        model, context=body_ctx,
        overall_height=height, overall_width=width,
    )
    if rep is not None:
        geometry.assign_representation(model, product=door_entity, representation=rep)
    return door_entity


def create_window_entity(model, body_ctx, window: dict, host_edges: list[Edge],
                          storey_placement):
    width = window["width_mm"] / 1000.0
    height = window.get("height_mm", 1200) / 1000.0
    sill = window.get("sill_mm", 900) / 1000.0
    cx, cy = window["position"]
    ux, uy = find_host_direction((cx, cy), host_edges)
    px = cx - ux * (width / 2)
    py = cy - uy * (width / 2)
    win_entity = root.create_entity(model, ifc_class="IfcWindow", name="Window")
    win_entity.OverallWidth = width
    win_entity.OverallHeight = height
    win_entity.ObjectPlacement = local_placement(
        model, location=(px, py, sill), x_axis_2d=(ux, uy),
        parent_placement=storey_placement,
    )
    rep = geometry.add_window_representation(
        model, context=body_ctx,
        overall_height=height, overall_width=width,
    )
    geometry.assign_representation(model, product=win_entity, representation=rep)
    return win_entity


def create_space(model, body_ctx, room: dict, height: float, storey_placement):
    space = root.create_entity(model, ifc_class="IfcSpace",
                                name=room["name"], predefined_type="INTERNAL")
    space.LongName = room["name"]
    space.ObjectPlacement = local_placement(
        model, location=(0.0, 0.0, 0.0), x_axis_2d=(1.0, 0.0),
        parent_placement=storey_placement,
    )
    # Build extruded area solid from polygon
    poly = polyline_2d(model, [tuple(p) for p in room["polygon"]], close=True)
    profile = model.create_entity(
        "IfcArbitraryClosedProfileDef",
        ProfileType="AREA",
        OuterCurve=poly,
    )
    extrusion = model.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=profile,
        Position=axis2_placement_3d(model),
        ExtrudedDirection=direction(model, 0.0, 0.0, 1.0),
        Depth=height,
    )
    rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=body_ctx,
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid",
        Items=[extrusion],
    )
    space.Representation = model.create_entity(
        "IfcProductDefinitionShape", Representations=[rep],
    )
    return space


def create_slab(model, body_ctx, polygon: list[list[float]], depth: float,
                 storey_placement):
    slab = root.create_entity(model, ifc_class="IfcSlab",
                               name="Floor", predefined_type="FLOOR")
    slab.ObjectPlacement = local_placement(
        model, location=(0.0, 0.0, -depth), x_axis_2d=(1.0, 0.0),
        parent_placement=storey_placement,
    )
    rep = geometry.add_slab_representation(
        model, context=body_ctx, depth=depth,
        polyline=[tuple(p) for p in polygon],
    )
    geometry.assign_representation(model, product=slab, representation=rep)
    return slab


# ---------- per-floor builder ----------

def _build_floor(model, body_ctx, storey, storey_placement,
                  boundary_polygon: list[list[float]],
                  rooms: list[dict],
                  doors: list[dict],
                  windows: list[dict],
                  thickness: float,
                  height: float):
    """Build all walls/doors/windows/spaces/slab for ONE floor into the
    given storey at its placement."""
    # Exterior walls
    for seg in compute_exterior_segments(boundary_polygon, doors, windows):
        wall = create_wall(model, body_ctx, seg, thickness, height,
                            storey_placement, external=True)
        if wall is not None:
            spatial.assign_container(model, products=[wall], relating_structure=storey)

    # Interior walls
    for seg in compute_interior_segments(rooms, doors):
        wall = create_wall(model, body_ctx, seg, thickness, height,
                            storey_placement, external=False)
        if wall is not None:
            spatial.assign_container(model, products=[wall], relating_structure=storey)

    # Host-edge list for door/window orientation
    boundary_edges: list[Edge] = []
    for i in range(len(boundary_polygon)):
        boundary_edges.append((tuple(boundary_polygon[i]),
                                tuple(boundary_polygon[(i + 1) % len(boundary_polygon)])))
    interior_shared_edges: list[Edge] = []
    edges = []
    for room in rooms:
        poly = room["polygon"]
        for i in range(len(poly)):
            edges.append((room["id"],
                           (tuple(poly[i]), tuple(poly[(i + 1) % len(poly)]))))
    seen = set()
    for i, (a_id, a_edge) in enumerate(edges):
        for b_id, b_edge in edges[i + 1:]:
            if a_id == b_id:
                continue
            ov = collinear_overlap(a_edge, b_edge)
            if ov is None:
                continue
            k = edge_key(ov)
            if k in seen:
                continue
            seen.add(k)
            interior_shared_edges.append(ov)
    host_edges = boundary_edges + interior_shared_edges

    for door in doors:
        d = create_door_entity(model, body_ctx, door, host_edges, storey_placement)
        spatial.assign_container(model, products=[d], relating_structure=storey)
    for window in windows:
        w = create_window_entity(model, body_ctx, window, host_edges, storey_placement)
        spatial.assign_container(model, products=[w], relating_structure=storey)

    for room in rooms:
        space = create_space(model, body_ctx, room, height, storey_placement)
        aggregate.assign_object(model, products=[space], relating_object=storey)

    slab = create_slab(model, body_ctx, boundary_polygon, 0.2, storey_placement)
    spatial.assign_container(model, products=[slab], relating_structure=storey)


# ---------- main build ----------

def build(template: dict, output_path: Path) -> ifcopenshell.file:
    model = project.create_file(version="IFC4")

    proj = root.create_entity(model, ifc_class="IfcProject",
                               name=template["metadata"]["description"][:60])
    # Force METRE (not MILLI METRE) so placement coords I author in meters
    # are interpreted correctly.
    unit.assign_unit(model, length={"is_metric": True, "raw": "METERS"})

    ctx = context.add_context(model, context_type="Model")
    body_ctx = context.add_context(
        model, context_type="Model", context_identifier="Body",
        target_view="MODEL_VIEW", parent=ctx,
    )

    site = root.create_entity(model, ifc_class="IfcSite", name="Site")
    building = root.create_entity(model, ifc_class="IfcBuilding", name="Building")
    aggregate.assign_object(model, relating_object=proj, products=[site])
    aggregate.assign_object(model, relating_object=site, products=[building])

    boundary_default = template["boundary"]["polygon"]
    t = template["boundary"]["wall_thickness_mm"] / 1000.0
    default_h = template["boundary"]["ceiling_height_mm"] / 1000.0

    floors_data = template.get("floors")
    if floors_data:
        # Multi-floor: stack each floor at its computed elevation.
        elevation_m = 0.0
        for fi, fl in enumerate(floors_data):
            fl_height = fl.get("ceiling_height_mm", template["boundary"]["ceiling_height_mm"]) / 1000.0
            fl_elev = (fl["elevation_mm"] / 1000.0
                       if "elevation_mm" in fl else elevation_m)
            storey = root.create_entity(model, ifc_class="IfcBuildingStorey",
                                          name=fl["name"])
            aggregate.assign_object(model, relating_object=building, products=[storey])
            storey.ObjectPlacement = local_placement(
                model, location=(0.0, 0.0, fl_elev), x_axis_2d=(1.0, 0.0),
            )
            poly = fl.get("boundary_polygon", boundary_default)
            _build_floor(model, body_ctx, storey, storey.ObjectPlacement,
                          poly, fl["rooms"], fl.get("doors", []),
                          fl.get("windows", []), t, fl_height)
            # Next floor stacks on top of this one
            elevation_m = fl_elev + fl_height
    else:
        # Single-floor (existing behavior)
        storey = root.create_entity(model, ifc_class="IfcBuildingStorey",
                                      name="Ground Floor")
        aggregate.assign_object(model, relating_object=building, products=[storey])
        storey.ObjectPlacement = local_placement(
            model, location=(0.0, 0.0, 0.0), x_axis_2d=(1.0, 0.0),
        )
        _build_floor(model, body_ctx, storey, storey.ObjectPlacement,
                      boundary_default, template["rooms"],
                      template["doors"], template["windows"], t, default_h)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(output_path))
    return model


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: build_template.py <template.json> <output.ifc>", file=sys.stderr)
        return 2
    template = json.loads(Path(argv[1]).read_text())
    build(template, Path(argv[2]))
    print(f"OK: wrote {argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
