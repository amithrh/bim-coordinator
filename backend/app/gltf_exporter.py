"""Convert a BIM template into a glTF 2.0 binary (GLB).

Output is a self-contained binary that:
  * Loads in PlayCanvas / three.js / Babylon / Blender
  * Carries per-room PBR materials so floors look like parquet/marble/tile
    according to the BIM room type
  * Exposes named meshes ("floor_r_wohnzimmer", "wall", "ceiling") so the
    frontend can attach behaviour (AABB collision, click-to-teleport) per
    room without a separate scene graph

Reuses the trimesh wall/floor/ceiling builders from depth_renderer so the
geometry is identical to what we feed to depth ControlNet.
"""

from __future__ import annotations

import io
import json
import math
from typing import Any

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from PIL import Image

from .depth_renderer import (
    _wall_segments_from_rooms,
    _shapely_polygon,
)


# ---------------------------------------------------------------------------
# Per-room PBR materials. Tuned to look like real surfaces in PlayCanvas's
# default PBR shader (no IBL, single sun + ambient).
# ---------------------------------------------------------------------------

# (basecolor RGB 0..1, roughness, metallic)
_FLOOR_MATERIALS = {
    "living":          ((0.55, 0.40, 0.27), 0.65, 0.0),  # warm oak parquet
    "dining":          ((0.55, 0.40, 0.27), 0.65, 0.0),
    "bedroom":         ((0.62, 0.48, 0.34), 0.65, 0.0),  # slightly lighter parquet
    "master_bedroom":  ((0.55, 0.40, 0.27), 0.65, 0.0),
    "kitchen":         ((0.85, 0.83, 0.78), 0.45, 0.0),  # light tile
    "kueche":          ((0.85, 0.83, 0.78), 0.45, 0.0),
    "kochnische":      ((0.85, 0.83, 0.78), 0.45, 0.0),
    "bathroom":        ((0.93, 0.94, 0.96), 0.30, 0.0),  # white tile
    "bad":             ((0.93, 0.94, 0.96), 0.30, 0.0),
    "wc":              ((0.93, 0.94, 0.96), 0.30, 0.0),
    "balcony":         ((0.55, 0.55, 0.50), 0.85, 0.0),  # weathered stone
    "loggia":          ((0.55, 0.55, 0.50), 0.85, 0.0),
    "terrace":         ((0.55, 0.55, 0.50), 0.85, 0.0),
    "corridor":        ((0.65, 0.55, 0.42), 0.65, 0.0),  # parquet
    "diele":           ((0.65, 0.55, 0.42), 0.65, 0.0),
    "entry":           ((0.65, 0.55, 0.42), 0.65, 0.0),
    "flur":            ((0.65, 0.55, 0.42), 0.65, 0.0),
    "study":           ((0.55, 0.40, 0.27), 0.65, 0.0),
    "office":          ((0.55, 0.40, 0.27), 0.65, 0.0),
    "utility":         ((0.78, 0.78, 0.78), 0.55, 0.0),
    "abstellraum":     ((0.78, 0.78, 0.78), 0.55, 0.0),
    "wardrobe":        ((0.62, 0.48, 0.34), 0.65, 0.0),
    "walk_in_closet":  ((0.62, 0.48, 0.34), 0.65, 0.0),
}
_DEFAULT_FLOOR_MAT = ((0.70, 0.62, 0.50), 0.70, 0.0)
_WALL_MAT          = ((0.94, 0.92, 0.86), 0.85, 0.0)  # warm white plaster
_CEILING_MAT       = ((0.97, 0.96, 0.93), 0.92, 0.0)  # off-white


def _pbr(rgb: tuple[float, float, float], roughness: float = 0.7,
         metallic: float = 0.0, name: str = "mat") -> PBRMaterial:
    """Build a PBRMaterial with proper alpha, base color, roughness."""
    return PBRMaterial(
        name=name,
        baseColorFactor=[rgb[0], rgb[1], rgb[2], 1.0],
        roughnessFactor=float(roughness),
        metallicFactor=float(metallic),
    )


# ---------------------------------------------------------------------------
# Mesh builders (re-using shapes from depth_renderer)
# ---------------------------------------------------------------------------

def _walls_mesh(rooms: list[dict], ceiling_height: float,
                wall_thickness: float = 0.15) -> trimesh.Trimesh:
    """One concatenated wall mesh for all interior + exterior walls."""
    boxes = []
    for a, b in _wall_segments_from_rooms(rooms):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 0.05:
            continue
        cx, cy = (ax + bx) / 2, (ay + by) / 2
        cz = ceiling_height / 2
        angle = math.atan2(dy, dx)
        box = trimesh.creation.box(extents=(length, wall_thickness, ceiling_height))
        rot = trimesh.transformations.rotation_matrix(angle, (0, 0, 1))
        box.apply_transform(rot)
        box.apply_translation((cx, cy, cz))
        boxes.append(box)
    if not boxes:
        return trimesh.Trimesh()
    out = trimesh.util.concatenate(boxes)
    out.visual.material = _pbr(_WALL_MAT[0], _WALL_MAT[1], _WALL_MAT[2], "wall")
    return out


def _ceiling_mesh(rooms: list[dict], ceiling_height: float) -> trimesh.Trimesh:
    parts = []
    for r in rooms:
        poly = r.get("polygon", [])
        if len(poly) < 3:
            continue
        try:
            pts = np.array(poly, dtype=float)
            ceil = trimesh.creation.extrude_polygon(_shapely_polygon(pts), height=0.05)
            ceil.apply_translation((0, 0, ceiling_height - 0.05))
            parts.append(ceil)
        except Exception:
            continue
    if not parts:
        return trimesh.Trimesh()
    out = trimesh.util.concatenate(parts)
    out.visual.material = _pbr(_CEILING_MAT[0], _CEILING_MAT[1], _CEILING_MAT[2], "ceiling")
    return out


def _floor_mesh_for_room(room: dict) -> trimesh.Trimesh | None:
    poly = room.get("polygon", [])
    if len(poly) < 3:
        return None
    try:
        pts = np.array(poly, dtype=float)
        floor = trimesh.creation.extrude_polygon(_shapely_polygon(pts), height=0.05)
    except Exception:
        return None
    rtype = (room.get("type") or "").lower()
    rgb, rough, metal = _FLOOR_MATERIALS.get(rtype, _DEFAULT_FLOOR_MAT)
    floor.visual.material = _pbr(rgb, rough, metal, f"floor_{room.get('id', rtype)}")
    return floor


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------

def _spawn_point(template: dict) -> tuple[float, float, float, float]:
    """Pick a sensible camera spawn: at the main entry door, looking inward.

    Returns (x, y, z, yaw_radians).  Yaw is the camera's heading (0 = +X).
    """
    ceiling_h = template.get("boundary", {}).get("ceiling_height_mm", 2700) / 1000.0
    rooms = template.get("rooms", []) or []
    if not rooms and template.get("floors"):
        for fl in template["floors"]:
            if fl.get("rooms"):
                rooms = fl["rooms"]
                break
    if not rooms:
        return (0.0, 0.0, 1.65, 0.0)

    # Find the main entry door, fall back to first door, fall back to first
    # circulation room centroid.
    doors = template.get("doors", [])
    if not doors and template.get("floors"):
        for fl in template["floors"]:
            doors = fl.get("doors", [])
            if doors:
                break

    main = next((d for d in doors if d.get("is_main_entry")), None)
    if main is None and doors:
        main = doors[0]

    if main and main.get("position"):
        # Find which room this door leads INTO and put the camera 1m inside it
        target_id = main.get("to") or main.get("from")
        target_room = next((r for r in rooms if r.get("id") == target_id), None)
        if target_room is None:
            target_room = rooms[0]
        rpoly = np.array(target_room["polygon"], dtype=float)
        cx, cy = rpoly.mean(axis=0)
        dx, dy = main["position"]
        # Aim the camera from the door toward the room centroid
        yaw = math.atan2(cy - dy, cx - dx)
        # Spawn 0.8m inside the room from the door
        sx = dx + math.cos(yaw) * 0.8
        sy = dy + math.sin(yaw) * 0.8
        return (sx, sy, 1.65, yaw)

    # No door info — drop into the largest room's centroid
    biggest = max(rooms, key=lambda r: r.get("area_sqm", 0))
    rpoly = np.array(biggest["polygon"], dtype=float)
    cx, cy = rpoly.mean(axis=0)
    return (cx, cy, 1.65, 0.0)


def template_to_glb(template: dict, include_ceiling: bool = False,
                       wall_height: float | None = None) -> bytes:
    """Render the BIM template into a glTF 2.0 binary (GLB) and return
    the raw bytes. Includes per-room floor materials, walls, optional
    ceiling, and a JSON 'extras' payload on the scene with metadata
    the frontend needs (room polygons for collision/teleport, spawn
    point, etc).

    Args:
      include_ceiling: when False (default for 3D walks), skip the
        ceiling so the sun can light the interior. The result reads
        as a Sims-style cutaway diorama, which is what we want for a
        first-person walk in a small apartment without modeled windows.
      wall_height: override the BIM ceiling height with a shorter wall
        height. Useful for top-down dollhouse renders.
    """
    rooms = template.get("rooms", []) or []
    if not rooms and template.get("floors"):
        # Multi-floor — flatten for now (future: stack floors with stair offsets)
        rooms = []
        for fl in template["floors"]:
            rooms.extend(fl.get("rooms", []))

    boundary = template.get("boundary", {})
    ceiling_h = boundary.get("ceiling_height_mm", 2700) / 1000.0
    wall_h = float(wall_height) if wall_height is not None else ceiling_h
    wall_t = boundary.get("wall_thickness_mm", 240) / 1000.0
    # Cap visual wall thickness same as the floor-plan renderer does.
    wall_t = min(wall_t, 0.18)

    scene = trimesh.Scene()

    # Floor per room (so each gets its own material)
    for r in rooms:
        floor = _floor_mesh_for_room(r)
        if floor is not None:
            scene.add_geometry(floor, geom_name=f"floor_{r.get('id', r.get('type','x'))}")

    # Walls (one mesh)
    walls = _walls_mesh(rooms, ceiling_height=wall_h, wall_thickness=wall_t)
    if not walls.is_empty:
        scene.add_geometry(walls, geom_name="walls")

    if include_ceiling:
        ceil = _ceiling_mesh(rooms, ceiling_height=ceiling_h)
        if not ceil.is_empty:
            scene.add_geometry(ceil, geom_name="ceiling")

    # Spawn point + room metadata for the frontend.
    spawn = _spawn_point(template)
    md = template.get("metadata", {}) or {}
    scene.metadata.update({
        "bim_template_id": template.get("id", ""),
        "bim_country": md.get("country", ""),
        "bim_city": md.get("city_inspiration", ""),
        "bim_total_area_sqm": md.get("total_area_sqm", 0),
        "bim_ceiling_h": ceiling_h,
        "bim_spawn": list(spawn),
        "bim_rooms": [
            {
                "id": r.get("id"),
                "name": r.get("name"),
                "type": r.get("type"),
                "polygon": r.get("polygon"),
                "area_sqm": r.get("area_sqm"),
            }
            for r in rooms
        ],
        "bim_doors": [
            {
                "from": d.get("from"), "to": d.get("to"),
                "position": d.get("position"),
                "width_mm": d.get("width_mm"),
            }
            for d in (template.get("doors") or [])
        ],
        "bim_windows": [
            {
                "room": w.get("room"),
                "position": w.get("position"),
                "width_mm": w.get("width_mm"),
            }
            for w in (template.get("windows") or [])
        ],
    })

    # IMPORTANT: trimesh writes the scene up-axis as Z; PlayCanvas / glTF
    # convention is Y-up. Apply a -90° rotation around X so the floor
    # plane lands on Y=0 in the export.
    rot_y_up = trimesh.transformations.rotation_matrix(-math.pi / 2, (1, 0, 0))
    for name, geom in scene.geometry.items():
        geom.apply_transform(rot_y_up)
    # And rotate the spawn point the same way so the frontend can use it
    # directly without having to re-transform.
    sx, sy, sz, yaw = spawn
    # After -90° about X: (x, y, z) → (x, z, -y). z was up; y was depth.
    # We're keeping the apartment "north" along +Z (was +Y) and "east" along +X.
    new_spawn_pos = [sx, sz, -sy]  # eye at z=1.65 stays as Y=1.65
    scene.metadata["bim_spawn"] = [*new_spawn_pos, yaw]

    # Export GLB bytes
    glb_bytes = scene.export(file_type="glb")
    return glb_bytes


def __main__():
    """CLI: python -m backend.app.gltf_exporter <template.json> [out.glb]"""
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: gltf_exporter <template.json> [out.glb]")
    template = json.loads(open(sys.argv[1]).read())
    out = sys.argv[2] if len(sys.argv) >= 3 else "/tmp/template.glb"
    glb = template_to_glb(template)
    with open(out, "wb") as f:
        f.write(glb)
    print(f"wrote {out}  ({len(glb)/1024:.1f} KB)")


if __name__ == "__main__":
    __main__()
