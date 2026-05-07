"""3D depth-map rendering from a BIM template.

Builds a 3D scene from the template's polygons (walls, floor, ceiling),
positions a perspective camera inside a chosen room, and renders a depth
map. The depth map drives Depth ControlNet so the SDXL output respects
the actual room geometry.

We avoid heavyweight 3D dependencies (no Blender, no PyOpenGL/EGL on Mac).
Instead we build wall planes as triangle meshes via trimesh and use trimesh's
embree-free raycasting for depth. This works headlessly on macOS.

Output:
  - depth_map: PIL.Image grayscale (closer = whiter)
  - canny_edges: PIL.Image grayscale edge map (top-down silhouette)
  - prompt suggestion based on the focused room
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
import trimesh


# ---------------------------------------------------------------------------
# Build 3D scene from template
# ---------------------------------------------------------------------------

def _wall_segments_from_rooms(rooms: list[dict]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Collect unique wall segments from room polygons.

    Each room polygon is a closed loop of points; consecutive points form
    edges. Many edges are shared between rooms — we dedup them.
    """
    seen: dict = {}
    for r in rooms:
        poly = r.get("polygon", [])
        for i in range(len(poly)):
            a = tuple(poly[i])
            b = tuple(poly[(i + 1) % len(poly)])
            key = tuple(sorted((a, b)))
            seen[key] = (a, b)
    return list(seen.values())


def _build_walls_mesh(rooms: list[dict], ceiling_height: float = 2.7,
                       wall_thickness: float = 0.15) -> trimesh.Trimesh:
    """Build a mesh of walls (boxes extruded between segment endpoints)."""
    boxes = []
    for a, b in _wall_segments_from_rooms(rooms):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 0.05:
            continue
        # Box centred at segment midpoint, oriented along segment
        cx, cy = (ax + bx) / 2, (ay + by) / 2
        cz = ceiling_height / 2
        angle = math.atan2(dy, dx)

        # Create box of size (length, wall_thickness, ceiling_height)
        box = trimesh.creation.box(extents=(length, wall_thickness, ceiling_height))
        # Rotate so the long axis aligns with the segment
        rot = trimesh.transformations.rotation_matrix(angle, (0, 0, 1))
        box.apply_transform(rot)
        # Translate to midpoint
        box.apply_translation((cx, cy, cz))
        boxes.append(box)

    if not boxes:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(boxes)


def _build_floor_ceiling_mesh(rooms: list[dict], ceiling_height: float = 2.7,
                                include_ceiling: bool = True) -> trimesh.Trimesh:
    """Build floor (z=0) and optionally ceiling (z=ceiling) as flat meshes.

    For dollhouse/cutaway views, set include_ceiling=False so the camera
    can see down into the rooms.
    """
    parts = []
    for r in rooms:
        poly = r.get("polygon", [])
        if len(poly) < 3:
            continue
        # Floor (planar polygon at z=0)
        try:
            pts2d = np.array(poly, dtype=float)
            triangulated = trimesh.creation.extrude_polygon(
                _shapely_polygon(pts2d), height=0.05
            )
            parts.append(triangulated)
            if include_ceiling:
                # Ceiling (planar polygon near z=ceiling)
                ceil = trimesh.creation.extrude_polygon(
                    _shapely_polygon(pts2d), height=0.05
                )
                ceil.apply_translation((0, 0, ceiling_height - 0.05))
                parts.append(ceil)
        except Exception:
            continue
    if not parts:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(parts)


def _shapely_polygon(pts: np.ndarray):
    from shapely.geometry import Polygon
    return Polygon(pts)


def build_scene(template: dict, focus_room_idx: int = 0) -> dict[str, Any]:
    """Build a 3D scene + camera positioned inside a chosen room.

    Returns a dict with 'mesh' (combined trimesh), 'camera_transform',
    'image_size', 'focal_length', 'focus_room' info.
    """
    rooms = template.get("rooms", []) or []
    if not rooms and template.get("floors"):
        # Multi-floor — use first non-lobby floor's rooms
        for fl in template["floors"]:
            if fl.get("rooms"):
                rooms = fl["rooms"]
                break
    if not rooms:
        return {"mesh": trimesh.Trimesh(), "camera_transform": np.eye(4),
                "image_size": (768, 512), "focus_room": None}

    # Pick focus room — prefer 'living' or largest non-circulation room
    habitable = [r for r in rooms if r.get("type") not in
                 ("entry", "stairs", "corridor", "wc")]
    if not habitable:
        habitable = rooms
    living = next((r for r in habitable if r.get("type") == "living"), None)
    focus_room = living or max(habitable, key=lambda r: r.get("area_sqm", 0))

    boundary = template.get("boundary", {})
    ceiling_h = boundary.get("ceiling_height_mm", 2700) / 1000.0

    walls = _build_walls_mesh(rooms, ceiling_height=ceiling_h)
    floors_ceilings = _build_floor_ceiling_mesh(rooms, ceiling_height=ceiling_h)
    parts = [m for m in (walls, floors_ceilings) if not m.is_empty]
    scene_mesh = trimesh.util.concatenate(parts) if parts else trimesh.Trimesh()

    # Position camera inside focus room, facing the long axis
    poly = np.array(focus_room["polygon"], dtype=float)
    cx, cy = poly.mean(axis=0)
    minp = poly.min(axis=0)
    maxp = poly.max(axis=0)
    w = maxp[0] - minp[0]
    h = maxp[1] - minp[1]
    # Camera at one short edge looking toward the opposite short edge
    if w >= h:
        # Wide room — camera at left, looking right
        cam_x = minp[0] + 0.6
        cam_y = (minp[1] + maxp[1]) / 2
        target_x = maxp[0]
        target_y = cam_y
    else:
        # Tall room — camera at bottom, looking up
        cam_x = (minp[0] + maxp[0]) / 2
        cam_y = minp[1] + 0.6
        target_x = cam_x
        target_y = maxp[1]
    cam_z = 1.65   # human eye height
    target_z = 1.4  # slightly downward gaze

    # Build look-at transform
    cam_pos = np.array([cam_x, cam_y, cam_z])
    target = np.array([target_x, target_y, target_z])
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    up = np.array([0, 0, 1])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    new_up = np.cross(right, forward)
    transform = np.eye(4)
    transform[:3, 0] = right
    transform[:3, 1] = new_up
    transform[:3, 2] = -forward  # camera looks along -Z by convention
    transform[:3, 3] = cam_pos

    return {
        "mesh": scene_mesh,
        "camera_transform": transform,
        "image_size": (768, 512),
        "focal_length": 500.0,
        "focus_room": focus_room,
        "ceiling_height": ceiling_h,
    }


# ---------------------------------------------------------------------------
# Depth render via raycasting (trimesh built-in, no GPU/OpenGL needed)
# ---------------------------------------------------------------------------

def render_depth_map(scene: dict[str, Any]) -> Image.Image:
    """Render a depth map by raycasting from the camera.

    Depth is mapped 0.5..max_d → white..black (closer = whiter, depth
    ControlNet convention). max_d defaults to 12m for interior views,
    but scenes can pass `depth_max` to override (e.g. 30m for dollhouse).
    """
    mesh = scene["mesh"]
    if mesh.is_empty:
        # Empty mesh — return blank
        return Image.new("L", scene["image_size"], 128)

    transform = scene["camera_transform"]
    width, height = scene["image_size"]
    focal = scene["focal_length"]
    depth_min = float(scene.get("depth_min", 0.5))
    depth_max = float(scene.get("depth_max", 12.0))

    # Camera center + axes from transform
    cam_pos = transform[:3, 3]
    right = transform[:3, 0]
    up = transform[:3, 1]
    forward = -transform[:3, 2]   # camera looks along its -Z

    # Build pixel rays
    cx_pix = width / 2
    cy_pix = height / 2
    # Use a coarser grid for speed (we'll upsample the image)
    grid_w, grid_h = 192, 128
    us = (np.arange(grid_w) + 0.5 - grid_w / 2) * (width / grid_w / focal)
    vs = -(np.arange(grid_h) + 0.5 - grid_h / 2) * (height / grid_h / focal)
    uu, vv = np.meshgrid(us, vs)
    dirs = (forward[None, None, :]
            + uu[..., None] * right[None, None, :]
            + vv[..., None] * up[None, None, :])
    dirs = dirs.reshape(-1, 3)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.tile(cam_pos, (dirs.shape[0], 1))

    # Cast rays
    locations, ray_indices, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=dirs, multiple_hits=False,
    )

    depth = np.full(dirs.shape[0], np.inf)
    if len(locations) > 0:
        dists = np.linalg.norm(locations - origins[ray_indices], axis=1)
        # Take nearest hit per ray
        for i, ri in enumerate(ray_indices):
            if dists[i] < depth[ri]:
                depth[ri] = dists[i]

    # Map far/no-hit to a max distance so the depth visualisation is consistent
    finite = np.where(np.isfinite(depth), depth, depth_max)
    finite = np.clip(finite, depth_min, depth_max)
    # Closer = whiter (Depth-ControlNet convention)
    span = max(depth_max - depth_min, 1e-3)
    norm = 1.0 - (finite - depth_min) / span
    norm = np.clip(norm, 0.0, 1.0)
    grid = (norm.reshape(grid_h, grid_w) * 255).astype(np.uint8)

    # Upsample to image_size
    img = Image.fromarray(grid).resize((width, height), Image.BICUBIC)
    return img


def render_template_depth(template: dict) -> dict[str, Any]:
    """Convenience: build scene + render depth map. Returns dict with image
    + scene metadata (focus_room name, etc) so prompts can be built."""
    scene = build_scene(template)
    img = render_depth_map(scene)
    return {
        "depth_image": img,
        "focus_room_name": (scene.get("focus_room") or {}).get("name", ""),
        "focus_room_type": (scene.get("focus_room") or {}).get("type", ""),
        "focus_room_area": (scene.get("focus_room") or {}).get("area_sqm", 0),
        "ceiling_height_m": scene.get("ceiling_height", 2.7),
    }


# ---------------------------------------------------------------------------
# Dollhouse / cutaway view — full apartment seen from above with no ceiling
# ---------------------------------------------------------------------------

def build_dollhouse_scene(template: dict, image_size: tuple[int, int] = (768, 512),
                           wall_height_factor: float = 1.0) -> dict[str, Any]:
    """3D scene + camera for a dollhouse cutaway of the WHOLE apartment.

    No ceiling, walls at full or reduced height, camera positioned at a
    high oblique angle so all rooms are visible (real-estate
    Matterport-style 3D model view).

    wall_height_factor < 1 makes walls shorter so rooms read more clearly
    from above (1.0 = full ceiling height).
    """
    rooms = template.get("rooms", []) or []
    if not rooms and template.get("floors"):
        for fl in template["floors"]:
            if fl.get("rooms"):
                rooms = fl["rooms"]
                break
    if not rooms:
        return {"mesh": trimesh.Trimesh(), "camera_transform": np.eye(4),
                "image_size": image_size, "rooms_count": 0}

    boundary = template.get("boundary", {})
    ceiling_h = boundary.get("ceiling_height_mm", 2700) / 1000.0
    wall_h = ceiling_h * wall_height_factor

    walls = _build_walls_mesh(rooms, ceiling_height=wall_h)
    floors_only = _build_floor_ceiling_mesh(rooms, ceiling_height=ceiling_h,
                                              include_ceiling=False)
    parts = [m for m in (walls, floors_only) if not m.is_empty]
    scene_mesh = trimesh.util.concatenate(parts) if parts else trimesh.Trimesh()

    # Compute apartment bounds across all room polygons
    all_pts = np.concatenate([np.array(r["polygon"], dtype=float) for r in rooms
                                if r.get("polygon")])
    minx, miny = all_pts.min(axis=0)
    maxx, maxy = all_pts.max(axis=0)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    width = maxx - minx
    depth = maxy - miny
    diag = math.hypot(width, depth)

    # Camera high above + offset from one corner, looking at apartment center.
    # Pull back symmetrically so the apartment isn't squeezed to one edge,
    # use enough height to clear the walls and look DOWN into the rooms.
    max_dim = max(width, depth)
    pull = max(max_dim * 0.9, 6.0)
    cam_height = max(max_dim * 1.15, wall_h * 2.5, 9.0)
    cam_x = cx - pull
    cam_y = cy - pull
    cam_z = cam_height
    target = np.array([cx, cy, wall_h * 0.3])

    cam_pos = np.array([cam_x, cam_y, cam_z])
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0, 0, 1])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    new_up = np.cross(right, forward)
    transform = np.eye(4)
    transform[:3, 0] = right
    transform[:3, 1] = new_up
    transform[:3, 2] = -forward
    transform[:3, 3] = cam_pos

    # Adjust focal so the projected diagonal of the apartment fills ~70% of frame.
    width_px, _ = image_size
    distance = float(np.linalg.norm(cam_pos - target))
    # apartment diagonal projected from this oblique angle is roughly diag * cos(pitch)
    # — but easier: just use diag itself as the size to fit, with margin.
    fit_dim = max(diag * 1.15, 4.0)
    focal = distance * width_px / fit_dim

    # Depth range scaled to scene: nearest visible point ~ pull*0.7,
    # furthest ~ distance + diag/2. Using these clamps the dynamic range so
    # the depth map has good contrast across the apartment.
    depth_min = max(distance - diag * 0.7, 1.0)
    depth_max = distance + diag * 0.7

    return {
        "mesh": scene_mesh,
        "camera_transform": transform,
        "image_size": image_size,
        "focal_length": float(focal),
        "rooms_count": len(rooms),
        "boundary_w": float(width),
        "boundary_d": float(depth),
        "ceiling_height": ceiling_h,
        "wall_height": wall_h,
        "depth_min": float(depth_min),
        "depth_max": float(depth_max),
    }


def render_template_dollhouse_depth(template: dict,
                                      image_size: tuple[int, int] = (768, 512),
                                      wall_height_factor: float = 1.0,
                                      ) -> dict[str, Any]:
    """Build a dollhouse scene of the full apartment + render its depth map."""
    scene = build_dollhouse_scene(template, image_size=image_size,
                                    wall_height_factor=wall_height_factor)
    img = render_depth_map(scene)
    md = template.get("metadata", {}) or {}
    return {
        "depth_image": img,
        "country": md.get("country", ""),
        "bedrooms": md.get("bedrooms", 0),
        "total_area_sqm": md.get("total_area_sqm", 0),
        "rooms_count": scene.get("rooms_count", 0),
        "boundary_w": scene.get("boundary_w", 0.0),
        "boundary_d": scene.get("boundary_d", 0.0),
        "ceiling_height_m": scene.get("ceiling_height", 2.7),
    }


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        sys.exit("usage: depth_renderer.py <template.json>")
    template = json.load(open(sys.argv[1]))
    info = render_template_depth(template)
    out = "/tmp/depth_test.png"
    info["depth_image"].save(out)
    print(f"Wrote {out}")
    print(f"Focus room: {info['focus_room_name']} ({info['focus_room_type']}, "
          f"{info['focus_room_area']} m²)")
