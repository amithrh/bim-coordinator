"""Synthesize a multi-view RGB dataset from a BIM template, ready to
train a 3D Gaussian Splat with msplat / nerfstudio / gsplat.

Pipeline per camera pose:
    BIM JSON  ->  trimesh scene + depth-map raycast (90° FOV)
              ->  SDXL-turbo + Depth ControlNet  ->  RGB PNG
    Camera pose (cam-to-world 4x4) recorded in nerfstudio transforms.json.

Output layout (compatible with msplat, nerfstudio, gsplat dataparsers):

    /tmp/splat_dataset/<template_id>/
        transforms.json
        images/
            frame_000.png
            frame_001.png
            ...

Camera path: for each habitable room we place 8 cameras around the
centroid, all looking inward — gives parallax (8 positions) AND overlap
(every camera sees the room centre) so 3DGS has enough signal to
densify Gaussians around the walls and floor.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import requests
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.depth_renderer import (
    _build_walls_mesh,
    _build_floor_ceiling_mesh,
    render_depth_map,
)
from backend.app import image_renderer  # imported as module so we read _PIPE_CN live
from backend.app.image_renderer import (
    _ensure_controlnet_pipeline_loaded,
    _INFER_LOCK,
    _build_faithful_prompt,
    _resolve_style,
)


# Skip these room types — no useful interior content for a tour
SKIP_ROOM_TYPES = {
    "corridor", "stairs", "balcony", "loggia", "terrace",
    "store", "wardrobe", "abstellraum", "passage", "duct",
}


# ---------------------------------------------------------------------------
# Camera path generation
# ---------------------------------------------------------------------------

def _room_centroid(room: dict) -> tuple[float, float]:
    poly = np.array(room["polygon"], dtype=float)
    c = poly.mean(axis=0)
    return float(c[0]), float(c[1])


def _room_radius(room: dict) -> float:
    """Inscribed-circle-ish radius — distance from centroid to nearest wall."""
    poly = np.array(room["polygon"], dtype=float)
    cx, cy = poly.mean(axis=0)
    minp = poly.min(axis=0)
    maxp = poly.max(axis=0)
    return min(cx - minp[0], maxp[0] - cx, cy - minp[1], maxp[1] - cy)


def _camera_poses_for_room(room: dict, eye_h: float = 1.65,
                              n_orbit: int = 12) -> list[dict]:
    """Dense camera coverage of one room — chosen to give 3DGS enough
    parallax signal to densify Gaussians on every wall + floor + ceiling.

    Per room (small):  ~22 poses
    Per room (large):  ~32 poses

    Layout:
      a) Outer orbit ring (12) — eye-height cameras 1m from centroid, looking IN
      b) Inner ring (8)        — 0.5m from centroid, looking OUT at far walls
      c) Centroid views (4)    — looking the 4 cardinal directions
      d) High views (4)        — eye height + 0.6m, slight downward pitch
                                  (covers ceiling + the parts of walls far cameras can't see)
      e) Low view (1, optional)— eye height - 0.5m, looking up

    All cameras share the same SEED at render time so SDXL paints the same
    interior style across views, dramatically reducing ghosting in the splat.
    """
    cx, cy = _room_centroid(room)
    R = _room_radius(room)
    r_outer = max(0.5, min(R - 0.4, 1.5))
    r_inner = max(0.2, r_outer * 0.5)

    poses: list[dict] = []

    # a) Outer ring — looking inward
    for i in range(n_orbit):
        ang = 2 * math.pi * i / n_orbit
        px = cx + r_outer * math.cos(ang)
        py = cy + r_outer * math.sin(ang)
        poses.append({
            "position": (px, py, eye_h),
            "target":   (cx, cy, 1.4),
            "tag": f"orbit_out_{i}",
        })

    # b) Inner ring — looking outward (gives the OPPOSITE walls a clear shot)
    for i in range(8):
        ang = 2 * math.pi * i / 8 + math.pi / 8  # offset from outer ring
        px = cx + r_inner * math.cos(ang)
        py = cy + r_inner * math.sin(ang)
        # Look toward a far point on a WALL in this direction
        tx = cx + 3.0 * math.cos(ang)
        ty = cy + 3.0 * math.sin(ang)
        poses.append({
            "position": (px, py, eye_h),
            "target":   (tx, ty, 1.55),
            "tag": f"orbit_in_{i}",
        })

    # c) Centroid — 4 cardinal views
    for i in range(4):
        ang = 2 * math.pi * i / 4
        poses.append({
            "position": (cx, cy, eye_h),
            "target":   (cx + 3.0 * math.cos(ang), cy + 3.0 * math.sin(ang), 1.55),
            "tag": f"centre_{i}",
        })

    # d) High views — slightly tilted down, covers ceiling+upper walls
    for i in range(4):
        ang = 2 * math.pi * i / 4 + math.pi / 4
        px = cx + r_outer * 0.5 * math.cos(ang)
        py = cy + r_outer * 0.5 * math.sin(ang)
        poses.append({
            "position": (px, py, eye_h + 0.5),
            "target":   (cx, cy, 1.0),
            "tag": f"high_{i}",
        })

    return poses


def _all_camera_poses(template: dict) -> list[dict]:
    """Walk every habitable room, collect all camera poses."""
    rooms = template.get("rooms", []) or []
    if not rooms and template.get("floors"):
        rooms = []
        for fl in template["floors"]:
            rooms.extend(fl.get("rooms", []))

    poses = []
    for r in rooms:
        rtype = (r.get("type") or "").lower()
        if rtype in SKIP_ROOM_TYPES:
            continue
        if not r.get("polygon") or len(r["polygon"]) < 3:
            continue
        for p in _camera_poses_for_room(r):
            p["room_id"] = r.get("id")
            p["room_name"] = r.get("name", "")
            p["room_type"] = rtype
            poses.append(p)
    return poses


# ---------------------------------------------------------------------------
# Camera math — build a look-at extrinsic matrix
# ---------------------------------------------------------------------------

def _look_at_transform(pos: tuple[float, float, float],
                          target: tuple[float, float, float],
                          up: tuple[float, float, float] = (0, 0, 1),
                          ) -> np.ndarray:
    """Returns a 4x4 cam-to-world matrix. OpenGL/NeRF convention:
       column 0 = camera right
       column 1 = camera up
       column 2 = camera back   (camera looks toward -Z)
       column 3 = camera position
    """
    p = np.array(pos, dtype=float)
    t = np.array(target, dtype=float)
    fwd = t - p
    n = np.linalg.norm(fwd)
    if n < 1e-6:
        fwd = np.array([1.0, 0.0, 0.0])
    else:
        fwd = fwd / n
    up_v = np.array(up, dtype=float)
    right = np.cross(fwd, up_v)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / rn
    new_up = np.cross(right, fwd)

    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = new_up
    T[:3, 2] = -fwd       # camera looks along its own -Z
    T[:3, 3] = p
    return T


# ---------------------------------------------------------------------------
# Per-pose render (depth raycast + SDXL+ControlNet)
# ---------------------------------------------------------------------------

def _build_scene_mesh(template: dict, include_ceiling: bool = True
                        ) -> trimesh.Trimesh:
    rooms = template.get("rooms", []) or []
    if not rooms and template.get("floors"):
        for fl in template["floors"]:
            if fl.get("rooms"):
                rooms = fl["rooms"]
                break
    boundary = template.get("boundary", {})
    ceiling_h = boundary.get("ceiling_height_mm", 2700) / 1000.0

    walls = _build_walls_mesh(rooms, ceiling_height=ceiling_h)
    floors_ceil = _build_floor_ceiling_mesh(rooms, ceiling_height=ceiling_h,
                                                include_ceiling=include_ceiling)
    parts = [m for m in (walls, floors_ceil) if not m.is_empty]
    if not parts:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(parts)


def _render_pose(template: dict, scene_mesh: trimesh.Trimesh,
                   transform: np.ndarray, room_meta: dict,
                   image_size: int = 768,
                   steps: int = 6,
                   controlnet_scale: float = 0.75,
                   seed: int = 42,
                   ) -> Image.Image:
    """Render one (depth + SDXL+ControlNet) frame from `transform`.

    Higher controlnet_scale (0.75 vs the interior render's 0.55) makes
    SDXL hew more strictly to the BIM geometry. We also use a fixed
    seed scoped to the ROOM (not the pose) so all views of the same
    room paint a consistent style — fewer ghosts when 3DGS averages
    pixels across views.
    """
    # Build scene dict that render_depth_map expects
    scene = {
        "mesh": scene_mesh,
        "camera_transform": transform,
        "image_size": (image_size, image_size),
        "focal_length": image_size / 2.0,  # 90° FOV
    }
    depth = render_depth_map(scene)
    depth_rgb = depth.convert("RGB").resize(
        (image_size, image_size), Image.BICUBIC,
    )

    # Build a lightweight prompt
    md = template.get("metadata", {}) or {}
    style = _resolve_style(md)
    rtype = room_meta.get("room_type", "living")
    room_words = {
        "living":         "living room, sofa and coffee table",
        "kitchen":        "kitchen with cabinetry",
        "kueche":         "kitchen with cabinetry",
        "kochnische":     "kitchen with cabinetry",
        "master_bedroom": "master bedroom with bed",
        "bedroom":        "bedroom with bed",
        "dining":         "dining area with table",
        "office":         "home office",
        "study":          "home office",
        "bathroom":       "bathroom with vanity and tub",
        "bad":            "bathroom with vanity and tub",
        "wc":             "small bathroom",
        "diele":          "entry foyer",
        "flur":           "entry foyer",
        "entry":          "entry foyer",
    }
    room_text = room_words.get(rtype, "interior room")
    prompt = (f"Photorealistic interior, {room_text}. {style}. "
              "natural daylight, sharp focus, real estate photography.")

    pipe = image_renderer._PIPE_CN  # read fresh — None at import, populated after load
    if pipe is None:
        raise RuntimeError("ControlNet pipeline not loaded")

    # Deterministic seed per ROOM so SDXL paints the same furniture/walls
    # in every view of that room. Different rooms get different seeds so
    # they don't all look identical.
    import torch
    room_id = room_meta.get("room_id") or room_meta.get("room_name") or ""
    room_seed = (seed + (abs(hash(room_id)) & 0xFFFF)) & 0xFFFFFFFF
    generator = torch.Generator(device="mps").manual_seed(int(room_seed))

    with _INFER_LOCK:
        img = pipe(
            prompt=prompt,
            image=depth_rgb,
            num_inference_steps=steps,
            guidance_scale=0.0,
            controlnet_conditioning_scale=controlnet_scale,
            height=image_size, width=image_size,
            generator=generator,
        ).images[0]
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(template_id: str, *,
            image_size: int = 768,
            include_ceiling: bool = True,
            depth_only: bool = False,
            output_dir: Path = Path("/tmp/splat_dataset"),
            api_base: str = "http://127.0.0.1:8009") -> Path:
    """Render the dataset. Returns the path it wrote to."""
    # Fetch template via API (so we use whatever's running, not stale disk)
    r = requests.get(f"{api_base}/api/templates/{template_id}/json")
    r.raise_for_status()
    template = r.json()

    out = output_dir / template_id
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)

    print(f"[splat] template = {template_id}")
    print(f"[splat] writing to {out}")

    # Build mesh once (re-used for every pose)
    scene_mesh = _build_scene_mesh(template, include_ceiling=include_ceiling)
    print(f"[splat] scene mesh: {len(scene_mesh.vertices)} verts, "
          f"{len(scene_mesh.faces)} faces")

    # Generate camera poses
    poses = _all_camera_poses(template)
    print(f"[splat] {len(poses)} camera poses across "
          f"{len({p['room_id'] for p in poses})} rooms")

    # Ensure SDXL+ControlNet is loaded (skip if depth_only)
    if not depth_only:
        if not _ensure_controlnet_pipeline_loaded():
            print("[splat] ERROR: ControlNet pipeline did not load")
            return out

    # transforms.json scaffold (nerfstudio / msplat / gsplat compatible)
    fl = image_size / 2.0   # 90° FOV
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": fl, "fl_y": fl,
        "cx": image_size / 2.0,
        "cy": image_size / 2.0,
        "w": image_size, "h": image_size,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "frames": [],
        # provenance
        "bim_template_id": template_id,
        "bim_country": template.get("metadata", {}).get("country", ""),
        "bim_total_area_sqm": template.get("metadata", {}).get("total_area_sqm", 0),
    }

    t0 = time.time()
    for i, pose in enumerate(poses):
        T = _look_at_transform(pose["position"], pose["target"])
        rel_path = f"images/frame_{i:03d}.png"
        out_path = out / rel_path

        ti0 = time.time()
        if depth_only:
            scene = {
                "mesh": scene_mesh,
                "camera_transform": T,
                "image_size": (image_size, image_size),
                "focal_length": image_size / 2.0,
            }
            img = render_depth_map(scene)
            img.convert("RGB").save(out_path)
        else:
            img = _render_pose(template, scene_mesh, T, pose,
                                  image_size=image_size)
            img.save(out_path)
        ti = time.time() - ti0

        transforms["frames"].append({
            "file_path": rel_path,
            "transform_matrix": T.tolist(),
            "bim_room_id": pose.get("room_id"),
            "bim_room_name": pose.get("room_name"),
            "bim_pose_tag": pose.get("tag"),
        })

        if (i + 1) % 5 == 0 or i == len(poses) - 1:
            elapsed = time.time() - t0
            est_total = elapsed / (i + 1) * len(poses)
            print(f"[splat] {i+1}/{len(poses)}  "
                  f"({pose['room_name']:18s} {pose['tag']:10s})  "
                  f"{ti:.2f}s  eta total {est_total:.1f}s")

    # Sample an initial point cloud from the BIM mesh — msplat /
    # nerfstudio init Gaussians from this. Our synthetic dataset has no
    # COLMAP SfM points, so we sample uniformly on the wall+floor+ceiling
    # surfaces, then colour each point by sampling the rendered images
    # that see it. ~30k points works well for a small apartment.
    n_init = 30000
    print(f"[splat] sampling {n_init} init points on the BIM mesh ...")
    pts, _ = trimesh.sample.sample_surface(scene_mesh, n_init)
    pts = np.asarray(pts, dtype=np.float32)

    # Color each point by averaging projections into a few rendered views
    # that see it. For each point: pick the camera whose forward best
    # aligns with the point-to-camera vector, project, sample the colour.
    if not depth_only:
        print("[splat] colouring points by projecting into nearest views ...")
        # Pre-load all images as numpy
        imgs_np: dict[str, np.ndarray] = {}
        for f in transforms["frames"]:
            ip = out / f["file_path"]
            imgs_np[f["file_path"]] = np.asarray(Image.open(ip)).astype(np.float32) / 255.0
        # Pre-extract camera position + cam-to-world matrices
        cams = []
        for f in transforms["frames"]:
            T = np.array(f["transform_matrix"], dtype=np.float32)
            cams.append((f["file_path"], T))
        # Camera intrinsics
        fl = transforms["fl_x"]
        cx_pix = transforms["cx"]
        cy_pix = transforms["cy"]
        W = transforms["w"]; H = transforms["h"]

        colors = np.zeros((n_init, 3), dtype=np.float32)
        votes = np.zeros(n_init, dtype=np.int32)
        for fp, T in cams:
            # World -> camera transform = inverse(c2w). For OpenGL c2w
            # convention, camera looks along -Z in camera space.
            R = T[:3, :3]
            t_w = T[:3, 3]
            # cam_pos in world; world to cam: x_cam = R^T (x_world - t_w)
            rel = pts - t_w
            cam_pts = rel @ R       # (N, 3) — col 0=right, 1=up, 2=back
            x = cam_pts[:, 0]; y = cam_pts[:, 1]; z = -cam_pts[:, 2]  # forward = +z after negation
            mask = z > 0.1
            if not mask.any():
                continue
            u = (x / z) * fl + cx_pix
            v = -(y / z) * fl + cy_pix
            inb = mask & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not inb.any():
                continue
            ui = np.round(u[inb]).astype(np.int32)
            vi = np.round(v[inb]).astype(np.int32)
            ui = np.clip(ui, 0, W - 1); vi = np.clip(vi, 0, H - 1)
            colors[inb] += imgs_np[fp][vi, ui, :]
            votes[inb] += 1
        votes_safe = np.maximum(votes, 1)
        colors /= votes_safe[:, None]
        # Points that no camera saw — keep them but use a neutral gray
        unseen = votes == 0
        colors[unseen] = np.array([0.55, 0.50, 0.45], dtype=np.float32)
    else:
        colors = np.full((n_init, 3), 0.55, dtype=np.float32)

    # Write points3D.ply (msplat's nerfstudio loader looks here automatically)
    ply_path = out / "points3D.ply"
    _write_ply_xyzrgb(ply_path, pts, colors)
    print(f"[splat] wrote {ply_path}  ({n_init} points)")
    transforms["ply_file_path"] = "points3D.ply"

    # Write transforms.json
    tj_path = out / "transforms.json"
    tj_path.write_text(json.dumps(transforms, indent=2))
    print(f"[splat] wrote {tj_path}  ({len(transforms['frames'])} frames)")

    # Validation contact sheet — quick visual to spot bad frames
    if not depth_only:
        try:
            _write_contact_sheet(out, transforms, image_size)
            print(f"[splat] wrote {out}/contact_sheet.jpg")
        except Exception as e:
            print(f"[splat] contact sheet failed: {e}")

    print(f"[splat] total: {time.time()-t0:.1f}s")
    return out


def _write_contact_sheet(out_dir: Path, transforms: dict, frame_size: int) -> None:
    """Write a JPG grid of all rendered frames so a human can scan
    quickly for SDXL failures (black frames, distorted geometry, etc).
    Tile is shrunk to keep the contact sheet under ~3 MB."""
    frames = transforms["frames"]
    n = len(frames)
    cols = min(8, max(4, int(math.sqrt(n))))
    rows = math.ceil(n / cols)
    tile = 256
    grid = Image.new("RGB", (cols * tile, rows * tile), (24, 24, 24))
    # Group frames by room with subtle dividers
    last_room = None
    for i, f in enumerate(frames):
        ip = out_dir / f["file_path"]
        if not ip.exists():
            continue
        img = Image.open(ip).convert("RGB").resize((tile, tile), Image.LANCZOS)
        x = (i % cols) * tile
        y = (i // cols) * tile
        grid.paste(img, (x, y))
    grid.save(out_dir / "contact_sheet.jpg", "JPEG", quality=82)


def _write_ply_xyzrgb(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a binary little-endian PLY with x,y,z float + r,g,b uchar."""
    n = xyz.shape[0]
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        # interleave xyz (3 floats) + rgb (3 uchars) per vertex
        for i in range(n):
            f.write(xyz[i].astype("<f4").tobytes())
            f.write(rgb_u8[i].tobytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("template_id", help="Template id, e.g. eu_de_1bed_munich_schwabing")
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--depth-only", action="store_true",
                     help="Skip SDXL — just save depth maps for camera-path validation")
    ap.add_argument("--output", type=Path, default=Path("/tmp/splat_dataset"))
    ap.add_argument("--api", default="http://127.0.0.1:8009")
    args = ap.parse_args()
    main(args.template_id,
            image_size=args.image_size,
            depth_only=args.depth_only,
            output_dir=args.output,
            api_base=args.api)
