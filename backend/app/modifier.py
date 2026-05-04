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

    # HARD VALIDATION GATE — the build plan §7.2 contract.
    errors = validate_dict(out)
    if errors:
        return None, errors
    return out, []
