"""Architectural quality constraints for generated layouts.

The IFC validator only checks polygon validity (no overlaps, doors on edges).
This module adds the *human* constraints — rooms have to be usable, not just
geometrically valid.

A 0.4m wide WC passes IFC validation but isn't a real toilet. We reject those.
"""

from __future__ import annotations

from dataclasses import dataclass


# Minimum room dimensions in meters (width AND depth must clear these).
# Tuned against real apartment plans (Indian, European, Japanese) — these are
# the *practically* usable minimums, not the aspirational maximums.
MIN_DIM_BY_TYPE: dict[str, tuple[float, float]] = {
    # type: (min_short_side, min_long_side)
    "entry":          (1.0, 1.2),  # foyer / hall — circulation
    "wc":             (1.0, 1.2),  # toilet only
    "bathroom":       (1.4, 1.8),  # full bath
    "kitchen":        (1.6, 2.0),  # functional kitchen
    "living":         (2.5, 3.0),  # real living room
    "dining":         (2.2, 2.5),  # dining area
    "master_bedroom": (2.4, 2.8),  # double bed + wardrobe
    "bedroom":        (2.2, 2.5),  # single/twin bed
    "balcony":        (1.0, 1.2),  # functional balcony
    "utility":        (0.9, 1.2),  # laundry/utility
    "store_room":     (0.8, 1.0),  # closet
}

# Default for any unmapped type (covers "Studio Living/Sleep" etc.)
DEFAULT_MIN_DIM = (1.4, 1.8)

# Maximum allowed aspect ratio (long_side / short_side). Above this rooms feel
# like corridors. We allow more leeway for narrow types (corridors, balconies).
MAX_ASPECT_RATIO = 3.0
CORRIDOR_TYPES = {"entry", "balcony", "store_room"}


@dataclass
class QualityIssue:
    room_name: str
    room_type: str
    width: float
    depth: float
    issue: str
    severity: int  # 1=minor, 2=warning, 3=fatal (room is unusable)


def _aspect(w: float, h: float) -> float:
    if min(w, h) <= 0:
        return 999
    return max(w, h) / min(w, h)


def check_room(room: dict) -> list[QualityIssue]:
    """Find all quality issues for a single room."""
    issues: list[QualityIssue] = []
    poly = room["polygon"]
    w = round(poly[1][0] - poly[0][0], 3)
    h = round(poly[3][1] - poly[0][1], 3)
    rtype = room.get("type", "")
    name = room.get("name", "?")

    short, long_ = sorted([w, h])
    min_short, min_long = MIN_DIM_BY_TYPE.get(rtype, DEFAULT_MIN_DIM)

    if short < min_short - 0.001:
        issues.append(QualityIssue(
            name, rtype, w, h,
            f"too narrow: short side {short:.2f}m < required {min_short}m",
            severity=3,
        ))
    if long_ < min_long - 0.001:
        issues.append(QualityIssue(
            name, rtype, w, h,
            f"too short: long side {long_:.2f}m < required {min_long}m",
            severity=3,
        ))

    # Aspect ratio (skip corridors)
    if rtype not in CORRIDOR_TYPES:
        a = _aspect(w, h)
        if a > MAX_ASPECT_RATIO + 0.001:
            issues.append(QualityIssue(
                name, rtype, w, h,
                f"awkward aspect: {a:.1f}:1 ({w:.1f}×{h:.1f}m) > max {MAX_ASPECT_RATIO}",
                severity=2 if a < 3.5 else 3,
            ))

    return issues


def audit_template(template: dict) -> list[QualityIssue]:
    """Find all quality issues in a template."""
    issues: list[QualityIssue] = []
    for r in template.get("rooms", []):
        issues.extend(check_room(r))
    return issues


def quality_score(template: dict) -> tuple[float, list[QualityIssue]]:
    """Score the template's *architectural* quality, not just geometric validity.

    Returns (score_0_to_100, list_of_issues).
    Score formula:
      - Start at 100
      - Each fatal issue: -25 (room unusable)
      - Each warning issue: -10
      - Each minor issue: -3
      - Bonus: living room close-to-square (+5 if aspect <1.4, +3 if <1.7)
      - Bonus: master bedroom close-to-square (+3 if aspect <1.7)
    """
    issues = audit_template(template)
    score = 100.0
    for issue in issues:
        if issue.severity == 3:
            score -= 25
        elif issue.severity == 2:
            score -= 10
        else:
            score -= 3

    # Bonuses for nice rooms
    rooms = template.get("rooms", [])
    living = next((r for r in rooms if r.get("type") == "living"), None)
    master = next((r for r in rooms if r.get("type") == "master_bedroom"), None)
    for r, bonus_full, bonus_half in [(living, 5, 3), (master, 3, 1)]:
        if not r:
            continue
        poly = r["polygon"]
        w = poly[1][0] - poly[0][0]
        h = poly[3][1] - poly[0][1]
        a = _aspect(w, h)
        if a < 1.4:
            score += bonus_full
        elif a < 1.7:
            score += bonus_half

    return max(0.0, score), issues


def has_fatal_issues(template: dict) -> bool:
    """True if any room is architecturally unusable (severity-3 issue)."""
    for r in template.get("rooms", []):
        for issue in check_room(r):
            if issue.severity == 3:
                return True
    return False


if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, '.')
    from backend.app.template_generator import TemplateProgram
    from backend.app.program_extractor import extract_program
    from backend.app.layout_strategies import generate_alternatives

    BRIEFS = [
        "1-bed Berlin Altbau couple 60 m²",
        "2 BHK Bangalore family 75 sqm balcony",
    ]
    for brief in BRIEFS:
        print(f"\n=== {brief} ===")
        prog = TemplateProgram.from_dict(extract_program(brief))
        for sn, t, _, _ in generate_alternatives(prog, n=3):
            score, issues = quality_score(t)
            fatal = sum(1 for i in issues if i.severity == 3)
            warn = sum(1 for i in issues if i.severity == 2)
            print(f"  [{sn:18s}] qscore={score:5.1f} fatal={fatal} warn={warn}")
