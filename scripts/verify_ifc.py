"""Programmatic verification of a generated IFC against the source template.

Checks: schema, units, spatial hierarchy, entity counts, placements,
representations, containment, world-coord rendering, boundary fit, z range,
placement uniqueness.

Usage:
    python verify_ifc.py <template.json> <ifc.ifc>
    # or verify all built templates:
    python verify_ifc.py --all
"""
from __future__ import annotations

import json
import multiprocessing
import sys
from collections import defaultdict
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.placement

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "data" / "templates"
IFC_DIR = REPO_ROOT / "data" / "ifc_samples"


def _bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), max(xs), min(ys), max(ys)


def verify(template_path: Path, ifc_path: Path,
            wall_thickness_buf: float | None = None) -> tuple[int, int, list[str]]:
    """Returns (passed, total, lines)."""
    template = json.loads(template_path.read_text())
    boundary = template["boundary"]["polygon"]
    bx_min, bx_max, by_min, by_max = _bbox(boundary)
    floors_data = template.get("floors")
    is_multifloor = bool(floors_data)
    expected_storeys = len(floors_data) if is_multifloor else 1
    if is_multifloor:
        expected_height = sum(
            (fl.get("ceiling_height_mm", template["boundary"]["ceiling_height_mm"]) / 1000.0)
            for fl in floors_data
        )
        n_doors = sum(len(fl.get("doors", [])) for fl in floors_data)
        n_windows = sum(len(fl.get("windows", [])) for fl in floors_data)
        n_rooms = sum(len(fl["rooms"]) for fl in floors_data)
        n_slabs_expected = len(floors_data)
    else:
        expected_height = template["boundary"]["ceiling_height_mm"] / 1000.0
        n_doors = len(template["doors"])
        n_windows = len(template["windows"])
        n_rooms = len(template["rooms"])
        n_slabs_expected = 1
    if wall_thickness_buf is None:
        t = template["boundary"]["wall_thickness_mm"] / 1000.0
        wall_thickness_buf = round(t / 2 + 0.05, 3)

    m = ifcopenshell.open(str(ifc_path))
    checks: list[tuple[str, bool, str]] = []

    def chk(name, ok, detail=""):
        checks.append((name, ok, detail))

    chk("Schema is IFC4", m.schema == "IFC4", m.schema)

    length_unit = next((u for u in m.by_type("IfcSIUnit")
                        if u.UnitType == "LENGTHUNIT"), None)
    chk("Length unit is METRE",
        length_unit and length_unit.Name == "METRE" and length_unit.Prefix is None,
        f"prefix={length_unit.Prefix} name={length_unit.Name}" if length_unit else "missing")

    proj = m.by_type("IfcProject")
    sites = m.by_type("IfcSite")
    bldgs = m.by_type("IfcBuilding")
    storeys = m.by_type("IfcBuildingStorey")
    chk(f"Spatial: 1 project / 1 site / 1 building / {expected_storeys} storey(s)",
        len(proj) == 1 and len(sites) == 1 and len(bldgs) == 1 and len(storeys) == expected_storeys,
        f"got {len(storeys)} storeys")

    chk("Site aggregated under project", bool(sites and sites[0].Decomposes))
    chk("Building aggregated under site", bool(bldgs and bldgs[0].Decomposes))
    chk("All storeys aggregated under building",
        all(bool(s.Decomposes) for s in storeys))

    # Entity-count expectations
    chk(f"IfcDoor count = {n_doors}",
        len(m.by_type("IfcDoor")) == n_doors,
        f"actual={len(m.by_type('IfcDoor'))}")
    chk(f"IfcWindow count = {n_windows}",
        len(m.by_type("IfcWindow")) == n_windows,
        f"actual={len(m.by_type('IfcWindow'))}")
    chk(f"IfcSpace count = {n_rooms}",
        len(m.by_type("IfcSpace")) == n_rooms,
        f"actual={len(m.by_type('IfcSpace'))}")
    chk(f"IfcSlab count = {n_slabs_expected}",
        len(m.by_type("IfcSlab")) == n_slabs_expected,
        f"actual={len(m.by_type('IfcSlab'))}")
    n_walls_actual = len(m.by_type("IfcWall"))
    chk(f"IfcWall count > 0 (built {n_walls_actual} segments)",
        n_walls_actual > 0)

    for cls in ("IfcWall", "IfcDoor", "IfcWindow", "IfcSlab", "IfcSpace"):
        miss_pl = sum(1 for e in m.by_type(cls) if e.ObjectPlacement is None)
        miss_rep = sum(1 for e in m.by_type(cls) if e.Representation is None)
        chk(f"{cls}: all have ObjectPlacement", miss_pl == 0, f"{miss_pl} missing")
        chk(f"{cls}: all have Representation", miss_rep == 0, f"{miss_rep} missing")

    for cls in ("IfcWall", "IfcDoor", "IfcWindow", "IfcSlab"):
        miss = sum(1 for e in m.by_type(cls) if not e.ContainedInStructure)
        chk(f"{cls}: contained in storey", miss == 0, f"{miss} not contained")
    sm = sum(1 for s in m.by_type("IfcSpace") if not s.Decomposes)
    chk("IfcSpace: aggregated into storey", sm == 0, f"{sm} not aggregated")

    # World-coord geom
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    it = ifcopenshell.geom.iterator(settings, m, multiprocessing.cpu_count())
    rendered = defaultdict(int)
    oob = []
    all_z = []
    if it.initialize():
        while True:
            s = it.get()
            v = s.geometry.verts
            if v:
                xs, ys, zs = v[0::3], v[1::3], v[2::3]
                all_z += list(zs)
                rendered[s.type] += 1
                if (min(xs) < bx_min - wall_thickness_buf or
                    max(xs) > bx_max + wall_thickness_buf or
                    min(ys) < by_min - wall_thickness_buf or
                    max(ys) > by_max + wall_thickness_buf):
                    oob.append(f"{s.type} {s.guid[:8]}")
            if not it.next():
                break

    chk(f"All {n_walls_actual} walls render", rendered["IfcWall"] == n_walls_actual,
        f"rendered={rendered['IfcWall']}")
    chk(f"All {n_doors} doors render", rendered["IfcDoor"] == n_doors,
        f"rendered={rendered['IfcDoor']}")
    chk(f"All {n_windows} windows render", rendered["IfcWindow"] == n_windows,
        f"rendered={rendered['IfcWindow']}")
    chk(f"All {n_rooms} spaces render", rendered["IfcSpace"] == n_rooms,
        f"rendered={rendered['IfcSpace']}")
    chk(f"All {n_slabs_expected} slab(s) render",
        rendered["IfcSlab"] == n_slabs_expected,
        f"rendered={rendered['IfcSlab']}")

    chk(f"No element extends past boundary (±{wall_thickness_buf}m allowance)",
        len(oob) == 0, f"{len(oob)} OOB: {', '.join(oob[:3])}")

    if all_z:
        chk("Slab depth z >= -0.21", min(all_z) >= -0.21, f"min z={min(all_z):.3f}")
        chk(f"Top z reaches expected ~{expected_height:.2f} (cumulative across floors)",
            abs(max(all_z) - expected_height) < 0.10, f"max z={max(all_z):.3f}")

    # Placement diversity
    wall_positions = []
    for w in m.by_type("IfcWall"):
        mat = ifcopenshell.util.placement.get_local_placement(w.ObjectPlacement)
        wall_positions.append((round(mat[0, 3], 3), round(mat[1, 3], 3)))
    if wall_positions:
        unique = len(set(wall_positions))
        chk(f"Walls have distinct placements ({unique}/{len(wall_positions)})",
            unique >= len(wall_positions) * 0.7,
            f"{unique}/{len(wall_positions)}")

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    lines = []
    for name, ok, detail in checks:
        sym = "PASS" if ok else "FAIL"
        suffix = (f"  [{detail}]" if detail and not ok
                   else (f"  ({detail})" if detail else ""))
        lines.append(f"  [{sym}] {name}{suffix}")
    return passed, total, lines


def main(argv):
    if len(argv) == 2 and argv[1] == "--all":
        any_failed = False
        for t in sorted(TEMPLATES_DIR.glob("*/*.json")):
            ifc = IFC_DIR / f"{t.stem}.ifc"
            if not ifc.exists():
                print(f"SKIP {t.stem} — IFC not built")
                continue
            print(f"\n{'=' * 72}\n  {t.stem}\n{'=' * 72}")
            passed, total, lines = verify(t, ifc)
            print("\n".join(lines))
            print(f"\n  Result: {passed}/{total} {'PASS' if passed == total else 'FAIL'}")
            if passed != total:
                any_failed = True
        return 1 if any_failed else 0
    if len(argv) != 3:
        print("usage: verify_ifc.py [<template.json> <ifc.ifc> | --all]", file=sys.stderr)
        return 2
    passed, total, lines = verify(Path(argv[1]), Path(argv[2]))
    print("\n".join(lines))
    print(f"\n  Result: {passed}/{total} {'PASS' if passed == total else 'FAIL'}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
