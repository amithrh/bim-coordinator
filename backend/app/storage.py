"""Template + asset loading. Single source of truth for data paths."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "data" / "templates"
IFC_DIR = REPO_ROOT / "data" / "ifc_samples"
SVG_DIR = REPO_ROOT / "data" / "svg_plans"


@lru_cache(maxsize=1)
def load_all_templates() -> list[dict]:
    templates: list[dict] = []
    for path in sorted(TEMPLATES_DIR.glob("*/*.json")):
        templates.append(json.loads(path.read_text()))
    return templates


def by_id(template_id: str) -> dict | None:
    for t in load_all_templates():
        if t["id"] == template_id:
            return t
    return None


def ifc_path(template_id: str) -> Path:
    return IFC_DIR / f"{template_id}.ifc"


def svg_path(template_id: str) -> Path:
    return SVG_DIR / f"{template_id}.svg"
