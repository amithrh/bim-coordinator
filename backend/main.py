"""FastAPI surface for the BIM Coordinator Phase 1 dashboard.

Endpoints:
  POST /api/brief     — extract a structured brief from free text
  POST /api/retrieve  — return top-N matching templates as cards
  POST /api/modify    — apply mods to a template, return modified IFC + JSON
  GET  /api/templates/{id}/svg
  GET  /api/templates/{id}/ifc
  GET  /api/templates/{id}/json   — raw template (for in-browser 3D)
  GET  /api/templates             — list all template IDs
"""
from __future__ import annotations

import json
import sys
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Make ./scripts importable for build_template
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_template import build  # noqa: E402

from backend.app import storage  # noqa: E402
from backend.app.brief_extractor import extract as extract_brief  # noqa: E402
from backend.app.modifier import apply_modifications  # noqa: E402
from backend.app.retrieval import retrieve  # noqa: E402

# Modified IFCs and SVGs land here; cleared on each restart.
MODIFIED_DIR = Path(tempfile.gettempdir()) / "bim_coordinator_modified"
MODIFIED_DIR.mkdir(exist_ok=True)

app = FastAPI(title="BIM Coordinator", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- request models ----------

class BriefRequest(BaseModel):
    text: str


class RetrieveRequest(BaseModel):
    brief: dict[str, Any]
    top_n: int = 4


class ModifyRequest(BaseModel):
    template_id: str
    mods: dict[str, Any]


class MoveOpeningRequest(BaseModel):
    template_id: str
    kind: str  # "door" or "window"
    index: int
    new_position: list[float]


# ---------- endpoints ----------

@app.get("/api/health")
def health():
    return {"ok": True, "templates": len(storage.load_all_templates())}


@app.get("/api/templates")
def list_templates():
    return {
        "templates": [
            {"id": t["id"], "metadata": t["metadata"]}
            for t in storage.load_all_templates()
        ]
    }


@app.get("/api/templates/{template_id}/json")
def template_json(template_id: str):
    t = storage.by_id(template_id)
    if t is None:
        raise HTTPException(404, f"unknown template {template_id}")
    return JSONResponse(t)


@app.get("/api/templates/{template_id}/svg")
def template_svg(template_id: str):
    p = storage.svg_path(template_id)
    if not p.exists():
        raise HTTPException(404, f"svg not built for {template_id}")
    return FileResponse(p, media_type="image/svg+xml")


@app.get("/api/templates/{template_id}/ifc")
def template_ifc(template_id: str):
    p = storage.ifc_path(template_id)
    if not p.exists():
        raise HTTPException(404, f"ifc not built for {template_id}")
    return FileResponse(
        p, media_type="application/octet-stream",
        filename=f"{template_id}.ifc",
    )


@app.post("/api/brief")
def brief(req: BriefRequest):
    return extract_brief(req.text)


@app.post("/api/retrieve")
def retrieve_endpoint(req: RetrieveRequest):
    cards = retrieve(req.brief, top_n=req.top_n)
    return {"cards": cards}


# ---------- modified template registry ----------
_modified_registry: dict[str, dict] = {}


@app.post("/api/modify")
def modify(req: ModifyRequest):
    base = storage.by_id(req.template_id)
    if base is None:
        # Maybe it's an already-modified template
        base = _modified_registry.get(req.template_id)
        if base is None:
            raise HTTPException(404, f"unknown template {req.template_id}")
    modified, errors = apply_modifications(base, req.mods)
    if errors:
        return JSONResponse(
            status_code=422,
            content={"ok": False, "errors": errors,
                     "message": "This modification would break the layout."},
        )
    # Generate a unique id and persist artifacts
    mod_id = f"mod_{uuid.uuid4().hex[:8]}"
    modified["id"] = mod_id
    _modified_registry[mod_id] = modified
    ifc_out = MODIFIED_DIR / f"{mod_id}.ifc"
    build(modified, ifc_out)
    # Render an SVG for the cards / detail view
    from render_svg import render_one  # noqa: E402
    import json
    json_path = MODIFIED_DIR / f"{mod_id}.json"
    json_path.write_text(json.dumps(modified))
    svg_out = MODIFIED_DIR / f"{mod_id}.svg"
    render_one(json_path, svg_out)
    return {
        "ok": True,
        "template": modified,
        "modified_id": mod_id,
    }


@app.get("/api/modified/{mod_id}/ifc")
def modified_ifc(mod_id: str):
    p = MODIFIED_DIR / f"{mod_id}.ifc"
    if not p.exists():
        raise HTTPException(404, f"unknown modified id {mod_id}")
    return FileResponse(p, media_type="application/octet-stream",
                          filename=f"{mod_id}.ifc")


@app.get("/api/modified/{mod_id}/svg")
def modified_svg(mod_id: str):
    p = MODIFIED_DIR / f"{mod_id}.svg"
    if not p.exists():
        raise HTTPException(404, f"unknown modified id {mod_id}")
    return FileResponse(p, media_type="image/svg+xml")


@app.get("/api/modified/{mod_id}/json")
def modified_json(mod_id: str):
    t = _modified_registry.get(mod_id)
    if t is None:
        raise HTTPException(404, f"unknown modified id {mod_id}")
    return JSONResponse(t)


@app.post("/api/move_opening")
def move_opening(req: MoveOpeningRequest):
    """Move a single door or window to a new position.

    The new position must lie on a valid edge — for doors, on a shared edge
    between the connected rooms (or boundary if 'outside'); for windows, on
    a boundary edge of the connected room. The hard validation gate runs
    over the full template, so any modification that would silently break
    the layout is rejected with a 422 + error list."""
    base = storage.by_id(req.template_id)
    if base is None:
        base = _modified_registry.get(req.template_id)
        if base is None:
            raise HTTPException(404, f"unknown template {req.template_id}")

    out = deepcopy(base)
    pos = [float(req.new_position[0]), float(req.new_position[1])]
    if req.kind == "door":
        if req.index < 0 or req.index >= len(out["doors"]):
            raise HTTPException(400, f"door index {req.index} out of range")
        out["doors"][req.index]["position"] = pos
    elif req.kind == "window":
        if req.index < 0 or req.index >= len(out["windows"]):
            raise HTTPException(400, f"window index {req.index} out of range")
        out["windows"][req.index]["position"] = pos
    else:
        raise HTTPException(400, f"invalid kind {req.kind}")

    # Validate full template — same hard gate as /api/modify
    from backend.app.modifier import validate_dict  # noqa: E402
    errors = validate_dict(out)
    if errors:
        return JSONResponse(
            status_code=422,
            content={"ok": False, "errors": errors,
                     "message": f"This position is not valid for the {req.kind}."},
        )

    mod_id = f"mov_{uuid.uuid4().hex[:8]}"
    out["id"] = mod_id
    _modified_registry[mod_id] = out

    ifc_out = MODIFIED_DIR / f"{mod_id}.ifc"
    build(out, ifc_out)

    from render_svg import render_one  # noqa: E402
    json_path = MODIFIED_DIR / f"{mod_id}.json"
    json_path.write_text(json.dumps(out))
    svg_out = MODIFIED_DIR / f"{mod_id}.svg"
    render_one(json_path, svg_out)

    return {"ok": True, "template": out, "modified_id": mod_id}
