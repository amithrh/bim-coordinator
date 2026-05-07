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
from backend.app.llm_client import reason as llm_reason  # noqa: E402
from backend.app.llm_generator import generate_from_brief as llm_generate  # noqa: E402
from backend.app.llm_modder import suggest_mods as llm_suggest_mods  # noqa: E402
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


# ---------- Stage 2: LLM reasoning ----------

class ReasonRequest(BaseModel):
    brief_text: str  # the original natural-language brief
    top_k: int = 10  # how many candidates to feed the model
    max_tokens: int = 700


class GenerateRequest(BaseModel):
    brief: str
    backend: str = "auto"        # "auto" | "claude_api" | "local_mlx" | "rules"
    max_attempts: int = 3


@app.post("/api/generate")
def generate_endpoint(req: GenerateRequest):
    """LEVEL 4: Free-form template generation from a brief (single best layout).

    The IFC produced is brand-new (never in the curated 500) AND verified
    by the same 35-check pipeline as the curated library.
    """
    result = llm_generate(req.brief, backend=req.backend, max_attempts=req.max_attempts)
    if not result.success or not result.template:
        return JSONResponse(status_code=422, content={
            "ok": False,
            "backend": result.backend,
            "attempts": result.attempts,
            "latency_s": result.latency_s,
            "errors_per_attempt": result.errors_per_attempt,
            "last_raw_response": result.raw_responses[-1] if result.raw_responses else "",
            "message": "Could not produce a valid template after retries.",
        })

    template = result.template
    template["id"] = f"mod_{uuid.uuid4().hex[:8]}_ai"
    _modified_registry[template["id"]] = template
    ifc_out = MODIFIED_DIR / f"{template['id']}.ifc"
    build(template, ifc_out)

    return {
        "ok": True,
        "modified_id": template["id"],
        "backend": result.backend,
        "attempts": result.attempts,
        "latency_s": result.latency_s,
        "program": result.program,
        "metadata": template["metadata"],
        "ifc_url": f"/api/modified/{template['id']}/ifc",
        "json_url": f"/api/modified/{template['id']}/json",
        "svg_url": f"/api/modified/{template['id']}/svg",
    }


class GenerateTowerRequest(BaseModel):
    brief: str  # natural-language tower brief


@app.post("/api/generate_tower")
def generate_tower_endpoint(req: GenerateTowerRequest):
    """LEVEL 5: Multi-story residential tower generation.

    Takes a brief like "Design a 20-story tower in Dubai inspired by Zaha Hadid"
    and produces a fully-validated multi-floor IFC with:
      - Lobby + amenity ground floor
      - N typical floors with K apartments each
      - Optional stepped massing at top (Zaha-inspired)
      - Penthouse on top floor

    Each floor is independently validated and the aggregated multi-story IFC
    passes the same 35-check pipeline as our curated 500 templates.
    """
    import time as _time
    from backend.app.tower_generator import parse_tower_brief, generate_tower

    t0 = _time.time()
    spec = parse_tower_brief(req.brief)
    template = generate_tower(spec)
    template["id"] = f"mod_{uuid.uuid4().hex[:8]}_tower"
    _modified_registry[template["id"]] = template
    ifc_out = MODIFIED_DIR / f"{template['id']}.ifc"
    try:
        build(template, ifc_out)
    except Exception as e:
        raise HTTPException(500, f"Failed to build tower IFC: {e}")
    return {
        "ok": True,
        "modified_id": template["id"],
        "spec": {
            "n_floors": spec.n_floors,
            "units_per_typical_floor": spec.units_per_typical_floor,
            "country": spec.country,
            "city": spec.city,
            "inspiration_architect": spec.inspiration_architect,
            "has_penthouse": spec.has_penthouse,
            "setback_top_n": spec.setback_top_n,
        },
        "metadata": template["metadata"],
        "n_floors": len(template["floors"]),
        "n_rooms_total": sum(len(f["rooms"]) for f in template["floors"]),
        "n_doors_total": sum(len(f["doors"]) for f in template["floors"]),
        "n_windows_total": sum(len(f["windows"]) for f in template["floors"]),
        "ifc_url": f"/api/modified/{template['id']}/ifc",
        "json_url": f"/api/modified/{template['id']}/json",
        "latency_s": round(_time.time() - t0, 4),
    }


class GenerateAlternativesRequest(BaseModel):
    brief: str
    n: int = 3


@app.post("/api/generate/alternatives")
def generate_alternatives_endpoint(req: GenerateAlternativesRequest):
    """LEVEL 4 (richer): generate N distinct layout strategies for the SAME brief.

    Returns N validated, geometrically-distinct layouts:
      - two_strip:        wet on top, dry on bottom
      - public_private:   vertical wing split
      - central_corridor: hallway down the middle (e.g. Berliner Korridor)

    Each is a brand-new IFC that wasn't in the curated 500. Each passes 35/35
    verification. They look visibly different and reflect different
    architectural traditions.
    """
    import time as _time
    from backend.app.layout_strategies import generate_alternatives
    from backend.app.template_generator import TemplateProgram
    from backend.app.program_extractor import extract_program

    t0 = _time.time()
    program_dict = extract_program(req.brief)
    program = TemplateProgram.from_dict(program_dict)
    alternatives = generate_alternatives(program, n=req.n)

    if not alternatives:
        raise HTTPException(422, "No layout strategy could produce a valid template")

    out = []
    for strategy_name, template, _errs, score in alternatives:
        template["id"] = f"mod_{uuid.uuid4().hex[:8]}_ai"
        _modified_registry[template["id"]] = template
        ifc_out = MODIFIED_DIR / f"{template['id']}.ifc"
        try:
            build(template, ifc_out)
        except Exception as e:
            # Skip this alternative if IFC build fails
            continue
        out.append({
            "strategy": strategy_name,
            "score": score,
            "modified_id": template["id"],
            "metadata": template["metadata"],
            "ifc_url": f"/api/modified/{template['id']}/ifc",
            "json_url": f"/api/modified/{template['id']}/json",
            "svg_url": f"/api/modified/{template['id']}/svg",
            "n_rooms": len(template["rooms"]),
            "n_doors": len(template["doors"]),
            "n_windows": len(template["windows"]),
        })

    return {
        "ok": True,
        "brief": req.brief,
        "program": program_dict,
        "alternatives": out,
        "total_latency_s": round(_time.time() - t0, 4),
    }


class LlmModifyRequest(BaseModel):
    template_id: str
    request: str  # natural-language modification request


@app.post("/api/llm_modify")
def llm_modify_endpoint(req: LlmModifyRequest):
    """LEVEL 2: LLM-driven modification of an existing template.

    Pipeline:
      1. Look up the template
      2. Ask LLM for parameter changes (area_scale, ceiling_height_mm, rotation_deg)
      3. Apply via existing modifier
      4. Re-build IFC + verify
      5. Return modified template + new IFC

    The output IFC is NEW (didn't exist before) AND validated.
    """
    base = storage.by_id(req.template_id)
    if base is None:
        raise HTTPException(404, f"unknown template {req.template_id}")

    # 1. LLM suggests structured mods
    suggestion = llm_suggest_mods(req.request, base)
    if suggestion.error:
        raise HTTPException(500, f"LLM error: {suggestion.error}")
    if not suggestion.mods:
        raise HTTPException(
            422,
            f"LLM did not propose any valid modifications. Raw: {suggestion.raw_response[:200]}",
        )

    # 2. Apply modifications using the existing validated path
    modified, errors = apply_modifications(base, suggestion.mods)
    if errors:
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "errors": errors,
                "llm_mods": suggestion.mods,
                "llm_reasoning": suggestion.reasoning,
                "message": "LLM-suggested modifications would break the layout.",
            },
        )

    # 3. Persist + build the new IFC (mirrors /api/modify)
    mod_id = f"mod_{uuid.uuid4().hex[:8]}"
    modified["id"] = mod_id
    _modified_registry[mod_id] = modified
    ifc_out = MODIFIED_DIR / f"{mod_id}.ifc"
    build(modified, ifc_out)

    return {
        "ok": True,
        "modified_id": mod_id,
        "base_template_id": req.template_id,
        "llm_mods": suggestion.mods,           # what the LLM proposed (after clamping)
        "llm_reasoning": suggestion.reasoning,  # 1-sentence rationale
        "llm_latency_s": suggestion.latency_s,
        "ifc_url": f"/api/modified/{mod_id}/ifc",
        "json_url": f"/api/modified/{mod_id}/json",
        "svg_url": f"/api/modified/{mod_id}/svg",
        "modified_metadata": modified["metadata"],
    }


@app.post("/api/reason")
def reason_endpoint(req: ReasonRequest):
    """Stage 1 retrieval (MiniLM) -> Stage 2 reasoning (fine-tuned LLM).

    Returns a structured response with the LLM's ranked picks and rationale,
    plus the raw retrieval cards in case the frontend wants to render them.
    """
    # Stage 1: structured brief -> top-K candidates via MiniLM
    structured_brief = extract_brief(req.brief_text)
    candidates = retrieve(structured_brief, top_n=req.top_k)
    if not candidates:
        raise HTTPException(404, "no candidates returned by retrieval")

    # Stage 2: LLM picks top 4 with reasoning
    resp = llm_reason(req.brief_text, candidates, max_tokens=req.max_tokens)

    return {
        "brief": req.brief_text,
        "structured_brief": structured_brief,
        "candidates": candidates,           # all top-K from retrieval
        "llm_response": resp.text,          # LLM-generated text
        "llm_latency_s": resp.latency_s,
        "llm_backend": resp.backend,
        "llm_model": resp.model,
        "llm_error": resp.error,
    }


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
