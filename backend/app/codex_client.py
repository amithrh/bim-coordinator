"""Codex CLI bridge — invoke the local Codex CLI for AI-driven architect
interpretation without needing an Anthropic/OpenAI API key in our process.

Codex CLI is OpenAI's command-line tool authenticated via the user's ChatGPT
account. We shell out to `codex exec` with a prompt + output schema, parse
the returned JSON, validate against safe bounds, and use it as a TowerSpec
override.

This replaces the hardcoded ARCHITECT_PROFILES dict for any architect Codex
knows about (which is hundreds — Sou Fujimoto, MAD, Kengo Kuma, Jeanne Gang,
SHoP, Diller Scofidio, anyone).

If Codex CLI is unavailable or returns an error, we fall back to the
hardcoded profile (or default) so the demo never breaks.
"""

from __future__ import annotations

import json
import re
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CODEX_BIN = shutil.which("codex") or "codex"
DEFAULT_MODEL = "gpt-5.4-mini"  # ChatGPT-account-supported model


SCHEMA_DESCRIPTION = """Output ONLY a JSON object (no prose, no markdown
fences). Keys + allowed values:
- setback_pattern: "none" | "stepped" | "pyramid" | "inverse_taper" | "mid_setback"
  * none = uniform tower (Foster, Pei, Ando)
  * stepped = top N floors smaller, right-aligned (Zaha, Gehry)
  * pyramid = stepped from middle to top, centered (Bjarke Ingels mountain)
  * inverse_taper = wide base for first M floors, narrow tower above (Calatrava civic)
  * mid_setback = no top setback but programmatic floor mid-tower (Koolhaas)
- n_setbacks: int 0-12 (more = more gradual taper)
- setback_amount_m: float 0.0-5.0 (size of each step)
- n_amenity_floors: int 1-3 (1=ground only, 2=+sky lobby, 3=multiple)
- sky_lobby_relative: float 0.0-1.0 (mid-tower amenity floor position)
- units_per_typical_floor: int 2-8 (density)
- typical_unit_area_sqm: float 40-200 (luxury 90+, normal 70, compact 55)
- floor_height_mm: int 2700-4000 (taller = more high-tech)
- footprint_aspect: float 0.8-2.5 (1.0 = square)
- asymmetric_units: boolean (true for sculpted/Gehry-style irregular)
- rationale: string (cite the architect's KNOWN projects/signature)
"""


@dataclass
class ArchitectInterpretation:
    spec: dict[str, Any]      # parsed JSON spec
    rationale: str            # AI's reasoning string
    backend: str              # "codex" or "fallback"
    raw: str = ""             # raw model output
    error: str | None = None


_JSON_OBJ_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    # Strip leading/trailing markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try first balanced object
    for m in _JSON_OBJ_RE.findall(text):
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _clamp_spec(spec: dict) -> dict:
    """Coerce model output to safe bounds — defends procedural builder."""
    out: dict[str, Any] = {}
    p = spec.get("setback_pattern", "stepped")
    if p in ("none", "stepped", "pyramid", "inverse_taper", "mid_setback"):
        out["setback_pattern"] = p
    else:
        out["setback_pattern"] = "stepped"

    def num(key: str, lo, hi, default):
        try:
            v = float(spec.get(key, default))
            v = max(lo, min(hi, v))
            return int(v) if isinstance(default, int) else v
        except (TypeError, ValueError):
            return default

    out["n_setbacks"] = num("n_setbacks", 0, 12, 2)
    out["setback_amount_m"] = num("setback_amount_m", 0.0, 5.0, 1.5)
    out["n_amenity_floors"] = num("n_amenity_floors", 1, 3, 1)
    out["sky_lobby_relative"] = num("sky_lobby_relative", 0.0, 1.0, 0.5)
    out["units_per_typical_floor"] = num("units_per_typical_floor", 2, 8, 4)
    out["typical_unit_area_sqm"] = num("typical_unit_area_sqm", 40, 200, 70.0)
    out["floor_height_mm"] = num("floor_height_mm", 2700, 4000, 3200)
    out["footprint_aspect"] = num("footprint_aspect", 0.8, 2.5, 1.4)
    out["asymmetric_units"] = bool(spec.get("asymmetric_units", False))

    return out


def codex_available() -> bool:
    """Check if codex CLI is callable on this machine."""
    try:
        r = subprocess.run([CODEX_BIN, "--version"], capture_output=True,
                           text=True, timeout=10)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# File-backed cache so repeated calls (e.g. dress-rehearsal demo briefs)
# don't re-pay the 30-90s Codex latency.
_CACHE_DIR = Path.home() / ".cache" / "bim-coordinator" / "codex_architect"


def _cache_path(architect: str) -> Path:
    safe = re.sub(r"[^a-z0-9]+", "_", architect.lower()).strip("_") or "unknown"
    return _CACHE_DIR / f"{safe}.json"


def _read_cache(architect: str) -> ArchitectInterpretation | None:
    p = _cache_path(architect)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return ArchitectInterpretation(
            spec=d["spec"], rationale=d["rationale"],
            backend="codex_cache", raw=d.get("raw", ""),
        )
    except (json.JSONDecodeError, KeyError):
        return None


def _write_cache(architect: str, interp: ArchitectInterpretation) -> None:
    if interp.backend != "codex":
        return  # only cache real codex results
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(architect).write_text(json.dumps({
        "spec": interp.spec, "rationale": interp.rationale, "raw": interp.raw,
    }, indent=2))


def interpret_architect(architect: str, model: str = DEFAULT_MODEL,
                        timeout: int = 180,
                        use_cache: bool = True) -> ArchitectInterpretation:
    """Ask Codex CLI to interpret an architect's signature into a tower spec.

    Cached on first success — subsequent calls for the same architect are
    instant. Returns backend="codex" (fresh), "codex_cache" (cached),
    or "fallback" (codex unavailable / errored).
    """
    if use_cache:
        cached = _read_cache(architect)
        if cached is not None:
            return cached

    if not codex_available():
        return ArchitectInterpretation(
            spec={}, rationale="", backend="fallback",
            error="codex CLI not available",
        )

    prompt = (
        f"Output ONLY a JSON object (no prose, no markdown fences) "
        f"describing the geometric signature parameters for a residential "
        f"tower inspired by {architect}.\n\n"
        f"{SCHEMA_DESCRIPTION}\n\n"
        f"Cite at least 2 of {architect}'s known projects in the rationale."
    )

    import tempfile
    out_file = Path(tempfile.mktemp(suffix=".json"))
    try:
        r = subprocess.run(
            [
                CODEX_BIN, "exec",
                "--model", model,
                "--sandbox", "read-only",
                "-c", 'reasoning_effort="low"',
                "--output-last-message", str(out_file),
                prompt,
            ],
            capture_output=True, text=True, timeout=timeout,
        )
        raw = out_file.read_text() if out_file.exists() else (r.stdout or "")
    except subprocess.TimeoutExpired:
        return ArchitectInterpretation(
            spec={}, rationale="", backend="fallback",
            error=f"codex timeout after {timeout}s",
        )
    except Exception as e:
        return ArchitectInterpretation(
            spec={}, rationale="", backend="fallback", error=repr(e),
        )
    finally:
        if out_file.exists():
            out_file.unlink()

    obj = _extract_json(raw or "")
    if obj is None:
        return ArchitectInterpretation(
            spec={}, rationale="", backend="fallback", raw=raw or "",
            error="codex returned no parseable JSON",
        )

    rationale = obj.pop("rationale", "") if isinstance(obj, dict) else ""
    spec = _clamp_spec(obj)
    interp = ArchitectInterpretation(
        spec=spec, rationale=rationale if isinstance(rationale, str) else "",
        backend="codex", raw=raw,
    )
    _write_cache(architect, interp)
    return interp


if __name__ == "__main__":
    import sys
    architect = " ".join(sys.argv[1:]) or "Sou Fujimoto"
    print(f"Asking Codex to interpret: {architect}")
    print(f"Codex available: {codex_available()}")
    result = interpret_architect(architect)
    print(f"\nBackend: {result.backend}")
    if result.error:
        print(f"Error: {result.error}")
    print(f"Spec:")
    for k, v in result.spec.items():
        print(f"  {k}: {v}")
    print(f"\nRationale: {result.rationale}")
