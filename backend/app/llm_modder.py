"""LEVEL 2: LLM-driven modification of an existing template.

Takes a chosen template + a natural-language modification request and
returns a structured `mods` dict that the existing apply_modifications()
function knows how to apply. The result is then re-validated and re-built
into a NEW IFC file that didn't exist before.

This is "model generates new validated IFC" without the difficulty of
free-form generation: the geometry stays correct because we modify a
known-valid template, and the verifier still gates the output.

Supported modifications (mirror apply_modifications):
  - area_scale: float between 0.5 and 2.0 (final size 50%-200% of original)
  - ceiling_height_mm: int between 2400 and 3800
  - rotation_deg: 0, 90, 180, or 270

Usage:
  mods, reasoning = suggest_mods("more space and taller ceilings", template)
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from .llm_client import get_backend


_BOUND = {
    "area_scale": (0.5, 2.0),
    "ceiling_height_mm": (2400, 3800),
    "rotation_deg": (0, 270),
}


SYSTEM_PROMPT = (
    "You are an architectural BIM assistant. Given a chosen floor plan template "
    "and a user's requested modification, output a JSON object with the parameter "
    "changes needed. ONLY output a single JSON object — no prose. "
    "Allowed keys (omit any you don't want to change):\n"
    "  area_scale     - float 0.5-2.0  (1.0 = no change, 1.2 = 20% more area)\n"
    "  ceiling_height_mm - int 2400-3800 (typical: 2700 modern, 3200 Altbau)\n"
    "  rotation_deg   - 0, 90, 180, or 270\n"
    "Plus a 'reasoning' string explaining your choices in one sentence.\n"
    "Example output:\n"
    '  {"area_scale": 1.2, "ceiling_height_mm": 3200, "reasoning": '
    '"20% larger and Altbau ceiling height to match the user\'s request for '
    'more space with high ceilings."}\n'
    "If the request is ambiguous, choose modest values (≤1.3× scale, "
    "ceiling height within 200mm of original)."
)


@dataclass
class ModSuggestion:
    mods: dict[str, Any]
    reasoning: str
    raw_response: str
    latency_s: float
    error: str | None = None


def _build_user_prompt(request: str, template: dict) -> str:
    md = template.get("metadata", {})
    boundary = template.get("boundary", {})
    return (
        f"CHOSEN TEMPLATE:\n"
        f"  id: {template.get('id', '')}\n"
        f"  location: {md.get('city_inspiration', '')}, {md.get('country', '')}\n"
        f"  size: {md.get('size_label', '')} ({md.get('total_area_sqm', 0)} sqm)\n"
        f"  current ceiling height: {boundary.get('ceiling_height_mm', 2700)}mm\n"
        f"  style: {md.get('style', '')}\n"
        f"\nUSER WANTS:\n  {request}\n"
        f"\nOutput a JSON object with the parameter changes."
    )


_JSON_OBJ_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Find the first complete JSON object in the model response."""
    # Try the whole text first
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try the first balanced {...} block
    matches = _JSON_OBJ_RE.findall(text)
    for m in matches:
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _coerce_area_scale(mods: dict, base_area_sqm: float) -> float | None:
    """Models often emit 'size' or 'area_m2' instead of 'area_scale'.
    Translate any variant into a multiplier relative to base_area_sqm."""
    # Direct multiplier
    if "area_scale" in mods:
        try:
            return float(mods["area_scale"])
        except (ValueError, TypeError):
            pass
    # Absolute area in m²
    for key in ("size", "area", "area_m2", "area_sqm", "total_area_sqm", "size_sqm", "size_m2"):
        if key in mods:
            try:
                v = float(mods[key])
                if base_area_sqm > 0 and v > 5:  # ignore tiny / zero
                    return v / base_area_sqm
            except (ValueError, TypeError):
                continue
    # Percentage strings: "+20%", "-50%"
    for key in ("scale", "scale_factor"):
        if key in mods:
            try:
                v = float(mods[key])
                # If it looks like 20 (meaning 20%) treat as percentage; else as multiplier
                return (1.0 + v / 100.0) if abs(v) > 5 else v
            except (ValueError, TypeError):
                continue
    return None


def _clamp(mods: dict, base_area_sqm: float = 50.0) -> dict[str, Any]:
    """Clamp model output to safe bounds — protects validator from bad LLM JSON."""
    clean: dict[str, Any] = {}

    # area_scale (with fallback to 'size'/'area_m2' translation)
    scale = _coerce_area_scale(mods, base_area_sqm)
    if scale is not None:
        lo, hi = _BOUND["area_scale"]
        clamped = max(lo, min(hi, scale))
        # Only include if meaningfully different from 1.0 (no-op otherwise)
        if abs(clamped - 1.0) > 0.05:
            clean["area_scale"] = round(clamped, 3)

    # ceiling_height_mm (also accept 'ceiling_height' meters)
    for key in ("ceiling_height_mm", "ceiling_height"):
        if key in mods:
            try:
                v = float(mods[key])
                # If looks like meters (e.g. 3.2), convert to mm
                if v < 100:
                    v *= 1000
                v = int(v)
                lo, hi = _BOUND["ceiling_height_mm"]
                clean["ceiling_height_mm"] = max(lo, min(hi, v))
                break
            except (ValueError, TypeError):
                continue

    # rotation_deg
    if "rotation_deg" in mods:
        try:
            v = int(mods["rotation_deg"])
            if v % 90 != 0:
                v = round(v / 90) * 90
            v = v % 360
            if v != 0:  # 0° is no-op
                clean["rotation_deg"] = v
        except (ValueError, TypeError):
            pass

    return clean


# ---------------------------------------------------------------------------
# Deterministic slot extractor — fast + reliable for common phrasings
# ---------------------------------------------------------------------------

_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_AREA_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:square\s*meters?|sq\s*m|sqm|m²|m2|metres|meters)",
    re.IGNORECASE,
)
_CEIL_M_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:m|meter|metre)\s*(?:high|ceilings?|tall)",
    re.IGNORECASE,
)
_CEIL_MM_RE = re.compile(r"(\d{4})\s*mm", re.IGNORECASE)
_ROT_RE = re.compile(r"rotate\s*(?:by\s*)?(\d{1,3})", re.IGNORECASE)


def _extract_slots(request: str, base_area_sqm: float) -> dict[str, Any]:
    """Pull mods from common phrasings without needing the LLM.

    Examples it handles:
      "20% bigger" → area_scale 1.2
      "half the area" / "50% smaller" → area_scale 0.5
      "100 sqm" → area_scale = 100/base_area
      "3.5m ceilings" → ceiling_height_mm 3500
      "3200mm" → ceiling_height_mm 3200
      "rotate 90" → rotation_deg 90
      "Altbau ceilings" → ceiling_height_mm 3300 (heuristic)
    """
    out: dict[str, Any] = {}
    txt = request.lower()

    # ---- Area changes ----
    pct = _PERCENT_RE.search(txt)
    if pct:
        v = float(pct.group(1))
        if any(w in txt for w in ("bigger", "larger", "more", "increase", "expand")):
            out["area_scale"] = round(1 + v / 100, 3)
        elif any(w in txt for w in ("smaller", "less", "reduce", "tight", "decrease")):
            out["area_scale"] = round(max(0.5, 1 - v / 100), 3)

    if "double" in txt and "area_scale" not in out:
        out["area_scale"] = 2.0
    if "half" in txt and "area_scale" not in out:
        out["area_scale"] = 0.5
    if "triple" in txt and "area_scale" not in out:
        out["area_scale"] = 2.0  # cap

    area = _AREA_RE.search(txt)
    if area and "area_scale" not in out:
        try:
            target = float(area.group(1))
            if base_area_sqm > 0 and 20 <= target <= 500:
                out["area_scale"] = round(target / base_area_sqm, 3)
        except ValueError:
            pass

    # ---- Ceiling height ----
    ceil_m = _CEIL_M_RE.search(txt)
    if ceil_m:
        try:
            v = float(ceil_m.group(1)) * 1000
            out["ceiling_height_mm"] = int(v)
        except ValueError:
            pass

    if "ceiling_height_mm" not in out:
        # Look for "<digit>m" near 'ceiling' or 'tall' or 'high'
        m_match = re.search(
            r"(\d(?:\.\d)?)\s*m(?!\w)",
            txt,
        )
        if m_match and any(w in txt for w in ("ceiling", "tall", "high")):
            try:
                v = float(m_match.group(1)) * 1000
                if 2000 <= v <= 4500:
                    out["ceiling_height_mm"] = int(v)
            except ValueError:
                pass

    ceil_mm = _CEIL_MM_RE.search(txt)
    if ceil_mm:
        try:
            v = int(ceil_mm.group(1))
            if 2000 <= v <= 4500:
                out["ceiling_height_mm"] = v
        except ValueError:
            pass

    if "ceiling_height_mm" not in out:
        if "altbau" in txt:
            out["ceiling_height_mm"] = 3300
        elif "high ceiling" in txt or "tall ceiling" in txt:
            out["ceiling_height_mm"] = 3200
        elif "low ceiling" in txt:
            out["ceiling_height_mm"] = 2400

    # ---- Rotation ----
    rot = _ROT_RE.search(txt)
    if rot:
        try:
            v = int(rot.group(1))
            v = round(v / 90) * 90
            v = v % 360
            if v != 0:
                out["rotation_deg"] = v
        except ValueError:
            pass
    if "rotation_deg" not in out:
        if "rotate ninety" in txt or "quarter turn" in txt:
            out["rotation_deg"] = 90
        elif "flip" in txt or "rotate 180" in txt:
            out["rotation_deg"] = 180

    return out


def _slots_reasoning(mods: dict[str, Any], base_area_sqm: float, base_ceil_mm: int) -> str:
    """Friendly one-line reasoning for deterministic-extracted mods."""
    parts = []
    if "area_scale" in mods:
        new_area = base_area_sqm * mods["area_scale"]
        parts.append(f"resize to ~{new_area:.0f} m² ({mods['area_scale']:.2f}× original)")
    if "ceiling_height_mm" in mods:
        parts.append(f"ceiling height {mods['ceiling_height_mm']}mm "
                     f"(was {base_ceil_mm}mm)")
    if "rotation_deg" in mods:
        parts.append(f"rotate {mods['rotation_deg']}°")
    if not parts:
        return ""
    return "Applying: " + ", ".join(parts) + "."


def suggest_mods(request: str, template: dict, max_tokens: int = 200) -> ModSuggestion:
    """Decide modification parameters from a natural-language request.

    Strategy:
      1. Try deterministic regex slot extraction (fast, always reliable)
      2. If we got at least one mod from regex, return immediately
      3. Otherwise fall back to the LLM (handles ambiguous wording)

    This gives us 100% reliable behavior on common phrasings while still
    supporting free-form requests.
    """
    base_area = float(template.get("metadata", {}).get("total_area_sqm", 50)) or 50.0
    base_ceil = int(template.get("boundary", {}).get("ceiling_height_mm", 2700))

    # ---- Path 1: deterministic slot extraction ----
    t0 = time.time()
    slot_mods = _extract_slots(request, base_area)
    if slot_mods:
        return ModSuggestion(
            mods=slot_mods,
            reasoning=_slots_reasoning(slot_mods, base_area, base_ceil),
            raw_response="(deterministic slot extractor)",
            latency_s=time.time() - t0,
            error=None,
        )

    # ---- Path 2: LLM fallback for ambiguous requests ----
    backend = get_backend()
    user_prompt = _build_user_prompt(request, template)

    # Call the LLM directly (we want low temperature + short output)
    if hasattr(backend, "_LocalMLXBackend") or backend.__class__.__name__ == "_LocalMLXBackend":
        backend._ensure_loaded()
        from mlx_lm import generate as _generate
        prompt = backend._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        t0 = time.time()
        try:
            text = _generate(
                backend._model, backend._tokenizer,
                prompt=prompt, max_tokens=max_tokens, verbose=False,
            )
            err = None
        except Exception as e:
            text = ""
            err = repr(e)
        latency = time.time() - t0
    else:
        # HTTP backend path
        import urllib.request, urllib.error
        body = json.dumps({
            "model": backend.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{backend.base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {backend.api_key}"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"]
            err = None
        except Exception as e:
            text = ""
            err = repr(e)
        latency = time.time() - t0

    # Parse JSON out of the model response
    obj = _extract_json(text) or {}
    reasoning = obj.pop("reasoning", "") if isinstance(obj, dict) else ""
    mods = _clamp(obj if isinstance(obj, dict) else {}, base_area_sqm=base_area)

    return ModSuggestion(
        mods=mods,
        reasoning=reasoning if isinstance(reasoning, str) else "",
        raw_response=text,
        latency_s=latency,
        error=err,
    )
