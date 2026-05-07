"""Architect interpretation — multiple backends with quality fallback.

Three backends, in order of preference at runtime:
  1. local_mlx (trained adapter): our fine-tuned Llama 3.2 3B + architect
     LoRA — 100% IFC-valid on held-out architects, 2s latency, no API.
  2. codex (Codex CLI): authenticated via user's ChatGPT account, 30-90s
     first call, instant after caching. Used as the offline TEACHER for
     knowledge distillation.
  3. fallback: hardcoded profiles in tower_generator.py.

For demo: only path 1 is used (no API dependency).
For training data generation: path 2 produced labelled examples.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import shutil
import threading
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


# ---------------------------------------------------------------------------
# Trained-Llama backend — preferred at demo runtime (no API dependency)
# ---------------------------------------------------------------------------

# Path to the architect-LoRA adapter (knowledge-distilled from Codex). Set via
# env var BIM_ARCHITECT_ADAPTER, or the default path.
ARCHITECT_ADAPTER_PATH = os.getenv(
    "BIM_ARCHITECT_ADAPTER",
    str(Path(__file__).resolve().parent.parent.parent /
        "training" / "checkpoints" / "llama32-architect-1778151040"),
)
ARCHITECT_BASE_MODEL = os.getenv(
    "BIM_ARCHITECT_BASE",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
)


_LLAMA_LOCK = threading.Lock()
_LLAMA_MODEL = None
_LLAMA_TOKENIZER = None
_LLAMA_ADAPTER_USED: str | None = None


def _ensure_llama_loaded() -> bool:
    """Lazily load the architect-fine-tuned Llama. Returns True if loaded."""
    global _LLAMA_MODEL, _LLAMA_TOKENIZER, _LLAMA_ADAPTER_USED
    if _LLAMA_MODEL is not None and _LLAMA_ADAPTER_USED == ARCHITECT_ADAPTER_PATH:
        return True
    if not Path(ARCHITECT_ADAPTER_PATH).exists():
        return False
    with _LLAMA_LOCK:
        if _LLAMA_MODEL is not None and _LLAMA_ADAPTER_USED == ARCHITECT_ADAPTER_PATH:
            return True
        try:
            from mlx_lm import load
            _LLAMA_MODEL, _LLAMA_TOKENIZER = load(
                ARCHITECT_BASE_MODEL,
                adapter_path=ARCHITECT_ADAPTER_PATH,
            )
            _LLAMA_ADAPTER_USED = ARCHITECT_ADAPTER_PATH
            return True
        except Exception:
            return False


_LLAMA_SYSTEM_PROMPT = (
    "You are an architectural BIM assistant. Given a design brief that "
    "names a famous architect, output a JSON object describing the geometric "
    "signature of a residential tower inspired by their work. Output only "
    "valid JSON — no prose, no markdown fences."
)


def interpret_via_trained_llama(architect: str, n_floors: int = 20,
                                 max_tokens: int = 400) -> ArchitectInterpretation | None:
    """Use the fine-tuned Llama (architect adapter) to interpret an architect.

    Returns None if the model can't be loaded; caller should fall back to
    Codex or hardcoded profile.
    """
    if not _ensure_llama_loaded():
        return None
    try:
        from mlx_lm import generate
    except ImportError:
        return None

    user_msg = f"Design me a {n_floors}-story residential tower inspired by {architect}."
    prompt = _LLAMA_TOKENIZER.apply_chat_template(
        [
            {"role": "system", "content": _LLAMA_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        add_generation_prompt=True, tokenize=False,
    )
    try:
        text = generate(_LLAMA_MODEL, _LLAMA_TOKENIZER,
                        prompt=prompt, max_tokens=max_tokens, verbose=False)
    except Exception as e:
        return ArchitectInterpretation(
            spec={}, rationale="", backend="fallback",
            error=f"trained_llama generation error: {e!r}",
        )

    obj = _extract_json(text or "")
    if obj is None:
        return ArchitectInterpretation(
            spec={}, rationale="", backend="fallback",
            raw=text or "", error="trained_llama returned no parseable JSON",
        )
    rationale = obj.pop("rationale", "") if isinstance(obj, dict) else ""
    spec = _clamp_spec(obj if isinstance(obj, dict) else {})
    return ArchitectInterpretation(
        spec=spec, rationale=rationale if isinstance(rationale, str) else "",
        backend="trained_llama", raw=text or "",
    )


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
                        use_cache: bool = True,
                        prefer: str = "trained_llama") -> ArchitectInterpretation:
    """Interpret an architect's signature into a TowerSpec.

    Backend preference order (when prefer='trained_llama'):
      1. Trained Llama (architect adapter) — 2s latency, no API
      2. Codex cache — instant for previously-seen architects
      3. Codex CLI (fresh) — 30-90s, costs API call
      4. Fallback (caller's hardcoded profile)

    Set prefer='codex' to skip the trained-Llama backend (e.g. for
    knowledge distillation data generation, where we want the teacher).
    """
    # Path 1: Trained Llama (preferred at runtime)
    if prefer == "trained_llama":
        result = interpret_via_trained_llama(architect)
        if result is not None and result.backend == "trained_llama" and result.spec:
            return result
        # If trained_llama returned None or fallback with no spec, try codex.

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
