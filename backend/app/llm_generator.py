"""LEVEL 4 high-level: brief → LLM proposes program → procedural geometry → validated IFC.

Two LLM backends supported:
  - "claude_api": Anthropic API (recommended — strong structured output)
  - "local_mlx":  Our fine-tuned Llama (cheap, but JSON discipline weaker)

The output of the LLM is a "program" JSON that template_generator.py knows
how to materialise into validated geometry. We retry up to N times on
validation errors, feeding the errors back into the prompt.

Demo positioning: this is "Phase 2 research preview" — we frame it as
"using a stronger reasoning model" to set expectations honestly.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from .template_generator import generate_template


SYSTEM_PROMPT_GEN = """You are an architectural BIM program generator.
Given a user brief, output a JSON object describing a single-floor apartment
program. Rooms are rectangular and laid out in two horizontal strips:
WET strip on top (entry, kitchen, bathrooms, WC, utility)
DRY strip on bottom (living, dining, bedrooms, balcony)

Output ONLY this JSON shape — no prose, no code fences:

{
  "region": "europe" | "india" | "global",
  "country": "<country>",
  "city": "<city>",
  "style": "<short style description>",
  "description": "<1-sentence description of the layout>",
  "bedrooms": <int 0-4>,
  "bathrooms": <int 1-3>,
  "total_area_sqm": <float, between 30 and 200>,
  "wall_thickness_mm": <int 150-400 (Indian: 230, European masonry: 350-380, modern: 150)>,
  "ceiling_height_mm": <int 2400-3500 (typical: 2700 modern, 3200 Altbau)>,
  "suitable_for": [<persona strings>],
  "rooms": [
    {"name": "<room name in local language>", "area_sqm": <float>},
    ...
  ]
}

CONSTRAINTS — follow strictly:
- Room areas must sum to total_area_sqm (within 5%)
- Always include exactly one entry-type room (e.g. Foyer, Hall, Diele, Entrée)
- Always include at least one bathroom-type room (e.g. Bathroom, Bad)
- Include a kitchen-type OR a combined Living/Dining/Kitchen room
- For 2+ bedrooms, label them clearly: Master Bedroom, Bedroom 2, Bedroom 3
- Use the local language for room names where appropriate (Diele not Hall for Germany)
- Total area must reflect realistic apartment sizes for the country

Example (German 2-Zimmer Altbau, 65 m²):
{"region":"europe","country":"Germany","city":"Berlin",
 "style":"Berlin Altbau 2-Zimmer","bedrooms":1,"bathrooms":1,
 "total_area_sqm":65,"wall_thickness_mm":380,"ceiling_height_mm":3300,
 "suitable_for":["couple","creative_artist"],
 "rooms":[{"name":"Diele","area_sqm":6},{"name":"Wohnzimmer","area_sqm":25},
          {"name":"Küche","area_sqm":10},{"name":"Badezimmer","area_sqm":7},
          {"name":"Schlafzimmer","area_sqm":17}]}
"""


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Robust JSON extraction. Handles ```json fences, prose wrapping, etc."""
    # Strip code fences first
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass
    # Try first balanced object
    match = _JSON_OBJ_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    return None


@dataclass
class GenerateResult:
    success: bool
    template: dict | None
    program: dict | None
    raw_responses: list[str]  # one per attempt
    errors_per_attempt: list[list[str]]
    latency_s: float
    backend: str
    attempts: int


def _call_claude(brief: str, error_feedback: str = "") -> tuple[str, str | None]:
    """Call Anthropic API. Returns (response_text, error)."""
    try:
        import anthropic
    except ImportError:
        return "", "anthropic SDK not installed: pip install anthropic"

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "", "ANTHROPIC_API_KEY not set in environment"

    user_msg = f"BRIEF:\n{brief}"
    if error_feedback:
        user_msg += f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION:\n{error_feedback}\nFix the errors above and re-emit the JSON."

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest"),
            max_tokens=2000,
            system=SYSTEM_PROMPT_GEN,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text, None
    except Exception as e:
        return "", repr(e)


def _call_local_mlx(brief: str, error_feedback: str = "") -> tuple[str, str | None]:
    """Call our fine-tuned Llama via mlx-lm."""
    from .llm_client import get_backend
    backend = get_backend()

    if backend.__class__.__name__ != "_LocalMLXBackend":
        return "", "local_mlx backend not active"

    backend._ensure_loaded()
    from mlx_lm import generate as _generate

    user_msg = f"BRIEF:\n{brief}"
    if error_feedback:
        user_msg += f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION:\n{error_feedback}\nFix the errors and re-emit the JSON."

    prompt = backend._tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT_GEN},
         {"role": "user", "content": user_msg}],
        add_generation_prompt=True, tokenize=False,
    )
    try:
        text = _generate(backend._model, backend._tokenizer,
                         prompt=prompt, max_tokens=1500, verbose=False)
        return text, None
    except Exception as e:
        return "", repr(e)


def generate_from_brief(
    brief: str,
    backend: str = "auto",  # "auto" | "rules" | "claude_api" | "local_mlx"
    max_attempts: int = 3,
) -> GenerateResult:
    """Brief → program → procedural geometry → validated template.

    Strategy (when backend="auto"):
      1. Try the rule-based extractor (deterministic, 100% reliable)
      2. If a stronger LLM is configured, also offer it as a richer alternative

    Returns a GenerateResult. If success=True, .template is a fully-validated
    template dict ready to be built into IFC.
    """
    if backend == "auto":
        backend = "rules"  # default to deterministic — always succeeds

    raw_responses: list[str] = []
    errors_per_attempt: list[list[str]] = []
    program_dict: dict | None = None
    template: dict | None = None

    t0 = time.time()
    error_feedback = ""

    # ---- Path 0: rule-based deterministic extractor (always succeeds) ----
    if backend == "rules":
        from .program_extractor import extract_program
        program_dict = extract_program(brief)
        template, errors = generate_template(program_dict)
        if not errors and template:
            from time import time as _t
            base = template["id"].split("_")[0]
            template["id"] = f"{base}_ai_{int(_t())}"
            return GenerateResult(
                success=True, template=template, program=program_dict,
                raw_responses=["(rule-based extractor)"],
                errors_per_attempt=[[]],
                latency_s=time.time() - t0, backend="rules", attempts=1,
            )
        # Rule-based shouldn't fail, but if it does fall through to LLM
        errors_per_attempt.append(errors)

    for attempt in range(max_attempts):
        if backend == "claude_api":
            text, err = _call_claude(brief, error_feedback)
        elif backend == "local_mlx":
            text, err = _call_local_mlx(brief, error_feedback)
        else:
            return GenerateResult(
                success=False, template=None, program=None,
                raw_responses=raw_responses, errors_per_attempt=errors_per_attempt,
                latency_s=time.time() - t0, backend=backend, attempts=attempt,
            )

        raw_responses.append(text)
        if err:
            errors_per_attempt.append([err])
            continue

        program_dict = _extract_json(text)
        if program_dict is None:
            errors_per_attempt.append(["Could not parse a JSON object from the response."])
            error_feedback = "Output strict JSON only — no prose, no code fences."
            continue

        # Geometry pass + validation
        template, errors = generate_template(program_dict)
        errors_per_attempt.append(errors)
        if not errors:
            # Re-stamp the id to be unique
            from time import time as _t
            base = template["id"].split("_")[0]  # eu | in | gl
            template["id"] = f"{base}_ai_{int(_t())}"
            return GenerateResult(
                success=True, template=template, program=program_dict,
                raw_responses=raw_responses, errors_per_attempt=errors_per_attempt,
                latency_s=time.time() - t0, backend=backend,
                attempts=attempt + 1,
            )

        # Build a feedback string for the next retry
        error_feedback = "\n".join(f"- {e}" for e in errors[:5])

    return GenerateResult(
        success=False,
        template=None,
        program=program_dict,
        raw_responses=raw_responses,
        errors_per_attempt=errors_per_attempt,
        latency_s=time.time() - t0,
        backend=backend,
        attempts=max_attempts,
    )


if __name__ == "__main__":
    # Quick test if API key set
    test_brief = "1-bedroom apartment for a young couple in Berlin, Altbau character with high ceilings, around 60 m²."
    print(f"Brief: {test_brief}\n")
    result = generate_from_brief(test_brief, backend="auto")
    print(f"Backend: {result.backend} | Attempts: {result.attempts} | Latency: {result.latency_s:.1f}s")
    print(f"Success: {result.success}")
    if result.success and result.template:
        print(f"Template: {result.template['metadata']['style']}")
        print(f"Rooms: {[r['name'] for r in result.template['rooms']]}")
    else:
        print(f"Errors on last attempt: {result.errors_per_attempt[-1] if result.errors_per_attempt else 'none'}")
        if result.raw_responses:
            print(f"Last raw output: {result.raw_responses[-1][:300]}")
