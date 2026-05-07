"""Live photorealistic rendering — SDXL-turbo on Apple Silicon.

Singleton model loaded once on first request (~30-60s cold start), then
reused across requests. Subsequent renders take ~1.5-3s on M4 Max.

Two entry points:
  - render_from_template(template_dict) — build prompt from BIM metadata
  - render_from_prompt(prompt) — direct prompt-to-image

Returns PIL Image + metadata (prompt, latency_s).
"""

from __future__ import annotations

import io
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Singleton model state
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_PIPE = None
_PIPE_DEVICE: str = ""
_PIPE_LOAD_TIME: float = 0.0
_DEFAULT_MODEL = os.getenv("BIM_RENDER_MODEL", "stabilityai/sdxl-turbo")
_DEFAULT_DEVICE = os.getenv("BIM_RENDER_DEVICE", "mps")  # mps | cuda | cpu


def _ensure_pipeline_loaded(model: str = _DEFAULT_MODEL,
                             device: str = _DEFAULT_DEVICE) -> bool:
    """Lazy-load the diffusers pipeline. Returns True if loaded."""
    global _PIPE, _PIPE_DEVICE, _PIPE_LOAD_TIME
    if _PIPE is not None and _PIPE_DEVICE == device:
        return True
    with _LOCK:
        if _PIPE is not None and _PIPE_DEVICE == device:
            return True
        try:
            t0 = time.time()
            import torch
            from diffusers import AutoPipelineForText2Image
            pipe = AutoPipelineForText2Image.from_pretrained(
                model,
                torch_dtype=torch.float16,
                variant="fp16",
            )
            pipe = pipe.to(device)
            _PIPE = pipe
            _PIPE_DEVICE = device
            _PIPE_LOAD_TIME = time.time() - t0
            return True
        except Exception as e:
            import traceback
            print(f"[image_renderer] failed to load pipeline: {e}")
            traceback.print_exc()
            return False


def warmup() -> dict[str, Any]:
    """Load the model + run one cheap inference to warm CUDA/Metal caches.

    Call this on server startup so the first user-visible render is fast
    (1.5-3s instead of 30-60s including model load).
    """
    if not _ensure_pipeline_loaded():
        return {"loaded": False, "error": "pipeline failed to load"}

    t0 = time.time()
    img = _PIPE(
        prompt="warmup",
        num_inference_steps=2,
        guidance_scale=0.0,
        height=512, width=512,
    ).images[0]
    return {
        "loaded": True,
        "device": _PIPE_DEVICE,
        "model": _DEFAULT_MODEL,
        "load_time_s": round(_PIPE_LOAD_TIME, 2),
        "warmup_inference_s": round(time.time() - t0, 2),
        "test_image_size": img.size,
    }


# ---------------------------------------------------------------------------
# Prompt builder — template metadata → photo-realistic SDXL prompt
# ---------------------------------------------------------------------------

# Style cues per country/style — translates BIM metadata to visual cues.
COUNTRY_STYLE_HINTS = {
    "Germany": "European Altbau-style apartment, parquet floor, white walls, "
               "tall windows with tilt-and-turn frames, Scandinavian-influenced furnishing",
    "France": "Parisian Haussmannian apartment, herringbone parquet, tall French "
              "windows, period mouldings, refined neutral palette",
    "United Kingdom": "British apartment, sash windows, period detailing, "
                       "neutral palette with classic English furniture",
    "Italy": "Italian apartment, terracotta tile floor, warm whites, "
             "textured plaster walls, Mediterranean character",
    "Spain": "Spanish apartment, terracotta or stone floor, warm whites, "
              "wrought-iron details, abundant natural light",
    "Greece": "Greek island-influenced interior, white walls, blue accents, "
              "stone floor, Mediterranean light",
    "Portugal": "Portuguese apartment, azulejo tile accents, warm whites, "
                 "dark wood floor, abundant natural light",
    "Netherlands": "Dutch canal-house apartment, large windows, white walls, "
                    "wood floors, contemporary Scandinavian furnishings",
    "Sweden": "Scandinavian apartment, white walls, light wood floors, "
              "minimal modern furniture, abundant daylight",
    "Norway": "Norwegian apartment, light wood, neutral palette, "
              "fjord daylight, hygge details",
    "Denmark": "Danish apartment, white walls, wide-plank oak floor, "
               "Hans Wegner-style furniture, soft natural light",
    "Finland": "Finnish apartment, blonde birch floor, white walls, "
                "minimalist Scandinavian furniture, Nordic daylight",
    "India": "Indian urban apartment, polished granite or vitrified tile floor, "
             "neutral walls, contemporary Indian furniture, balcony with city view",
    "Japan": "Japanese apartment, clean wood floors, sliding doors (shoji-influenced), "
              "minimalist palette, low furniture, paper-shaded lighting",
    "Singapore": "Singapore HDB apartment, vinyl plank floor, white walls, "
                  "compact modern furniture, balcony with tropical city view",
    "United States": "American urban apartment, hardwood floor, exposed brick or "
                      "modern white walls, contemporary furniture",
    "Canada": "Canadian apartment, hardwood floor, neutral palette, contemporary furniture",
    "Australia": "Australian apartment, polished concrete or hardwood floor, "
                  "neutral palette, indoor-outdoor flow",
    "United Arab Emirates": "Modern Dubai apartment, marble or polished stone floor, "
                             "neutral luxe palette, tall windows with skyline view",
    "Saudi Arabia": "Modern Riyadh apartment, marble floor, traditional Arabic-influenced "
                     "furniture (majlis-style), warm neutral palette",
    "Turkey": "Turkish apartment, parquet floor, Anatolian-influenced textiles, "
               "warm tones, Bosphorus-style daylight",
    "Brazil": "Brazilian apartment, polished concrete or wood floor, tropical "
               "modernist furniture, lush plants, abundant daylight",
    "Argentina": "Buenos Aires apartment, parquet floor, period mouldings, "
                  "contemporary South American furniture",
    "Mexico": "Mexican apartment, talavera tile or wood floor, warm whites, "
               "rustic-modern furniture, abundant light",
    "Morocco": "Moroccan apartment, geometric zellige tile accents, warm earthy tones, "
                "intricate patterned textiles, rattan and wood furniture",
    "Egypt": "Cairo apartment, marble or stone floor, warm neutral palette, "
              "Mediterranean influences, large windows",
    "Kenya": "Nairobi apartment, polished concrete or wood floor, contemporary "
              "African-influenced furniture, bright daylight",
    "Nigeria": "Lagos apartment, polished tile floor, warm neutral palette, "
                "contemporary West African furniture, large windows",
    "South Africa": "Cape Town apartment, polished concrete or wood floor, "
                     "indoor-outdoor flow, contemporary furniture, abundant daylight",
    "China": "Modern Chinese apartment, polished tile floor, neutral palette, "
              "contemporary furniture with Chinese-influenced details",
    "South Korea": "Seoul apartment, polished tile or laminate floor, "
                    "minimalist contemporary furniture, abundant city light",
    "Vietnam": "Vietnamese apartment, polished tile floor, tropical-modernist "
                "furniture, contemporary Asian palette",
    "Indonesia": "Jakarta apartment, hardwood floor, tropical contemporary "
                  "furniture, abundant daylight",
}


ROOM_TYPE_TO_FOCUS = {
    "living": "spacious living room with sofa, coffee table, and view of windows",
    "kitchen": "modern kitchen with island, cabinetry, and natural daylight",
    "master_bedroom": "calm master bedroom with bed and large window",
    "bedroom": "tranquil bedroom",
    "dining": "dining area with table and chairs",
    "balcony": "balcony or terrace with city view",
}


def build_prompt_from_template(template: dict, focus_room_type: str | None = None) -> str:
    """Build an SDXL-turbo-friendly photorealistic prompt from BIM metadata.

    If focus_room_type is given (e.g. 'living'), the prompt focuses on that
    room. Otherwise picks the largest room.
    """
    md = template.get("metadata", {})
    country = md.get("country", "")
    style = md.get("style", "")
    total_area = md.get("total_area_sqm")
    bedrooms = md.get("bedrooms", 0)
    boundary = template.get("boundary", {})
    ceiling_mm = boundary.get("ceiling_height_mm", 2700)
    ceiling_m = ceiling_mm / 1000.0

    # Pick a room to focus on
    rooms = template.get("rooms", []) or []
    if not rooms and template.get("floors"):
        # Multi-floor — pick a typical floor's largest room
        for fl in template["floors"]:
            for r in fl.get("rooms", []):
                if r.get("type") in ROOM_TYPE_TO_FOCUS:
                    rooms.append(r)

    focus_room = None
    if focus_room_type:
        for r in rooms:
            if r.get("type") == focus_room_type:
                focus_room = r
                break
    if focus_room is None and rooms:
        # Largest non-circulation room
        habitable = [r for r in rooms if r.get("type") not in ("entry", "stairs", "corridor")]
        if habitable:
            focus_room = max(habitable, key=lambda r: r.get("area_sqm", 0))

    style_hint = COUNTRY_STYLE_HINTS.get(country, "modern apartment, neutral palette, abundant natural light")
    focus_text = ""
    if focus_room is not None:
        rtype = focus_room.get("type", "living")
        focus_text = ROOM_TYPE_TO_FOCUS.get(rtype, "interior")
        if focus_room.get("area_sqm"):
            focus_text += f", approximately {focus_room['area_sqm']:.0f} square meters"

    ceiling_text = ""
    if ceiling_m >= 3.0:
        ceiling_text = f", high {ceiling_m:.1f}m ceilings"

    area_text = f"{int(total_area)} m² total" if total_area else ""

    parts = [
        "Photorealistic interior architectural photography",
        f"of a {bedrooms}-bedroom" if bedrooms else "of an",
        f"apartment in {country}" if country else "apartment",
        f"({area_text})" if area_text else "",
        ceiling_text,
        ".",
        style_hint,
        ".",
        f"Focus: {focus_text}" if focus_text else "",
        ".",
        "Soft natural daylight, wide-angle lens, sharp focus, magazine quality.",
    ]
    return " ".join(p for p in parts if p).replace(" .", ".").replace(" ,", ",")


def build_prompt_from_tower(template: dict) -> str:
    """Tower-specific prompt — emphasises exterior architectural photography."""
    md = template.get("metadata", {})
    country = md.get("country", "")
    city = md.get("city_inspiration") or country
    n_floors = md.get("n_floors", "")
    architect = md.get("inspiration_architect", "")
    style_word = (md.get("style") or "").split("(")[0].strip()

    architect_hint = ""
    if architect:
        architect_hint = f", in the architectural style of {architect}"

    return (
        f"Photorealistic architectural photograph of a {n_floors}-story "
        f"residential tower in {city}{architect_hint}. "
        f"Modern building exterior, glass and concrete construction, blue sky, "
        f"sunlit facade, ground-level perspective, professional architectural "
        f"photography, sharp focus, magazine quality. {style_word}"
    )


# ---------------------------------------------------------------------------
# Render entrypoints
# ---------------------------------------------------------------------------

@dataclass
class RenderResult:
    image_bytes: bytes
    image_format: str
    prompt: str
    latency_s: float
    width: int
    height: int
    backend: str
    error: str | None = None


def _render(prompt: str, width: int = 768, height: int = 512,
             steps: int = 2) -> RenderResult:
    if not _ensure_pipeline_loaded():
        return RenderResult(
            image_bytes=b"", image_format="png", prompt=prompt,
            latency_s=0.0, width=width, height=height,
            backend="error", error="pipeline failed to load",
        )
    t0 = time.time()
    try:
        img = _PIPE(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,
            height=height, width=width,
        ).images[0]
    except Exception as e:
        return RenderResult(
            image_bytes=b"", image_format="png", prompt=prompt,
            latency_s=time.time() - t0, width=width, height=height,
            backend="error", error=repr(e),
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return RenderResult(
        image_bytes=buf.getvalue(), image_format="png", prompt=prompt,
        latency_s=time.time() - t0, width=img.width, height=img.height,
        backend=f"sdxl-turbo:{_PIPE_DEVICE}",
    )


def render_from_prompt(prompt: str, width: int = 768, height: int = 512,
                        steps: int = 2) -> RenderResult:
    return _render(prompt, width=width, height=height, steps=steps)


def render_from_template(template: dict, focus_room_type: str | None = None,
                          width: int = 768, height: int = 512,
                          steps: int = 2) -> RenderResult:
    if template.get("metadata", {}).get("tower"):
        prompt = build_prompt_from_tower(template)
    else:
        prompt = build_prompt_from_template(template, focus_room_type)
    return _render(prompt, width=width, height=height, steps=steps)


if __name__ == "__main__":
    import sys
    print("Warming up...")
    info = warmup()
    print(info)
    if info.get("loaded"):
        print("\nGenerating 3 demo images...")
        prompts = [
            ("Berlin Altbau", "Photorealistic Berlin Altbau apartment interior, 60 sqm, parquet floor, large windows, natural daylight"),
            ("Tokyo studio", "Photorealistic Tokyo studio apartment interior, 30 sqm, modern minimalist, wood floor, paper screens"),
            ("Indian 2BHK", "Photorealistic Indian 2BHK apartment living room, polished granite floor, contemporary Indian furniture, balcony with city view"),
        ]
        for name, p in prompts:
            r = render_from_prompt(p, width=768, height=512)
            out = f"/tmp/render_demo_{name.lower().replace(' ', '_')}.png"
            Path(out).write_bytes(r.image_bytes)
            print(f"  {name}: {r.latency_s:.2f}s -> {out}")
