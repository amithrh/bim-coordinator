"""Live photorealistic rendering — SDXL-turbo on Apple Silicon.

Singleton model loaded once on first request (~30-60s cold start), then
reused across requests. Subsequent renders take ~1.5-3s on M4 Max.

Two pipelines, both lazy-loaded:
  - _PIPE: base SDXL-turbo (stylistic, prompt-only). 1-2s/image.
  - _PIPE_CN: SDXL-turbo + Depth ControlNet (faithful to floor plan
    layout — uses depth_renderer to project the BIM 3D scene to a depth
    map, conditions SDXL on it). ~3-5s/image.

Three entry points:
  - render_from_template(template_dict) — stylistic, prompt-only
  - render_from_prompt(prompt) — direct prompt-to-image
  - render_faithful_from_template(template_dict) — depth-conditioned,
    matches the actual floor plan geometry

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

_LOCK_CN = threading.Lock()
_PIPE_CN = None
_PIPE_CN_DEVICE: str = ""
_PIPE_CN_LOAD_TIME: float = 0.0

# Apple MPS can't actually run multiple SDXL inferences in parallel —
# concurrent calls all stall and finish at roughly N× single latency,
# which makes the walkthrough feel hung. Serialize all SDXL calls
# (both base + ControlNet) through a single inference mutex so each
# request finishes in ~2s back-to-back instead of all stalling.
_INFER_LOCK = threading.Lock()

_DEFAULT_MODEL = os.getenv("BIM_RENDER_MODEL", "stabilityai/sdxl-turbo")
_DEFAULT_DEVICE = os.getenv("BIM_RENDER_DEVICE", "mps")  # mps | cuda | cpu
_DEFAULT_CONTROLNET = os.getenv(
    "BIM_RENDER_CONTROLNET", "diffusers/controlnet-depth-sdxl-1.0-small"
)


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


def _ensure_controlnet_pipeline_loaded(
    base_model: str = _DEFAULT_MODEL,
    controlnet: str = _DEFAULT_CONTROLNET,
    device: str = _DEFAULT_DEVICE,
) -> bool:
    """Lazy-load SDXL + Depth ControlNet pipeline (separate from the base).

    We load this on demand the first time someone requests a 'faithful'
    render so the simple stylistic path stays fast and lightweight.
    """
    global _PIPE_CN, _PIPE_CN_DEVICE, _PIPE_CN_LOAD_TIME
    if _PIPE_CN is not None and _PIPE_CN_DEVICE == device:
        return True
    with _LOCK_CN:
        if _PIPE_CN is not None and _PIPE_CN_DEVICE == device:
            return True
        try:
            t0 = time.time()
            import torch
            from diffusers import (
                StableDiffusionXLControlNetPipeline,
                ControlNetModel,
            )
            cn = ControlNetModel.from_pretrained(
                controlnet, torch_dtype=torch.float16, variant="fp16",
            )
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                base_model,
                controlnet=cn,
                torch_dtype=torch.float16,
                variant="fp16",
            )
            pipe = pipe.to(device)
            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass
            _PIPE_CN = pipe
            _PIPE_CN_DEVICE = device
            _PIPE_CN_LOAD_TIME = time.time() - t0
            print(f"[image_renderer] ControlNet pipeline loaded in {_PIPE_CN_LOAD_TIME:.1f}s")
            return True
        except Exception as e:
            import traceback
            print(f"[image_renderer] failed to load ControlNet pipeline: {e}")
            traceback.print_exc()
            return False


def warmup(include_controlnet: bool = False) -> dict[str, Any]:
    """Load the model + run one cheap inference to warm CUDA/Metal caches.

    Call this on server startup so the first user-visible render is fast
    (1.5-3s instead of 30-60s including model load).

    If include_controlnet=True, also pre-loads + warms the depth-conditioned
    pipeline (an extra ~10-30s and ~1GB RAM, but the first faithful render
    is then instant).
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
    info: dict[str, Any] = {
        "loaded": True,
        "device": _PIPE_DEVICE,
        "model": _DEFAULT_MODEL,
        "load_time_s": round(_PIPE_LOAD_TIME, 2),
        "warmup_inference_s": round(time.time() - t0, 2),
        "test_image_size": img.size,
    }

    if include_controlnet:
        if _ensure_controlnet_pipeline_loaded():
            from PIL import Image
            t1 = time.time()
            depth = Image.new("L", (512, 512), 128)
            try:
                _PIPE_CN(
                    prompt="warmup",
                    image=depth,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    controlnet_conditioning_scale=0.5,
                    height=512, width=512,
                ).images[0]
                info["controlnet_loaded"] = True
                info["controlnet_model"] = _DEFAULT_CONTROLNET
                info["controlnet_load_time_s"] = round(_PIPE_CN_LOAD_TIME, 2)
                info["controlnet_warmup_inference_s"] = round(time.time() - t1, 2)
            except Exception as e:
                info["controlnet_loaded"] = False
                info["controlnet_error"] = repr(e)
        else:
            info["controlnet_loaded"] = False
            info["controlnet_error"] = "pipeline failed to load"

    return info


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
        # Serialize — see note on _INFER_LOCK above.
        with _INFER_LOCK:
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


# ---------------------------------------------------------------------------
# Faithful render: depth-conditioned via Depth ControlNet
# ---------------------------------------------------------------------------

def _render_with_depth(prompt: str, depth_image, width: int = 768,
                        height: int = 512, steps: int = 5,
                        controlnet_scale: float = 0.55) -> RenderResult:
    """Render an image conditioned on a grayscale depth map.

    SDXL-turbo + Depth ControlNet. We use slightly more steps (5 vs 2)
    than pure turbo to give ControlNet enough denoising headroom; latency
    on M4 Max is ~3-4s warm.
    """
    if not _ensure_controlnet_pipeline_loaded():
        return RenderResult(
            image_bytes=b"", image_format="png", prompt=prompt,
            latency_s=0.0, width=width, height=height,
            backend="error", error="controlnet pipeline failed to load",
        )
    # Resize depth map to render dimensions if needed
    if depth_image.size != (width, height):
        from PIL import Image
        depth_image = depth_image.resize((width, height), Image.BICUBIC)
    # ControlNet expects an RGB image even for depth
    if depth_image.mode != "RGB":
        depth_image = depth_image.convert("RGB")

    t0 = time.time()
    try:
        # Serialize SDXL calls — Apple MPS doesn't multiplex SDXL well,
        # concurrent calls just stall together. Better to queue them.
        with _INFER_LOCK:
            img = _PIPE_CN(
                prompt=prompt,
                image=depth_image,
                num_inference_steps=steps,
                guidance_scale=0.0,
                controlnet_conditioning_scale=controlnet_scale,
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
        backend=f"sdxl-turbo+depth-controlnet:{_PIPE_CN_DEVICE}",
    )


# City-level overrides for the faithful prompt. When metadata.city_inspiration
# matches one of these, we use it instead of the country-level hint, so a
# Munich Schwabing apartment doesn't get rendered as a Berlin Altbau.
_FAITHFUL_CITY_STYLE = {
    "Berlin":     "Berlin Altbau Wilhelminian, herringbone parquet, tall sash windows",
    "Munich":     "Munich Schwabing Jugendstil, oak parquet, ornate stucco mouldings, tall casement windows",
    "Hamburg":    "Hamburg Hanseatic apartment, white walls, parquet, tall windows",
    "Cologne":    "Cologne Belgian Quarter, parquet, classical mouldings, large windows",
    "Frankfurt":  "Frankfurt Westend Gründerzeit, parquet, period mouldings",
    "Leipzig":    "Leipzig Gründerzeit, parquet, classical mouldings",
    "Vienna":     "Vienna Altbau, herringbone parquet, ornate Habsburg-era mouldings",
    "Paris":      "Paris Haussmann, herringbone parquet, French casement windows, period mouldings",
    "Lyon":       "Lyon Presqu'île, parquet, period detail",
    "London":     "London period apartment, sash windows, white walls, neutral classic furniture",
    "Edinburgh":  "Edinburgh New Town, sash windows, classical mouldings",
    "Rome":       "Roman apartment, terracotta floor, warm whites, plaster walls",
    "Milan":      "Milanese apartment, herringbone wood, refined Italian modernism",
    "Barcelona":  "Barcelona Eixample, hydraulic tile floor, ornate ceiling rosettes",
    "Madrid":     "Madrid apartment, parquet, classical detail, abundant light",
    "Lisbon":     "Lisbon Pombaline, azulejo accents, dark wood floor",
    "Amsterdam":  "Dutch canal-house, large windows, parquet, contemporary nordic",
    "Copenhagen": "Danish apartment, wide-plank oak, Hans Wegner furniture, soft daylight",
    "Stockholm":  "Stockholm apartment, white walls, light wood floor, minimal modern",
    "Tokyo":      "Tokyo mansion, wood floor, low Japanese furniture, sliding doors",
    "Kyoto":      "Kyoto machiya, tatami detail, warm wood, paper screens",
    "Osaka":      "Osaka modern apartment, wood floor, low furniture, urban view",
    "Seoul":      "Seoul apartment, laminate floor, minimalist Korean modern",
    "Singapore":  "Singapore HDB, vinyl plank, compact modern, tropical light",
    "Hong Kong":  "Hong Kong apartment, polished tile, dense modern, harbour view",
    "Dubai":      "Dubai luxe apartment, marble floor, neutral palette, skyline view",
    "Mumbai":     "Mumbai apartment, vitrified tile, contemporary Indian, balcony",
    "Bangalore":  "Bangalore apartment, granite floor, contemporary Indian, balcony",
    "Delhi":      "Delhi apartment, polished granite floor, contemporary Indian",
    "Sydney":     "Sydney apartment, hardwood, indoor-outdoor flow, garden view",
    "Melbourne":  "Melbourne apartment, hardwood, contemporary, neutral palette",
    "New York":   "New York apartment, hardwood floor, exposed brick or modern white walls",
    "Brooklyn":   "Brooklyn brownstone, hardwood, exposed brick",
    "Los Angeles":"LA apartment, polished concrete, mid-century modern, abundant light",
    "Toronto":    "Toronto apartment, hardwood, neutral contemporary",
    "Mexico City":"CDMX apartment, talavera tile, warm whites, plants",
    "São Paulo":  "São Paulo apartment, polished concrete, tropical modernist",
    "Rio de Janeiro":"Rio apartment, polished wood, tropical modernist, ocean light",
}

# Compact style hints for the faithful pipeline (CLIP truncates at 77 tokens
# — depth map already supplies geometry, so the prompt only needs style words).
_FAITHFUL_STYLE = {
    "Germany": "Berlin Altbau, parquet floor, tall sash windows",
    "France": "Haussmannian Paris, herringbone parquet, French windows, period mouldings",
    "United Kingdom": "British apartment, sash windows, neutral period interior",
    "Italy": "Italian apartment, terracotta tile, warm whites, plaster walls",
    "Spain": "Spanish apartment, terracotta floor, warm whites, wrought iron",
    "Greece": "Greek interior, white walls, blue accents, stone floor",
    "Portugal": "Portuguese apartment, azulejo accents, dark wood floor",
    "Netherlands": "Dutch canal-house, large windows, white walls, wood floor",
    "Sweden": "Scandinavian, white walls, light wood floor, minimal furniture",
    "Norway": "Norwegian, light wood, neutral palette, hygge details",
    "Denmark": "Danish apartment, wide-plank oak, Hans Wegner furniture",
    "Finland": "Finnish apartment, blonde birch floor, Nordic minimalism",
    "India": "Indian apartment, granite or vitrified tile floor, balcony with city view",
    "Japan": "Japanese apartment, wood floor, shoji-style sliding doors, low furniture",
    "Singapore": "Singapore HDB, vinyl plank, compact modern, tropical view",
    "United States": "American apartment, hardwood floor, contemporary furniture",
    "Canada": "Canadian apartment, hardwood floor, neutral contemporary",
    "Australia": "Australian apartment, polished concrete, indoor-outdoor flow",
    "United Arab Emirates": "Dubai apartment, marble floor, neutral luxe, skyline view",
    "Saudi Arabia": "Riyadh apartment, marble floor, Arabic majlis-style furniture",
    "Turkey": "Turkish apartment, parquet, Anatolian textiles, warm tones",
    "Brazil": "Brazilian apartment, polished concrete, tropical modernist, plants",
    "Argentina": "Buenos Aires apartment, parquet, period mouldings",
    "Mexico": "Mexican apartment, talavera tile, rustic-modern, abundant light",
    "Morocco": "Moroccan apartment, zellige tile, earthy tones, patterned textiles",
    "Egypt": "Cairo apartment, marble floor, warm neutrals, Mediterranean light",
    "Kenya": "Nairobi apartment, polished concrete, African-influenced furniture",
    "Nigeria": "Lagos apartment, polished tile, contemporary West African",
    "South Africa": "Cape Town apartment, indoor-outdoor flow, contemporary",
    "China": "Modern Chinese apartment, polished tile, Chinese-influenced details",
    "South Korea": "Seoul apartment, laminate floor, minimalist contemporary",
    "Vietnam": "Vietnamese apartment, tile floor, tropical modernist",
    "Indonesia": "Jakarta apartment, hardwood, tropical contemporary",
}


def _resolve_style(md: dict) -> str:
    """Pick the most specific style hint available: city > country > default."""
    city = (md.get("city_inspiration") or "").strip()
    if city and city in _FAITHFUL_CITY_STYLE:
        return _FAITHFUL_CITY_STYLE[city]
    country = md.get("country", "")
    return _FAITHFUL_STYLE.get(country, "modern apartment, neutral palette")


def _build_faithful_prompt(template: dict, focus_room_type: str,
                             focus_room_name: str, ceiling_m: float) -> str:
    """Compact prompt (<=77 CLIP tokens) for depth-conditioned rendering.

    The depth map gives the room geometry, so we only need:
      room type + city/country style + lighting + quality cues.
    """
    md = template.get("metadata", {})
    style = _resolve_style(md)

    rtype = focus_room_type or "living"
    room_words = {
        "living": "living room, sofa and coffee table",
        "kitchen": "kitchen with cabinetry and natural light",
        "master_bedroom": "master bedroom with bed and window",
        "bedroom": "bedroom with bed",
        "dining": "dining area with table",
        "balcony": "balcony with city view",
        "office": "home office with desk",
        "bathroom": "bathroom with sink and mirror",
    }
    room_text = room_words.get(rtype, "interior room")

    high_ceiling = "high ceilings, " if ceiling_m >= 3.0 else ""

    # Keep this terse — every word costs CLIP tokens.
    return (
        f"Photorealistic interior, {room_text}. "
        f"{style}. {high_ceiling}natural daylight, wide-angle, magazine quality."
    )


def _build_dollhouse_prompt(template: dict) -> str:
    """Compact prompt (<=77 CLIP tokens) for full-apartment dollhouse view.

    The depth map is a cutaway model from oblique-above; we want SDXL to
    render this as an architect's 3D model / Matterport-style dollhouse,
    not a single-room interior.
    """
    md = template.get("metadata", {}) or {}
    bedrooms = md.get("bedrooms", 0)
    area = md.get("total_area_sqm", 0)
    style = _resolve_style(md)

    bed_text = f"{bedrooms}-bedroom " if bedrooms else ""
    area_text = f"{int(area)} m² " if area else ""

    # Architectural model / dollhouse keywords push SDXL toward the
    # cutaway visualisation that matches the depth geometry.
    return (
        f"Architectural 3D dollhouse cutaway model of a {bed_text}"
        f"{area_text}apartment, {style}. "
        f"Top-down oblique view, all rooms visible, walls cut away, "
        f"furnished interior, photoreal architectural visualization."
    )


def render_faithful_dollhouse_from_template(
    template: dict,
    width: int = 768, height: int = 512,
    steps: int = 5,
    controlnet_scale: float = 0.65,
    wall_height_factor: float = 1.0,
) -> dict[str, Any]:
    """Photorealistic FULL-FLAT cutaway render via depth ControlNet.

    Builds a 3D scene of the entire apartment with NO ceiling, positions
    a camera high above one corner looking down at the whole footprint,
    raycasts a depth map of the cutaway, and feeds it to SDXL +
    Depth ControlNet. The result is a photoreal Matterport-style 3D
    model image that shows ALL rooms at once.

    Tower templates fall back to stylistic exterior (no interior to model).
    """
    if template.get("metadata", {}).get("tower"):
        result = render_from_template(template, width=width, height=height)
        return {
            "result": result,
            "view": "dollhouse",
            "rooms_count": 0,
            "fallback": "tower-exterior-stylistic",
        }

    from .depth_renderer import render_template_dollhouse_depth

    info = render_template_dollhouse_depth(
        template, image_size=(width, height),
        wall_height_factor=wall_height_factor,
    )
    depth = info["depth_image"]
    prompt = _build_dollhouse_prompt(template)
    result = _render_with_depth(
        prompt, depth, width=width, height=height,
        steps=steps, controlnet_scale=controlnet_scale,
    )
    return {
        "result": result,
        "view": "dollhouse",
        "rooms_count": info.get("rooms_count", 0),
        "boundary_w": info.get("boundary_w", 0.0),
        "boundary_d": info.get("boundary_d", 0.0),
        "ceiling_height_m": info.get("ceiling_height_m", 2.7),
        "country": info.get("country", ""),
        "bedrooms": info.get("bedrooms", 0),
        "total_area_sqm": info.get("total_area_sqm", 0),
        "fallback": None,
    }


def render_faithful_from_template(template: dict,
                                   focus_room_type: str | None = None,
                                   focus_room_id: str | None = None,
                                   width: int = 768, height: int = 512,
                                   steps: int = 5,
                                   controlnet_scale: float = 0.55,
                                   ) -> dict[str, Any]:
    """Photorealistic render that respects the actual floor plan layout.

    Builds a 3D scene from the template (walls/floor/ceiling), positions
    a camera inside the chosen room (focus_room_id if given, else the
    largest non-circulation room), raycasts a depth map, and uses
    SDXL + Depth ControlNet to render. The result is an interior
    photograph whose room proportions and wall angles match the BIM
    geometry — not just stylistic stock imagery.

    Pass focus_room_id to render any specific room (used for the
    per-room virtual walkthrough).

    Returns a dict containing the RenderResult plus depth-map metadata
    (focus room id/name/type/area, ceiling height) so callers can show
    the user which room is depicted.
    """
    # Tower templates → no usable interior; fall back to stylistic exterior.
    if template.get("metadata", {}).get("tower"):
        result = render_from_template(template, width=width, height=height)
        return {
            "result": result,
            "focus_room_id": "",
            "focus_room_name": "",
            "focus_room_type": "",
            "focus_room_area": 0,
            "ceiling_height_m": 0.0,
            "fallback": "tower-exterior-stylistic",
        }

    from .depth_renderer import render_template_depth

    info = render_template_depth(
        template, focus_room_id=focus_room_id,
        focus_room_type=focus_room_type,
    )
    depth = info["depth_image"]
    fr_type = info.get("focus_room_type") or focus_room_type or "living"
    prompt = _build_faithful_prompt(
        template,
        focus_room_type=fr_type,
        focus_room_name=info.get("focus_room_name", ""),
        ceiling_m=info.get("ceiling_height_m", 2.7),
    )
    result = _render_with_depth(
        prompt, depth, width=width, height=height,
        steps=steps, controlnet_scale=controlnet_scale,
    )
    return {
        "result": result,
        "focus_room_id": info.get("focus_room_id", ""),
        "focus_room_name": info.get("focus_room_name", ""),
        "focus_room_type": info.get("focus_room_type", ""),
        "focus_room_area": info.get("focus_room_area", 0),
        "ceiling_height_m": info.get("ceiling_height_m", 2.7),
        "fallback": None,
    }


# ---------------------------------------------------------------------------
# Cubemap photoreal — Matterport-style 360° walk per room
# ---------------------------------------------------------------------------

# Per-face look hints. SDXL paints the back wall the most attention, so we
# tag what's likely visible looking each direction (purely cosmetic — the
# depth map enforces geometry).
_FACE_HINT = {
    "posx": "looking forward across the room",
    "negx": "looking back across the room",
    "posy": "looking up at the ceiling",
    "negy": "looking down at the floor",
    "posz": "looking forward to the far wall",
    "negz": "looking back from the far wall",
}


def _build_cubemap_face_prompt(template: dict, room_type: str,
                                  face: str, ceiling_m: float) -> str:
    """Compact CLIP-77 prompt per cubemap face. Same style hint as the
    interior render, but with a face-specific framing tag so SDXL doesn't
    repeat the same wall on all 6 sides.
    """
    md = template.get("metadata", {})
    style = _resolve_style(md)
    rtype = (room_type or "living").lower()
    room_words = {
        "living":         "living room, sofa and coffee table",
        "kitchen":        "kitchen with cabinetry",
        "kueche":         "kitchen with cabinetry",
        "kochnische":     "kitchen with cabinetry",
        "master_bedroom": "master bedroom with bed",
        "bedroom":        "bedroom with bed",
        "dining":         "dining area with table",
        "balcony":        "balcony with view",
        "office":         "home office",
        "study":          "home office",
        "bathroom":       "bathroom with vanity and tub",
        "bad":            "bathroom with vanity and tub",
        "wc":             "small bathroom",
        "diele":          "entry foyer",
        "flur":           "entry foyer",
        "entry":          "entry foyer",
    }
    room_text = room_words.get(rtype, "interior room")

    # The "looking up/down" faces want different prompts to avoid SDXL
    # painting a sofa on the ceiling.
    if face == "posy":
        face_text = "ceiling view, ornate plaster ceiling, light fixture"
    elif face == "negy":
        face_text = "floor view, top-down on rug and floor"
    else:
        face_text = f"{room_text}, {_FACE_HINT.get(face, '')}"

    high_ceiling = "high ceilings, " if ceiling_m >= 3.0 else ""
    return (
        f"Photorealistic interior, {face_text}. "
        f"{style}. {high_ceiling}natural daylight, magazine quality."
    )


def render_room_cubemap_photoreal(template: dict, room_id: str,
                                       face_size: int = 512,
                                       steps: int = 5,
                                       controlnet_scale: float = 0.55,
                                       ) -> dict[str, Any]:
    """Generate the 6 photoreal cubemap faces for a single room.

    Pipeline per face:
      1. trimesh raycast → 90°-FOV depth map (face_size × face_size)
      2. SDXL-turbo + Depth ControlNet → photoreal face

    Total: ~12-15s for a cold room (warm pipeline), serialised by
    _INFER_LOCK so the GPU never thrashes.

    Returns a dict { "faces": {label: PIL.Image}, "room_*": metadata,
                     "latency_s": float, "errors": [str, ...] }.
    """
    from .depth_renderer import render_room_cubemap_depth

    t0 = time.time()
    depth_info = render_room_cubemap_depth(template, room_id, image_size=face_size)
    depth_faces = depth_info["faces"]
    room_type = depth_info.get("room_type", "")
    ceiling_m = template.get("boundary", {}).get("ceiling_height_mm", 2700) / 1000.0

    if not _ensure_controlnet_pipeline_loaded():
        return {
            "faces": {}, "room_id": room_id,
            "errors": ["controlnet pipeline failed to load"],
            "latency_s": time.time() - t0,
        }

    out_faces: dict[str, Any] = {}
    errors: list[str] = []
    for label, depth in depth_faces.items():
        prompt = _build_cubemap_face_prompt(
            template, room_type, label, ceiling_m,
        )
        # Inline render — sharing the interior pipeline's helper:
        try:
            depth_rgb = depth.convert("RGB").resize(
                (face_size, face_size), depth.BICUBIC
            ) if hasattr(depth, "BICUBIC") else depth.convert("RGB")
            with _INFER_LOCK:
                img = _PIPE_CN(
                    prompt=prompt,
                    image=depth_rgb,
                    num_inference_steps=steps,
                    guidance_scale=0.0,
                    controlnet_conditioning_scale=controlnet_scale,
                    height=face_size, width=face_size,
                ).images[0]
            out_faces[label] = img
        except Exception as e:
            errors.append(f"{label}: {e!r}")
            # Fallback: use the depth map as a placeholder so the cubemap
            # still loads (better than a hole in the skybox).
            out_faces[label] = depth.convert("RGB")

    return {
        "faces": out_faces,
        "room_id": room_id,
        "room_name": depth_info.get("room_name", ""),
        "room_type": room_type,
        "room_area": depth_info.get("room_area", 0),
        "ceiling_height_m": ceiling_m,
        "position_bim": depth_info.get("position_bim"),
        "latency_s": time.time() - t0,
        "errors": errors,
    }


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
