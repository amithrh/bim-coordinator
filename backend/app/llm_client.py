"""Stage 2 reasoning client for the BIM Coordinator demo.

Wraps the fine-tuned Llama 3.2 3B (or whichever model wins) for use by the
FastAPI backend. Stage 1 retrieval (MiniLM in retrieval.py) returns top-N
candidates; this module turns brief + candidates into a structured ranked
response with architectural reasoning.

Two backends:
  - local_mlx: load the fine-tuned model directly via mlx-lm (Mac dev mode)
  - http:     POST to an OpenAI-compatible endpoint (Cloud Run / Ollama / vLLM)

Configured via env vars:
  BIM_SLM_BACKEND   = "local_mlx" | "http" (default: "local_mlx" if MLX is
                      importable on this host, else "http")
  BIM_SLM_MODEL     = HF id for local mlx, model name for http
  BIM_SLM_ADAPTER   = (local mlx only) path to LoRA adapter directory
  BIM_SLM_BASE_URL  = (http only) e.g. https://bim-slm-xxx.run.app
  BIM_SLM_API_KEY   = (http only) Bearer token; "anything" works for Ollama
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator


SYSTEM_PROMPT = (
    "You are an expert architectural consultant for the BIM Coordinator. "
    "You help users find floor plan templates that match their lifestyle, "
    "budget, and location preferences from a curated library of 500 real-world "
    "templates spanning 74 countries. "
    "Always ground your recommendations in the specific metadata of the candidate "
    "templates provided. Pick the top 4 best fits, with the strongest match first, "
    "and explain each choice in 2-4 sentences using real architectural reasoning "
    "(layout, ceiling height, room flow, cultural conventions, persona fit). "
    "Be honest about trade-offs."
)


def _candidate_block(card: dict[str, Any], idx: int) -> str:
    """Render one candidate template the way the model was trained to read.

    Accepts either:
      - a raw template dict {id, metadata, rooms, ...}
      - a retrieval card {template: {...}, score: int, reasoning: [...]}
    """
    template = card["template"] if "template" in card else card
    md = template.get("metadata", {})
    rooms = (
        template.get("room_names")
        or [r.get("name", "") for r in template.get("rooms", [])]
    )
    suit = md.get("suitable_for", [])

    parts = [
        f"id: {template['id']}",
        f"location: {md.get('city_inspiration', '')}, {md.get('country', '')}",
        (
            f"size: {md.get('size_label', '')} "
            f"({md.get('total_area_sqm', 0)} sqm, "
            f"{md.get('bedrooms', 0)}-bed/{md.get('bathrooms', 0)}-bath)"
        ),
        f"style: {md.get('style', '')}",
        f"rooms: {', '.join(rooms)}",
        f"suitable_for: {', '.join(suit) if suit else 'general'}",
    ]
    return f"[{idx}] {template['id']}\n" + "\n".join(parts)


def build_user_prompt(brief_text: str, candidates: list[dict]) -> str:
    """Assemble the user prompt the model expects (matches training data)."""
    blocks = [_candidate_block(c, i) for i, c in enumerate(candidates, 1)]
    return (
        f"BRIEF:\n{brief_text}\n\n"
        f"CANDIDATE TEMPLATES (top retrieval matches):\n\n"
        + "\n\n".join(blocks)
        + (
            "\n\nPick the top 4 candidates that best fit the brief. For each, "
            "give 2-4 sentences of architectural reasoning grounded in the "
            "template's actual metadata. List the top pick first, then 3 alternatives."
        )
    )


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    backend: str
    model: str
    error: str | None = None


# ---------------------------------------------------------------------------
# Local MLX backend
# ---------------------------------------------------------------------------

class _LocalMLXBackend:
    """Lazy-loaded mlx-lm model. One instance per process."""

    _lock = threading.Lock()
    _model = None
    _tokenizer = None
    _model_path: str = ""
    _adapter_path: str | None = None

    def __init__(self, model_path: str, adapter_path: str | None = None) -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path

    def _ensure_loaded(self):
        with self._lock:
            need_reload = (
                self.__class__._model is None
                or self.__class__._model_path != self.model_path
                or self.__class__._adapter_path != self.adapter_path
            )
            if need_reload:
                from mlx_lm import load  # imported lazily so server starts fast
                if self.adapter_path:
                    model, tok = load(self.model_path, adapter_path=self.adapter_path)
                else:
                    model, tok = load(self.model_path)
                self.__class__._model = model
                self.__class__._tokenizer = tok
                self.__class__._model_path = self.model_path
                self.__class__._adapter_path = self.adapter_path

    def generate(
        self,
        brief_text: str,
        candidates: list[dict],
        max_tokens: int = 700,
    ) -> LLMResponse:
        self._ensure_loaded()
        from mlx_lm import generate as _generate

        prompt = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(brief_text, candidates)},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        t0 = time.time()
        try:
            text = _generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            )
            err = None
        except Exception as e:
            text = ""
            err = repr(e)
        return LLMResponse(
            text=text,
            latency_s=time.time() - t0,
            backend="local_mlx",
            model=self.model_path + ("+adapter" if self.adapter_path else ""),
            error=err,
        )


# ---------------------------------------------------------------------------
# HTTP backend (OpenAI-compatible — Cloud Run / Ollama / vLLM)
# ---------------------------------------------------------------------------

class _HttpBackend:
    def __init__(self, base_url: str, model: str, api_key: str = "anything") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def generate(
        self,
        brief_text: str,
        candidates: list[dict],
        max_tokens: int = 700,
    ) -> LLMResponse:
        import urllib.request
        import urllib.error

        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(brief_text, candidates)},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"]
            err = None
        except urllib.error.URLError as e:
            text = ""
            err = f"connect failed: {e.reason}"
        except Exception as e:
            text = ""
            err = repr(e)
        return LLMResponse(
            text=text,
            latency_s=time.time() - t0,
            backend="http",
            model=f"{self.base_url}/{self.model}",
            error=err,
        )


# ---------------------------------------------------------------------------
# Factory + module-level singleton
# ---------------------------------------------------------------------------

def _backend_from_env():
    backend_kind = os.getenv("BIM_SLM_BACKEND")
    if not backend_kind:
        # Default: prefer local_mlx if mlx is available, else http
        try:
            import mlx_lm  # noqa: F401
            backend_kind = "local_mlx"
        except ImportError:
            backend_kind = "http"

    if backend_kind == "local_mlx":
        model_path = os.getenv(
            "BIM_SLM_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit"
        )
        adapter_path = os.getenv("BIM_SLM_ADAPTER")
        return _LocalMLXBackend(model_path, adapter_path)

    if backend_kind == "http":
        base_url = os.getenv("BIM_SLM_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "BIM_SLM_BACKEND=http but BIM_SLM_BASE_URL is not set"
            )
        model = os.getenv("BIM_SLM_MODEL", "bim-slm")
        api_key = os.getenv("BIM_SLM_API_KEY", "anything")
        return _HttpBackend(base_url, model, api_key)

    raise RuntimeError(f"Unknown BIM_SLM_BACKEND: {backend_kind}")


_BACKEND = None
_BACKEND_LOCK = threading.Lock()


def get_backend():
    """Singleton accessor — lazily initialised so server can boot without model."""
    global _BACKEND
    if _BACKEND is None:
        with _BACKEND_LOCK:
            if _BACKEND is None:
                _BACKEND = _backend_from_env()
    return _BACKEND


def reason(brief_text: str, candidates: list[dict], max_tokens: int = 700) -> LLMResponse:
    """Top-level entry: brief + candidates -> ranked picks with reasoning."""
    return get_backend().generate(brief_text, candidates, max_tokens=max_tokens)
