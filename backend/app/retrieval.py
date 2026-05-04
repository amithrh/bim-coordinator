"""Sentence-transformers retrieval over the 8 template pool.

Builds embeddings on import (warm cache for the FastAPI worker). The
all-MiniLM-L6-v2 model is ~80MB and downloads once on first import.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .storage import load_all_templates


_MODEL: SentenceTransformer | None = None


def _model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def template_text(t: dict) -> str:
    m = t["metadata"]
    tags = ", ".join(m.get("tags", []))
    return (
        f"{m['description']} | {m['style']} | tags: {tags} | "
        f"{m['size_label']} {m['bedrooms']} bedrooms {m['total_area_sqm']} sqm "
        f"in {m.get('city_inspiration', m.get('country', ''))}"
    )


@lru_cache(maxsize=1)
def _embeddings() -> tuple[list[dict], np.ndarray]:
    templates = list(load_all_templates())
    texts = [template_text(t) for t in templates]
    embeds = _model().encode(texts, convert_to_numpy=True)
    return templates, embeds


def _compute_reasons(brief: dict, t: dict) -> list[str]:
    reasons: list[str] = []
    m = t["metadata"]
    if brief.get("region") and m.get("region") == brief["region"]:
        reasons.append(f"matches region {brief['region']}")
    if brief.get("city") and brief["city"].lower() in m.get("city_inspiration", "").lower():
        reasons.append(f"{brief['city']} typology")
    if brief.get("bedrooms") is not None and m.get("bedrooms") == brief["bedrooms"]:
        reasons.append(f"{brief['bedrooms']} BR match")
    if brief.get("vastu_compliant") and m.get("vastu_compliant"):
        reasons.append("Vastu-compliant")
    target = brief.get("total_area_sqm")
    if target:
        diff = abs(m["total_area_sqm"] - target)
        ratio = diff / target
        if ratio < 0.10:
            reasons.append(f"area within 10% of {target} sqm")
        elif ratio < 0.25:
            reasons.append(f"area near {target} sqm")
    if not reasons:
        reasons.append("semantic match on style and tags")
    return reasons


def retrieve(brief: dict, top_n: int = 4) -> list[dict[str, Any]]:
    """Return ranked card payloads. Cards include the full template + a
    score in [60, 99] and a list of reasoning bullets."""
    templates, embeds = _embeddings()
    if not templates:
        return []

    # Hard filters: region (if specified) and bedrooms (within ±1)
    candidate_idx: list[int] = []
    target_bedrooms = brief.get("bedrooms")
    target_region = brief.get("region")
    for i, t in enumerate(templates):
        m = t["metadata"]
        if target_region and m["region"] != target_region:
            continue
        if target_bedrooms is not None:
            if abs(m["bedrooms"] - target_bedrooms) > 1:
                continue
        candidate_idx.append(i)

    if not candidate_idx:
        # Fallback to all templates if hard filters eliminate everything
        candidate_idx = list(range(len(templates)))

    brief_embed = _model().encode([brief["raw"] or ""], convert_to_numpy=True)[0]
    brief_norm = np.linalg.norm(brief_embed) or 1.0

    scored: list[tuple[float, int]] = []
    for i in candidate_idx:
        t_embed = embeds[i]
        denom = (brief_norm * (np.linalg.norm(t_embed) or 1.0))
        cosine = float(np.dot(brief_embed, t_embed) / denom)
        bonus = 0.0
        m = templates[i]["metadata"]
        if brief.get("vastu_compliant") and m.get("vastu_compliant"):
            bonus += 0.15
        if brief.get("city") and brief["city"].lower() in m.get("city_inspiration", "").lower():
            bonus += 0.10
        target = brief.get("total_area_sqm")
        if target:
            ratio = abs(m["total_area_sqm"] - target) / target
            bonus -= ratio * 0.2
        scored.append((cosine + bonus, i))

    scored.sort(key=lambda s: s[0], reverse=True)
    top = scored[:top_n]
    if not top:
        return []
    max_score = max(s[0] for s in top) or 1.0

    cards: list[dict[str, Any]] = []
    for s, i in top:
        cards.append({
            "template": templates[i],
            "score": max(60, min(99, int((s / max_score) * 95) + 4)),
            "raw_cosine": round(s, 4),
            "reasoning": _compute_reasons(brief, templates[i]),
        })
    return cards
