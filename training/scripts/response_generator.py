"""Generate ideal model responses for (brief, target, candidates) tuples.

The response is what the fine-tuned model learns to produce. Quality here =
quality of the trained model. We use the actual template metadata so every
fact in the response is grounded.

Response format (markdown - readable + parseable):

  Based on your brief, <one-line interpretation>.

  **Top 4 matches:**

  **1. <style>** (`<template_id>`)
  <2-4 sentence rationale grounded in template metadata>

  **2. <style>** (`<template_id>`)
  <2-4 sentence rationale>

  **3. <style>** (`<template_id>`)
  <2-4 sentence rationale>

  **4. <style>** (`<template_id>`)
  <2-4 sentence rationale>

  <optional 1-2 sentence comparison summary>
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from brief_generator import Brief
from candidate_selector import CandidatePool
from templates_index import Template


# ---------------------------------------------------------------------------
# Reusable prose fragments
# ---------------------------------------------------------------------------

INTERPRET_OPENINGS = [
    "Based on your brief",
    "Looking at what you described",
    "From your requirements",
    "Reading your brief",
    "Given your needs",
]


def _persona_summary(brief: Brief) -> str:
    """Translate the brief's persona into a one-line interpretation."""
    p = brief.persona
    summaries = {
        "single": "you're looking for a compact place that suits one person",
        "single_professional": "you want a clean professional flat that supports work-life balance",
        "couple": "you want a flat for two with room to cook and entertain",
        "young_family": "you need family-friendly layouts with separated bedrooms and good light",
        "growing_family": "you want a layout with room to grow into",
        "family": "you need real family bedrooms, not boxes",
        "large_family": "you need maximum bedroom count without sacrificing common space",
        "single_parent": "you want a calm, well-organised space for you and your child",
        "creative_artist": "natural light and a flexible work corner are non-negotiable",
        "academic": "you want quiet, lots of wall space, and a proper study area",
        "student": "you need affordable, walkable, and low-maintenance",
        "retired": "single-floor, accessible, and easy to maintain matter most",
        "executive_pied_a_terre": "you want a polished city flat that you only occupy part-time",
        "weekend_retreat": "you want something easy to lock and leave",
        "remote_worker": "you need a real home office, not a corner of the living room",
        "investor": "you want strong tenant appeal and rental yield",
    }
    return summaries.get(p, "you want a well-designed home that fits your lifestyle")


def _format_size(t: Template) -> str:
    """e.g. '1-bed studio (50 m²)' or '2 BHK (75 m²)'."""
    if t.bedrooms == 0:
        return f"studio ({int(t.total_area_sqm)} m²)"
    return f"{t.bedrooms}-bed ({int(t.total_area_sqm)} m²)"


def _layout_summary(t: Template) -> str:
    """e.g. 'Foyer → Living/Dining → Kitchen, Master + Bedroom 2 + Bath/WC.'"""
    rooms = t.room_names
    if not rooms:
        return ""
    # Group rooms by entry/public/private
    entry = [r for r in rooms if r.lower() in
             ("foyer", "entry", "hall", "hol", "diele", "ingresso", "entrada",
              "recibidor", "vestibule", "lobby", "inkom", "genkan")]
    public = [r for r in rooms if any(k in r.lower() for k in
              ("living", "salon", "soggiorno", "stue", "wohn", "salone",
               "dining", "kitchen", "küche", "cocina", "cucina", "vardagsrum",
               "olohuone", "ldk", "saloni", "drawing"))]
    rest = [r for r in rooms if r not in entry and r not in public]
    parts = []
    if entry:
        parts.append(entry[0])
    if public:
        parts.append(" / ".join(public[:3]))
    if rest:
        parts.append(", ".join(rest[:5]))
    return " → ".join(parts)


def _ceiling_note(t: Template) -> str | None:
    """Returns a sentence about ceiling height if it's notable."""
    h = t.ceiling_height_mm
    if h >= 3500:
        return f"the {h}mm ceilings are exceptionally tall — strong vertical feel"
    if h >= 3000:
        return f"the {h}mm ceilings give it real airiness"
    if h >= 2700:
        return f"the {h}mm ceilings are standard but generous for the size"
    if h > 0 and h < 2500:
        return f"compact {h}mm ceilings — practical not lofty"
    return None


def _wall_note(t: Template) -> str | None:
    """Returns context about wall thickness when culturally meaningful."""
    w = t.wall_thickness_mm
    if w >= 350 and t.country in ("Germany", "Italy", "France", "Spain", "Portugal", "Greece", "Belgium"):
        return f"thick {w}mm masonry walls give it solidity and natural sound insulation"
    if w == 230 and t.country in ("India", "Pakistan", "Bangladesh", "Sri Lanka"):
        return f"standard {w}mm brick-RCC construction"
    return None


def _persona_fit(t: Template, brief: Brief) -> str | None:
    """One sentence linking template features to the brief's persona need."""
    p = brief.persona
    has_balcony = t.has_balcony
    has_separate_kitchen = t.has_separate_kitchen
    is_open_plan = t.is_open_plan

    # Persona-specific connections
    if p == "young_family" and t.bedrooms >= 2:
        if "Master" in " ".join(t.room_names) or "Master Bedroom" in t.room_names:
            return "the master + second bedroom layout gives kids and parents acoustic separation"
        return "the bedroom split works well for parents + kids"

    if p == "couple":
        if has_separate_kitchen:
            return "the separate kitchen suits people who cook a lot"
        if is_open_plan:
            return "the open plan suits couples who like to entertain"

    if p == "remote_worker" and t.bedrooms >= 2:
        return "the second bedroom can become a real home office"

    if p == "creative_artist" and has_balcony:
        return "the balcony is useful as an indoor-outdoor work overflow space"

    if p == "single_professional":
        if has_balcony:
            return "the balcony adds outdoor breathing room without inflating size"
        return "the layout is right-sized — no wasted space"

    if p == "executive_pied_a_terre":
        return "compact and lock-and-leave friendly"

    if p == "retired" and t.bedrooms >= 1:
        return "single-floor living, no stairs to navigate"

    return None


def _why_target(target: Template, brief: Brief) -> str:
    """2-4 sentences explaining why this is the top pick."""
    sentences = []

    # Sentence 1: locate + size
    loc_sentence = (
        f"This is a {_format_size(target)} in {target.short_locale}, "
        f"in the {target.style.split('(')[0].strip().split('—')[0].strip()} style."
    )
    sentences.append(loc_sentence)

    # Sentence 2: layout
    layout = _layout_summary(target)
    if layout:
        sentences.append(f"Layout: {layout}.")

    # Sentence 3: a meaningful technical detail (ceiling or walls)
    detail = _ceiling_note(target) or _wall_note(target)
    if detail:
        sentences.append(detail.capitalize() + ".")

    # Sentence 4: fit to the persona
    fit = _persona_fit(target, brief)
    if fit:
        sentences.append(fit.capitalize() + ".")

    return " ".join(sentences)


_DIFFERENT_CITY_PHRASES = [
    "Same country, different city — {city} brings its own neighborhood character.",
    "{city} has a different feel than the top pick but the same architectural lineage.",
    "{city} is the alternative if you'd rather be in this part of {country}.",
    "Switch to {city} if the local character there appeals more.",
    "Same size and country, but {city} instead of the top pick's location.",
]

_DIFFERENT_COUNTRY_PHRASES = [
    "Different country ({country}) — only consider if you're geographically flexible.",
    "If you're open to {country} instead, this is a strong size-class match.",
    "Drops the {target_country} setting for {country}-style {style_kw}.",
    "A non-{target_country} alternative — useful as a sanity check on your specs.",
]

_SIZE_UP_PHRASES = [
    "A step up in bedroom count — more room but likely higher cost.",
    "Larger than the top pick — pick this if your headcount or guest needs grew.",
    "More space but you'll feel it in the rent.",
]

_SIZE_DOWN_PHRASES = [
    "Smaller than the top pick — pick this if budget or simplicity wins.",
    "Tighter on space but more efficient and easier to maintain.",
    "Compact alternative — solid if you'd rather optimise for cost over square metres.",
]


def _why_alternative(
    template: Template,
    target: Template,
    brief: Brief,
    rng: random.Random,
) -> str:
    """2-3 sentences for an alternative, with honest comparison."""
    sentences = []

    # Sentence 1: locate + size
    sentences.append(
        f"A {_format_size(template)} alternative in {template.short_locale}."
    )

    # Sentence 2: layout if interesting
    layout = _layout_summary(template)
    if layout:
        sentences.append(f"{layout}.")

    # Sentence 3: trade-off vs target (varied phrasings)
    if template.country != target.country:
        style_kw = "construction" if template.country in ("India", "Pakistan", "Bangladesh") else "architecture"
        phrase = rng.choice(_DIFFERENT_COUNTRY_PHRASES)
        sentences.append(phrase.format(
            country=template.country,
            target_country=target.country,
            style_kw=style_kw,
        ))
    elif template.size_band != target.size_band:
        if template.bedrooms > target.bedrooms:
            sentences.append(rng.choice(_SIZE_UP_PHRASES))
        else:
            sentences.append(rng.choice(_SIZE_DOWN_PHRASES))
    elif template.city != target.city:
        phrase = rng.choice(_DIFFERENT_CITY_PHRASES)
        sentences.append(phrase.format(city=template.city, country=template.country))

    return " ".join(sentences)


def _clean_style_title(style: str) -> str:
    """Strip parenthetical clutter from style strings used as headers."""
    main = style.split("(")[0].strip()
    # Don't strip if the main part is empty (rare)
    return main if main else style


def _comparison_note(picks: list[Template], target: Template, rng: random.Random) -> str | None:
    """A summary note — what trade-offs to think about."""
    if len(picks) < 2:
        return None
    cities = sorted(set(p.city for p in picks))
    countries = sorted(set(p.country for p in picks))

    if len(countries) > 1:
        return (
            f"All four cross {len(countries)} countries — the top pick "
            f"({target.short_locale}) is closest to your specifications, the others "
            f"are useful if you're geographically flexible."
        )
    if len(cities) > 1:
        return (
            f"All four are in {countries[0]} but spread across {len(cities)} cities — "
            f"compare on neighborhood character as well as layout."
        )
    return None


# ---------------------------------------------------------------------------
# Pick the top 4 from the candidate set
# ---------------------------------------------------------------------------

def _rank_candidates(
    target: Template,
    candidates: list[Template],
    brief: Brief,
) -> list[Template]:
    """Rank candidates so target is #1, and the next 3 are the closest matches."""
    def score(t: Template) -> float:
        if t.id == target.id:
            return 1e9  # target wins
        s = 0.0
        if t.country == target.country:
            s += 5
        if t.size_band == target.size_band:
            s += 4
        # persona overlap
        overlap = set(t.suitable_for) & set(target.suitable_for)
        s += len(overlap) * 1.5
        # tag overlap (style coherence)
        tag_overlap = set(t.tags) & set(target.tags)
        s += len(tag_overlap) * 0.3
        # area similarity
        if target.total_area_sqm > 0:
            ratio = min(t.total_area_sqm, target.total_area_sqm) / max(
                t.total_area_sqm, target.total_area_sqm
            )
            s += ratio * 2
        return s

    return sorted(candidates, key=score, reverse=True)


# ---------------------------------------------------------------------------
# Top-level: build the assistant response
# ---------------------------------------------------------------------------

@dataclass
class TrainingExample:
    """A single training pair: input prompt + ideal assistant response."""

    user_prompt: str
    assistant_response: str
    target_template_id: str
    candidate_ids: list[str]  # in the order they appeared in the prompt
    persona: str


def build_user_prompt(brief: Brief, candidates: list[Template]) -> str:
    """The text the user (or upstream retrieval) sends the model."""
    candidate_blocks = []
    for i, c in enumerate(candidates, 1):
        block = f"[{i}] {c.id}\n{c.metadata_summary()}"
        candidate_blocks.append(block)
    candidates_text = "\n\n".join(candidate_blocks)

    return (
        f"BRIEF:\n{brief.text}\n\n"
        f"CANDIDATE TEMPLATES (top retrieval matches):\n\n"
        f"{candidates_text}\n\n"
        f"Pick the top 4 candidates that best fit the brief. For each, give 2-4 sentences "
        f"of architectural reasoning grounded in the template's actual metadata. "
        f"List the top pick first, then 3 alternatives."
    )


def build_assistant_response(
    brief: Brief,
    target: Template,
    candidates: list[Template],
    rng: random.Random,
) -> str:
    ranked = _rank_candidates(target, candidates, brief)
    top4 = ranked[:4]

    opening = rng.choice(INTERPRET_OPENINGS)
    interp = _persona_summary(brief)
    parts = [f"{opening}, {interp}.", "", "**Top 4 matches:**", ""]

    for i, t in enumerate(top4, 1):
        if i == 1:
            rationale = _why_target(target, brief)
        else:
            rationale = _why_alternative(t, target, brief, rng)
        title = _clean_style_title(t.style)
        parts.append(f"**{i}. {title}** (`{t.id}`)")
        parts.append(rationale)
        parts.append("")

    note = _comparison_note(top4, target, rng)
    if note:
        parts.append(note)

    return "\n".join(parts).rstrip() + "\n"


def build_training_example(
    brief: Brief,
    target: Template,
    candidates: list[Template],
    rng: random.Random,
) -> TrainingExample:
    user_prompt = build_user_prompt(brief, candidates)
    assistant = build_assistant_response(brief, target, candidates, rng)
    return TrainingExample(
        user_prompt=user_prompt,
        assistant_response=assistant,
        target_template_id=target.id,
        candidate_ids=[c.id for c in candidates],
        persona=brief.persona,
    )


if __name__ == "__main__":
    from brief_generator import briefs_for_template
    from templates_index import load_all

    templates = load_all()
    pool = CandidatePool(templates)
    rng = random.Random(7)

    samples = rng.sample(templates, 2)
    for t in samples:
        briefs = briefs_for_template(t, rng, n=2)
        for b in briefs:
            cands = pool.select(t, rng=rng)
            ex = build_training_example(b, t, cands, rng)
            print("=" * 80)
            print("USER PROMPT:")
            print(ex.user_prompt[:1200])
            print("\n... [truncated] ...\n")
            print("ASSISTANT RESPONSE:")
            print(ex.assistant_response)
            print()
