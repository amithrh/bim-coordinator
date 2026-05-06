"""Generate diverse, natural-language briefs that map to specific templates.

The output of this module is the (brief, target_template_id) pairs that drive
the rest of the training data pipeline.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterable

from templates_index import Template


# ---------------------------------------------------------------------------
# Persona library
# ---------------------------------------------------------------------------

PERSONA_NARRATIVES = {
    "single": [
        "I'm a single person looking for my first place",
        "Just me, looking for a compact place I can call home",
        "Single, looking for somewhere quiet and well-organised",
    ],
    "single_professional": [
        "I'm a working professional looking for a place close to the office",
        "Single professional, work-from-home a few days a week",
        "Mid-career professional, want a clean and modern space",
    ],
    "couple": [
        "My partner and I are looking for our first apartment together",
        "We're a young couple, both work full-time",
        "Couple in our 30s, no kids yet but maybe soon",
        "Looking for a place for the two of us, we like cooking and entertaining",
    ],
    "young_family": [
        "Young family with one toddler, expecting another soon",
        "Family of four — two parents, two young kids",
        "We have one child and want a place with room to grow",
    ],
    "growing_family": [
        "Family of three but we want space to grow",
        "Two kids and we're thinking about a third",
    ],
    "family": [
        "Family of four looking for a more permanent home",
        "Married couple with two school-age kids",
    ],
    "large_family": [
        "Multi-generational household — parents, two kids, and one grandparent",
        "Big family, six of us in total",
    ],
    "single_parent": [
        "Single parent with one school-age child",
        "Just me and my daughter — we want something cosy",
    ],
    "creative_artist": [
        "I'm an illustrator and need a corner that gets good natural light for my work",
        "Artist working from home, need at least a dedicated work corner",
        "Creative professional, want character over modernity",
    ],
    "academic": [
        "Academic researcher, lots of books and a quiet study area is essential",
        "PhD candidate, work mostly from home",
    ],
    "student": [
        "University student looking for an affordable studio",
        "Postgrad student, need somewhere walkable to campus",
    ],
    "student_share": [
        "Two students sharing — split a 2-bed",
        "Student looking for a flat-share-friendly layout",
    ],
    "retired": [
        "Recently retired, looking to downsize",
        "Retired couple, want single-level living",
    ],
    "retiree_couple": [
        "Retired couple, no stairs please",
        "Senior couple, looking for accessible single-floor living",
    ],
    "executive_pied_a_terre": [
        "I travel a lot for work and need a city pied-à-terre",
        "Looking for a small executive-style flat for occasional use",
    ],
    "weekend_retreat": [
        "Looking for a weekend place, somewhere we can escape to",
        "Second home for weekends and short stays",
    ],
    "remote_worker": [
        "I work fully remote and need a proper home office",
        "Remote worker, video calls daily, need a quiet dedicated space",
    ],
    "investor": [
        "Looking at this as a rental investment, want something marketable",
        "Investor — focused on yield and tenant-friendly layout",
    ],
}


# ---------------------------------------------------------------------------
# Country / region phrasings — used to vary how location is described
# ---------------------------------------------------------------------------

# Country phrases - country-level only, no specific cities (cities go through city specificity)
COUNTRY_PHRASES = {
    "Germany": ["Germany", "a German city", "Deutschland"],
    "France": ["France", "a French city", "somewhere in France"],
    "Italy": ["Italy", "an Italian city", "northern Italy", "central Italy"],
    "Spain": ["Spain", "a Spanish city"],
    "United Kingdom": ["the UK", "Britain", "a British city"],
    "Greece": ["Greece", "a Greek city"],
    "Portugal": ["Portugal", "a Portuguese city"],
    "India": ["India", "an Indian city", "South India", "North India"],
    "Japan": ["Japan", "a Japanese city"],
    "Australia": ["Australia", "an Australian city"],
    "United States": ["the US", "an American city"],
    "Canada": ["Canada", "a Canadian city"],
    "Netherlands": ["the Netherlands", "a Dutch city"],
    "Denmark": ["Denmark", "a Danish city"],
    "Sweden": ["Sweden", "a Swedish city"],
    "Norway": ["Norway", "a Norwegian city"],
    "Finland": ["Finland"],
    "Belgium": ["Belgium"],
    "Switzerland": ["Switzerland", "a Swiss city"],
    "Austria": ["Austria"],
    "Brazil": ["Brazil"],
    "Argentina": ["Argentina"],
    "Chile": ["Chile"],
    "Colombia": ["Colombia"],
    "Mexico": ["Mexico", "a Mexican city"],
    "Singapore": ["Singapore"],
    "Malaysia": ["Malaysia"],
    "Thailand": ["Thailand"],
    "Vietnam": ["Vietnam"],
    "Indonesia": ["Indonesia"],
    "Philippines": ["the Philippines"],
    "Pakistan": ["Pakistan"],
    "Bangladesh": ["Bangladesh"],
    "Sri Lanka": ["Sri Lanka"],
    "Turkey": ["Turkey"],
    "United Arab Emirates": ["the UAE"],
    "Saudi Arabia": ["Saudi Arabia"],
    "Egypt": ["Egypt"],
    "Morocco": ["Morocco"],
    "Kenya": ["Kenya"],
    "Nigeria": ["Nigeria"],
    "South Africa": ["South Africa"],
    "New Zealand": ["New Zealand", "an NZ city"],
    "South Korea": ["South Korea"],
    "China": ["China"],
}

# Country -> appropriate style/era keywords. Prevents using "Altbau" for Italian etc.
COUNTRY_STYLE_VOCAB = {
    "Germany": {"altbau", "altbau_high_ceilings", "modern", "mid_century", "bauhaus", "historic", "compact", "balcony"},
    "Austria": {"altbau", "altbau_high_ceilings", "historic", "modern"},
    "Switzerland": {"modern", "altbau_high_ceilings", "historic"},
    "France": {"haussmann", "historic", "modern", "compact", "balcony"},
    "Italy": {"historic", "modern", "compact", "balcony", "loft"},
    "Spain": {"historic", "modern", "balcony", "compact"},
    "Portugal": {"historic", "modern", "compact"},
    "Greece": {"historic", "compact", "modern", "balcony"},
    "United Kingdom": {"victorian", "historic", "modern", "compact"},
    "Ireland": {"victorian", "historic"},
    "India": {"modern", "compact", "balcony"},
    "Japan": {"compact", "modern"},
}

# Always-safe styles that work everywhere
UNIVERSAL_STYLES = {"modern", "compact", "open_plan", "single_floor", "balcony", "terrace",
                    "garden", "boutique", "coastal", "historic"}


# ---------------------------------------------------------------------------
# Size phrasings
# ---------------------------------------------------------------------------

SIZE_PHRASES = {
    "studio": ["studio", "small studio", "compact studio"],
    "1bed": ["1-bedroom", "one-bedroom", "1-bed", "compact 1-bed", "small one-bed"],
    "2bed": ["2-bedroom", "two-bedroom", "2-bed", "2BHK"],
    "3bed": ["3-bedroom", "three-bedroom", "3-bed", "3BHK"],
    "4bed": ["4-bedroom", "four-bedroom", "spacious 4-bed"],
    "4bed+": ["4+ bedroom", "large family home", "4 or 5 bed"],
    "5plus_bed": ["5+ bedroom", "very large family home"],
}


# ---------------------------------------------------------------------------
# Style references derived from template tags / descriptions
# ---------------------------------------------------------------------------

STYLE_KEYWORDS = {
    "altbau": ["Altbau character", "pre-war character", "high-ceiling Altbau", "old building feel"],
    "victorian": ["Victorian character", "period property feel", "Victorian terrace style"],
    "haussmann": ["Haussmann-style", "classical Parisian apartment"],
    "modern": ["modern", "contemporary", "newly built"],
    "mid_century": ["mid-century", "post-war", "1960s/70s era"],
    "bauhaus": ["Bauhaus-inspired", "modernist"],
    "historic": ["historic", "with character", "older building"],
    "compact": ["compact", "well-organised small space", "efficient layout"],
    "open_plan": ["open-plan", "open layout", "flowing layout"],
    "single_floor": ["single-floor", "no stairs", "all on one level"],
    "altbau_high_ceilings": ["high ceilings", "tall ceilings", "Altbau ceilings"],
    "balcony": ["with a balcony", "balcony is important", "outdoor space"],
    "terrace": ["with a terrace", "outdoor terrace"],
    "garden": ["with garden access", "garden", "ground floor garden"],
    "boutique": ["boutique-style", "small high-end"],
    "loft": ["loft-style", "industrial loft", "open warehouse feel"],
    "coastal": ["coastal", "near the sea", "sea views"],
}


# ---------------------------------------------------------------------------
# Brief assembly
# ---------------------------------------------------------------------------

OPENING_TEMPLATES = [
    "Hi, ",
    "Hi there. ",
    "",  # No opening, just dive in
    "I'm looking for help finding a flat. ",
    "Looking for a place. ",
    "Need some help shortlisting apartments. ",
    "We need a new place. ",
]

CLOSING_TEMPLATES = [
    "",  # No closing
    " Any thoughts?",
    " What do you recommend?",
    " Show me what fits.",
    " Anything in your library that fits?",
    " Looking for 4 options to compare.",
]


@dataclass
class Brief:
    """A generated user brief paired with the ground-truth target template id."""

    text: str
    target_template_id: str
    persona: str
    style_dimensions: list[str]  # which dimensions the brief emphasises


def _persona_for_template(template: Template, rng: random.Random) -> str:
    """Pick a persona consistent with the template, fall back gracefully."""
    candidates = [p for p in template.suitable_for if p in PERSONA_NARRATIVES]
    if candidates:
        return rng.choice(candidates)
    # Fall back: infer from bedrooms
    if template.bedrooms == 0:
        return rng.choice(["single", "student", "single_professional"])
    if template.bedrooms == 1:
        return rng.choice(["couple", "single_professional"])
    if template.bedrooms == 2:
        return rng.choice(["young_family", "couple", "remote_worker"])
    if template.bedrooms >= 3:
        return rng.choice(["family", "growing_family"])
    return "couple"


def _size_phrase(template: Template, rng: random.Random) -> str:
    return rng.choice(SIZE_PHRASES.get(template.size_band, [template.size_label]))


def _location_phrase(template: Template, rng: random.Random, specificity: str) -> str:
    """Returns a phrase WITHOUT leading 'in' so callers can prepend safely.

    specificity: 'city' | 'country' | 'region'.
    """
    if specificity == "city":
        return template.city
    if specificity == "country":
        choices = COUNTRY_PHRASES.get(template.country, [template.country])
        return rng.choice(choices)
    # 'region' - vague continental hint
    region_map = {
        "europe": ["somewhere in Europe", "a European city"],
        "india": ["somewhere in India", "an Indian city"],
        "global": ["abroad", "an international city"],
    }
    return rng.choice(region_map.get(template.region, ["somewhere"]))


def _format_location(loc: str) -> str:
    """Add 'in' prefix unless the phrase already starts with a preposition."""
    lower = loc.lower()
    if lower.startswith(("in ", "somewhere ", "abroad", "internationally")):
        return loc
    return f"in {loc}"


def _style_phrases(template: Template, rng: random.Random, max_styles: int = 2) -> list[str]:
    """Pull 0-2 style references from template tags, filtered by country appropriateness."""
    # Only use exact tag matches, then filter by country-appropriate vocabulary
    country_vocab = COUNTRY_STYLE_VOCAB.get(template.country, UNIVERSAL_STYLES)
    allowed = country_vocab | UNIVERSAL_STYLES
    available = [k for k in STYLE_KEYWORDS if k in template.tags and k in allowed]
    n = rng.randint(0, min(max_styles, len(available)))
    if n == 0:
        return []
    chosen = rng.sample(available, n)
    return [rng.choice(STYLE_KEYWORDS[k]) for k in chosen]


def _feature_phrases(template: Template, rng: random.Random) -> list[str]:
    out = []
    if template.has_balcony and rng.random() < 0.5:
        out.append(rng.choice(["with a balcony", "outdoor space matters", "needs a balcony"]))
    if template.has_separate_kitchen and rng.random() < 0.3:
        out.append(rng.choice(["separate kitchen", "I prefer a closed kitchen"]))
    if template.is_open_plan and rng.random() < 0.3:
        out.append(rng.choice(["open-plan layout", "I like open layouts"]))
    if template.ceiling_height_mm >= 3000 and rng.random() < 0.2:
        out.append(rng.choice(["high ceilings", "tall ceilings"]))
    return out


def _budget_phrase(template: Template, rng: random.Random) -> str | None:
    """Sometimes mention a fuzzy budget framing."""
    if rng.random() > 0.35:
        return None
    if template.total_area_sqm < 50:
        return rng.choice(["budget-conscious", "affordable", "tight budget"])
    if template.total_area_sqm < 80:
        return rng.choice(["mid-range budget", "reasonable budget"])
    return rng.choice(["upper-mid budget", "willing to spend"])


def _make_brief(
    template: Template,
    rng: random.Random,
    style: str,
) -> Brief:
    """Build one brief in a particular style."""
    persona = _persona_for_template(template, rng)
    persona_text = rng.choice(PERSONA_NARRATIVES[persona])
    size = _size_phrase(template, rng)

    location_specificity = rng.choices(
        ["city", "country", "region"], weights=[0.2, 0.65, 0.15]
    )[0]
    location = _location_phrase(template, rng, location_specificity)

    style_phrases = _style_phrases(template, rng)
    feature_phrases = _feature_phrases(template, rng)
    budget = _budget_phrase(template, rng)

    dims = []
    if style_phrases:
        dims.append("style")
    if feature_phrases:
        dims.append("features")
    if budget:
        dims.append("budget")
    dims.append(f"location:{location_specificity}")

    formatted_location = _format_location(location)

    if style == "direct":
        # Short, to-the-point
        bits = [f"{size} {formatted_location}"]
        if style_phrases:
            bits.append(rng.choice(style_phrases))
        if feature_phrases:
            bits.extend(feature_phrases[:1])
        if budget:
            bits.append(budget)
        text = ", ".join(bits)
        text = text[0].upper() + text[1:] + "."

    elif style == "descriptive":
        # Multi-sentence, with personality
        opening = rng.choice(OPENING_TEMPLATES)
        intro = persona_text
        need = f"We're looking for a {size} {formatted_location}"
        if budget:
            need += f", {budget}"
        need += "."
        extras = []
        if style_phrases:
            extras.append(
                f"Ideally with {style_phrases[0]}"
                + (f" and {style_phrases[1]}" if len(style_phrases) > 1 else "")
                + "."
            )
        if feature_phrases:
            extras.append(
                f"{feature_phrases[0][0].upper() + feature_phrases[0][1:]}."
            )
        closing = rng.choice(CLOSING_TEMPLATES)
        text = (opening + intro + ". " + need + " " + " ".join(extras) + closing).strip()

    elif style == "constraint":
        # Constraint-heavy
        bits = [f"{size}", formatted_location]
        if budget:
            bits.append(budget)
        bits.extend(style_phrases)
        bits.extend(feature_phrases)
        # Add a clear constraint
        if template.total_area_sqm > 0:
            target = template.total_area_sqm
            low = max(20, int(target - 10))
            high = int(target + 15)
            bits.append(f"around {low}-{high} sqm")
        text = "Need: " + "; ".join(bits) + "."

    elif style == "lifestyle":
        # Lifestyle-driven - lifestyle phrases must match the *chosen* persona
        opening = rng.choice(OPENING_TEMPLATES)
        intro = persona_text
        size_loc = f"Want a {size} {_format_location(location)}"
        persona_lifestyle = {
            "remote_worker": ["I work from home most days, so a quiet workspace matters",
                              "Daily video calls — need somewhere I can shut a door"],
            "creative_artist": ["natural light for my work matters",
                                "I need a corner that gets good morning light"],
            "young_family": ["we have a small kid so good light and a separate bedroom matter",
                             "kids need their own space to play and sleep"],
            "growing_family": ["thinking about expanding the family soon"],
            "family": ["four of us, plus the dog — we need real bedrooms not boxes"],
            "single_professional": ["I cook a lot so kitchen quality matters",
                                    "want somewhere I can wind down after long days"],
            "couple": ["we cook together most nights and entertain occasionally",
                       "we like having friends over"],
            "academic": ["lots of books — wall space for shelves matters"],
            "retired": ["no stairs is non-negotiable", "we want easy living"],
            "retiree_couple": ["single-floor living, near amenities"],
            "executive_pied_a_terre": ["I'm only in the city ~10 days a month"],
            "weekend_retreat": ["we use it Friday-Sunday, easy to lock and leave"],
            "investor": ["focused on rental yield and tenant appeal"],
        }
        lifestyle_options = persona_lifestyle.get(persona, [])
        lifestyle_text = (
            f" {rng.choice(lifestyle_options)}." if lifestyle_options else ""
        )
        extras = []
        if style_phrases:
            extras.append(f"Looking for {style_phrases[0]}.")
        if feature_phrases:
            extras.append(f"{feature_phrases[0][0].upper() + feature_phrases[0][1:]}.")
        closing = rng.choice(CLOSING_TEMPLATES)
        text = (opening + intro + ". " + size_loc + "." + lifestyle_text + " "
                + " ".join(extras) + closing).strip()

    else:  # 'compact_search'
        # Search-bar style query - bare keywords, no preposition
        words = [size, location]
        words.extend(style_phrases[:1])
        words.extend([s.replace("with a ", "").replace("with ", "") for s in feature_phrases[:1]])
        text = " ".join(w for w in words if w).strip()

    # Clean up double spaces, multiple periods, etc.
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\.\s*\.", ".", text)
    # Capitalize first letter after sentence-ending punctuation
    def _cap_after_period(match: re.Match[str]) -> str:
        return match.group(1) + match.group(2).upper()
    text = re.sub(r"([.!?]\s+)([a-z])", _cap_after_period, text)
    # Capitalize the very first letter of the text
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    # Remove space before punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    return Brief(
        text=text,
        target_template_id=template.id,
        persona=persona,
        style_dimensions=dims,
    )


def briefs_for_template(
    template: Template,
    rng: random.Random,
    n: int = 6,
) -> list[Brief]:
    """Generate `n` diverse briefs that target this template."""
    styles = ["direct", "descriptive", "constraint", "lifestyle", "compact_search"]
    out = []
    for _ in range(n):
        style = rng.choice(styles)
        out.append(_make_brief(template, rng, style))
    return out


def all_briefs(
    templates: Iterable[Template],
    n_per_template: int = 6,
    seed: int = 42,
) -> list[Brief]:
    rng = random.Random(seed)
    out = []
    for t in templates:
        out.extend(briefs_for_template(t, rng, n_per_template))
    return out


if __name__ == "__main__":
    from templates_index import load_all

    templates = load_all()
    rng = random.Random(7)
    sample_templates = rng.sample(templates, 5)

    for t in sample_templates:
        print(f"\n{'='*70}\nTEMPLATE: {t.id} ({t.short_locale}, {t.size_label})")
        print(f"  style: {t.style}")
        print(f"  suitable_for: {t.suitable_for}")
        briefs = briefs_for_template(t, rng, n=4)
        for i, b in enumerate(briefs, 1):
            print(f"\n  [{i}] persona={b.persona} dims={b.style_dimensions}")
            print(f"      {b.text!r}")
