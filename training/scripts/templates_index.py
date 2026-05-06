"""Load and index the 500 floor plan templates for training data generation."""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


REGION_HUMAN = {
    "europe": "Europe",
    "global": "Global",
    "india": "India",
}

# Map country -> broader region hint for retrieval queries
SUBREGION_HINTS = {
    "Germany": ["central_europe", "german_speaking"],
    "Austria": ["central_europe", "german_speaking"],
    "Switzerland": ["central_europe", "german_speaking"],
    "France": ["western_europe", "french_speaking"],
    "Belgium": ["western_europe", "french_speaking"],
    "Italy": ["southern_europe", "mediterranean"],
    "Spain": ["southern_europe", "mediterranean"],
    "Portugal": ["southern_europe", "iberian"],
    "Greece": ["southern_europe", "mediterranean"],
    "United Kingdom": ["western_europe", "british_isles"],
    "Ireland": ["western_europe", "british_isles"],
    "Netherlands": ["western_europe", "low_countries"],
    "Denmark": ["northern_europe", "scandinavia"],
    "Sweden": ["northern_europe", "scandinavia"],
    "Norway": ["northern_europe", "scandinavia"],
    "Finland": ["northern_europe", "nordic"],
    "Poland": ["central_europe", "eastern_europe"],
    "Czech Republic": ["central_europe", "eastern_europe"],
    "Croatia": ["southern_europe", "balkans"],
    "Turkey": ["west_asia", "mediterranean"],
    "United States": ["north_america"],
    "Canada": ["north_america"],
    "Mexico": ["north_america", "latin_america"],
    "Brazil": ["south_america", "latin_america"],
    "Argentina": ["south_america", "latin_america"],
    "Chile": ["south_america", "latin_america"],
    "Colombia": ["south_america", "latin_america"],
    "Australia": ["oceania"],
    "New Zealand": ["oceania"],
    "Japan": ["east_asia"],
    "South Korea": ["east_asia"],
    "China": ["east_asia"],
    "Singapore": ["southeast_asia"],
    "Malaysia": ["southeast_asia"],
    "Thailand": ["southeast_asia"],
    "Vietnam": ["southeast_asia"],
    "Indonesia": ["southeast_asia"],
    "Philippines": ["southeast_asia"],
    "India": ["south_asia"],
    "Pakistan": ["south_asia"],
    "Bangladesh": ["south_asia"],
    "Sri Lanka": ["south_asia"],
    "United Arab Emirates": ["middle_east", "gulf"],
    "Saudi Arabia": ["middle_east", "gulf"],
    "Egypt": ["middle_east", "north_africa"],
    "Morocco": ["north_africa", "maghreb"],
    "Kenya": ["east_africa", "sub_saharan"],
    "Nigeria": ["west_africa", "sub_saharan"],
    "South Africa": ["southern_africa"],
}


@dataclass
class Template:
    """One floor plan template with metadata + computed fields for retrieval."""

    id: str
    region: str
    country: str
    city: str
    size_label: str
    size_band: str  # studio | 1bed | 2bed | 3bed | 4bed+
    total_area_sqm: float
    bedrooms: int
    bathrooms: int
    style: str
    description: str
    suitable_for: list[str]
    tags: list[str]
    room_names: list[str]
    room_count: int
    has_balcony: bool
    has_separate_kitchen: bool
    has_separate_dining: bool
    is_open_plan: bool
    ceiling_height_mm: int
    wall_thickness_mm: int
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def display_name(self) -> str:
        """Human-friendly name for the template, used in responses."""
        return self.style

    @property
    def short_locale(self) -> str:
        """e.g. 'Trichy, India' or 'Athens, Greece'."""
        return f"{self.city}, {self.country}"

    @property
    def subregion_hints(self) -> list[str]:
        return SUBREGION_HINTS.get(self.country, [])

    def metadata_summary(self) -> str:
        """Compact summary used as a candidate description in prompts."""
        parts = [
            f"id: {self.id}",
            f"location: {self.short_locale}",
            f"size: {self.size_label} ({self.total_area_sqm} sqm, "
            f"{self.bedrooms}-bed/{self.bathrooms}-bath)",
            f"style: {self.style}",
            f"rooms: {', '.join(self.room_names)}",
            f"suitable_for: {', '.join(self.suitable_for)}",
        ]
        return "\n".join(parts)


def _detect_features(rooms: list[dict[str, Any]]) -> dict[str, bool]:
    """Look at room types/names to detect features."""
    types = {r.get("type", "").lower() for r in rooms}
    names = {r.get("name", "").lower() for r in rooms}
    name_text = " | ".join(names)

    has_balcony = (
        "balcony" in types
        or "terrace" in types
        or "balkon" in name_text
        or "balcón" in name_text
        or "veranda" in name_text
        or "balcony" in name_text
        or "loggia" in name_text
        or "tuin" in name_text
    )
    has_separate_kitchen = "kitchen" in types
    has_separate_dining = "dining" in types
    # Open plan if living/kitchen are combined or there is a "living/dining" room
    is_open_plan = any(
        keyword in name_text
        for keyword in ("living/dining", "wohnkueche", "ldk", "open plan", "living-comedor")
    )
    return {
        "has_balcony": has_balcony,
        "has_separate_kitchen": has_separate_kitchen,
        "has_separate_dining": has_separate_dining,
        "is_open_plan": is_open_plan,
    }


def _normalize_size_band(metadata: dict[str, Any]) -> str:
    """Normalise into studio/1bed/2bed/3bed/4bed+."""
    explicit = metadata.get("size_band")
    if explicit:
        return explicit
    bedrooms = metadata.get("bedrooms", 0)
    if bedrooms == 0:
        return "studio"
    if bedrooms >= 4:
        return "4bed+"
    return f"{bedrooms}bed"


def _city_from_style(style: str, country: str) -> str:
    """Best-effort fallback: pull city from the style string when metadata lacks it."""
    if not style:
        return country  # last-ditch fallback
    first_word = style.split()[0]
    return first_word


def load_template(path: str) -> Template:
    """Read a single template JSON and parse into a Template object."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    md = data["metadata"]
    rooms = data.get("rooms", [])
    boundary = data.get("boundary", {})
    features = _detect_features(rooms)
    city = md.get("city_inspiration") or md.get("city") or _city_from_style(
        md.get("style", ""), md.get("country", "")
    )
    return Template(
        id=data["id"],
        region=md["region"],
        country=md["country"],
        city=city,
        size_label=md.get("size_label", ""),
        size_band=_normalize_size_band(md),
        total_area_sqm=float(md.get("total_area_sqm", 0)),
        bedrooms=int(md.get("bedrooms", 0)),
        bathrooms=int(md.get("bathrooms", 0)),
        style=md.get("style", ""),
        description=md.get("description", ""),
        suitable_for=list(md.get("suitable_for", [])),
        tags=list(md.get("tags", [])),
        room_names=[r.get("name", "") for r in rooms],
        room_count=len(rooms),
        ceiling_height_mm=int(boundary.get("ceiling_height_mm", 0)),
        wall_thickness_mm=int(boundary.get("wall_thickness_mm", 0)),
        raw=data,
        **features,
    )


def load_all(repo_root: str | None = None) -> list[Template]:
    """Load all 500 templates from data/templates/**/*.json."""
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pattern = os.path.join(repo_root, "data", "templates", "**", "*.json")
    paths = sorted(glob.glob(pattern, recursive=True))
    return [load_template(p) for p in paths]


def index_by_country(templates: list[Template]) -> dict[str, list[Template]]:
    out: dict[str, list[Template]] = defaultdict(list)
    for t in templates:
        out[t.country].append(t)
    return dict(out)


def index_by_size_band(templates: list[Template]) -> dict[str, list[Template]]:
    out: dict[str, list[Template]] = defaultdict(list)
    for t in templates:
        out[t.size_band].append(t)
    return dict(out)


def index_by_persona(templates: list[Template]) -> dict[str, list[Template]]:
    out: dict[str, list[Template]] = defaultdict(list)
    for t in templates:
        for p in t.suitable_for:
            out[p].append(t)
    return dict(out)


if __name__ == "__main__":
    ts = load_all()
    print(f"Loaded {len(ts)} templates")
    print(f"Countries: {len(index_by_country(ts))}")
    print(f"Size bands: {sorted(index_by_size_band(ts).keys())}")
    personas = index_by_persona(ts)
    print(f"Personas (top 10):")
    for p, items in sorted(personas.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {p}: {len(items)}")
