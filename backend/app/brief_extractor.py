"""Phase 1 brief extractor — regex + simple parser.

Hackathon Day 1 swaps this for an LLM call (Qwen via Ollama). The
output schema is the contract:

    {
      "raw": str,
      "region": "india" | "europe" | None,
      "country": str | None,
      "city": str | None,
      "size_label": str | None,
      "bedrooms": int | None,
      "total_area_sqm": float | None,
      "vastu_compliant": bool | None,
    }
"""
from __future__ import annotations

import re

INDIAN_CITIES = {
    "bangalore", "bengaluru", "mumbai", "delhi", "pune", "chennai",
    "hyderabad", "kolkata", "ahmedabad", "kochi", "noida", "gurgaon",
    "gurugram",
}
EUROPEAN_CITIES = {
    "munich", "muenchen", "berlin", "hamburg", "frankfurt", "vienna",
    "zurich", "london", "paris", "amsterdam", "milan", "stuttgart",
    "köln", "koeln", "cologne", "düsseldorf", "duesseldorf",
    "dresden", "leipzig", "bremen", "hannover", "nürnberg", "nuremberg",
    "edinburgh", "glasgow", "manchester", "liverpool", "birmingham",
    "bristol", "leeds", "rome", "venice", "florence", "naples",
    "barcelona", "madrid", "sevilla", "valencia",
    "stockholm", "copenhagen", "oslo", "helsinki",
    "rotterdam", "the hague", "utrecht",
    "lyon", "marseille", "toulouse", "bordeaux", "nice",
    "brussels", "antwerp", "lisbon", "porto", "warsaw", "prague",
}

INDIAN_COUNTRY_HINTS = {"india", "indian", "vastu", "bhk", "pooja"}
EUROPEAN_COUNTRY_HINTS = {
    "germany", "german", "deutschland", "neubau", "altbau", "wohnkueche",
    "wohnküche", "zimmer", "diele", "uk", "england", "victorian",
    "terraced", "europe", "european",
}


def extract(text: str) -> dict:
    raw = text.strip()
    lower = raw.lower()

    out: dict = {
        "raw": raw,
        "region": None,
        "country": None,
        "city": None,
        "size_label": None,
        "bedrooms": None,
        "total_area_sqm": None,
        "vastu_compliant": None,
    }

    # City detection
    for c in INDIAN_CITIES:
        if c in lower:
            out["city"] = c.title()
            out["region"] = "india"
            out["country"] = "India"
            break
    if out["region"] is None:
        for c in EUROPEAN_CITIES:
            if c in lower:
                out["city"] = c.title()
                out["region"] = "europe"
                break

    # Region from broader hints
    if out["region"] is None:
        if any(h in lower for h in INDIAN_COUNTRY_HINTS):
            out["region"] = "india"
            out["country"] = "India"
        elif any(h in lower for h in EUROPEAN_COUNTRY_HINTS):
            out["region"] = "europe"

    # Vastu
    if "vastu" in lower:
        out["vastu_compliant"] = True

    # Size label and bedroom count
    bhk = re.search(r"(\d+)\s*bhk", lower)
    zimmer = re.search(r"(\d+)\s*[- ]?zimmer", lower)
    bed_match = re.search(r"(\d+)[\s-]?bed(room)?", lower)
    if bhk:
        n = int(bhk.group(1))
        out["size_label"] = f"{n}bhk"
        out["bedrooms"] = n
    elif zimmer:
        n = int(zimmer.group(1))
        out["size_label"] = f"{n}zimmer"
        # German Zimmer counts living room, so bedrooms = Zimmer - 1
        out["bedrooms"] = max(0, n - 1)
    elif bed_match:
        n = int(bed_match.group(1))
        out["size_label"] = f"{n}bed"
        out["bedrooms"] = n
    elif "studio" in lower:
        out["size_label"] = "studio"
        out["bedrooms"] = 0

    # Area (m² / sqm / square meters / qm)
    area = re.search(
        r"(?:around\s+|about\s+|roughly\s+|~\s*)?(\d+(?:\.\d+)?)\s*"
        r"(?:sqm|m²|m2|sq\s*m|square\s*meters?|qm)",
        lower,
    )
    if area:
        out["total_area_sqm"] = float(area.group(1))

    return out
