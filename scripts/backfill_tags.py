#!/usr/bin/env python3
"""Backfill typology + era tags on existing templates by inferring from
filename and metadata. Run once to bring legacy templates up to controlled
vocabulary. Idempotent — re-running adds nothing if tags are already present.

Usage: python scripts/backfill_tags.py [--dry]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIRS = [ROOT / "data" / "templates" / "europe", ROOT / "data" / "templates" / "india"]

# Filename → (typology, era) inference table. Values are lists; first existing tag wins.
INFERENCE = {
    # German Altbau era pieces
    "berlin_altbau": ("altbau", "wilhelminian"),
    "berlin_kreuzberg": ("hinterhaus", "wilhelminian"),
    "berlin_plattenbau": ("plattenbau", "modernist_1970s"),
    "berlin_neubau": ("neubau", "contemporary_2000_2020"),
    "leipzig_grunderzeit": ("altbau", "gruenderzeit"),
    "wiesbaden_jugendstil": ("altbau", "jugendstil"),
    "dresden_altbau": ("altbau", "wilhelminian"),
    "heidelberg_altstadt": ("altbau", "pre_1900"),
    "nuremberg_altstadt": ("altbau", "pre_1900"),

    # Munich
    "munich_studio": ("neubau", "current_2020plus"),
    "munich_neubau": ("neubau", "current_2020plus"),
    "munich_penthouse": ("penthouse", "contemporary_2000_2020"),
    "munich_maisonette": ("maisonette", "contemporary_2000_2020"),
    "munich_familienwohnung": ("neubau", "contemporary_2000_2020"),
    "munich_villa": ("stadtvilla", "contemporary_2000_2020"),

    # Hamburg
    "hamburg_eppendorf": ("altbau", "wilhelminian"),
    "hamburg_hafencity": ("neubau", "current_2020plus"),

    # Frankfurt / Bonn / Mainz / Düsseldorf
    "frankfurt_hochhaus": ("hochhaus", "contemporary_2000_2020"),
    "frankfurt_familie": ("neubau", "contemporary_2000_2020"),
    "bonn_rheinblick": ("neubau", "contemporary_2000_2020"),
    "mainz_modern": ("neubau", "contemporary_2000_2020"),
    "dusseldorf_dachgeschoss": ("dachgeschoss", "wilhelminian"),

    # Köln / Bremen / Augsburg / Hannover / Stuttgart / Karlsruhe / Freiburg / Kiel
    "koln_neubau": ("neubau", "contemporary_2000_2020"),
    "bremen_bremerhaus": ("bremerhaus", "wilhelminian"),
    "augsburg_genossenschaft": ("genossenschaft", "interwar_1920_1939"),
    "hannover_doppelhaus": ("doppelhaus", "interwar_1920_1939"),
    "stuttgart_doppelhaus": ("doppelhaus", "postwar_1945_1969"),
    "karlsruhe_einfamilienhaus": ("einfamilienhaus", "contemporary_2000_2020"),
    "freiburg_passivhaus": ("passivhaus", "current_2020plus"),
    "kiel_kuste": ("neubau", "contemporary_2000_2020"),

    # UK
    "london_modern_flat": ("purpose_built_flat", "contemporary_2000_2020"),
    "london_duplex_townhouse": ("victorian_terrace", "victorian"),
    "manchester_edwardian": ("edwardian_semi", "edwardian"),
    "birmingham_1930s_semi": ("1930s_semi", "interwar_1920_1939"),
    "edinburgh_newtown": ("georgian_townhouse", "georgian"),
    "glasgow_tenement": ("tenement", "victorian"),
    "cotswolds_cottage": ("cottage", "vintage_traditional"),
    "liverpool_waterfront": ("warehouse_conversion", "contemporary_2000_2020"),
    "bristol_townhouse": ("georgian_townhouse", "georgian"),
    "uk_terraced_2bed": ("victorian_terrace", "victorian"),

    # India
    "in_studio_compact": ("apartment_indian", "contemporary_2000_2020"),
    "in_studio_pune": ("apartment_indian", "current_2020plus"),
    "in_1bhk_mumbai": ("apartment_indian", "current_2020plus"),
    "in_2bhk_bangalore": ("apartment_indian", "current_2020plus"),
    "in_3bhk_with_pooja": ("apartment_indian", "current_2020plus"),
    "in_4bhk_bangalore_villa_duplex": ("duplex_indian", "current_2020plus"),
    "in_5bhk_delhi_premium_villa": ("villa_indian", "current_2020plus"),
}


def infer(name: str) -> tuple[str | None, str | None]:
    for key, (typ, era) in INFERENCE.items():
        if key in name:
            return typ, era
    return None, None


def normalize_country(c: str) -> str:
    """UK → United Kingdom etc."""
    return {
        "UK": "United Kingdom",
        "Deutschland": "Germany",
    }.get(c, c)


def backfill(dry: bool = False) -> int:
    changed = 0
    for d in TEMPLATE_DIRS:
        for f in sorted(d.glob("*.json")):
            t = json.loads(f.read_text())
            meta = t.setdefault("metadata", {})
            tags = meta.setdefault("tags", [])
            modified = False

            # Normalize country
            old_country = meta.get("country", "")
            new_country = normalize_country(old_country)
            if old_country != new_country:
                meta["country"] = new_country
                modified = True

            typ, era = infer(f.stem)
            if typ and typ not in tags:
                tags.append(typ)
                modified = True
            if era and era not in tags:
                tags.append(era)
                modified = True

            if modified:
                changed += 1
                print(f"{'(dry) ' if dry else ''}updated {f.name}: typ={typ} era={era}")
                if not dry:
                    f.write_text(json.dumps(t, indent=2, ensure_ascii=False) + "\n")
    print(f"\n{changed} files {'would be' if dry else ''} modified")
    return 0


if __name__ == "__main__":
    sys.exit(backfill(dry="--dry" in sys.argv))
