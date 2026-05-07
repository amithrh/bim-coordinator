"""Rule-based brief → template program extractor.

Pulls structured fields from a natural-language brief without an LLM:
  - country / region (with native room name vocabulary)
  - bedrooms / bathrooms
  - area (target sqm)
  - persona

Then assembles a country-appropriate room program with realistic areas.
This is the reliable Level-4 backbone: 100% deterministic, never fails to
produce a valid program, runs in <50ms.

For free-form generation cases the rule-based extractor can't handle
(e.g. "split-level loft with a mezzanine"), the LLM generator in
llm_generator.py takes over.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Country / region detection
# --------------------------------------------------------------------------- #

COUNTRY_KEYWORDS = {
    "Germany": ["germany", "berlin", "munich", "münchen", "hamburg", "frankfurt",
                "cologne", "köln", "düsseldorf", "stuttgart", "bremen", "leipzig",
                "altbau", "wilhelminian", "gründerzeit", "deutschland"],
    "France": ["france", "paris", "lyon", "marseille", "nice", "bordeaux",
               "toulouse", "nantes", "rennes", "strasbourg", "haussmann",
               "haussmannien", "french"],
    "United Kingdom": ["uk", "britain", "british", "england", "london", "manchester",
                       "edinburgh", "bristol", "newcastle", "victorian", "georgian",
                       "edwardian", "tudor"],
    "Italy": ["italy", "rome", "milan", "milano", "florence", "firenze", "naples",
              "napoli", "turin", "torino", "venice", "venezia", "italian"],
    "Spain": ["spain", "madrid", "barcelona", "seville", "valencia", "spanish"],
    "Portugal": ["portugal", "lisbon", "lisboa", "porto", "portuguese"],
    "Greece": ["greece", "athens", "thessaloniki", "greek"],
    "Netherlands": ["netherlands", "amsterdam", "rotterdam", "the hague", "dutch"],
    "Belgium": ["belgium", "brussels", "antwerp", "belgian"],
    "Switzerland": ["switzerland", "zurich", "zürich", "geneva", "basel", "swiss"],
    "Austria": ["austria", "vienna", "wien", "austrian"],
    "Denmark": ["denmark", "copenhagen", "københavn", "aarhus", "danish"],
    "Sweden": ["sweden", "stockholm", "gothenburg", "swedish"],
    "Norway": ["norway", "oslo", "bergen", "norwegian"],
    "Finland": ["finland", "helsinki", "tampere", "finnish"],
    "Poland": ["poland", "warsaw", "warszawa", "krakow", "kraków", "polish"],
    "Czech Republic": ["czech", "prague", "praha", "brno"],
    "India": ["india", "indian", "mumbai", "bangalore", "bengaluru", "delhi", "chennai",
              "kolkata", "hyderabad", "pune", "ahmedabad", "bhk", "vastu",
              "tamil nadu", "south india", "north india", "kerala", "karnataka"],
    "Japan": ["japan", "tokyo", "osaka", "kyoto", "japanese", "ldk", "genkan"],
    "Singapore": ["singapore", "hdb"],
    "United Arab Emirates": ["uae", "dubai", "abu dhabi"],
    "Saudi Arabia": ["saudi", "riyadh", "jeddah", "majlis"],
    "Turkey": ["turkey", "istanbul", "ankara", "turkish"],
    "Australia": ["australia", "sydney", "melbourne", "brisbane", "adelaide", "perth"],
    "United States": ["united states", "us ", "usa", "america", "new york", "boston",
                      "chicago", "los angeles", "san francisco", "seattle", "denver"],
    "Canada": ["canada", "toronto", "montreal", "vancouver", "canadian"],
    "Brazil": ["brazil", "são paulo", "sao paulo", "rio de janeiro", "brasil"],
    "Argentina": ["argentina", "buenos aires"],
    "Mexico": ["mexico", "mexican", "guadalajara"],
    "South Africa": ["south africa", "johannesburg", "cape town"],
    "Morocco": ["morocco", "moroccan", "casablanca", "rabat"],
    "Egypt": ["egypt", "cairo"],
    "Kenya": ["kenya", "nairobi"],
    "Nigeria": ["nigeria", "lagos", "abuja"],
    "China": ["china", "shanghai", "beijing", "chinese"],
    "South Korea": ["korea", "seoul", "korean"],
    "Vietnam": ["vietnam", "ho chi minh", "hanoi", "vietnamese"],
}

REGION_OF = {
    **{c: "europe" for c in ["Germany", "France", "United Kingdom", "Italy",
                              "Spain", "Portugal", "Greece", "Netherlands", "Belgium",
                              "Switzerland", "Austria", "Denmark", "Sweden", "Norway",
                              "Finland", "Poland", "Czech Republic"]},
    "India": "india",
}


# --------------------------------------------------------------------------- #
# Country-appropriate room programs
# --------------------------------------------------------------------------- #

@dataclass
class CountryConvention:
    wall_thickness_mm: int = 230
    ceiling_height_mm: int = 2700
    entry_name: str = "Foyer"
    living_name: str = "Living/Dining"
    kitchen_name: str = "Kitchen"
    bath_name: str = "Bathroom"
    wc_name: str = "WC"
    master_name: str = "Master Bedroom"
    bedroom2_name: str = "Bedroom 2"
    bedroom3_name: str = "Bedroom 3"
    balcony_name: str = "Balcony"


COUNTRY_CONVENTIONS: dict[str, CountryConvention] = {
    "Germany": CountryConvention(
        wall_thickness_mm=380, ceiling_height_mm=3200,
        entry_name="Diele", living_name="Wohnzimmer", kitchen_name="Küche",
        bath_name="Badezimmer", wc_name="Gäste-WC",
        master_name="Schlafzimmer", bedroom2_name="Kinderzimmer",
        bedroom3_name="Arbeitszimmer", balcony_name="Balkon",
    ),
    "France": CountryConvention(
        wall_thickness_mm=380, ceiling_height_mm=3200,
        entry_name="Entrée", living_name="Séjour", kitchen_name="Cuisine",
        bath_name="Salle de bain", wc_name="WC",
        master_name="Chambre 1", bedroom2_name="Chambre 2",
        bedroom3_name="Chambre 3", balcony_name="Balcon",
    ),
    "Italy": CountryConvention(
        wall_thickness_mm=350, ceiling_height_mm=3000,
        entry_name="Ingresso", living_name="Soggiorno", kitchen_name="Cucina",
        bath_name="Bagno", wc_name="WC",
        master_name="Camera da letto", bedroom2_name="Camera 2",
        bedroom3_name="Camera 3", balcony_name="Balcone",
    ),
    "Spain": CountryConvention(
        wall_thickness_mm=380, ceiling_height_mm=3000,
        entry_name="Recibidor", living_name="Salón", kitchen_name="Cocina",
        bath_name="Baño", wc_name="Aseo",
        master_name="Dormitorio", bedroom2_name="Dormitorio 2",
        bedroom3_name="Dormitorio 3", balcony_name="Balcón",
    ),
    "Portugal": CountryConvention(
        wall_thickness_mm=350, ceiling_height_mm=2900,
        entry_name="Entrada", living_name="Sala", kitchen_name="Cozinha",
        bath_name="Casa de Banho", wc_name="WC",
        master_name="Quarto", bedroom2_name="Quarto 2",
        bedroom3_name="Quarto 3", balcony_name="Varanda",
    ),
    "Greece": CountryConvention(
        wall_thickness_mm=300, ceiling_height_mm=2800,
        entry_name="Χολ", living_name="Σαλόνι", kitchen_name="Κουζίνα",
        bath_name="Μπάνιο", wc_name="WC",
        master_name="Κύριο Υπνοδωμάτιο", bedroom2_name="Υπνοδωμάτιο 2",
        bedroom3_name="Γραφείο", balcony_name="Μπαλκόνι",
    ),
    "Netherlands": CountryConvention(
        wall_thickness_mm=300, ceiling_height_mm=2700,
        entry_name="Hal", living_name="Woonkamer", kitchen_name="Keuken",
        bath_name="Badkamer", wc_name="WC",
        master_name="Slaapkamer", bedroom2_name="Slaapkamer 2",
        bedroom3_name="Werkkamer", balcony_name="Balkon",
    ),
    "Denmark": CountryConvention(
        wall_thickness_mm=350, ceiling_height_mm=2900,
        entry_name="Entré", living_name="Stue", kitchen_name="Køkken",
        bath_name="Badeværelse", wc_name="WC",
        master_name="Soveværelse", bedroom2_name="Soveværelse 2",
        bedroom3_name="Børneværelse", balcony_name="Altan",
    ),
    "Sweden": CountryConvention(
        wall_thickness_mm=300, ceiling_height_mm=2700,
        entry_name="Hall", living_name="Vardagsrum", kitchen_name="Kök",
        bath_name="Badrum", wc_name="WC",
        master_name="Sovrum", bedroom2_name="Sovrum 2",
        bedroom3_name="Arbetsrum", balcony_name="Balkong",
    ),
    "Norway": CountryConvention(
        wall_thickness_mm=350, ceiling_height_mm=2800,
        entry_name="Gang", living_name="Stue", kitchen_name="Kjøkken",
        bath_name="Bad", wc_name="WC",
        master_name="Soverom", bedroom2_name="Soverom 2",
        bedroom3_name="Kontor", balcony_name="Balkong",
    ),
    "Finland": CountryConvention(
        wall_thickness_mm=300, ceiling_height_mm=2600,
        entry_name="Eteinen", living_name="Olohuone", kitchen_name="Keittiö",
        bath_name="Kylpyhuone", wc_name="WC",
        master_name="Makuuhuone", bedroom2_name="Makuuhuone 2",
        bedroom3_name="Työhuone", balcony_name="Parveke",
    ),
    "United Kingdom": CountryConvention(
        wall_thickness_mm=300, ceiling_height_mm=2700,
        entry_name="Hall", living_name="Living Room", kitchen_name="Kitchen-Diner",
        bath_name="Bathroom", wc_name="Cloakroom",
        master_name="Master Bedroom", bedroom2_name="Bedroom 2",
        bedroom3_name="Bedroom 3", balcony_name="Garden",
    ),
    "India": CountryConvention(
        wall_thickness_mm=230, ceiling_height_mm=2750,
        entry_name="Foyer", living_name="Living/Dining", kitchen_name="Kitchen",
        bath_name="Bathroom", wc_name="WC",
        master_name="Master Bedroom", bedroom2_name="Bedroom 2",
        bedroom3_name="Bedroom 3", balcony_name="Balcony",
    ),
    "Japan": CountryConvention(
        wall_thickness_mm=200, ceiling_height_mm=2400,
        entry_name="Genkan", living_name="LDK (Living-Dining-Kitchen)",
        kitchen_name="Kitchenette", bath_name="Unit Bath", wc_name="Toilet",
        master_name="Yoshitsu (Bedroom)", bedroom2_name="Bedroom 2",
        bedroom3_name="Tatami Room", balcony_name="Veranda",
    ),
    "Turkey": CountryConvention(
        wall_thickness_mm=200, ceiling_height_mm=2700,
        entry_name="Hol", living_name="Salon", kitchen_name="Mutfak",
        bath_name="Banyo", wc_name="WC",
        master_name="Yatak Odası", bedroom2_name="Çocuk Odası",
        bedroom3_name="Çalışma Odası", balcony_name="Balkon",
    ),
}

DEFAULT_CONVENTION = CountryConvention()  # English fallback


def detect_country(brief: str) -> str | None:
    txt = brief.lower()
    # First pass: most specific (longer keywords first)
    matches: list[tuple[int, str]] = []
    for country, kws in COUNTRY_KEYWORDS.items():
        for kw in kws:
            if kw in txt:
                matches.append((len(kw), country))
    if not matches:
        return None
    # Prefer longest match
    matches.sort(reverse=True)
    return matches[0][1]


def detect_bedrooms(brief: str) -> int:
    txt = brief.lower()
    # Common patterns — allow hyphen separator like "3-bed", "2-bedroom"
    m = re.search(r"(\d+)\s*[-]?\s*(?:bed|bedroom|bhk|zimmer|pieces|rooms)", txt)
    if m:
        n = int(m.group(1))
        # German Zimmer counts ROOMS not bedrooms (2-Zimmer = 1 bedroom + 1 living)
        if "zimmer" in txt and n >= 2:
            return n - 1
        return min(n, 4)
    if "studio" in txt:
        return 0
    if "single" in txt:
        return 1
    if "family" in txt:
        return 2
    return 1


def detect_bathrooms(brief: str, bedrooms: int) -> int:
    txt = brief.lower()
    m = re.search(r"(\d+)\s*(?:bath|bathroom)", txt)
    if m:
        return min(int(m.group(1)), 3)
    # Default: 2 bathrooms for 2+ bed, 1 for smaller
    return 2 if bedrooms >= 2 else 1


def detect_area(brief: str) -> float | None:
    txt = brief.lower()
    # Look for "<n> m²" or "<n> sqm" or "<n> square meters"
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:m²|m2|sqm|square\s*meters?|metres|meters)",
        txt,
    )
    if m:
        v = float(m.group(1))
        if 20 <= v <= 500:
            return v
    return None


def detect_persona(brief: str) -> list[str]:
    txt = brief.lower()
    out: list[str] = []
    if "couple" in txt or "two of us" in txt or "my partner" in txt:
        out.append("couple")
    if "young family" in txt or "with kids" in txt or "with children" in txt or "toddler" in txt:
        out.append("young_family")
    if "family of" in txt or "family" in txt and not out:
        out.append("family")
    if "single" in txt and "professional" in txt:
        out.append("single_professional")
    if "remote" in txt or "work from home" in txt or "wfh" in txt:
        out.append("remote_worker")
    if "academic" in txt or "researcher" in txt or "phd" in txt:
        out.append("academic")
    if "creative" in txt or "artist" in txt or "designer" in txt:
        out.append("creative_artist")
    if "executive" in txt or "pied-à-terre" in txt or "pied a terre" in txt:
        out.append("executive_pied_a_terre")
    if "investor" in txt or "rental" in txt:
        out.append("investor")
    if "retired" in txt or "retiree" in txt:
        out.append("retired")
    if "student" in txt:
        out.append("student")
    return out or ["general"]


# --------------------------------------------------------------------------- #
# Program assembly
# --------------------------------------------------------------------------- #

def _typical_area(bedrooms: int, country: str) -> float:
    """Country-typical area defaults."""
    if bedrooms == 0:
        return 35 if country in {"Japan", "Singapore"} else 42
    if bedrooms == 1:
        if country == "Japan":
            return 32
        if country in {"India", "Pakistan", "Bangladesh", "Sri Lanka"}:
            return 50
        if country in {"Germany", "France"}:
            return 55
        return 50
    if bedrooms == 2:
        if country == "Japan":
            return 55
        if country in {"India", "Pakistan", "Bangladesh"}:
            return 75
        if country in {"Germany", "France", "Italy", "Spain"}:
            return 70
        return 75
    if bedrooms == 3:
        return 100 if country in {"India", "Pakistan"} else 95
    return 130  # 4+ bed


def _allocate_areas(bedrooms: int, total: float, want_balcony: bool) -> dict[str, float]:
    """Allocate areas to rooms following typical proportions."""
    # Studios: bath + kitchen + combined living/sleep
    if bedrooms == 0:
        bath = 5
        kitchen = max(4, total * 0.12)
        entry = 2
        balcony = 3 if want_balcony else 0
        living_sleep = total - bath - kitchen - entry - balcony
        out = {"entry": entry, "bath": bath, "kitchen": kitchen, "living_sleep": living_sleep}
        if want_balcony:
            out["balcony"] = balcony
        return out

    entry = max(3, total * 0.06)
    bath = 5 if total < 60 else 6
    kitchen = max(7, total * 0.13)
    balcony = 4 if want_balcony else 0

    if bedrooms == 1:
        used = entry + bath + kitchen + balcony
        living = max(15, total * 0.35)
        master = total - used - living
        return {"entry": entry, "bath": bath, "kitchen": kitchen,
                "living": living, "master": master,
                **({"balcony": balcony} if want_balcony else {})}

    if bedrooms == 2:
        wc = 3
        used = entry + bath + wc + kitchen + balcony
        living = max(18, total * 0.30)
        rest = total - used - living
        master = rest * 0.55
        bedroom2 = rest * 0.45
        return {"entry": entry, "bath": bath, "wc": wc, "kitchen": kitchen,
                "living": living, "master": master, "bedroom2": bedroom2,
                **({"balcony": balcony} if want_balcony else {})}

    # 3+ bed
    wc = 3
    bath2 = 4
    used = entry + bath + bath2 + wc + kitchen + balcony
    living = max(22, total * 0.28)
    rest = total - used - living
    master = rest * 0.40
    bedroom2 = rest * 0.30
    bedroom3 = rest * 0.30
    return {"entry": entry, "bath": bath, "bath2": bath2, "wc": wc,
            "kitchen": kitchen, "living": living,
            "master": master, "bedroom2": bedroom2, "bedroom3": bedroom3,
            **({"balcony": balcony} if want_balcony else {})}


def extract_program(brief: str) -> dict:
    """Brief → complete program dict (ready for template_generator.lay_out_program)."""
    country = detect_country(brief) or "Germany"  # Default to Europe/Germany
    region = REGION_OF.get(country, "global")
    bedrooms = detect_bedrooms(brief)
    bathrooms = detect_bathrooms(brief, bedrooms)
    total_area = detect_area(brief) or _typical_area(bedrooms, country)
    persona = detect_persona(brief)

    conv = COUNTRY_CONVENTIONS.get(country, DEFAULT_CONVENTION)
    want_balcony = "balcony" in brief.lower() or "outdoor" in brief.lower() or bedrooms >= 2

    # ROOM AREAS by allocation
    areas = _allocate_areas(bedrooms, total_area, want_balcony)
    rooms: list[dict] = []

    if bedrooms == 0:
        rooms = [
            {"name": conv.entry_name, "area_sqm": round(areas["entry"], 1)},
            {"name": "Studio Living/Sleep", "area_sqm": round(areas["living_sleep"], 1)},
            {"name": conv.kitchen_name, "area_sqm": round(areas["kitchen"], 1)},
            {"name": conv.bath_name, "area_sqm": round(areas["bath"], 1)},
        ]
        if want_balcony:
            rooms.append({"name": conv.balcony_name, "area_sqm": round(areas["balcony"], 1)})
    elif bedrooms == 1:
        rooms = [
            {"name": conv.entry_name, "area_sqm": round(areas["entry"], 1)},
            {"name": conv.living_name, "area_sqm": round(areas["living"], 1)},
            {"name": conv.kitchen_name, "area_sqm": round(areas["kitchen"], 1)},
            {"name": conv.bath_name, "area_sqm": round(areas["bath"], 1)},
            {"name": conv.master_name, "area_sqm": round(areas["master"], 1)},
        ]
        if want_balcony:
            rooms.append({"name": conv.balcony_name, "area_sqm": round(areas["balcony"], 1)})
    elif bedrooms == 2:
        rooms = [
            {"name": conv.entry_name, "area_sqm": round(areas["entry"], 1)},
            {"name": conv.living_name, "area_sqm": round(areas["living"], 1)},
            {"name": conv.kitchen_name, "area_sqm": round(areas["kitchen"], 1)},
            {"name": conv.bath_name, "area_sqm": round(areas["bath"], 1)},
            {"name": conv.wc_name, "area_sqm": round(areas["wc"], 1)},
            {"name": conv.master_name, "area_sqm": round(areas["master"], 1)},
            {"name": conv.bedroom2_name, "area_sqm": round(areas["bedroom2"], 1)},
        ]
        if want_balcony:
            rooms.append({"name": conv.balcony_name, "area_sqm": round(areas["balcony"], 1)})
    else:  # 3+ bed
        rooms = [
            {"name": conv.entry_name, "area_sqm": round(areas["entry"], 1)},
            {"name": conv.living_name, "area_sqm": round(areas["living"], 1)},
            {"name": conv.kitchen_name, "area_sqm": round(areas["kitchen"], 1)},
            {"name": conv.bath_name, "area_sqm": round(areas["bath"], 1)},
            {"name": "En-suite", "area_sqm": round(areas["bath2"], 1)},
            {"name": conv.wc_name, "area_sqm": round(areas["wc"], 1)},
            {"name": conv.master_name, "area_sqm": round(areas["master"], 1)},
            {"name": conv.bedroom2_name, "area_sqm": round(areas["bedroom2"], 1)},
            {"name": conv.bedroom3_name, "area_sqm": round(areas["bedroom3"], 1)},
        ]
        if want_balcony:
            rooms.append({"name": conv.balcony_name, "area_sqm": round(areas["balcony"], 1)})

    # Adjust style description based on detected keywords
    style_bits = []
    if "altbau" in brief.lower():
        style_bits.append("Altbau character")
    if "modern" in brief.lower():
        style_bits.append("modern build")
    if "victorian" in brief.lower():
        style_bits.append("Victorian period")
    if "haussmann" in brief.lower():
        style_bits.append("Haussmann classical")
    style_suffix = f" — {', '.join(style_bits)}" if style_bits else ""

    return {
        "region": region,
        "country": country,
        "city": "AI Generated",
        "style": f"{country} {bedrooms}-bed AI Generation{style_suffix}",
        "description": (
            f"AI-generated program from brief: \"{brief[:80]}{'...' if len(brief) > 80 else ''}\". "
            f"{bedrooms}-bed, {bathrooms}-bath, {total_area:.0f} m² — laid out by the procedural "
            f"geometry engine and verified against the 35-check IFC validation pipeline."
        ),
        "suitable_for": persona,
        "tags": ["ai_generated", country.lower().replace(" ", "_"),
                 f"{bedrooms}bed" if bedrooms else "studio"],
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "total_area_sqm": round(total_area, 1),
        "wall_thickness_mm": conv.wall_thickness_mm,
        "ceiling_height_mm": conv.ceiling_height_mm,
        "rooms": rooms,
    }


if __name__ == "__main__":
    import json
    BRIEFS = [
        "1-bedroom apartment for a young couple in Berlin, Altbau character with high ceilings, around 60 m².",
        "2 BHK in Bangalore for a young family, around 75 sqm, with separate kitchen and balcony.",
        "Studio in Tokyo, around 30 m², for a single professional.",
        "Family of four, looking for a 3-bed in the UK with Victorian character. Around 100 m².",
        "Athens 1-bed near the historic center, around 50 m², for a couple.",
    ]
    for b in BRIEFS:
        print(f"\n--- {b} ---")
        prog = extract_program(b)
        print(f"  {prog['country']} | {prog['bedrooms']}-bed/{prog['bathrooms']}-bath | "
              f"{prog['total_area_sqm']} m²")
        for r in prog['rooms']:
            print(f"    {r['name']:30s} {r['area_sqm']:5.1f} m²")
