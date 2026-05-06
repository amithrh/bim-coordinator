# BIM Coordinator — Phase 1 (Walking Skeleton)

Voice-driven AI architect for the Nemetschek ecosystem. Phase 1 is a static demo: text prompt → 4 floor plan cards → detail view (2D + 3D) → adjustment sliders.

See `Day-Zero-Build-Plan-v3.2.docx` for the full spec.

## Template Library

**500 hand-curated templates, all 36/36 IFC-verified.** Real-listing-grounded room programs with cited source URLs spanning 50+ countries across Europe, India, USA, Asia-Pacific, Africa, MENA, and Latin America.

### Coverage (50+ countries)

| Region | Count | |
|---|---|---|
| Germany | 56+ | Altbau (Wilhelminian/Gründerzeit/Jugendstil), Plattenbau (P2/M10/WBR83/WBS70), Neubau, Maisonette, Stadtvilla, Bauhaus, Friesenhaus, Bremerhaus, Dachgeschoss, Souterrain, Hochparterre, Gartengeschoss, Bremen Tabakquartier Loft, Erkrath 70er Bungalow, NURDA Stadthaus S-150/S-190, Bauhaus B-140, Pultdachbungalow |
| United Kingdom | 36+ | Victorian terrace, Edwardian semi, Georgian (Edinburgh/Bath/Brighton), Tenement (Glasgow), Welsh slate cottage, Welsh longhouse, Cornish fisherman, Northumbrian stone, Lake District slate, Devon Dartmoor longhouse, Tudor mock, Surrey detached, Suffolk Rectory, Yorkshire Hebden Bridge, Hampstead Edwardian villa, Liverpool wharf, Bristol harbourside, Belfast Edwardian, Cardiff bay-window, Newcastle Tyneside flat, Sheffield student, Brighton Regency, Modernist Span, Didcot eco, Kent Oast, Manchester Detached, London Whittington 1960s, London Vauxhall conv |
| India | 25+ | Vastu+Pooja (Chennai T-Nagar, Kolkata, Bangalore Sarjapur), NE-corner pooja, SW Master, SE Kitchen, Servant quarter (Gurgaon DLF, Mumbai Bandra, Hyderabad Banjara, Bangalore), Walk-in wardrobe, Dry/service balcony, Modular kitchen, Kerala thinnai+nadumuttam, Bengali verandah, Gujarati otla utility, Mumbai Powai/Andheri/Bandra, Delhi Lajpat, Hyderabad Madhapur/Gachibowli, Pune Wakad/Hinjewadi/Kharadi/NIBM, Bangalore Whitefield 1/2/3 BHK |
| France | 11 | Haussmannien (3p+5p), Echoppe Bordeaux (simple+étage), Lyon Canut, Marseille Trois Fenêtres+Le Panier, Strasbourg Alsacienne, Brittany Longère, Annecy Chalet, Provence Mas, Lille Maison de ville, Toulouse, Paris Marais |
| Italy | 8 | Milano Liberty, Milano compatto, Roma Trastevere, Trullo, Firenze palazzo, Napoli QS, Sicily Palazzotto, Masseria Pugliese |
| Netherlands | 4 | Amsterdam Jordaan trapgevel (multi-floor), Amsterdam compact, Utrecht Jaren-30, Rotterdam Kralingen, Friese Stelpboerderij |
| Austria | 3 | Wien Gründerzeit, Salzburg Bürgerhaus, Tirolerhaus |
| Spain | 3 | Madrid Lavapiés Corrala, Valencia Modernista, Andalucía Cortijo |
| USA | 3 | NYC Tenement, Suburban Ranch, Suburban Colonial |
| Switzerland | 2 | Zürich Altbau, Engadinerhaus |
| Norway | 2 | Hytte (fjord cabin), Oslo Bygård |
| Sweden | 2 | Sommarstuga, Stockholm Sekelskifte |
| Denmark | 1 | København Klassisk |
| Ireland | 1 | Dublin Georgian Terrace |
| Poland | 1 | Warsaw Kamienica |
| Australia | 1 | Sydney Victorian Terrace |
| Japan | 1 | Tokyo 1LDK Mansion |
| Portugal | 2 | Lisbon Pombaline, Porto townhouse |
| Greece | 1 | Athens Neoclassical |
| Belgium | 1 | Brussels Maison de Maître |
| Czech Republic | 1 | Prague pavlač |
| Finland | 1 | Helsinki kerrostalo (with sauna) |
| Hungary | 1 | Budapest Neoklassicista |
| Turkey | 1 | Istanbul Bosphorus |
| Russia | 1 | Moscow Stalinka |
| Slovakia | 1 | Bratislava Panelák |
| Cyprus | 1 | Limassol coastal |
| Estonia | 1 | Tallinn old town |
| Latvia | 1 | Riga Jugendstil |
| Lithuania | 1 | Vilnius Baroque |

### Size distribution

15 studios | 25 1-bed | 49 2-bed | 34 3-bed | 18 4-bed | 6 5+ bed

### Distinctive cultural features captured

Vastu compliance (NE pooja / SW master / SE kitchen), pooja rooms, servant quarters (4'×5' WC standard), dry/service balconies, modular kitchen vs traditional split, walk-in wardrobes, Kerala thinnai+nadumuttam courtyard, Bengali verandah, Gujarati otla utility verandah, Kachelofen tile stoves, Norwegian fjord hytte sleeping loft, Japanese genkan+unit-bath separate from toilet, Mediterranean cortile/patio, English inglenook+bay window+conservatory, Lithuanian/Latvian baroque ornament, Finnish in-unit sauna, Russian wide stalinka balcony.

### Multi-floor templates

7 multi-floor templates (Stuttgart Maisonette, Bordeaux Echoppe à étage, Amsterdam Jordaan trapgevel, Bangalore villa duplex, Munich Maisonette, Hannover Doppelhaushälfte, Kerala Kochi).

## Quick start

```bash
# 1. Activate venv (already created)
source .venv/bin/activate

# 2. Build all templates → IFC + SVG + .frag
make all

# 3. Backend
make backend     # http://localhost:8000

# 4. Frontend (separate terminal)
make frontend    # http://localhost:3000
```

## Pipeline

Every template passes a strict validation + verification pipeline:

- **Validate** (`scripts/validate_template.py`) — JSON schema, geometry (polygon containment, room overlap, area-sum tile, hole detection), door/window edge-host check
- **Build IFC** (`scripts/build_template.py`) — Segmented-walls IFC4 with IfcSpace via aggregate, IfcWall via spatial.assign_container, METRE units, walls placed at p1, doors+windows centered on host edges with edge detection
- **Render SVG** (`scripts/render_svg.py`) — 2D plan with furniture icons, door swing arcs, dimensions, palette
- **Verify** (`scripts/verify_ifc.py`) — 35-check verification (schema, units, spatial hierarchy, entity counts, world-coord rendering, boundary fit, z-range, distinct wall placements)

## Repo layout

```
bim-coordinator/
├── data/
│   ├── _schema.json              # Template JSON schema
│   ├── tag_vocabulary.json       # Controlled tag vocabulary (10 facets)
│   ├── templates/
│   │   ├── europe/               # 322 European templates
│   │   ├── india/                # 74 Indian templates
│   │   └── global/               # 104 global templates (US/JP/AU/etc.)
│   ├── build/                    # Generated IFCs
│   ├── svg/                      # Generated SVGs
│   └── research/                 # Research source data (.md files)
│       ├── de_research.md        # 49 German templates source
│       ├── uk_research.md        # 40 UK templates source
│       ├── eu_other_research.md  # 35 EU other templates source
│       ├── india_research.md     # 18 India templates source
│       └── global_research.json  # 10 global templates source
├── scripts/
│   ├── build_template.py
│   ├── validate_template.py
│   ├── verify_ifc.py
│   ├── render_svg.py
│   ├── tag_audit.py              # Verify templates against vocabulary
│   ├── backfill_tags.py          # One-shot tag backfill helper
│   └── preview_footprint.py
├── backend/                      # FastAPI surface
└── frontend/                     # Next.js + Three.js
```

See §4 of the build plan for full architecture.
