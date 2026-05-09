# HyperFrames tour pipeline — image generation, composition, render

This is how a BIM template becomes a 47-second vertical broker tour MP4.
End-to-end runtime: **~50 seconds per template** on M-series Apple Silicon.

The pipeline has three stages:

```
  data/templates/<region>/<id>.json
            │
            ▼
  ┌─────────────────────────────────────┐
  │  STAGE 1 — image generation         │     SDXL-turbo + diffusers
  │  scripts/render_template_tour.py    │     (per-room panoramic photos)
  │      ↳ 5×6s per template            │
  └─────────────────────────────────────┘
            │  emits  experiments/.../assets/room_<id>.png
            ▼
  ┌─────────────────────────────────────┐
  │  STAGE 2 — composition emission     │     Python string-template patcher
  │  scripts/render_template_tour.py    │     against the seed Munich HTML
  │      ↳ <1s                          │
  └─────────────────────────────────────┘
            │  emits  experiments/<exp>/index.html + meta.json + hyperframes.json
            ▼
  ┌─────────────────────────────────────┐
  │  STAGE 3 — HTML→MP4 render          │     `npx hyperframes render`
  │  HyperFrames headless Puppeteer     │     (1080×1920, 30 fps, 47 s)
  │      ↳ ~17 s per template           │
  └─────────────────────────────────────┘
            │  emits  out/<id>_tour.mp4
            ▼
  frontend/public/<id>_tour.mp4   →   /tours playlist page (Next.js)
```

## Inputs

A single BIM template JSON file under `data/templates/<region>/<id>.json`
provides everything the pipeline needs. Example shape (from the
Munich Schwabing 1-Zimmer template):

```json
{
  "id": "eu_de_1bed_munich_schwabing",
  "metadata": {
    "country": "Germany",
    "city_inspiration": "Munich",
    "size_label": "1-Zimmer-Wohnung",
    "total_area_sqm": 50,
    "bedrooms": 1,
    "style": "München Schwabing 1-Zimmer (Jugendstil Universitätsviertel)"
  },
  "boundary": {
    "polygon": [[0,0],[10,0],[10,5],[0,5]],
    "ceiling_height_mm": 3100
  },
  "rooms": [
    { "id": "r_flur", "name": "Flur", "type": "entry",
      "polygon": [[0,0],[2,0],[2,2],[0,2]], "area_sqm": 4 },
    { "id": "r_wohnzimmer", "name": "Wohnzimmer", "type": "living",
      "polygon": [[2,0],[7,0],[7,3],[2,3]], "area_sqm": 15 },
    ...
  ],
  "doors": [
    { "from": "outside", "to": "r_flur", "position": [0,1],
      "is_main_entry": true }
  ],
  "windows": [
    { "room": "r_wohnzimmer", "position": [4.5,0], "width_mm": 1800 },
    ...
  ]
}
```

Every visual decision in the tour (room layout on the floor plan, the
photo content, the ENTRY marker position, the room-area captions) is
**derived from this JSON** — no hand-tuning per template.

## Stage 1 — image generation (SDXL-turbo)

### Model choice — SDXL-turbo, not Flux

We use **`stabilityai/sdxl-turbo`** (Stability AI's distilled SDXL,
released Nov 2023). Configured at `backend/app/image_renderer.py:53`:

```python
_DEFAULT_MODEL = os.getenv("BIM_RENDER_MODEL", "stabilityai/sdxl-turbo")
```

The env var lets you swap models without code changes, but the default —
and what every render in the demo runs against — is SDXL-turbo.

**Why not Flux.1 or vanilla SDXL?**

| Property | SDXL-turbo (in use) | Flux.1-dev | Vanilla SDXL |
|---|---|---|---|
| Inference steps | **2–4** | 25–50 | 25–50 |
| Latency per 1536×768 on M4 Max | **~5–6s** warm | ~30–60s | ~10–20s |
| License | OpenRAIL++ | Non-commercial (`dev` variant) | OpenRAIL++ |
| Parameter count | 3.5 B (UNet) | 12 B | 3.5 B |

A full bake of all 7 templates is **35 panoramas** (5 rooms × 7 cities).
At Flux speeds that's 17–35 min per bake; at SDXL-turbo it's ~3 min. The
quality delta at 1536×768 is real but not enough to justify a 10×
slowdown for the iteration loop. **No Flux model is loaded anywhere in
the codebase** (verified by grep).

**Optional Depth ControlNet** (`diffusers/controlnet-depth-sdxl-1.0-small`,
defined at `image_renderer.py:55`) is wired for a separate
`render_faithful_from_template()` path that conditions on a BIM-dollhouse
depth map. **The broker-tour pipeline deliberately does NOT use
ControlNet** because we want freeform interior compositions per room,
not strict geometric matching, and ControlNet adds ~2× to render
latency.

### Why panoramic 1536×768

SDXL-turbo is trained at 1024×1024 native resolution but handles 1536×768
(2:1 aspect) well in 4 inference steps. We use 2:1 aspect because
**the broker-tour photo frame animates a horizontal pan across each photo
at runtime** (alternating L→R and R→L per stop, scale 1.45×). A wider
source image means the pan reveals genuinely different parts of the room
at t=4s vs t=8s — which is what sells the "360° look" feel.

A square 1024×1024 panned the same way would just zoom in and out on the
same content. We tried that first; it didn't read as "looking around."

### Prompt construction (per room)

`scripts/render_template_tour.py::build_room_prompt()` builds each prompt
deterministically from the BIM template:

```
Photorealistic panoramic interior photograph, wide horizontal view, of
{ROOM_TYPE_PROMPT[room.type]}, approximately {area_sqm} square meters
({width_m}m × {height_m}m), {ceiling_m} meter ceiling{window_count_clause}.
{COUNTRY_STYLE_ANCHOR}. Wide-angle, sharp focus, no people.
```

Concretely:

| Component | Source | Example for Munich Wohnzimmer |
|---|---|---|
| `ROOM_TYPE_PROMPT[room.type]` | `ROOM_TYPE_PROMPT` dict in script (12 room types) | "a spacious living room with sofa, coffee table, framed art and refined furnishings" |
| `area_sqm` | `template.rooms[i].area_sqm` | 15 |
| `width_m × height_m` | computed from `template.rooms[i].polygon` bounding box | 5m × 3m |
| `ceiling_m` | `template.boundary.ceiling_height_mm / 1000` | 3.1 |
| `window_count_clause` | `len(template.windows where room == this)` | ", 2 large windows letting in soft daylight" |
| `COUNTRY_STYLE_ANCHOR` | `STYLE_ANCHORS[template.metadata.country]` | "Paris Marais Haussmannian, herringbone parquet, tall French windows, period mouldings, refined neutrals" |

This is the BIM-conditioning that prevents "generic stock photo" output.
A 2 m × 2 m Flur with 3.1 m ceiling and 0 windows generates a *small,
tall foyer*, not a random hallway shot. A 5 m × 3 m Wohnzimmer with
2 windows generates a *wide living room with two windows*. The window
count surfaces in both the prompt AND the photo caption ("15 m² · living
room · 2 windows"), so plan and photo stay coherent.

### Country style anchor

Eight country anchors are defined in `scripts/render_template_tour.py`:

```python
STYLE_ANCHORS = {
    "Germany":             "Munich Schwabing Jugendstil Altbau, oak parquet, period mouldings, soft daylight",
    "France":              "Paris Marais Haussmannian, herringbone parquet, tall French windows, period mouldings, refined neutrals",
    "India":               "Bangalore vastu-compliant apartment, polished granite floor, neutral walls, contemporary Indian design",
    "Japan":               "Tokyo modern minimalist, light wood floor, white walls, low furniture, paper-shaded lighting, Nordic-Japanese",
    "United Arab Emirates":"Dubai Marina luxe modern, marble floor, neutral palette, floor-to-ceiling windows, skyline view",
    "United States":       "NYC tenement-style apartment, hardwood floor, white walls, exposed brick accents, contemporary",
    "Australia":           "Sydney modern coastal, polished concrete or oak floor, abundant daylight, indoor-outdoor flow",
}
```

Each is **deliberately ≤25 words** because CLIP's tokenizer (used by SDXL)
truncates at 77 tokens. The room-specific detail at the *start* of the
prompt always survives; the style suffix may be partially clipped on
verbose rooms, which is fine — the room geometry is what carries the photo.

### Renderer call

The actual SDXL invocation is in `backend/app/image_renderer.py`:

```python
from backend.app.image_renderer import render_from_prompt
result = render_from_prompt(prompt, width=1536, height=768, steps=4)
out_path.write_bytes(result.image_bytes)
```

Backed by `diffusers.AutoPipelineForText2Image` with
`stabilityai/sdxl-turbo` (fp16 on MPS / CUDA). Cold load is ~10s; warm
inference at 1536×768 is **~5.5 seconds per panorama**. Five per template
= ~28 seconds of image generation per tour.

### Why not ML-based scene generation?

We considered Holodeck / AnyHome / SceneTeller (CVPR/ECCV 2024 papers).
They produce 3D scenes from text but require GPT-4 + Unity + Objaverse and
are not browser-deployable in 2026. SDXL panoramas + a horizontal pan
animation gives ~80% of the "look around the room" feel at <1% of the
deployment cost. See the IFC→Browser-Walkable strategic research doc §C
("Furniture & material auto-placement").

## Stage 2 — composition emission

`scripts/render_template_tour.py::emit_index_html()` reads the seed
Munich composition (`experiments/hyperframes-bim-floorplan-tour/index.html`)
and patches six chunks:

1. **Header** — city, BR title + accent word (Jugendstil/Haussmann/Vastu/...), area + room count + ceiling
2. **Floor plan rooms** — one `<div class="room">` per BIM room, positioned by polygon bbox (CSS-y is inverted because BIM north grows up, CSS top grows down)
3. **ENTRY marker** — class `.entry-{south,north,west,east}` decided by which boundary edge the BIM main entry door sits on; pill auto-positioned and arrow rotated to point INTO the plan
4. **Photo frames** — one `<div class="photo-frame">` per stop with `<img src="assets/room_<id>.png">`, the `360° look` pill, and a 3-line caption ("Stop X / N · ROLE", `<name>`, `area · type · window_count`)
5. **Outro** — city + BR title + total room count + area
6. **`STOPS` JS array** — `[{id, x, y, photoId, roomId}, ...]` with x/y as %-of-canvas centroids; drives the GSAP timeline at runtime

Stop selection is canonical: at most 5 in priority order
`entry → corridor → living → kitchen → bathroom → master_bedroom → bedroom → balcony → dining`.
A 9-room Sydney apartment shows 5 stops; a 4-room NYC tenement shows 4.

Output: a per-template directory `experiments/hyperframes-bim-tour-<id>/`
containing a self-contained `index.html`, `meta.json`, and
`hyperframes.json` that HyperFrames can render directly.

## Stage 3 — HyperFrames render

[HyperFrames](https://github.com/heygen-com/hyperframes) is HeyGen's
HTML→video renderer. It launches headless Chromium via Puppeteer, seeks
a paused GSAP timeline (`window.__timelines.main`) frame-by-frame at
1080×1920 / 30 fps for the duration declared in
`<div data-duration="47">`, captures each frame, and encodes to MP4.

```bash
cd experiments/hyperframes-bim-tour-<id>
npx hyperframes render . --output out/<id>_tour.mp4 --quality draft
```

Render time: **~17 seconds** for a 47-second tour on M4 Max (6 parallel
Chromium workers). Output is ~10–15 MB per tour (H.264 / MP4, no audio).

### Critical HTML idioms inside the composition

A few non-obvious gotchas, all baked into the seed `index.html` and
inherited by every template:

- **Timeline must be paused.** HyperFrames seeks the GSAP timeline per
  frame; if it auto-plays you get duplicate / dropped frames.
  ```js
  const tl = gsap.timeline({ paused: true });
  ...
  window.__timelines["main"] = tl;
  ```

- **No `vector-effect: non-scaling-stroke` on dasharray-animated paths.**
  When non-scaling-stroke is set, `stroke-dasharray` is interpreted in
  *screen pixels* but `getTotalLength()` returns user-space units →
  the dash math goes wrong and the path renders mid-segment dashes
  even when "fully hidden". (We hit this on the original walker-trail
  experiment; the broker-mode rebuild dropped the trail entirely.)

- **`tl.set()` has `immediateRender: true` by default.** If you use
  `set()` to pin start values for a sequence of GSAP tweens on the same
  property, all the sets fire at script load and corrupt the initial
  state. Use zero-duration `tl.to({..., immediateRender: false})`
  instead when you need a "pin" inside the timeline.

- **GSAP CSS-variable tweens** (`--rot`, etc.) work, but require the var
  to be declared on the target element with a default value, otherwise
  the first tween snaps instead of interpolating.

These are documented as comments inside `experiments/hyperframes-bim-floorplan-tour/index.html`.

## What lands in the repo

Source (committed):

- `scripts/render_munich_panoramas.py` — one-shot for the seed Munich panoramas
- `scripts/render_template_tour.py` — generalised pipeline; `python … <template_id>`
- `experiments/hyperframes-bim-floorplan-tour/{index.html, meta.json, hyperframes.json}` — seed composition; the parameterised renderer reads index.html and patches it per template
- `frontend/app/tours/page.tsx` — Next.js playlist UI at `/tours`
- `docs/HYPERFRAMES_TOUR_PIPELINE.md` — this file

Generated (gitignored, rebuild with the script):

- `experiments/hyperframes-bim-tour-<id>/` — per-template emitted compositions
- `experiments/*/assets/*.png` — SDXL room panoramas
- `experiments/*/out/*.mp4` — local HyperFrames render outputs
- `frontend/public/*_tour.mp4` — tour MP4s served by Next.js
- `frontend/public/tour_thumbs/*.jpg` — playlist page thumbnails (extracted from MP4s with ffmpeg)

To rebuild any tour from scratch:

```bash
# 1. Generate panoramas + composition + render the MP4
.venv/bin/python scripts/render_template_tour.py eu_fr_1bed_paris_marais

# 2. Make a thumbnail for the playlist
ffmpeg -ss 12 -i frontend/public/eu_fr_1bed_paris_marais_tour.mp4 \
       -frames:v 1 -vf scale=540:-1 -q:v 4 \
       frontend/public/tour_thumbs/eu_fr_1bed_paris_marais.jpg

# 3. (If new template) add an entry to TOURS in frontend/app/tours/page.tsx
```

## Where this fits in the bigger product roadmap

The HyperFrames tour pipeline produces **marketing assets** — vertical
MP4 reels suitable for WhatsApp / Instagram / portal embeds. It is
**not** the productionised "browser-walkable BIM tour" that the
strategic research roadmap (see PDF) calls for in Stage 1. That product
needs Three.js + `@thatopen/components` + `@thatopen/fragments` + Rapier
physics on top of the same template library — a separate workstream
estimated at 12.5 person-weeks for one region.

The two pipelines share the same input contract (`data/templates/<region>/<id>.json`),
so a builder who clicks "generate web tour" inside Allplan PythonParts
should eventually be able to fan out to *both* the marketing reel and the
walkable 3D tour from a single IFC export.

## 3D / splat path (reference, currently hidden)

There is a separate 3D Gaussian-splat experiment under
`backend/app/splat_*` and `scripts/render_splat_dataset.py` /
`scripts/train_splat.py` that trains a Splat from SDXL multi-view renders
of a BIM mesh. It works but the multi-view consistency of SDXL views
isn't tight enough for a clean splat — output looked muddy on the
demo, so the "🪐 Splat walk" tab is hidden behind a `?splat` URL flag
in `frontend/components/Walkthrough.tsx`. This is *not* the route
forward for the productisation roadmap; the strategic research doc
explicitly recommends `@thatopen/fragments` over splat-based capture
because Fragments preserves the IFC semantics (room types, areas,
window positions) that the rule-based furniture engine needs. The splat
path is left in the repo as a research artifact, not a demo asset.
