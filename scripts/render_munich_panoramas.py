#!/usr/bin/env python3
"""Generate BIM-conditioned panoramic per-room photos for the Munich Schwabing
1-Zimmer template, used by the HyperFrames floor-plan tour.

Each photo is rendered at 1536×768 (panoramic 2:1 aspect) so the broker-mode
tour's horizontal pan animation has actual room width to sweep across — not
the same 16:9 frame cropped twice.

Source of truth: data/templates/europe/eu_de_1bed_munich_schwabing.json
Output: experiments/hyperframes-bim-floorplan-tour/assets/room_<name>.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root (scripts/ → ..)
sys.path.insert(0, str(ROOT))

from backend.app.image_renderer import render_from_prompt  # noqa: E402

TEMPLATE_PATH = ROOT / "data/templates/europe/eu_de_1bed_munich_schwabing.json"
ASSETS_DIR = ROOT / "experiments/hyperframes-bim-floorplan-tour/assets"

# Panoramic dimensions: SDXL-turbo handles 1536×768 well (close to native 1024×1024
# in pixel count, just stretched to 2:1). Higher than 2048×1024 starts to ghost.
WIDTH, HEIGHT = 1536, 768
STEPS = 4  # SDXL-turbo gets sharp results at 2-4 steps

# Stylistic anchors — every room shares these so the 5 photos read as ONE
# apartment, not 5 different homes.
STYLE_ANCHOR = (
    "München Schwabing Jugendstil Altbau 1900-1925, period mouldings, "
    "stucco ornament, oak herringbone parquet, tall tilt-and-turn windows, "
    "soft Munich daylight, refined neutral palette with warm whites"
)

# Per-room prompt overrides. Each prompt is built from the BIM polygon
# (room dims), windows, and the room's narrative role in the tour.
ROOM_PROMPTS = {
    "r_flur": {
        "filename": "room_flur.png",
        "prompt": (
            f"Photorealistic panoramic interior photograph, wide horizontal view, "
            f"of a small Altbau entry foyer, 4 square meters (2m × 2m), "
            f"3.1 meter ornate stucco ceiling with central pendant lantern, "
            f"oak parquet floor, white panelled walls, double doors at far end leading "
            f"to the living room, period brass coat hooks, framed art on side wall, "
            f"console table with antique lamp. {STYLE_ANCHOR}. "
            f"Wide-angle lens, sharp focus, magazine quality, no people."
        ),
    },
    "r_wohnzimmer": {
        "filename": "room_wohnzimmer.png",
        "prompt": (
            f"Photorealistic panoramic interior photograph, wide horizontal view, "
            f"of a Munich Schwabing Jugendstil living room, 15 square meters (5m × 3m), "
            f"3.1 meter ornate ceiling with stucco rose and crystal chandelier, "
            f"oak herringbone parquet, "
            f"two tall French windows on the south wall (1.8m wide) overlooking the "
            f"Englischer Garten with linden trees in soft afternoon light, "
            f"period mouldings on cornice and dado, comfortable cream sofa, "
            f"walnut coffee table, vintage Persian rug, framed prints on wall. "
            f"{STYLE_ANCHOR}. Wide-angle, sharp focus, no people."
        ),
    },
    "r_kueche": {
        "filename": "room_kueche.png",
        "prompt": (
            f"Photorealistic panoramic interior photograph, wide horizontal view, "
            f"of a classic Munich kitchen, 9 square meters (3m × 3m), "
            f"3.1 meter ceiling with surface-mounted period fixture, "
            f"oak parquet floor, custom cream cabinetry with brass handles, "
            f"white subway tile backsplash, marble countertop, integrated appliances, "
            f"single tilt-and-turn window (1.2m wide) on the south wall above the sink "
            f"with cafe curtains, small breakfast nook in the corner. "
            f"{STYLE_ANCHOR}. Wide-angle, sharp focus, no people."
        ),
    },
    "r_bad": {
        "filename": "room_bad.png",
        "prompt": (
            f"Photorealistic panoramic interior photograph, wide horizontal view, "
            f"of a period Munich Altbau bathroom, 6 square meters (2m × 3m), "
            f"3.1 meter ceiling, hexagonal black-and-white encaustic tile floor, "
            f"white subway tile wainscot, freestanding cast-iron clawfoot bathtub on "
            f"the far wall, period pedestal sink with brass fittings, "
            f"vintage gilded mirror above sink, narrow obscured-glass window (0.6m wide) "
            f"with frosted texture letting in soft diffused light, towel ladder, "
            f"small marble shelf with rolled towels. {STYLE_ANCHOR}. "
            f"Wide-angle, sharp focus, no people."
        ),
    },
    "r_schlafzimmer": {
        "filename": "room_schlafzimmer.png",
        "prompt": (
            f"Photorealistic panoramic interior photograph, wide horizontal view, "
            f"of a Munich Schwabing master bedroom, 16 square meters (8m × 2m), "
            f"3.1 meter ceiling with ornate stucco coving, oak herringbone parquet, "
            f"queen bed with white linen and walnut headboard centered on the back wall, "
            f"two matching nightstands with brass lamps, tall French window (1.5m wide) "
            f"on the north wall overlooking a quiet courtyard with morning light, "
            f"vintage armoire on the side wall, persian rug under the bed. "
            f"{STYLE_ANCHOR}. Wide-angle, sharp focus, no people."
        ),
    },
}


def main() -> int:
    template = json.loads(TEMPLATE_PATH.read_text())
    rooms = {r["id"]: r for r in template["rooms"]}

    # Sanity: every prompt key matches a real room
    for room_id in ROOM_PROMPTS:
        assert room_id in rooms, f"Unknown room {room_id!r} in prompts"

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Render each room
    for room_id, cfg in ROOM_PROMPTS.items():
        out_path = ASSETS_DIR / cfg["filename"]
        room = rooms[room_id]
        print(f"\n[{room['name']:<14}] {WIDTH}×{HEIGHT}  {STEPS} steps")
        print(f"  prompt: {cfg['prompt'][:120]}...")
        t0 = time.time()
        result = render_from_prompt(
            cfg["prompt"], width=WIDTH, height=HEIGHT, steps=STEPS,
        )
        dt = time.time() - t0
        if result.error:
            print(f"  ERROR ({dt:.1f}s): {result.error}")
            return 1
        out_path.write_bytes(result.image_bytes)
        print(f"  wrote {out_path.name} ({len(result.image_bytes)//1024} KB) in {dt:.1f}s")

    print(f"\nAll 5 panoramic room photos written to {ASSETS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
